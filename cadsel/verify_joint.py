"""Pre-flight checks for the joint WLE+EUS pipeline.

Every check here corresponds to a defect that has actually bitten this project, so
they are assertions rather than comments. Run before launching any joint training:

    python -m cadsel.verify_joint
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .config import MODALITY_INDEX, MODALITY_NAMES, MODALITY_STATS, TASK_CLASSES
from .data import (
    CadSelDetectionDataset,
    ModalityBatchSampler,
    build_joint_manifest,
    build_joint_dataloaders,
    collate_fn,
    compute_class_weights,
    joint_strata_key,
    make_normalizer,
    patient_level_split,
    read_yolo_label,
    split_summary,
)
from .engine import _batch_modality
from .models import build_joint_detection_model, get_active_modality, set_active_modality


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.n = 0

    def check(self, name: str, condition: bool, detail: str = "") -> bool:
        self.n += 1
        if condition:
            print(f"  [ok]   {name}" + (f" — {detail}" if detail else ""))
        else:
            print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
            self.failures.append(name)
        return condition


# ---------------------------------------------------------------------------
def check_split(c: Checks, samples, seed: int = 42) -> tuple[list, list]:
    """Split integrity — the defect that makes the existing baselines unusable here."""
    print("\n1. Shared patient-level split")
    train, val = patient_level_split(samples, seed=seed, strata_key=joint_strata_key)

    train_keys = {s.patient_key for s in train}
    val_keys = {s.patient_key for s in val}
    c.check("no patient straddles the split", not (train_keys & val_keys),
            f"{len(train_keys & val_keys)} overlapping")

    # The specific cross-modal form: a patient's WLE frames in train while its EUS
    # frames are in val. Measured at 67/85 on the per-modality baseline splits.
    cross = 0
    for modality, other in (("WLE", "EUS"), ("EUS", "WLE")):
        a = {s.patient_key for s in train if s.modality == modality}
        b = {s.patient_key for s in val if s.modality == other}
        cross += len(a & b)
    c.check("no cross-modal leak (WLE in train / EUS in val)", cross == 0,
            f"{cross} patients")

    summary = split_summary(train, val)
    for side in ("train", "val"):
        s = summary[side]
        print(f"         {side}: {s['images']} images / {s['patients']} patients "
              f"{s.get('by_modality')} paired={s.get('paired_patients')}")

    c.check("val holds paired patients for the fusion head",
            summary["val"].get("paired_patients", 0) >= 40,
            f"{summary['val'].get('paired_patients')} paired val patients")

    # Every class must be scoreable in val, or AP silently drops that class.
    for task in TASK_CLASSES:
        _, counts = compute_class_weights(val, task)
        c.check(f"every {task} class present in val", bool((counts > 0).all()),
                dict(zip(TASK_CLASSES[task], counts.astype(int).tolist())))
    return train, val


def check_normalisation(c: Checks) -> None:
    """Guards the double-/255 bug that made Exp 8 report all-zero detections."""
    print("\n2. Per-modality normalisation")
    normalize = make_normalizer("reference", per_modality=True)
    shared = make_normalizer("reference", per_modality=False)

    for modality in MODALITY_NAMES:
        mean, std = MODALITY_STATS[modality]
        # A synthetic uint8 batch drawn to match this modality's statistics.
        rng = np.random.default_rng(0)
        arr = np.stack([
            rng.normal(mean[ch], std[ch], size=(64, 64)) for ch in range(3)
        ])
        img = torch.from_numpy(
            np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        )

        out = normalize([img.clone()], modality)[0]
        ok_mean = abs(float(out.mean())) < 0.25
        ok_std = 0.7 < float(out.std()) < 1.4
        c.check(f"{modality}: per-modality norm centres near 0 / unit std",
                ok_mean and ok_std,
                f"mean={float(out.mean()):+.3f} std={float(out.std()):.3f}")

        # And demonstrate why it is needed: the shared statistics do not.
        bad = shared([img.clone()])[0]
        if modality == "EUS":
            c.check("EUS under shared stats is badly off-centre (why per-modality "
                    "norm exists)", abs(float(bad.mean())) > 1.0,
                    f"mean={float(bad.mean()):+.3f}")

        # uint8 input is the contract; a float input would divide by 255 twice.
        c.check(f"{modality}: normaliser did not mangle a uint8 input to ~0",
                float(out.abs().max()) > 0.1,
                f"max|x|={float(out.abs().max()):.3f}")


def check_batches(c: Checks, train, val, task: str = "multiclass") -> None:
    print("\n3. Modality-homogeneous batching")
    sampler = ModalityBatchSampler(train, batch_size=8, shuffle=True, seed=0)
    batches = list(iter(sampler))
    homogeneous = all(
        len({train[i].modality for i in b}) == 1 for b in batches
    )
    c.check("every batch is single-modality", homogeneous,
            f"{len(batches)} batches")
    c.check("__len__ matches what __iter__ yields", len(sampler) == len(batches),
            f"len={len(sampler)} iter={len(batches)}")

    # Successive epochs must differ, or shuffling silently does nothing.
    first, second = list(iter(sampler)), list(iter(sampler))
    c.check("successive epochs reshuffle", first != second)

    interleaved = sum(
        1 for a, b in zip(batches, batches[1:])
        if train[a[0]].modality != train[b[0]].modality
    )
    c.check("batch order interleaves the two domains", interleaved > len(batches) // 4,
            f"{interleaved} modality switches across {len(batches)} batches")

    # A deliberately mixed batch must raise, not silently pick one modality.
    ds = CadSelDetectionDataset(train, task, img_size=320, train=False)
    mixed = [train[i].modality for i in range(len(train))]
    i_wle = mixed.index("WLE")
    i_eus = mixed.index("EUS")
    _, t0 = ds[i_wle]
    _, t1 = ds[i_eus]
    try:
        _batch_modality([t0, t1])
        c.check("mixed-modality batch raises", False, "no exception")
    except ValueError:
        c.check("mixed-modality batch raises", True)

    c.check("target carries the modality",
            int(t0["modality"].item()) == MODALITY_INDEX["WLE"]
            and int(t1["modality"].item()) == MODALITY_INDEX["EUS"])


def check_conditioning(c: Checks, task: str = "multiclass") -> None:
    print("\n4. Modality conditioning")
    model = build_joint_detection_model(
        "fasterrcnn_mobilenet_320", task, conditioning="film",
        pretrained=False, verbose=False,
    )
    films = getattr(model.backbone, "modality_films", None)
    c.check("FiLM layers exist before the optimizer is built", films is not None
            and len(films) > 0, f"{0 if films is None else len(films)} levels")

    named = dict(model.named_parameters())
    film_params = [k for k in named if "modality_films" in k]
    c.check("FiLM parameters are reachable via named_parameters()",
            len(film_params) > 0, f"{len(film_params)} tensors")

    # Identity at init: an untrained conditioner must not perturb the pretrained
    # features, so a run starts from the modality-blind model.
    identity = all(
        torch.allclose(films[k].gamma, torch.ones_like(films[k].gamma))
        and torch.allclose(films[k].beta, torch.zeros_like(films[k].beta))
        for k in films
    )
    c.check("FiLM initialised to the identity", identity)

    # And it must actually differ per modality once the parameters diverge.
    model.eval()
    key = list(films)[0]
    with torch.no_grad():
        films[key].gamma[MODALITY_INDEX["EUS"]].fill_(2.0)
        x = torch.rand(1, 3, 320, 320)
        set_active_modality("WLE")
        wle_feat = model.backbone(x)[key]
        set_active_modality("EUS")
        eus_feat = model.backbone(x)[key]
        set_active_modality(None)
        blind_feat = model.backbone(x)[key]

    c.check("conditioning changes the features per modality",
            not torch.allclose(wle_feat, eus_feat),
            f"max|delta|={float((wle_feat - eus_feat).abs().max()):.4f}")
    # WLE's FiLM is still the identity, so the unconditioned path must match it
    # exactly — that is what makes `set_active_modality(None)` safe for the
    # feature extraction the fusion head does.
    c.check("modality None bypasses modulation entirely",
            torch.allclose(blind_feat, wle_feat),
            f"max|delta|={float((blind_feat - wle_feat).abs().max()):.2e}")
    c.check("EUS branch is the one that diverged",
            not torch.allclose(eus_feat, blind_feat))
    c.check("get_active_modality round-trips", get_active_modality() is None)


def check_loader_smoke(c: Checks, train, val, task: str = "multiclass") -> None:
    print("\n5. Loader smoke test (num_workers=0)")
    train_loader, val_loader = build_joint_dataloaders(
        train[:64], val[:32], task=task, img_size=320, batch_size=4,
        num_workers=0, input_mode="reference", seed=0,
    )
    normalize = make_normalizer("reference", per_modality=True)
    dtypes: set[torch.dtype] = set()
    stds: list[float] = []
    finite = True
    seen = Counter()
    for images, targets in train_loader:
        modality = _batch_modality(targets)
        seen[modality] += 1
        dtypes.add(images[0].dtype)
        out = normalize(images, modality)
        finite = finite and bool(torch.isfinite(out[0]).all())
        stds.append(float(out[0].std()))
        if sum(seen.values()) >= 8:
            break

    c.check("loader yields uint8 images (shared-memory IPC contract)",
            dtypes == {torch.uint8}, str(dtypes))
    c.check("normalised batches are finite and non-degenerate",
            finite and min(stds) > 0.05, f"min std={min(stds):.3f}")
    c.check("copy_paste is rejected for joint runs", _rejects_copy_paste(train, val, task))


def _rejects_copy_paste(train, val, task) -> bool:
    try:
        build_joint_dataloaders(train[:8], val[:8], task=task, img_size=320,
                                batch_size=2, num_workers=0,
                                augment_ops=["flip", "copy_paste"])
        return False
    except ValueError:
        return True


def check_unimodal_untouched(c: Checks) -> None:
    """The shared edits to data.py/engine.py must not disturb the unimodal path."""
    print("\n6. Unimodal path unchanged")
    from .data import build_dataloaders, build_manifest, patient_level_split as pls

    wle = build_manifest("WLE")
    tr, va = pls(wle, seed=42)
    # Default strata_key must reproduce the split the completed baselines used.
    import json
    ref = Path("runs/fasterrcnn_mobilenet__WLE__multiclass/split.json")
    if ref.exists():
        expected = set(json.loads(ref.read_text())["val_patients"])
        got = {s.patient_key for s in va}
        c.check("default split reproduces the completed baseline's split",
                got == expected,
                f"{len(got ^ expected)} differing patients")
    else:
        print("         (skipped: no reference split.json)")

    normalize = make_normalizer("reference")           # legacy one-arg form
    img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
    out = normalize([img])
    c.check("make_normalizer still accepts the one-argument call", len(out) == 1
            and bool(torch.isfinite(out[0]).all()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-conditioning", action="store_true",
                        help="skip the model checks (they build a detector)")
    args = parser.parse_args()

    c = Checks()
    samples = build_joint_manifest(data_root=args.data_root)
    print(f"joint manifest: {len(samples)} images "
          f"{dict(sorted(Counter(s.modality for s in samples).items()))}")

    train, val = check_split(c, samples, seed=args.seed)
    check_normalisation(c)
    check_batches(c, train, val)
    if not args.skip_conditioning:
        check_conditioning(c)
    check_loader_smoke(c, train, val)
    check_unimodal_untouched(c)

    print(f"\n{c.n - len(c.failures)}/{c.n} checks passed")
    if c.failures:
        print("failed: " + ", ".join(c.failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
