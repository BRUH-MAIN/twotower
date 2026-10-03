"""Experiment 36 -- patient-level counterfactual gate for the joint two-tower
WLE+EUS fusion model (`cadsel.joint_fusion_model.JointTwoTowerModel`,
FINDINGS.md §15). `27_classifier_counterfactual.py`'s protocol -- score a
checkpoint on `ablation_inputs.build_five_variants`'s 5 variants
(full/lesion/bg_mask/bg_patch/boxmap) and check whether lesion-removed
variants (bg_patch/boxmap) retain the model's accuracy -- was built for a
single-image, single-modality classifier and doesn't directly apply to a
patient-level, multi-frame, two-modality model. This adapts it: build the 5
variants separately for each modality's val frames, then swap BOTH towers to
the same variant simultaneously (so "boxmap" means neither tower sees any
real tissue) while keeping each patient's *frame selection* identical to a
normal eval pass (only pixel content changes, not which frames are used) --
so retention is comparable across variants for the same patient, and the
"full" variant should reproduce the checkpoint's own recorded validation
balanced accuracy as a sanity check.

    python -m experiments.src.36_joint_fusion_counterfactual --run-name \\
        joint_efficientnet_b0__binary__freeze-partial__fusion-gated__loss-weighted_ce__moddrop0.3__mf4__ep100__es20__s42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ablation_inputs import FIVE_VARIANTS, build_five_variants
from cadsel.config import MODALITY_NAMES
from cadsel.data import build_joint_manifest, joint_strata_key, patient_level_folds
from cadsel.fusion import patient_classes, patient_label
from cadsel.joint_fusion_model import JointTwoTowerModel, tower_normalizers
from cadsel.train_fusion import score

RESULTS_DIR = Path("experiments/results/joint_fusion_runs")
OUT_DIR = Path("experiments/results")


def _normalize_frames(images: dict, normalizers: dict, device: torch.device) -> dict:
    out = {}
    for m in MODALITY_NAMES:
        x = images[m].to(device, non_blocking=True)
        B, T, C, H, W = x.shape
        flat = normalizers[m](x.reshape(B * T, C, H, W).float())
        out[m] = flat.reshape(B, T, C, H, W)
    return out


def build_frame_selection(patient_keys, by_patient_modality, max_frames, seed):
    """patient_key -> modality -> list of chosen Sample objects (len max_frames,
    or [] if the patient has no frames in that modality). Reuses
    ``JointPatientDataset``'s exact ``random.Random((seed*1_000_003)^index)``
    scheme so this reproduces the same frame selection a normal eval pass
    would use -- swapping only pixel content, never which frames are picked.
    """
    selection = {}
    for index, pkey in enumerate(patient_keys):
        rng = random.Random((seed * 1_000_003) ^ index)
        sel = {}
        for modality in MODALITY_NAMES:
            pool = by_patient_modality[pkey].get(modality, [])
            sel[modality] = [pool[rng.randrange(len(pool))] for _ in range(max_frames)] if pool else []
        selection[pkey] = sel
    return selection


def pack_variant_batch(patient_keys, selection, variant_arrays, modality_idx,
                        variant_name, max_frames, img_size, device):
    images, masks = {}, {}
    for modality in MODALITY_NAMES:
        arr = np.zeros((len(patient_keys), max_frames, img_size, img_size, 3), dtype=np.uint8)
        mask = np.zeros((len(patient_keys), max_frames), dtype=np.float32)
        for i, pkey in enumerate(patient_keys):
            for j, s in enumerate(selection[pkey][modality]):
                idx = modality_idx[modality][s.image]
                arr[i, j] = variant_arrays[modality][variant_name][idx]
                mask[i, j] = 1.0
        images[modality] = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 1, 4, 2, 3))).to(device)
        masks[modality] = torch.from_numpy(mask).to(device)
    return images, masks


def counterfactual_for(run_name: str, device: torch.device) -> dict:
    run = json.loads((RESULTS_DIR / f"{run_name}.json").read_text())
    task = run["task"]
    classes = patient_classes(task)
    n_classes = len(classes)
    chance = 1 / n_classes
    max_frames = run["max_frames"]
    img_size = run["img_size"]
    normalizers = tower_normalizers(run.get("backbone", "efficientnet_b0"))

    samples = build_joint_manifest()
    if run.get("paired_only"):
        # the run's folds were drawn over the paired patients only
        paired_keys = {s.patient_key for s in samples if s.modality == "EUS"}
        samples = [s for s in samples if s.patient_key in paired_keys]
    folds = patient_level_folds(samples, n_folds=run["n_splits"], seed=run["seed"],
                                 strata_key=joint_strata_key)

    per_fold = []
    for f in run["folds"]:
        k = f["fold"]
        ckpt_path = RESULTS_DIR / run_name / f"fold{k}_best.pt"
        if not ckpt_path.exists():
            print(f"  missing {ckpt_path}, skipping fold {k}")
            continue
        _, val_samples = folds[k]

        by_patient_modality: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        subtype_of = {}
        for s in val_samples:
            by_patient_modality[s.patient_key][s.modality].append(s)
            subtype_of[s.patient_key] = s.subtype
        patient_keys = sorted(by_patient_modality)
        y = np.array([classes.index(patient_label(subtype_of[pk], task)) for pk in patient_keys])

        variant_arrays, modality_idx = {}, {}
        for modality in MODALITY_NAMES:
            mod_samples = [s for s in val_samples if s.modality == modality]
            print(f"  fold {k} {modality}: building 5 counterfactual variants over {len(mod_samples)} frames ...")
            variant_arrays[modality] = build_five_variants(mod_samples, img_size=img_size, progress_every=0)
            modality_idx[modality] = {s.image: i for i, s in enumerate(mod_samples)}

        selection = build_frame_selection(patient_keys, by_patient_modality, max_frames, run["seed"])

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = JointTwoTowerModel(n_classes, backbone=ck.get("backbone", "efficientnet_b0"),
                                    freeze=ck["freeze"], fusion=ck["fusion"],
                                    modality_dropout=ck["modality_dropout"],
                                    unfreeze_last_n=ck.get("unfreeze_last_n", 4)).to(device).eval()
        model.load_state_dict(ck["model_state_dict"])

        row = {"fold": k}
        eval_batch = 16
        with torch.no_grad():
            for v in FIVE_VARIANTS:
                preds = []
                for b in range(0, len(patient_keys), eval_batch):
                    chunk_keys = patient_keys[b:b + eval_batch]
                    images, masks = pack_variant_batch(chunk_keys, selection, variant_arrays,
                                                         modality_idx, v, max_frames, img_size, device)
                    images_norm = _normalize_frames(images, normalizers, device)
                    logits = model(images_norm, masks)
                    preds.append(torch.softmax(logits.float(), dim=1).argmax(1).cpu().numpy())
                pred = np.concatenate(preds)
                row[v] = score(pred, y, n_classes)["bal"]
        per_fold.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_row = {"run_name": run_name, "task": task, "chance": chance}
    for v in FIVE_VARIANTS:
        mean_row[v] = float(np.mean([r[v] for r in per_fold]))
    # Retention is measured against the model's own normal input: the full
    # frame, or -- for a --lesion-crop run, which never saw a full frame --
    # the lesion crop (``build_five_variants``'s ``lesion`` uses the same box
    # and 10% padding as ``joint_fusion_data._lesion_crop``).
    ref = "lesion" if run.get("lesion_crop") else "full"
    mean_row["reference"] = ref
    denom = mean_row[ref] - chance
    for v in FIVE_VARIANTS:
        if v != ref:
            mean_row[f"ret_{v}"] = (mean_row[v] - chance) / denom if denom > 1e-6 else None
    mean_row["per_fold"] = per_fold
    return mean_row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True,
                    help="matches experiments/results/joint_fusion_runs/<run-name>.json")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== target run: {args.run_name} ===")
    target = counterfactual_for(args.run_name, device)

    print("\n" + "=" * 100)
    print("JOINT FUSION COUNTERFACTUAL CHECK -- patient-level balanced accuracy "
          f"(retention vs {target['reference']} in brackets)")
    print("=" * 100)
    hdr = f"{'run':<75}{'chance':>7}" + "".join(f"{v:>16}" for v in FIVE_VARIANTS)
    print(hdr); print("-" * len(hdr))

    def cell(row, name):
        if name == row["reference"]:
            return f"{row[name]:.3f} (  ref)"
        r = row.get(f"ret_{name}")
        return f"{row[name]:.3f} ({r:>5.0%})" if r is not None else "n/a"

    print(f"{target['run_name']:<75}{target['chance']:>7.3f}"
          + "".join(f"{cell(target, v):>16}" for v in FIVE_VARIANTS))

    out_path = OUT_DIR / f"36_joint_fusion_counterfactual__{args.run_name}.json"
    out_path.write_text(json.dumps(target, indent=2))
    print(f"\nwrote {out_path}")

    print("\n" + "=" * 100)
    print("VERDICT (both towers swapped to the same variant simultaneously)")
    print("=" * 100)
    ret_bgp, ret_box = target.get("ret_bg_patch"), target.get("ret_boxmap")
    print("Gate: bg_patch/boxmap retention must be BELOW the win's own retention on the "
          f"{target['reference']} image to be trusted -- if a variant with no real lesion "
          f"tissue at all still matches or beats the {target['reference']}-image score, the "
          "fused win is shortcut-driven, not genuine lesion signal.")
    print(f"  ret_bg_patch={ret_bgp}  ret_boxmap={ret_box}")
    for name, ret in (("bg_patch", ret_bgp), ("boxmap", ret_box)):
        if ret is None:
            continue
        if ret >= 1.0 - 1e-6:
            print(f"  -> {name}: SHORTCUT-DRIVEN ({ret:.0%} retention). Do NOT record this "
                  f"as a validated fusion improvement.")
        else:
            print(f"  -> {name}: retention {ret:.0%} < 100%, consistent with genuine "
                  f"lesion-dependent signal, not purely shortcut-driven.")


if __name__ == "__main__":
    main()
