"""Experiment 32 (Track A follow-up, Tier 2a) — probability-average ensemble
of the same config's checkpoints across multiple seeds. Zero additional GPU
training: every checkpoint already exists on disk (e.g. the 3-seed DINOv3
`--freeze partial` confirmation, FINDINGS.md §12.2), this only adds
inference.

    python experiments/src/32_classifier_multiseed_ensemble.py --run-names \\
        dinov3_vits16__WLE__binary__freeze-partial__os-none__loss-ce__s42 \\
        dinov3_vits16__WLE__binary__freeze-partial__os-none__loss-ce__s123 \\
        dinov3_vits16__WLE__binary__freeze-partial__os-none__loss-ce__s7

Correctness note this script exists specifically to get right: each seed's
``patient_level_folds(samples, n_splits, seed=seed)`` partitions patients
*differently* -- fold 0 of seed 42 is not the same patients as fold 0 of
seed 123. Averaging "fold k of seed A" with "fold k of seed B" would silently
mix predictions from different held-out sets. Instead, each seed's own 5
folds are used to build one full out-of-fold (OOF) prediction covering every
sample exactly once (fold 0's checkpoint predicts fold 0's val images, fold
1's predicts fold 1's, etc. -- together they cover the whole dataset), keyed
by ``Sample.image`` (a stable per-image identity). Only *then* are the
per-seed OOF probability arrays averaged, aligned by that key -- never by
list position or fold index across seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from cadsel.classification_data import make_classification_normalizer
from cadsel.data import build_manifest, patient_level_folds
from finetune_classifier import build_model
from harness import classes_for, encode_labels
from probes import balanced_accuracy

RESULTS_DIR = Path("experiments/results/classifier_runs")
OUT_DIR = Path("experiments/results")


def decode_fold(val_samples, img_size: int) -> np.ndarray:
    """Byte-identical to ``30_classifier_tta_eval.py``'s own copy -- not
    imported from there since a module name starting with a digit
    (``30_classifier_tta_eval``) isn't importable via a normal ``import``
    statement; duplicated rather than reached for ``importlib`` machinery
    for 7 lines of code."""
    arrs = np.zeros((len(val_samples), img_size, img_size, 3), dtype=np.uint8)
    for i, s in enumerate(val_samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            arrs[i] = np.asarray(im, dtype=np.uint8)
    return arrs


def _predict_probs(model, normalize, arr: np.ndarray, device: torch.device, batch: int = 32) -> np.ndarray:
    out = []
    for k in range(0, len(arr), batch):
        chunk = arr[k:k + batch]
        x = torch.from_numpy(np.ascontiguousarray(chunk.transpose(0, 3, 1, 2))).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(normalize(x))
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


def collect_oof_probs(run_name: str, device: torch.device) -> dict:
    """Returns {"probs": {image_path: prob_vector}, "y": {image_path: label},
    "modality", "task", "classes", "n_classes", "own_bal_mean"}."""
    run = json.loads((RESULTS_DIR / f"{run_name}.json").read_text())
    modality, task = run["modality"], run["task"]
    classes = classes_for(task)
    n_classes = len(classes)

    samples = build_manifest(modality)
    folds = patient_level_folds(samples, n_folds=run["n_splits"], seed=run["seed"])

    probs_by_key: dict[str, np.ndarray] = {}
    y_by_key: dict[str, int] = {}
    for f in run["folds"]:
        k = f["fold"]
        ckpt_path = RESULTS_DIR / run_name / f"fold{k}_best.pt"
        if not ckpt_path.exists():
            print(f"  [{run_name}] missing {ckpt_path}, skipping fold {k}")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _, val_samples = folds[k]
        y = encode_labels(val_samples, task)

        model = build_model(ck["backbone"], n_classes, ck["freeze"]).to(device).eval()
        model.load_state_dict(ck["model_state_dict"])
        normalize = make_classification_normalizer(ck["mean"], ck["std"])
        arr = decode_fold(val_samples, ck["img_size"])
        probs = _predict_probs(model, normalize, arr, device)

        for s, yy, p in zip(val_samples, y, probs):
            probs_by_key[s.image] = p
            y_by_key[s.image] = int(yy)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y_arr = np.array([y_by_key[k] for k in probs_by_key])
    pred_arr = np.array([probs_by_key[k].argmax() for k in probs_by_key])
    own_bal = balanced_accuracy(y_arr, pred_arr, n_classes)
    print(f"  [{run_name}] reconstructed OOF balanced accuracy: {own_bal:.4f} "
          f"(run's own recorded 5-fold mean: {run['balanced_accuracy_mean']:.4f} -- "
          f"should match closely, this is a sanity check on the reconstruction)")

    return {
        "probs": probs_by_key, "y": y_by_key, "modality": modality, "task": task,
        "classes": classes, "n_classes": n_classes, "own_bal": own_bal,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-names", nargs="+", required=True,
                    help="2+ run_names of the SAME config differing only in --seed")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_run = [collect_oof_probs(rn, device) for rn in args.run_names]

    modality = per_run[0]["modality"]
    task = per_run[0]["task"]
    classes = per_run[0]["classes"]
    n_classes = per_run[0]["n_classes"]
    for r in per_run[1:]:
        if r["modality"] != modality or r["task"] != task:
            raise SystemExit(
                f"--run-names must all share the same modality/task to ensemble "
                f"(got {modality}/{task} and {r['modality']}/{r['task']})"
            )

    keys = sorted(per_run[0]["probs"].keys())
    for r, rn in zip(per_run[1:], args.run_names[1:]):
        other_keys = sorted(r["probs"].keys())
        if other_keys != keys:
            raise SystemExit(
                f"{rn} covers a different image set than {args.run_names[0]} -- "
                f"can't ensemble (missing checkpoints for some folds?)"
            )

    y_true = np.array([per_run[0]["y"][k] for k in keys])
    avg_probs = np.mean([[r["probs"][k] for k in keys] for r in per_run], axis=0)
    pred = avg_probs.argmax(1)

    acc = float((pred == y_true).mean())
    bal = balanced_accuracy(y_true, pred, n_classes)
    own_bals = [r["own_bal"] for r in per_run]

    print(f"\n[{modality}/{task}] {len(args.run_names)}-seed probability ensemble:")
    print(f"  individual seed OOF balanced accuracy: "
          f"{', '.join(f'{b:.4f}' for b in own_bals)} "
          f"(mean {np.mean(own_bals):.4f} +/- {np.std(own_bals):.4f})")
    print(f"  ensemble accuracy: {acc:.4f}  ensemble balanced accuracy: {bal:.4f}")
    print(f"  delta vs. mean individual seed: {bal - np.mean(own_bals):+.4f}")

    out = {
        "run_names": args.run_names, "modality": modality, "task": task,
        "individual_bal": own_bals, "individual_bal_mean": float(np.mean(own_bals)),
        "individual_bal_std": float(np.std(own_bals)),
        "ensemble_accuracy": acc, "ensemble_balanced_accuracy": bal,
        "delta_vs_mean_individual": bal - float(np.mean(own_bals)),
    }
    out_path = OUT_DIR / f"32_classifier_multiseed_ensemble__{modality}_{task}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
