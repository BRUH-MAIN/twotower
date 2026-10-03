"""Experiment 33 (Track A follow-up, Tier 3b) — multimodal WLE+EUS
patient-level fusion built on the new trainable classifier backbones
(DINOv3 `--freeze partial`, this project's confirmed-best single-modality
classifier, FINDINGS.md §12.2) instead of `cadsel.fusion`'s original
detector-ROI-feature front-end.

Reuses `cadsel.fusion`'s `FusionHead`/`make_batch`/`PatientFeatures` and
`cadsel.train_fusion`'s `train_fold`/`ABLATIONS` completely unchanged --
only the feature-extraction front-end differs
(`cadsel.classifier_fusion_features` instead of `cadsel.fusion.extract_features`).

    python experiments/src/33_classifier_fusion.py \\
        --checkpoint-wle dinov3_vits16__WLE__subtype__freeze-partial__os-none__loss-weighted_ce__s42 \\
        --checkpoint-eus dinov3_vits16__EUS__subtype__freeze-partial__os-none__loss-ce__s42

Each classifier run trains 5 *per-fold* checkpoints (unlike the detector
pipeline's single `best_model.pth`); `--fold` (default 0) picks which one
serves as the frozen feature extractor for every patient. This mirrors the
existing detector-based fusion pipeline's own structure exactly: one frozen
checkpoint (trained on its own patient split) extracts features for every
patient, and the fusion head is then evaluated under a *separate*
patient-level CV split of its own -- so some patients the frozen extractor
was trained on can land in the fusion CV's validation folds, the same
possible-overlap structure `cadsel.train_fusion.py` already has. Not a new
risk introduced here; inherited from the pipeline being reused as-is.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from cadsel.classifier_fusion_features import extract_classifier_features, load_classifier
from cadsel.config import MODALITY_NAMES, PROJECT_ROOT
from cadsel.data import build_joint_manifest, joint_strata_key, patient_level_folds
from cadsel.fusion import load_features, patient_classes, save_features
from cadsel.train_fusion import ABLATIONS, train_fold

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "classifier_runs"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-wle", required=True, help="finetune_classifier.py run_name (dinov3_vits16 only)")
    p.add_argument("--checkpoint-eus", required=True, help="finetune_classifier.py run_name (dinov3_vits16 only)")
    p.add_argument("--fold", type=int, default=0,
                   help="which of the classifier's own 5 folds to use as the frozen feature extractor")
    p.add_argument("--task", choices=["binary", "subtype"], default="subtype")
    p.add_argument("--folds", type=int, default=5, help="fusion CV folds")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="AdamW weight decay for the FusionHead (default: every prior "
                        "confirmed result in this project used 1e-4)")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="FusionHead encoder/classifier dropout (default: 0.3, the value "
                        "every prior confirmed result used)")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="stop a fold once val_bal hasn't improved for this many epochs "
                        "(default: 0 = off, train the full --epochs budget, matching "
                        "every prior confirmed result)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paired-only", action="store_true",
                   help="restrict to the patients that have both modalities")
    p.add_argument("--fusion", choices=["gated", "symmetric"], default="gated")
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments" / "results" / "classifier_fusion.json")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for modality, run_name in (("WLE", args.checkpoint_wle), ("EUS", args.checkpoint_eus)):
        run_meta = json.loads((RESULTS_DIR / f"{run_name}.json").read_text())
        if run_meta["task"] != args.task:
            raise SystemExit(
                f"--checkpoint-{modality.lower()} {run_name} was trained on task="
                f"{run_meta['task']!r}, but --task={args.task!r} -- they must match "
                f"(the classifier head's output dimension depends on it)"
            )
        if run_meta["modality"] != modality:
            raise SystemExit(f"--checkpoint-{modality.lower()} {run_name} is a {run_meta['modality']} run, expected {modality}")

    samples = build_joint_manifest()
    n_classes = len(patient_classes(args.task))

    features_path = args.features or (
        RESULTS_DIR.parent / f"classifier_fusion_features__{args.checkpoint_wle}__{args.checkpoint_eus}.npz"
    )
    if features_path.exists() and not args.force_extract:
        print(f"loading cached features from {features_path}")
        features = load_features(features_path)
    else:
        models, specs = {}, {}
        for modality, run_name in (("WLE", args.checkpoint_wle), ("EUS", args.checkpoint_eus)):
            ckpt_path = RESULTS_DIR / run_name / f"fold{args.fold}_best.pt"
            model, spec = load_classifier(ckpt_path, n_classes, device)
            models[modality] = model
            specs[modality] = spec
            print(f"  loaded {modality} classifier from {ckpt_path}")

        print(f"extracting features over {len(samples)} frames ...")
        started = time.time()
        features = extract_classifier_features(models, specs, samples, n_classes, device)
        print(f"  {len(features)} patients in {time.time() - started:.0f}s")
        save_features(features_path, features)
        print(f"  cached to {features_path}")
        del models
        torch.cuda.empty_cache()

    if args.paired_only:
        features = {k: v for k, v in features.items() if all(v.has(m) for m in MODALITY_NAMES)}
        print(f"paired-only: {len(features)} patients")

    keep = set(features)
    folds = patient_level_folds(
        [s for s in samples if s.patient_key in keep],
        n_folds=args.folds, seed=args.seed, strata_key=joint_strata_key,
    )

    classes = patient_classes(args.task)
    results: dict[str, dict] = {}
    for name, modalities, moddrop in ABLATIONS:
        per_fold = []
        for i, (train_samples, val_samples) in enumerate(folds):
            train_keys = {s.patient_key for s in train_samples}
            val_keys = {s.patient_key for s in val_samples}
            assert not (train_keys & val_keys), "fold leaks patients"

            train_p = [features[k] for k in sorted(train_keys) if k in features]
            val_p = [features[k] for k in sorted(val_keys) if k in features]
            if name == "eus_only":
                train_p = [p for p in train_p if p.has("EUS")]
                val_p = [p for p in val_p if p.has("EUS")]
            if len(val_p) < 2 or len(train_p) < 10:
                continue

            fold_result = train_fold(
                train_p, val_p, args.task, list(modalities), moddrop,
                epochs=args.epochs, seed=args.seed + i, device=device, fusion=args.fusion,
                weight_decay=args.weight_decay, dropout=args.dropout,
                early_stop_patience=args.early_stop_patience, capture_predictions=True,
            )
            per_fold.append(fold_result)

        agg = {
            metric: {
                "mean": round(float(np.mean([f[metric] for f in per_fold])), 4),
                "std": round(float(np.std([f[metric] for f in per_fold])), 4),
            }
            for metric in ("acc", "bal", "macro_f1")
        }
        results[name] = {"folds": per_fold, **agg}
        print(f"  {name:<16} bal {agg['bal']['mean']:.4f} +/- {agg['bal']['std']:.4f}   "
              f"macroF1 {agg['macro_f1']['mean']:.4f} +/- {agg['macro_f1']['std']:.4f}   "
              f"({len(per_fold)} folds)")

    payload = {
        "checkpoint_wle": args.checkpoint_wle, "checkpoint_eus": args.checkpoint_eus,
        "extraction_fold": args.fold, "task": args.task, "folds": args.folds,
        "paired_only": args.paired_only, "fusion": args.fusion, "classes": classes,
        "seed": args.seed, "epochs": args.epochs, "weight_decay": args.weight_decay,
        "dropout": args.dropout, "early_stop_patience": args.early_stop_patience,
        "protocol": "patient-level cross-validation; NOT comparable to Table 4/5 AP "
                    "or the per-image classifier balanced accuracy in FINDINGS.md §10-13",
        "ablations": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
