"""Experiment 38 -- leak-free DINOv3 dual-stream fusion (nested CV).

``33_classifier_fusion.py`` uses fold 0 of each per-modality DINOv3 classifier
as ONE frozen extractor for every patient, then runs its own fusion CV over the
head -- so ~80% of the fusion validation patients (510/641 WLE, 335/421 EUS)
were in the extractor's training set, and the head's best epoch is picked on
the same validation fold it is scored on. This rebuilds the extractors per
fusion fold and keeps every selection step off the scored patients:

for each outer fold k (the same folds as 33: ``joint_strata_key``, ``--seed``):
  1. split the outer-train patients into *fit* / *select* (patient-level,
     ``joint_strata_key``, ``--inner-val-fraction``)
  2. train one DINOv3 classifier per modality on the fit patients' frames;
     checkpoint and LR schedule are chosen on the select patients' frames
     (``finetune_classifier.run_fold``, unchanged)
  3. extract CLS + softmax-posterior descriptors for every patient with those
     two extractors (``cadsel.classifier_fusion_features``, unchanged)
  4. train each ablation's ``FusionHead`` on the fit patients, pick its epoch
     on the select patients, and report the outer-val patients at that epoch
     (``cadsel.train_fusion.train_fold`` with ``test_patients``)

Outer-val patients are never trained on and never used to pick anything.
Extractor defaults reproduce the checkpoints 33 was run with
(``--freeze partial``, 4 unfrozen blocks, 20 epochs, early-stop 8, lr 3e-4,
WLE ``weighted_ce`` / EUS ``ce``).

    python experiments/src/38_classifier_fusion_nested.py --task binary
    python experiments/src/38_classifier_fusion_nested.py --task subtype

Resumable: each fold's extractor checkpoints and feature cache are reused if
already on disk; the head stage is cheap and always re-run.
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
from cadsel.data import build_joint_manifest, joint_strata_key, patient_level_folds, patient_level_split
from cadsel.fusion import load_features, patient_classes, save_features
from cadsel.train_fusion import ABLATIONS, train_fold
from finetune_classifier import build_parser as build_classifier_parser
from finetune_classifier import run_fold as train_classifier_fold
from harness import classes_for

OUT_DEFAULT = PROJECT_ROOT / "experiments" / "results" / "classifier_fusion_nested"


def train_extractor(modality, fit_samples, select_samples, args, fold_idx, ckpt_dir, device):
    """One per-modality DINOv3 classifier for outer fold ``fold_idx``."""
    ckpt = ckpt_dir / f"fold{fold_idx}_best.pt"
    summary_path = ckpt_dir / f"fold{fold_idx}_summary.json"
    if ckpt.exists() and summary_path.exists():
        print(f"    {modality} extractor fold {fold_idx}: reusing {ckpt}")
        return json.loads(summary_path.read_text())

    loss = args.wle_loss if modality == "WLE" else args.eus_loss
    clf_args = build_classifier_parser().parse_args([
        "--backbone", "dinov3_vits16", "--modality", modality, "--task", args.task,
        "--freeze", args.freeze, "--unfreeze-last-n", str(args.unfreeze_last_n),
        "--loss", loss, "--img-size", str(args.img_size),
        "--epochs", str(args.extractor_epochs), "--lr", str(args.extractor_lr),
        "--batch-size", str(args.extractor_batch_size),
        "--early-stop-patience", str(args.extractor_early_stop),
        "--seed", str(args.seed), "--num-workers", str(args.num_workers),
    ])
    torch.manual_seed(args.seed + fold_idx)
    np.random.seed(args.seed + fold_idx)
    r = train_classifier_fold(fit_samples, select_samples, clf_args, args.task,
                              classes_for(args.task), device, fold_idx, ckpt_dir)
    summary = {"modality": modality, "loss": loss, "n_fit_frames": r["n_train"],
               "n_select_frames": r["n_val"], "select_best": r["best"], "history": r["history"]}
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["binary", "subtype"], default="subtype")
    p.add_argument("--folds", type=int, default=5, help="outer (fusion) CV folds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inner-val-fraction", type=float, default=0.2,
                   help="share of each outer-train fold held out as the select set")
    # extractor (per-modality DINOv3 classifier)
    p.add_argument("--freeze", choices=["partial", "head", "none"], default="partial")
    p.add_argument("--unfreeze-last-n", type=int, default=4)
    p.add_argument("--wle-loss", choices=["ce", "weighted_ce", "seesaw_ce"], default="weighted_ce")
    p.add_argument("--eus-loss", choices=["ce", "weighted_ce", "seesaw_ce"], default="ce")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--extractor-epochs", type=int, default=20)
    p.add_argument("--extractor-lr", type=float, default=3e-4)
    p.add_argument("--extractor-batch-size", type=int, default=32)
    p.add_argument("--extractor-early-stop", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    # fusion head
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--fusion", choices=["gated", "symmetric"], default="gated")
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--max-outer-folds", type=int, default=None,
                   help="stop after this many outer folds (smoke tests only)")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = (
        f"dinov3_vits16__{args.task}__freeze-{args.freeze}__wle-{args.wle_loss}__eus-{args.eus_loss}"
        f"__fusion-{args.fusion}__inner{args.inner_val_fraction:g}__s{args.seed}"
    )
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(patient_classes(args.task))
    assert n_classes == len(classes_for(args.task))

    samples = build_joint_manifest()
    outer = patient_level_folds(samples, n_folds=args.folds, seed=args.seed, strata_key=joint_strata_key)
    print(f"[{run_name}] device={device} {len({s.patient_key for s in samples})} patients, "
          f"{args.folds}-fold outer CV")

    started = time.time()
    fold_meta = []
    results: dict[str, list] = {name: [] for name, _, _ in ABLATIONS}
    for k, (outer_train, outer_val) in enumerate(outer):
        if args.max_outer_folds is not None and k >= args.max_outer_folds:
            break
        t0 = time.time()
        fit, select = patient_level_split(outer_train, val_fraction=args.inner_val_fraction,
                                          seed=args.seed + 1000 + k, strata_key=joint_strata_key)
        fit_keys = {s.patient_key for s in fit}
        select_keys = {s.patient_key for s in select}
        test_keys = {s.patient_key for s in outer_val}
        assert not (fit_keys & select_keys) and not (fit_keys & test_keys) and not (select_keys & test_keys), \
            "fit/select/test patient sets overlap"
        print(f"\n  outer fold {k}: fit={len(fit_keys)} select={len(select_keys)} test={len(test_keys)} patients")

        # 1-2. per-modality extractors, trained on fit patients only
        extractors = {}
        for modality in MODALITY_NAMES:
            fit_m = [s for s in fit if s.modality == modality]
            select_m = [s for s in select if s.modality == modality]
            assert {s.patient_key for s in fit_m} <= fit_keys
            ckpt_dir = run_dir / "extractors" / modality
            extractors[modality] = train_extractor(modality, fit_m, select_m, args, k, ckpt_dir, device)
            sb = extractors[modality]["select_best"]
            print(f"    {modality} extractor: select-set frame bal {sb['bal']:.4f} @ epoch {sb['epoch']}")

        # 3. descriptors for every patient from this fold's extractors
        features_path = run_dir / f"features_fold{k}.npz"
        if features_path.exists():
            features = load_features(features_path)
        else:
            models, specs = {}, {}
            for modality in MODALITY_NAMES:
                models[modality], specs[modality] = load_classifier(
                    run_dir / "extractors" / modality / f"fold{k}_best.pt", n_classes, device)
            features = extract_classifier_features(models, specs, samples, n_classes, device, progress=False)
            save_features(features_path, features)
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # 4. fusion heads: fit on fit, select on select, score on outer val
        for name, modalities, moddrop in ABLATIONS:
            fit_p = [features[key] for key in sorted(fit_keys)]
            select_p = [features[key] for key in sorted(select_keys)]
            test_p = [features[key] for key in sorted(test_keys)]
            if name == "eus_only":
                fit_p, select_p, test_p = ([pf for pf in ps if pf.has("EUS")] for ps in (fit_p, select_p, test_p))
            r = train_fold(
                fit_p, select_p, args.task, list(modalities), moddrop,
                epochs=args.epochs, seed=args.seed + k, device=device, fusion=args.fusion,
                weight_decay=args.weight_decay, dropout=args.dropout,
                early_stop_patience=args.early_stop_patience, capture_predictions=True,
                test_patients=test_p,
            )
            test = r.pop("test")
            results[name].append({
                **{m: test[m] for m in ("acc", "bal", "macro_f1")},
                "fold": k, "epoch": r["epoch"], "n_train": r["n_train"], "n_select": r["n_val"],
                "n_test": r["n_test"], "select": {m: r[m] for m in ("acc", "bal", "macro_f1")},
                "test_patient_keys": [pf.patient_key for pf in test_p],
                "pred": test["pred"], "probs": test["probs"], "history": r["history"],
            })
            print(f"    {name:<14} test bal {test['bal']:.4f}  (select bal {r['bal']:.4f} @ epoch {r['epoch']})")

        fold_meta.append({
            "fold": k, "n_fit": len(fit_keys), "n_select": len(select_keys), "n_test": len(test_keys),
            "extractors": {m: {key: v for key, v in e.items() if key != "history"} for m, e in extractors.items()},
            "elapsed_sec": round(time.time() - t0, 1),
        })
        print(f"  outer fold {k} done in {time.time() - t0:.0f}s")

    def agg(per_fold, metric):
        vals = [f[metric] for f in per_fold]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    ablations = {
        name: {"folds": per_fold, **{m: agg(per_fold, m) for m in ("acc", "bal", "macro_f1")}}
        for name, per_fold in results.items()
    }
    print(f"\n[{run_name}] outer-fold test metrics (mean +/- std over {len(fold_meta)} folds):")
    for name, a in ablations.items():
        print(f"  {name:<14} bal {a['bal']['mean']:.4f} +/- {a['bal']['std']:.4f}   "
              f"macroF1 {a['macro_f1']['mean']:.4f} +/- {a['macro_f1']['std']:.4f}")

    payload = {
        "run_name": run_name, "task": args.task, "classes": patient_classes(args.task),
        "protocol": "nested patient-level CV: per-outer-fold DINOv3 extractors trained on the "
                    "fit split, extractor checkpoint and head epoch chosen on the select split, "
                    "metrics on outer-fold patients only; outer folds identical to "
                    "33_classifier_fusion.py for the same --seed",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "outer_folds_done": len(fold_meta), "fold_meta": fold_meta, "ablations": ablations,
        "elapsed_sec": round(time.time() - started, 1),
    }
    out_path = args.output_dir / f"{run_name}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
