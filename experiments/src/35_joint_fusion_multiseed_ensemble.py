"""Experiment 35 -- probability-average ensemble of the joint two-tower
fusion model's checkpoints across multiple seeds. Zero additional GPU
training: the 3-seed binary confirmation (FINDINGS.md §15.4) already has all
15 fold checkpoints on disk, this only adds inference. Mirrors
``32_classifier_multiseed_ensemble.py``'s pattern -- reconstruct each seed's
own fold split (a joint-fusion seed also reshuffles ``patient_level_folds``,
not just the model init), collect one full out-of-fold prediction per seed
keyed by ``patient_key`` (never by fold index across seeds), then average.

    python -m experiments.src.35_joint_fusion_multiseed_ensemble --run-names \\
        joint_efficientnet_b0__binary__freeze-partial__fusion-gated__loss-weighted_ce__moddrop0.3__mf4__ep100__es20__s42 \\
        joint_efficientnet_b0__binary__freeze-partial__fusion-gated__loss-weighted_ce__moddrop0.3__mf4__ep100__es20__s123 \\
        joint_efficientnet_b0__binary__freeze-partial__fusion-gated__loss-weighted_ce__moddrop0.3__mf4__ep100__es20__s7
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import MODALITY_NAMES, PROJECT_ROOT
from cadsel.data import build_joint_manifest, joint_strata_key, patient_level_folds
from cadsel.fusion import patient_classes, patient_label
from cadsel.joint_fusion_data import JointPatientDataset, joint_collate
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


def collect_oof_probs(run_name: str, device: torch.device) -> dict:
    """Returns {"probs": {patient_key: prob_vector}, "y": {patient_key: label},
    "paired": {patient_key: bool}, "task", "classes", "n_classes", "own_bal"}."""
    run = json.loads((RESULTS_DIR / f"{run_name}.json").read_text())
    task = run["task"]
    classes = patient_classes(task)
    n_classes = len(classes)

    samples = build_joint_manifest()
    if run.get("paired_only"):
        # the run's folds were drawn over the paired patients only
        paired_keys = {s.patient_key for s in samples if s.modality == "EUS"}
        samples = [s for s in samples if s.patient_key in paired_keys]
    folds = patient_level_folds(samples, n_folds=run["n_splits"], seed=run["seed"],
                                 strata_key=joint_strata_key)
    normalizers = tower_normalizers(run.get("backbone", "efficientnet_b0"))

    probs_by_key: dict[str, np.ndarray] = {}
    y_by_key: dict[str, int] = {}
    paired_by_key: dict[str, bool] = {}
    for f in run["folds"]:
        k = f["fold"]
        ckpt_path = RESULTS_DIR / run_name / f"fold{k}_best.pt"
        if not ckpt_path.exists():
            print(f"  [{run_name}] missing {ckpt_path}, skipping fold {k}")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _, val_samples = folds[k]

        val_ds = JointPatientDataset(val_samples, task, img_size=ck["img_size"],
                                      max_frames=ck["max_frames"], train=False, seed=run["seed"],
                                      lesion_crop=run.get("lesion_crop", False),
                                      bg_remove_prob=run.get("bg_remove_prob", 0.0))
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=joint_collate,
                                 num_workers=0)

        model = JointTwoTowerModel(n_classes, backbone=ck.get("backbone", "efficientnet_b0"),
                                    freeze=ck["freeze"], fusion=ck["fusion"],
                                    modality_dropout=ck["modality_dropout"],
                                    unfreeze_last_n=ck.get("unfreeze_last_n", 4)).to(device).eval()
        model.load_state_dict(ck["model_state_dict"])

        idx = 0
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images_norm = _normalize_frames(images, normalizers, device)
                masks_dev = {m: masks[m].to(device) for m in MODALITY_NAMES}
                logits = model(images_norm, masks_dev)
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                paired = ((masks["WLE"].sum(dim=1) > 0.5) & (masks["EUS"].sum(dim=1) > 0.5)).numpy()
                for j in range(len(labels)):
                    pkey = val_ds.patient_keys[idx]
                    probs_by_key[pkey] = probs[j]
                    y_by_key[pkey] = int(labels[j])
                    paired_by_key[pkey] = bool(paired[j])
                    idx += 1

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    y_arr = np.array([y_by_key[k] for k in probs_by_key])
    pred_arr = np.array([probs_by_key[k].argmax() for k in probs_by_key])
    own = score(pred_arr, y_arr, n_classes)
    print(f"  [{run_name}] reconstructed OOF: acc={own['acc']:.4f} bal={own['bal']:.4f} "
          f"(run's own recorded fused/unpaired 5-fold mean: "
          f"{run['summary']['fused']['unpaired']['bal']['mean']:.4f} -- should match closely)")

    return {
        "probs": probs_by_key, "y": y_by_key, "paired": paired_by_key,
        "task": task, "classes": classes, "n_classes": n_classes, "own_bal": own["bal"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-names", nargs="+", required=True,
                    help="2+ run_names of the SAME config differing only in --seed")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    per_run = [collect_oof_probs(rn, device) for rn in args.run_names]

    task = per_run[0]["task"]
    classes = per_run[0]["classes"]
    n_classes = per_run[0]["n_classes"]
    for r in per_run[1:]:
        if r["task"] != task:
            raise SystemExit(f"--run-names must all share the same task to ensemble")

    keys = sorted(per_run[0]["probs"].keys())
    for r, rn in zip(per_run[1:], args.run_names[1:]):
        if sorted(r["probs"].keys()) != keys:
            raise SystemExit(f"{rn} covers a different patient set -- can't ensemble")

    y_true = np.array([per_run[0]["y"][k] for k in keys])
    paired = np.array([per_run[0]["paired"][k] for k in keys])
    avg_probs = np.mean([[r["probs"][k] for k in keys] for r in per_run], axis=0)
    pred = avg_probs.argmax(1)

    own_bals = [r["own_bal"] for r in per_run]
    unpaired_stats = score(pred, y_true, n_classes)
    paired_stats = score(pred[paired], y_true[paired], n_classes)

    print(f"\n[{task}] {len(args.run_names)}-seed patient-level probability ensemble (fused arm):")
    print(f"  individual seed OOF fused/unpaired bal: "
          f"{', '.join(f'{b:.4f}' for b in own_bals)} "
          f"(mean {np.mean(own_bals):.4f} +/- {np.std(own_bals):.4f})")
    print(f"  ensemble, unpaired ({len(keys)} patients): acc={unpaired_stats['acc']:.4f} "
          f"bal={unpaired_stats['bal']:.4f} macroF1={unpaired_stats['macro_f1']:.4f}")
    print(f"  ensemble, paired-only ({int(paired.sum())} patients): acc={paired_stats['acc']:.4f} "
          f"bal={paired_stats['bal']:.4f} macroF1={paired_stats['macro_f1']:.4f}")
    print(f"  delta (unpaired) vs. mean individual seed: {unpaired_stats['bal'] - np.mean(own_bals):+.4f}")

    out = {
        "run_names": args.run_names, "task": task,
        "individual_bal": own_bals, "individual_bal_mean": float(np.mean(own_bals)),
        "individual_bal_std": float(np.std(own_bals)),
        "ensemble_unpaired": unpaired_stats, "ensemble_paired_only": paired_stats,
        "n_unpaired": len(keys), "n_paired": int(paired.sum()),
    }
    # Tag the output by the shared config (first run_name with its trailing
    # __s<seed> stripped), not just `task` -- otherwise ensembling a
    # different config (e.g. --freeze none vs --freeze partial, or
    # --lesion-crop) on the same task silently overwrites an earlier
    # ensemble's result under the same filename, the same collision-by-omission
    # bug class this campaign has hit repeatedly with run_name construction.
    config_tag = re.sub(r"__s\d+$", "", args.run_names[0])
    out_path = OUT_DIR / f"35_joint_fusion_multiseed_ensemble__{config_tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
