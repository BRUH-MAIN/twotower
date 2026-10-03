"""Experiment 37 -- patient-level multi-frame test-time augmentation for the
joint two-tower WLE+EUS fusion model. Distinct from the single-image flip/
5-crop TTA already found null on the plain classifiers (FINDINGS.md
§12.4/§14.6): this architecture's eval already fixes ONE deterministic
frame-subset draw per patient (`JointPatientDataset`'s frame selection is
seeded by patient index, unlike each frame's own augmentation which is
already deterministic at eval time via a plain ``Resize``). Since a patient
with more raw frames than ``max_frames`` only ever gets one fixed subset at
eval time, averaging predictions over K *different* frame-subset draws per
patient is a new axis this architecture specifically enables that a
single-image classifier never had.

    python -m experiments.src.37_joint_fusion_frame_tta --run-name \\
        joint_efficientnet_b0__binary__freeze-none__fusion-gated__loss-weighted_ce__moddrop0.3__mf4__ep100__es20__lesioncrop__s42 \\
        --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import MODALITY_NAMES
from cadsel.data import build_joint_manifest, joint_strata_key, patient_level_folds
from cadsel.fusion import patient_classes
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


@torch.no_grad()
def eval_pass(model, val_ds, normalizers, device, batch_size=16) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, labels) arrays, patient order matching val_ds.patient_keys."""
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=joint_collate,
                         num_workers=0)
    all_probs, all_labels = [], []
    for images, masks, labels in loader:
        images_norm = _normalize_frames(images, normalizers, device)
        masks_dev = {m: masks[m].to(device) for m in MODALITY_NAMES}
        logits = model(images_norm, masks_dev)
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--k", type=int, default=5, help="number of independent frame-subset draws to average")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = json.loads((RESULTS_DIR / f"{args.run_name}.json").read_text())
    task = run["task"]
    classes = patient_classes(task)
    n_classes = len(classes)
    normalizers = tower_normalizers(run.get("backbone", "efficientnet_b0"))

    samples = build_joint_manifest()
    if run.get("paired_only"):
        # the run's folds were drawn over the paired patients only
        paired_keys = {s.patient_key for s in samples if s.modality == "EUS"}
        samples = [s for s in samples if s.patient_key in paired_keys]
    folds = patient_level_folds(samples, n_folds=run["n_splits"], seed=run["seed"],
                                 strata_key=joint_strata_key)

    plain_bals, tta_bals = [], []
    for f in run["folds"]:
        k = f["fold"]
        ckpt_path = RESULTS_DIR / args.run_name / f"fold{k}_best.pt"
        if not ckpt_path.exists():
            print(f"  missing {ckpt_path}, skipping fold {k}")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _, val_samples = folds[k]

        model = JointTwoTowerModel(n_classes, backbone=ck.get("backbone", "efficientnet_b0"),
                                    freeze=ck["freeze"], fusion=ck["fusion"],
                                    modality_dropout=ck["modality_dropout"],
                                    unfreeze_last_n=ck.get("unfreeze_last_n", 4)).to(device).eval()
        model.load_state_dict(ck["model_state_dict"])

        # Draw 0 uses the run's own seed (reproduces the plain recorded eval
        # exactly); draws 1..k-1 use distinct offsets so each picks a
        # different frame subset per patient (frame *selection* RNG only --
        # eval-mode augmentation is already the deterministic Resize).
        draw_probs = []
        patient_keys = None
        for draw in range(args.k):
            draw_seed = run["seed"] if draw == 0 else run["seed"] + draw * 1000
            val_ds = JointPatientDataset(val_samples, task, img_size=ck["img_size"],
                                          max_frames=ck["max_frames"], train=False, seed=draw_seed,
                                          lesion_crop=run.get("lesion_crop", False),
                                          bg_remove_prob=run.get("bg_remove_prob", 0.0))
            if patient_keys is None:
                patient_keys = val_ds.patient_keys
            probs, labels = eval_pass(model, val_ds, normalizers, device)
            draw_probs.append(probs)
            if draw == 0:
                plain_probs, y = probs, labels

        tta_probs = np.mean(draw_probs, axis=0)
        plain_pred = plain_probs.argmax(1)
        tta_pred = tta_probs.argmax(1)
        plain_bal = score(plain_pred, y, n_classes)["bal"]
        tta_bal = score(tta_pred, y, n_classes)["bal"]
        plain_bals.append(plain_bal)
        tta_bals.append(tta_bal)
        print(f"  fold {k}: plain={plain_bal:.4f}  tta(k={args.k})={tta_bal:.4f}  "
              f"delta={tta_bal - plain_bal:+.4f}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    plain_mean, plain_std = float(np.mean(plain_bals)), float(np.std(plain_bals))
    tta_mean, tta_std = float(np.mean(tta_bals)), float(np.std(tta_bals))
    delta = tta_mean - plain_mean
    print(f"\n[{args.run_name}] k={args.k}  plain {plain_mean:.4f}+/-{plain_std:.4f}  "
          f"tta {tta_mean:.4f}+/-{tta_std:.4f}  delta {delta:+.4f}")

    out = {
        "run_name": args.run_name, "k": args.k, "task": task,
        "plain_mean": plain_mean, "plain_std": plain_std,
        "tta_mean": tta_mean, "tta_std": tta_std, "delta": delta,
        "per_fold": [{"fold": i, "plain": pl, "tta": tt} for i, (pl, tt) in enumerate(zip(plain_bals, tta_bals))],
    }
    out_path = OUT_DIR / f"37_joint_fusion_frame_tta__k{args.k}__{args.run_name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
