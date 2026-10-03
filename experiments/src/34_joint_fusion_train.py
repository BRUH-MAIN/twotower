"""Experiment 34 -- joint end-to-end two-tower WLE+EUS fusion.

Unlike ``cadsel/train_fusion.py`` (trains ``cadsel.fusion.FusionHead`` on
FROZEN, pre-extracted per-modality features -- the backbones never see a
fusion-loss gradient) and ``experiments/src/33_classifier_fusion.py`` (same
frozen-feature pattern, using a DINOv3 classifier checkpoint as the frozen
extractor instead of a detector), this trains two EfficientNet-B0 backbones
(``cadsel.joint_fusion_model.JointTwoTowerModel``) end-to-end together with
the same ``FusionHead``, each modality independently augmented every step
(``cadsel.joint_fusion_data.JointPatientDataset``).

Compute cost forces one deliberate simplification vs. ``train_fusion.py``'s
4-way ``ABLATIONS`` protocol: rather than training 4 separate models per fold
(``wle_only`` / ``eus_only`` / ``fused`` / ``fused_moddrop``, each from
scratch), this trains ONE joint model per fold with modality dropout active
throughout training (matching ``train_fusion.py``'s ``fused_moddrop`` regime,
since that is the setting that lets the 220 WLE-only patients contribute
training signal), then reports ``wle_only``/``eus_only``/``fused`` as three
*masked-inference* views of that single trained model (``model.eval()``
already disables modality dropout, so this is not the same experiment as
``train_fusion.py``'s separately-trained ``fused`` vs. ``fused_moddrop`` rows
-- flagged here rather than silently conflated). Both the "unpaired" (all val
patients) and "paired-only" (patients with both modalities) denominators are
reported for every ablation, mirroring FINDINGS.md's established §5i/§14.4
convention; ``eus_only`` is always restricted to patients who actually have
EUS frames, since a WLE-masked prediction for a WLE-only patient is a
zero-padding artifact, not a real "EUS-only" read (same restriction
``train_fusion.py`` already applies).

``--inner-val-fraction`` > 0 switches to the leak-free nested protocol (same
split as ``38_classifier_fusion_nested.py``): each outer-train fold is split
into *fit* / *select* patients, the model trains on fit, and the best
checkpoint, ``ReduceLROnPlateau`` and early stopping all run on select; the
ablations are then scored on the untouched outer fold. The default (0) keeps
the original protocol, where the scored fold also picks the checkpoint --
an optimistic estimate.

    python -m experiments.src.34_joint_fusion_train --task subtype --seed 42
    python -m experiments.src.34_joint_fusion_train --task binary --backbone resnet50 \\
        --freeze none --lesion-crop --epochs 100 --early-stop-patience 20 --inner-val-fraction 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import MODALITY_NAMES, PROJECT_ROOT
from cadsel.data import build_joint_manifest, joint_strata_key, patient_level_folds, patient_level_split
from cadsel.fusion import patient_classes, patient_label
from cadsel.joint_fusion_data import JointPatientDataset, joint_collate
from cadsel.joint_fusion_model import TOWERS, JointTwoTowerModel, tower_normalizers
from cadsel.train_fusion import score
from finetune_classifier import compute_classification_class_weights
from probes import balanced_accuracy

OUT_DEFAULT = PROJECT_ROOT / "experiments" / "results" / "joint_fusion_runs"
ABLATION_MODALITIES = {"wle_only": ["WLE"], "eus_only": ["EUS"], "fused": MODALITY_NAMES}


def _normalize_frames(images: dict, normalizers: dict, device: torch.device) -> dict:
    """``images[m]``: (B, T, C, H, W) uint8 -> (B, T, C, H, W) normalised float."""
    out = {}
    for m in MODALITY_NAMES:
        x = images[m].to(device, non_blocking=True)
        B, T, C, H, W = x.shape
        flat = normalizers[m](x.reshape(B * T, C, H, W).float())
        out[m] = flat.reshape(B, T, C, H, W)
    return out


def run_fold(train_samples, val_samples, args, task: str, classes: list[str],
             device: torch.device, fold_idx: int, ckpt_dir: Path | None) -> dict:
    n_classes = len(classes)
    # Outer-fold patients are the test set either way; under the nested
    # protocol, selection moves to a select split carved out of train.
    test_samples = val_samples
    if args.inner_val_fraction > 0:
        train_samples, val_samples = patient_level_split(
            train_samples, val_fraction=args.inner_val_fraction,
            seed=args.seed + 1000 + fold_idx, strata_key=joint_strata_key,
        )
        fit_keys = {s.patient_key for s in train_samples}
        select_keys = {s.patient_key for s in val_samples}
        test_keys = {s.patient_key for s in test_samples}
        assert not (fit_keys & select_keys or fit_keys & test_keys or select_keys & test_keys), \
            "fit/select/test patient sets overlap"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_ds = JointPatientDataset(train_samples, task, img_size=args.img_size,
                                    max_frames=args.max_frames, train=True, seed=args.seed,
                                    lesion_crop=args.lesion_crop, bg_remove_prob=args.bg_remove_prob)
    val_ds = JointPatientDataset(val_samples, task, img_size=args.img_size,
                                  max_frames=args.max_frames, train=False, seed=args.seed,
                                  lesion_crop=args.lesion_crop, bg_remove_prob=args.bg_remove_prob)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=joint_collate,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=joint_collate,
        num_workers=min(2, args.num_workers), pin_memory=device.type == "cuda",
    )
    if test_samples is val_samples:
        test_ds, test_loader = val_ds, val_loader
    else:
        test_ds = JointPatientDataset(test_samples, task, img_size=args.img_size,
                                       max_frames=args.max_frames, train=False, seed=args.seed,
                                       lesion_crop=args.lesion_crop, bg_remove_prob=args.bg_remove_prob)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=joint_collate,
            num_workers=min(2, args.num_workers), pin_memory=device.type == "cuda",
        )

    model = JointTwoTowerModel(
        n_classes, backbone=args.backbone, freeze=args.freeze, fusion=args.fusion,
        modality_dropout=args.modality_dropout, dropout=args.dropout,
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)
    n_params = {"total": sum(p.numel() for p in model.parameters()),
                "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    normalizers = tower_normalizers(args.backbone)

    ytr = np.array([
        classes.index(patient_label(train_ds.subtype[k], task)) for k in train_ds.patient_keys
    ])
    weight_tensor = None
    if args.loss == "weighted_ce":
        w = compute_classification_class_weights(ytr, n_classes)
        weight_tensor = torch.tensor(w, dtype=torch.float32, device=device)

    if args.backbone_lr is not None and args.backbone_lr != args.lr:
        backbone_params = [p for p in model.backbones.parameters() if p.requires_grad]
        head_params = [p for p in model.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [{"params": backbone_params, "lr": args.backbone_lr},
             {"params": head_params, "lr": args.lr}],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr,
            weight_decay=args.weight_decay,
        )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = {"bal": -1.0, "acc": 0.0, "epoch": 0}
    history = []
    start_epoch = 0
    resume_path = (ckpt_dir / f"fold{fold_idx}_resume.pt") if ckpt_dir is not None else None
    if resume_path is not None and resume_path.exists():
        rs = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(rs["model_state_dict"])
        optimizer.load_state_dict(rs["optimizer_state_dict"])
        scheduler.load_state_dict(rs["scheduler_state_dict"])
        scaler.load_state_dict(rs["scaler_state_dict"])
        best, history, start_epoch = rs["best"], rs["history"], rs["epoch"]
        print(f"    fold {fold_idx}: resuming from epoch {start_epoch+1}/{args.epochs} "
              f"(best so far: bal={best['bal']:.4f} @ epoch {best['epoch']})", flush=True)

    for epoch in range(start_epoch, args.epochs):
        epoch_t0 = time.time()
        model.train()
        train_loss_sum, train_n = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, masks, labels) in enumerate(train_loader):
            images_norm = _normalize_frames(images, normalizers, device)
            masks_dev = {m: masks[m].to(device, non_blocking=True) for m in MODALITY_NAMES}
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images_norm, masks_dev)
                loss = F.cross_entropy(logits, labels, weight=weight_tensor)
            scaler.scale(loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_loss_sum += float(loss.item()) * len(labels)
            train_n += len(labels)
        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images_norm = _normalize_frames(images, normalizers, device)
                masks_dev = {m: masks[m].to(device, non_blocking=True) for m in MODALITY_NAMES}
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(images_norm, masks_dev)
                    loss = F.cross_entropy(logits, labels)
                val_loss_sum += float(loss.item()) * len(labels)
                val_n += len(labels)
                all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        val_loss = val_loss_sum / max(val_n, 1)
        probs = np.concatenate(all_probs) if all_probs else np.zeros((0, n_classes))
        yva = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.int64)
        pred = probs.argmax(1) if len(probs) else np.zeros(0, dtype=np.int64)
        acc = float((pred == yva).mean()) if len(pred) else 0.0
        bal = balanced_accuracy(yva, pred, n_classes)
        scheduler.step(bal)
        lr = optimizer.param_groups[-1]["lr"]   # head LR; also last group in the single-group case
        backbone_lr = optimizer.param_groups[0]["lr"]

        improved = bal > best["bal"]
        if improved:
            best = {"bal": bal, "acc": acc, "epoch": epoch + 1}
            if ckpt_dir is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(), "task": task,
                    "backbone": args.backbone, "unfreeze_last_n": args.unfreeze_last_n,
                    "freeze": args.freeze, "fusion": args.fusion,
                    "modality_dropout": args.modality_dropout, "img_size": args.img_size,
                    "max_frames": args.max_frames, "fold": fold_idx, "bal": bal,
                    "inner_val_fraction": args.inner_val_fraction,
                    "lesion_crop": args.lesion_crop, "bg_remove_prob": args.bg_remove_prob,
                }, ckpt_dir / f"fold{fold_idx}_best.pt")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                         "lr": lr, "backbone_lr": backbone_lr, "acc": acc, "bal": bal})
        marker = " *" if improved else ""
        print(f"    fold {fold_idx} epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr:.2e}  bb_lr={backbone_lr:.2e}  "
              f"acc={acc:.4f}  bal={bal:.4f}  ({time.time()-epoch_t0:.0f}s){marker}", flush=True)

        if resume_path is not None:
            torch.save({
                "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best": best, "history": history,
            }, resume_path)

        if args.early_stop_patience > 0 and (epoch + 1) - best["epoch"] >= args.early_stop_patience:
            break

    # Fold finished cleanly (full budget or early stop, not a crash) -- the
    # per-epoch resume state is no longer needed for this fold.
    if resume_path is not None and resume_path.exists():
        resume_path.unlink()

    # Reload the best checkpoint for the masked-inference ablation pass -- the
    # in-memory `model` is whatever epoch the loop last left it at, not
    # necessarily the best one.
    if ckpt_dir is not None:
        ck = torch.load(ckpt_dir / f"fold{fold_idx}_best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
    model.eval()
    peak_mem_gb = (round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
                   if device.type == "cuda" else None)

    per_patient = []
    with torch.no_grad():
        for images, masks, labels in test_loader:
            images_norm = _normalize_frames(images, normalizers, device)
            paired = (masks["WLE"].sum(dim=1) > 0.5) & (masks["EUS"].sum(dim=1) > 0.5)
            for name, use_modalities in ABLATION_MODALITIES.items():
                masks_dev = {
                    m: (masks[m].to(device) if m in use_modalities
                        else torch.zeros_like(masks[m]).to(device))
                    for m in MODALITY_NAMES
                }
                logits = model(images_norm, masks_dev)
                pred_b = torch.softmax(logits.float(), dim=1).argmax(1).cpu().numpy()
                for i in range(len(labels)):
                    per_patient.append({
                        "ablation": name, "y_true": int(labels[i]), "y_pred": int(pred_b[i]),
                        "paired": bool(paired[i]),
                    })

    def _agg(ablation: str, restrict_paired: bool):
        sel = [r for r in per_patient if r["ablation"] == ablation
               and (not restrict_paired or r["paired"])
               and (ablation != "eus_only" or r["paired"])]
        if not sel:
            return None
        pred_arr = np.array([r["y_pred"] for r in sel])
        truth_arr = np.array([r["y_true"] for r in sel])
        return {**score(pred_arr, truth_arr, n_classes), "n": len(sel)}

    ablation_results = {
        ablation: {"unpaired": _agg(ablation, False), "paired_only": _agg(ablation, True)}
        for ablation in ABLATION_MODALITIES
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "fold": fold_idx, "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "best": best, "history": history, "ablations": ablation_results,
        "n_params": n_params, "peak_mem_gb": peak_mem_gb,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["binary", "subtype"], default="subtype")
    p.add_argument("--freeze", choices=["none", "partial"], default="partial")
    p.add_argument("--fusion", choices=["gated", "symmetric"], default="gated")
    p.add_argument("--modality-dropout", type=float, default=0.3)
    p.add_argument("--loss", choices=["ce", "weighted_ce"], default="weighted_ce")
    p.add_argument("--max-frames", type=int, default=4)
    p.add_argument("--lesion-crop", action="store_true",
                    help="crop each frame to its lesion box + 10%% context padding before "
                         "the standard augmentation pipeline, instead of feeding the whole "
                         "frame -- global average pooling over the whole frame otherwise "
                         "mostly encodes context (cadsel/fusion.py's own rationale for using "
                         "ROI features instead, in the original detector-based fusion)")
    p.add_argument("--bg-remove-prob", type=float, default=0.0,
                    help="stochastic train-time augmentation (mutually exclusive with "
                         "--lesion-crop): with this probability per frame, erase everything "
                         "outside the lesion box (filled with the frame's own mean colour) "
                         "before the standard augmentation pipeline, forcing that frame to "
                         "be classified without background signal -- directly targets "
                         "Experiment 3's finding that labels are 68-95%% recoverable from "
                         "lesion-free crops alone. Default 0.0 (off).")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4,
                    help="AdamW weight decay (default: every prior confirmed result "
                         "in this project used 1e-4)")
    p.add_argument("--dropout", type=float, default=0.3,
                    help="FusionHead encoder/classifier dropout (default: 0.3, the "
                         "value every prior confirmed result used)")
    p.add_argument("--backbone-lr", type=float, default=None,
                    help="discriminative LR for the two backbone towers, separate from "
                         "--lr (used for the FusionHead) -- e.g. 1e-5 while the head trains "
                         "at 1e-4. Default None means one shared LR for everything (the "
                         "original behaviour).")
    p.add_argument("--batch-size", type=int, default=8, help="patients per batch")
    p.add_argument("--grad-accum", type=int, default=1,
                    help="optimizer step every N batches (e.g. --batch-size 4 --grad-accum 2 "
                         "keeps an effective 8 patients when a large tower doesn't fit)")
    p.add_argument("--backbone", choices=sorted(TOWERS), default="efficientnet_b0",
                    help="tower backbone, same for both modalities (see cadsel.joint_fusion_model.TOWERS)")
    p.add_argument("--unfreeze-last-n", type=int, default=4,
                    help="ViT towers with --freeze partial: trainable transformer blocks from the end")
    p.add_argument("--inner-val-fraction", type=float, default=0.0,
                    help="> 0: leak-free nested protocol -- this share of each outer-train fold "
                         "becomes the select set (checkpoint, LR plateau, early stop) and the "
                         "outer fold is only ever scored. 0 (default): the original protocol, "
                         "where the scored fold also picks the checkpoint.")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--max-folds", type=int, default=None, help="stop after this many folds (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paired-only", action="store_true",
                    help="restrict to the 421 patients that have both modalities")
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--save-checkpoint", action="store_true", default=True)
    p.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = (
        f"joint_{args.backbone}__{args.task}__freeze-{args.freeze}__fusion-{args.fusion}"
        f"__loss-{args.loss}__moddrop{args.modality_dropout:g}__mf{args.max_frames}"
        f"{'' if args.epochs == 30 else f'__ep{args.epochs}'}"
        f"{'' if args.early_stop_patience == 0 else f'__es{args.early_stop_patience}'}"
        f"{'' if args.backbone_lr is None or args.backbone_lr == args.lr else f'__blr{args.backbone_lr:g}'}"
        f"{'__lesioncrop' if args.lesion_crop else ''}"
        f"{'' if args.bg_remove_prob == 0.0 else f'__bgremove{args.bg_remove_prob:g}'}"
        f"{'' if args.weight_decay == 1e-4 else f'__wd{args.weight_decay:g}'}"
        f"{'' if args.dropout == 0.3 else f'__drop{args.dropout:g}'}"
        f"{'' if args.unfreeze_last_n == 4 or args.freeze != 'partial' else f'__unfreeze{args.unfreeze_last_n}'}"
        f"{'' if args.batch_size == 8 else f'__bs{args.batch_size}'}"
        f"{'' if args.grad_accum == 1 else f'__ga{args.grad_accum}'}"
        f"{'' if args.inner_val_fraction == 0 else f'__nested{args.inner_val_fraction:g}'}"
        f"{'__paired' if args.paired_only else ''}__s{args.seed}"
    )
    out_path = args.output_dir / f"{run_name}.json"
    ckpt_dir = (args.output_dir / run_name) if args.save_checkpoint else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_joint_manifest()
    if args.paired_only:
        paired_keys = {s.patient_key for s in samples if s.modality == "EUS"}
        samples = [s for s in samples if s.patient_key in paired_keys]

    classes = patient_classes(args.task)
    folds = patient_level_folds(samples, n_folds=args.n_splits, seed=args.seed,
                                 strata_key=joint_strata_key)
    n_patients = len({s.patient_key for s in samples})

    # Resume support: reuse folds a prior (interrupted) run of this exact
    # run_name already completed, rather than re-training them from scratch --
    # combined with run_fold's own per-epoch resume state, a job killed
    # mid-fold (session restart, etc.) picks back up at the next epoch rather
    # than losing the whole fold, and a job killed between folds skips
    # straight to the first incomplete one.
    fold_results = []
    completed = set()
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            fold_results = prior.get("folds", [])
            completed = {f["fold"] for f in fold_results}
        except (json.JSONDecodeError, KeyError):
            fold_results = []
    if completed:
        print(f"[{run_name}] resuming: fold(s) {sorted(completed)} already complete, skipping")

    def _summarize(ablation: str, denom: str):
        vals = [f["ablations"][ablation][denom] for f in fold_results
                if f["ablations"][ablation][denom] is not None]
        if not vals:
            return None
        out = {}
        for metric in ("acc", "bal", "macro_f1"):
            arr = [v[metric] for v in vals]
            out[metric] = {"mean": round(float(np.mean(arr)), 4), "std": round(float(np.std(arr)), 4)}
        return out

    def _write_result() -> dict:
        summary = {
            ablation: {denom: _summarize(ablation, denom) for denom in ("unpaired", "paired_only")}
            for ablation in ABLATION_MODALITIES
        }
        result = {
            "run_name": run_name, "backbone": args.backbone, "task": args.task,
            "freeze": args.freeze, "unfreeze_last_n": args.unfreeze_last_n,
            "fusion": args.fusion, "loss": args.loss,
            "modality_dropout": args.modality_dropout, "max_frames": args.max_frames,
            "img_size": args.img_size, "epochs": args.epochs, "lr": args.lr,
            "weight_decay": args.weight_decay, "dropout": args.dropout,
            "backbone_lr": args.backbone_lr, "early_stop_patience": args.early_stop_patience,
            "lesion_crop": args.lesion_crop, "bg_remove_prob": args.bg_remove_prob,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "n_splits": args.n_splits, "seed": args.seed,
            "paired_only": args.paired_only, "n_patients": n_patients,
            "inner_val_fraction": args.inner_val_fraction,
            "protocol": ("nested: checkpoint/LR/early-stop chosen on an inner select split, "
                         "summary scored on the outer fold only") if args.inner_val_fraction > 0
                        else "original: checkpoint chosen on the scored fold (optimistic)",
            "chance": round(1 / len(classes), 4),
            "folds_done": len(fold_results),
            "summary": summary,
            "folds": fold_results,
            "elapsed_sec": round(sum(f.get("elapsed_sec", 0.0) for f in fold_results), 1),
        }
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    print(f"[{run_name}] device={device} {n_patients} patients, {args.n_splits}-fold CV")
    for k, (train_samples, val_samples) in enumerate(folds):
        if args.max_folds is not None and k >= args.max_folds:
            break
        if k in completed:
            continue
        t0 = time.time()
        r = run_fold(train_samples, val_samples, args, args.task, classes, device, k, ckpt_dir)
        r["elapsed_sec"] = round(time.time() - t0, 1)
        fold_results.append(r)
        fused = r["ablations"]["fused"]["unpaired"]
        print(f"  fold {k}: select bal={r['best']['bal']:.4f} (epoch {r['best']['epoch']})  "
              f"test fused bal={fused['bal']:.4f}  n_train={r['n_train']} n_val={r['n_val']} "
              f"n_test={r['n_test']}  peak_mem={r['peak_mem_gb']}GB  {r['elapsed_sec']:.0f}s")
        _write_result()

    result = _write_result()
    print(f"\n[{run_name}] {result['folds_done']}-fold summary (mean +/- std, chance={result['chance']:.3f}):")
    for ablation in ABLATION_MODALITIES:
        for denom in ("unpaired", "paired_only"):
            s = result["summary"][ablation][denom]
            if s is not None:
                print(f"  {ablation:<10} {denom:<12} bal {s['bal']['mean']:.4f} +/- {s['bal']['std']:.4f}  "
                      f"macroF1 {s['macro_f1']['mean']:.4f} +/- {s['macro_f1']['std']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
