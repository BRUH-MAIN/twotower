"""Train one joint WLE+EUS detector.

A single Faster R-CNN over both modalities, evaluated per modality so its rows stay
comparable to the paper's Table 4/5 (which are per-modality).

Two things motivate this over the unimodal baselines:

* **Rare-class supervision.** Table 4's mAP is a mean over classes and is sunk by
  the rare ones — Leiomyoma has 139 WLE boxes and 81 EUS. Pooling the modalities
  takes that to 220, and Lipoma to 762.
* **A leak-free split.** Splitting each modality independently, as the unimodal
  baselines do, leaves 67 of 85 EUS validation patients with their WLE frames in
  the training set. `build_joint_manifest` + `patient_level_split` assign whole
  patients across both modalities at once.

Example
-------
python -m cadsel.train_joint --conditioning film --task multiclass
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import (
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    MODALITIES,
    MODEL_NAMES,
    MODEL_REGISTRY,
    PROJECT_ROOT,
    TASKS,
    TASK_CLASSES,
)
from .data import (
    AUGMENT_OPS,
    DEFAULT_AUGMENT,
    build_joint_dataloaders,
    build_joint_manifest,
    compute_class_weights,
    image_level_split,
    joint_strata_key,
    make_hard_negatives,
    make_normalizer,
    patient_level_split,
    save_split,
    split_summary,
)
from .engine import evaluate, train_one_epoch
from .models import (
    CONDITIONING_MODES,
    build_class_weight_tensor,
    build_joint_detection_model,
    configure_transform_for_input_mode,
    install_weighted_loss,
    set_class_weights,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a joint WLE+EUS CAD-SEL detector")
    p.add_argument("--model", choices=MODEL_NAMES, default="fasterrcnn_mobilenet")
    p.add_argument("--task", choices=TASKS, default="multiclass")
    p.add_argument(
        "--conditioning",
        choices=CONDITIONING_MODES,
        default="film",
        help="'none' is the modality-blind control — required to attribute any "
        "gain to conditioning rather than to 1.75x the training images",
    )
    p.add_argument(
        "--modalities",
        default=",".join(MODALITIES),
        help="comma-separated; the default trains on both",
    )

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None, help="default: per-model (see config)")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--img-size", type=int, default=None, help="default: per-model (640 / 320)")
    p.add_argument("--num-workers", type=int,
                   default=min(8, max(2, (os.cpu_count() or 4) // 2)))
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="stop after N epochs with no macro AP@0.5 improvement; 0 disables")
    p.add_argument(
        "--split-level",
        choices=["patient", "image"],
        default="patient",
        help="'image' leaks correlated frames across the split; kept only to "
        "produce the column that compares against the paper's published figures",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)

    p.add_argument(
        "--augment",
        default=",".join(DEFAULT_AUGMENT),
        help=f"comma-separated from {AUGMENT_OPS}. 'copy_paste' is rejected for "
        "joint runs — see build_joint_dataloaders.",
    )
    p.add_argument("--hard-negatives", type=float, default=0.0,
                   help="fraction of lesion-free negatives to add (0.3 recommended)")
    p.add_argument("--input-mode", choices=["reference", "torchvision"], default="reference")
    p.add_argument(
        "--shared-norm",
        action="store_true",
        help="use the single global CADSEL_MEAN/STD for both modalities instead of "
        "MODALITY_STATS. Diagnostic only: it centres EUS near -1.7.",
    )
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    p.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)

    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-checkpoint", action="store_true", default=True)
    p.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p


def assert_split_is_leak_free(train_samples, val_samples) -> None:
    """No patient may straddle the split, in either modality.

    This is the specific defect the per-modality baselines have: 67 of their 85
    EUS validation patients have WLE frames in the WLE training set. Any model
    that sees both modalities would read straight through it, so it is checked
    rather than commented.
    """
    train_keys = {s.patient_key for s in train_samples}
    val_keys = {s.patient_key for s in val_samples}
    overlap = train_keys & val_keys
    if overlap:
        raise AssertionError(
            f"{len(overlap)} patient(s) appear in both splits, e.g. "
            f"{sorted(overlap)[:3]} — the joint split must assign whole patients"
        )


def _write_result(out_dir, args, spec, run_name, img_size, batch_size,
                  summary, best, best_epoch, epochs_run, elapsed) -> dict:
    result = {
        "run_name": run_name,
        "model": args.model,
        "paper_name": spec["paper_name"],
        "modality": "JOINT",
        "modalities": [m.strip() for m in args.modalities.split(",") if m.strip()],
        "conditioning": args.conditioning,
        "task": args.task,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": args.epochs,
        "epochs_run": epochs_run,
        "learning_rate": args.learning_rate,
        "input_mode": args.input_mode,
        "per_modality_norm": not args.shared_norm,
        "augment": [o.strip() for o in args.augment.split(",") if o.strip()],
        "hard_negatives": args.hard_negatives,
        "split_level": args.split_level,
        "class_weighting": not args.no_class_weights,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "elapsed_sec": elapsed,
        "split": summary,
        "best": best,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    spec = MODEL_REGISTRY[args.model]
    img_size = args.img_size or spec["default_img_size"]
    batch_size = args.batch_size or spec["batch_size"]
    class_names = TASK_CLASSES[args.task]
    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]

    # The split level goes in the name: the two protocols are reported side by
    # side, so both runs must coexist rather than overwrite each other.
    suffix = "" if args.split_level == "patient" else f"__{args.split_level}level"
    run_name = (args.run_name
                or f"joint_{args.conditioning}__{args.model}__{args.task}{suffix}")
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    show_bars = not (args.quiet or args.no_progress)

    # -- data ---------------------------------------------------------------
    samples = build_joint_manifest(modalities, data_root=args.data_root)
    if args.split_level == "image":
        print("  WARNING: image-level split — correlated frames leak across it. "
              "Valid only as the column that compares to the paper's figures.")
        train_samples, val_samples = image_level_split(
            samples, val_fraction=args.val_fraction, seed=args.seed
        )
    else:
        train_samples, val_samples = patient_level_split(
            samples, val_fraction=args.val_fraction, seed=args.seed,
            strata_key=joint_strata_key,
        )
        assert_split_is_leak_free(train_samples, val_samples)

    save_split(out_dir / "split.json", train_samples, val_samples)
    summary = split_summary(train_samples, val_samples)

    augment_ops = [o.strip() for o in args.augment.split(",") if o.strip()]

    if args.hard_negatives > 0:
        negatives = make_hard_negatives(train_samples, args.hard_negatives, args.seed)
        print(f"  hard negatives: +{len(negatives)} lesion-free images "
              f"({args.hard_negatives:.0%} of {len(train_samples)})")
        train_samples = list(train_samples) + negatives

    train_loader, val_loader = build_joint_dataloaders(
        train_samples, val_samples, task=args.task, img_size=img_size,
        batch_size=batch_size, num_workers=args.num_workers,
        input_mode=args.input_mode, seed=args.seed, augment_ops=augment_ops,
    )

    # -- class weights (over the union) --------------------------------------
    fg_weights, counts = compute_class_weights(train_samples, args.task)
    install_weighted_loss()
    if args.no_class_weights:
        set_class_weights(None)
        weight_tensor = None
    else:
        weight_tensor = build_class_weight_tensor(fg_weights).to(device)
        set_class_weights(weight_tensor)

    # -- model --------------------------------------------------------------
    model = build_joint_detection_model(
        args.model, args.task, conditioning=args.conditioning,
        pretrained=not args.no_pretrained, verbose=not args.quiet,
    )
    configure_transform_for_input_mode(model, args.input_mode, img_size)
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    normalize = make_normalizer(args.input_mode, per_modality=not args.shared_norm)

    if not args.quiet:
        print(f"[{run_name}] device={device} img_size={img_size} "
              f"batch_size={batch_size} workers={args.num_workers} amp={use_amp}",
              flush=True)
        for side in ("train", "val"):
            s = summary[side]
            print(f"  {side}: {s['images']} images / {s['patients']} patients "
                  f"{s.get('by_modality', {})} paired={s.get('paired_patients')}")
        print(f"  box counts per class {dict(zip(class_names, counts.astype(int).tolist()))}")
        if weight_tensor is not None:
            print(f"  class weights (bg first) {weight_tensor.cpu().numpy().round(4).tolist()}")
        print(f"  normalisation: {'per-modality' if not args.shared_norm else 'SHARED (diagnostic)'}")
        print(flush=True)

    # -- training loop ------------------------------------------------------
    history = []
    best = {"AP@0.5_macro": -1.0}
    best_epoch = -1
    started = time.time()

    for epoch in range(args.epochs):
        epoch_started = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, scaler=scaler, epoch=epoch,
            total_epochs=args.epochs, progress=show_bars,
            warmup_iters=args.warmup_iters, base_lr=args.learning_rate,
            normalize=normalize, modality_aware=True,
        )
        metrics = evaluate(
            model, val_loader, device, class_names,
            iou_threshold=args.iou_threshold, conf_threshold=args.conf_threshold,
            use_amp=use_amp, progress=show_bars, with_loss=True,
            normalize=normalize, modality_aware=True,
        )
        val_loss = metrics.pop("val_loss")
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{k: v for k, v in metrics.items() if k not in ("per_class", "per_modality")},
            "per_class": metrics["per_class"],
            "per_modality": {
                m: {k: v for k, v in s.items() if k != "per_class"}
                for m, s in metrics.get("per_modality", {}).items()
            },
        })

        if not args.quiet:
            improved = metrics["AP@0.5_macro"] > best["AP@0.5_macro"]
            per_mod = "  ".join(
                f"{m}={s['AP@0.5']:.4f}" for m, s in metrics.get("per_modality", {}).items()
            )
            print(f"  epoch {epoch + 1:3d}/{args.epochs}  "
                  f"train_loss={train_stats['loss']:.4f}  val_loss={val_loss:.4f}  "
                  f"macroAP={metrics['AP@0.5_macro']:.4f}  [{per_mod}]  "
                  f"F1={metrics['f1']:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}  "
                  f"({time.time() - epoch_started:.0f}s)"
                  f"{' *' if improved else ''}", flush=True)

        if metrics["AP@0.5_macro"] > best["AP@0.5_macro"]:
            best = {k: v for k, v in metrics.items()}
            best_epoch = epoch + 1
            if args.save_checkpoint:
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "model_type": args.model,
                    "task": args.task,
                    "modality": "JOINT",
                    "modalities": modalities,
                    "conditioning": args.conditioning,
                    "img_size": img_size,
                    "input_mode": args.input_mode,
                    "per_modality_norm": not args.shared_norm,
                    "class_names": class_names,
                    "metrics": {k: v for k, v in metrics.items() if k != "predictions"},
                }, out_dir / "best_model.pth")

        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        _write_result(out_dir, args, spec, run_name, img_size, batch_size,
                      summary, best, best_epoch, len(history), time.time() - started)

        if args.early_stop_patience > 0:
            stale = (epoch + 1) - best_epoch
            if stale >= args.early_stop_patience:
                if not args.quiet:
                    print(f"  early stop: no macro AP@0.5 improvement for {stale} "
                          f"epochs (best was epoch {best_epoch})", flush=True)
                break

    result = _write_result(out_dir, args, spec, run_name, img_size, batch_size,
                           summary, best, best_epoch, len(history), time.time() - started)

    if not args.quiet:
        per_mod = "  ".join(
            f"{m}={s['AP@0.5']:.4f}" for m, s in best.get("per_modality", {}).items()
        )
        print(f"[{run_name}] best epoch {best_epoch}: "
              f"macroAP={best['AP@0.5_macro']:.4f}  [{per_mod}]")
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
