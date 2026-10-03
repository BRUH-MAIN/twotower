"""Two-stage fine-tuning for rare classes (TFA-style).

Wang et al. 2020, "Frustratingly Simple Few-Shot Object Detection": after
normal training, freeze everything except the box predictor (classification +
regression head) and briefly fine-tune at a low learning rate on a
class-balanced subset. CAD-SEL isn't literally few-shot (112 Leiomyoma boxes,
not 1-10), but the mechanism this corrects for is the same one FSOD papers use
for the tail -- the frequent class's gradient dominates a jointly-trained
decision boundary even with loss weighting, because loss weighting changes
*how much* a box's gradient counts, not *how many* rare-class boxes the
optimiser ever sees relative to frequent ones in a given step.

Stage 1 is the normal `cadsel.train` run; this is stage 2, run against its
checkpoint:

    python -m cadsel.finetune_rare --checkpoint runs/.../best_model.pth --epochs 10
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import torch

from .config import MODEL_REGISTRY, PROJECT_ROOT, TASK_CLASSES, TASK_ID_MAP
from .data import Sample, build_dataloaders, build_manifest, make_normalizer, patient_level_split, read_yolo_label
from .engine import evaluate, train_one_epoch
from .models import build_detection_model, configure_transform_for_input_mode


def build_balanced_subset(
    samples: Sequence[Sample], task: str, target_per_class: int | None = None, seed: int = 0
) -> list[Sample]:
    """Images grouped by which foreground classes they contain; from each
    class's image pool, sample up to ``target_per_class`` (default: the
    rarest class's own image count). An image containing multiple classes can
    be pulled in by any of them and is included once, so the class balance of
    the *returned set* is much flatter than the original training split even
    though per-image class co-occurrence isn't touched.
    """
    id_map = TASK_ID_MAP[task]
    by_class: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        if getattr(s, "negative", False):
            continue
        classes = {id_map[c] for c, *_ in read_yolo_label(s.label)}
        for c in classes:
            by_class[c].append(s)

    if not by_class:
        return []
    if target_per_class is None:
        target_per_class = min(len(pool) for pool in by_class.values())

    rng = random.Random(seed)
    chosen: dict[str, Sample] = {}
    for pool in by_class.values():
        shuffled = list(pool)
        rng.shuffle(shuffled)
        for s in shuffled[:target_per_class]:
            chosen[s.image] = s
    return list(chosen.values())


def build_balanced_subset_by_center(
    samples: Sequence[Sample], target_per_center: int | None = None, seed: int = 0
) -> list[Sample]:
    """DFR-style (Kirichenko et al. 2023) center-balanced subset -- sibling of
    ``build_balanced_subset``, bucketing by ``Sample.center`` instead of
    foreground class. Targets a leave-one-center-out training pool's
    imbalance (e.g. holding out Center-3 leaves a pool that's 93.7% Center-1
    by patient count) the same way ``build_balanced_subset`` targets class
    imbalance for TFA.
    """
    by_center: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        if getattr(s, "negative", False):
            continue
        by_center[s.center].append(s)

    if not by_center:
        return []
    if target_per_center is None:
        target_per_center = min(len(pool) for pool in by_center.values())

    rng = random.Random(seed)
    chosen: dict[str, Sample] = {}
    for pool in by_center.values():
        shuffled = list(pool)
        rng.shuffle(shuffled)
        for s in shuffled[:target_per_center]:
            chosen[s.image] = s
    return list(chosen.values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument(
        "--target-per-class", type=int, default=None,
        help="images per class in the balanced subset (default: the rarest class's own count)",
    )
    p.add_argument(
        "--balance-by", choices=["class", "center"], default="class",
        help="'class' (default) is the original TFA recipe, class-balanced. 'center' is "
        "DFR-style, center-balanced -- targets a leave-one-center-out training pool's "
        "imbalance instead of class imbalance.",
    )
    p.add_argument(
        "--target-per-center", type=int, default=None,
        help="images per center in the balanced subset (--balance-by center only; "
        "default: the smallest center's own count)",
    )
    p.add_argument(
        "--centers", default=None,
        help="comma-separated center filter, matching a leave-one-center-out base "
        "checkpoint's own --centers exclusion (see cadsel.train --centers) -- without "
        "this, fine-tuning would silently pull in the checkpoint's held-out center and "
        "invalidate any later external-AP measurement on it. Default: no filter.",
    )
    p.add_argument("--batch-size", type=int, default=None, help="default: the model's registry batch size")
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-seed", type=int, default=None,
        help="seed for patient_level_split (default: same as --seed, for backward "
        "compatibility). Pass the base checkpoint's own split seed explicitly when "
        "sweeping --seed, so every sweep run validates on the SAME held-out patients "
        "as the base checkpoint rather than a reshuffled set -- --seed alone drives "
        "both the split and training randomness, which silently confounds a seed "
        "sweep meant to isolate only fine-tuning noise.",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    split_seed = args.split_seed if args.split_seed is not None else args.seed

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type, task, modality = ckpt["model_type"], ckpt["task"], ckpt["modality"]
    img_size = ckpt["img_size"]
    multiscale_sizes = ckpt.get("multiscale_sizes")
    input_mode = ckpt.get("input_mode", "reference")
    class_names = TASK_CLASSES[task]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detection_model(model_type, task, pretrained=False)
    configure_transform_for_input_mode(model, input_mode, multiscale_sizes or img_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Freeze everything except the box predictor -- the TFA recipe.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.roi_heads.box_predictor.parameters():
        param.requires_grad = True
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"frozen all but the box predictor: {n_trainable:,} / {n_total:,} params trainable")

    centers = [c.strip() for c in args.centers.split(",") if c.strip()] if args.centers else None
    samples = build_manifest(modality, centers=centers)
    train_samples, val_samples = patient_level_split(samples, args.val_fraction, split_seed)
    if args.balance_by == "center":
        subset = build_balanced_subset_by_center(train_samples, args.target_per_center, args.seed)
    else:
        subset = build_balanced_subset(train_samples, task, args.target_per_class, args.seed)
    print(f"balanced subset ({args.balance_by}): {len(subset)} images "
          f"(from {len(train_samples)} full training images)")

    batch_size = args.batch_size or MODEL_REGISTRY[model_type]["batch_size"]
    train_loader, val_loader = build_dataloaders(
        subset, val_samples, task, img_size, batch_size=batch_size,
        num_workers=args.num_workers, input_mode=input_mode, seed=args.seed,
    )
    normalize = make_normalizer(input_mode)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad], lr=args.learning_rate
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    suffix = "finetune_rare" if args.balance_by == "class" else f"finetune_{args.balance_by}"
    run_name = args.run_name or f"{Path(args.checkpoint).parent.name}__{suffix}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best = {"AP@0.5": -1.0}
    started = time.time()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, scaler=scaler,
            epoch=epoch, total_epochs=args.epochs, progress=not args.no_progress, normalize=normalize,
        )
        metrics = evaluate(
            model, val_loader, device, class_names,
            iou_threshold=args.iou_threshold, conf_threshold=args.conf_threshold,
            use_amp=use_amp, progress=not args.no_progress, normalize=normalize,
        )
        history.append({"epoch": epoch + 1, "train_loss": train_stats["loss"],
                         **{k: v for k, v in metrics.items() if k != "per_class"},
                         "per_class": metrics["per_class"]})
        marker = ""
        if metrics["AP@0.5"] > best["AP@0.5"]:
            best = metrics
            marker = " *"
            torch.save(
                {
                    "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                    "model_type": model_type, "task": task, "modality": modality,
                    "img_size": img_size, "multiscale_sizes": multiscale_sizes,
                    "input_mode": input_mode, "class_names": class_names,
                    "metrics": {k: v for k, v in metrics.items() if k != "predictions"},
                    "finetune_rare_from": str(args.checkpoint),
                },
                out_dir / "best_model.pth",
            )
        per_class_rare = " ".join(f"{c}={metrics['per_class'][c]['AP@0.5']:.3f}" for c in class_names)
        print(f"  epoch {epoch + 1:3d}/{args.epochs}  train_loss={train_stats['loss']:.4f}  "
              f"AP@0.5={metrics['AP@0.5']:.4f}  [{per_class_rare}]{marker}")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    result = {
        "run_name": run_name, "finetuned_from": str(args.checkpoint),
        "model": model_type, "modality": modality, "task": task,
        "balance_by": args.balance_by,
        "target_per_class": args.target_per_class, "target_per_center": args.target_per_center,
        "centers": args.centers, "subset_size": len(subset),
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "seed": args.seed, "split_seed": split_seed,
        "elapsed_sec": time.time() - started, "best": best,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[{run_name}] best AP@0.5={best['AP@0.5']:.4f}; wrote {out_dir}")


if __name__ == "__main__":
    main()
