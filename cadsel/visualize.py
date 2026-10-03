"""Reproduce Fig 4: ground-truth boxes beside the model's predictions.

python -m cadsel.visualize --checkpoint runs/fasterrcnn_mobilenet__WLE__binary/best_model.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import PROJECT_ROOT, TASK_CLASSES, TASK_ID_MAP
from .data import (
    build_dataloaders,
    build_manifest,
    make_normalizer,
    patient_level_split,
    read_yolo_label,
)
from .engine import evaluate
from .models import build_detection_model, configure_transform_for_input_mode


def _draw(ax, image, boxes, labels, class_names, scores=None, title=""):
    ax.imshow(image)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none"
            )
        )
        name = class_names[int(label) - 1] if 0 < int(label) <= len(class_names) else str(label)
        text = f"{name} {scores[i]:.2f}" if scores is not None else name
        ax.text(
            x1, max(y1 - 6, 8), text, color="white", fontsize=8,
            bbox=dict(facecolor="red", alpha=0.75, pad=1.5, edgecolor="none"),
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Qualitative predictions (Fig 4 style)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-images", type=int, default=6)
    p.add_argument("--conf-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    task = ckpt["task"]
    modality = ckpt["modality"]
    img_size = ckpt["img_size"]
    input_mode = ckpt.get("input_mode", "reference")
    class_names = TASK_CLASSES[task]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detection_model(model_type, task, pretrained=False)
    configure_transform_for_input_mode(model, input_mode, img_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    samples = build_manifest(modality)
    _, val_samples = patient_level_split(samples, args.val_fraction, args.seed)
    _, val_loader = build_dataloaders(
        val_samples[:1], val_samples, task, img_size, batch_size=4,
        num_workers=args.num_workers, input_mode=input_mode, seed=args.seed,
    )

    metrics = evaluate(
        model, val_loader, device, class_names,
        conf_threshold=args.conf_threshold, progress=False, collect_predictions=True,
        normalize=make_normalizer(input_mode),
    )

    # Prefer confident, correctly-typed examples spread across classes.
    id_map = TASK_ID_MAP[task]
    by_class: dict[int, list[dict]] = {i + 1: [] for i in range(len(class_names))}
    for pred in metrics["predictions"]:
        keep = pred["scores"] > args.conf_threshold
        if not keep.any() or len(pred["gt_labels"]) == 0:
            continue
        top = int(np.argmax(pred["scores"]))
        if pred["labels"][top] != pred["gt_labels"][0]:
            continue
        by_class[int(pred["gt_labels"][0])].append(pred)

    chosen: list[dict] = []
    round_robin = True
    while round_robin and len(chosen) < args.num_images:
        round_robin = False
        for cid in sorted(by_class):
            bucket = by_class[cid]
            if bucket and len(chosen) < args.num_images:
                bucket.sort(key=lambda d: -float(d["scores"].max()))
                chosen.append(bucket.pop(0))
                round_robin = True

    if not chosen:
        print("No confident correct predictions to visualise.")
        return

    fig, axes = plt.subplots(len(chosen), 2, figsize=(9, 4.2 * len(chosen)))
    axes = np.atleast_2d(axes)

    for row, pred in enumerate(chosen):
        sample = val_samples[pred["index"]]
        with Image.open(sample.image) as im:
            raw = im.convert("RGB")
            if input_mode == "reference":
                raw = raw.resize((img_size, img_size), Image.BILINEAR)
            image = np.asarray(raw)

        _draw(axes[row, 0], image, pred["gt_boxes"], pred["gt_labels"], class_names,
              title=f"Ground truth — {sample.subtype} ({modality})")

        keep = pred["scores"] > args.conf_threshold
        _draw(axes[row, 1], image, pred["boxes"][keep], pred["labels"][keep], class_names,
              scores=pred["scores"][keep], title=f"Prediction — {model_type}")

    fig.tight_layout()
    out = args.output or (PROJECT_ROOT / "results" / f"predictions_{model_type}_{modality}_{task}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
