"""Experiment 29 (Track B, Phase 3) — fair AP@0.5 eval bridge, scoring YOLO
predictions through the exact same ``cadsel/metrics.py`` code (
``DetectionAccumulator``/``summarize``) that produced the existing Table 4/5
Faster R-CNN numbers.

``--model-family fasterrcnn`` is this bridge's own correctness check, not a
second production path: it runs an *existing* torchvision checkpoint's raw
predictions through this script's own ground-truth-parsing +
``DetectionAccumulator`` code (the same path ``--model-family yolo`` uses)
and reports how closely it reproduces that checkpoint's already-published
AP@0.5. Run this validation first -- a silent off-by-one in the class-id
remap here would invalidate every downstream YOLO number.

    # validate the bridge itself against a known-good Faster R-CNN checkpoint
    python experiments/src/29_yolo_eval_bridge.py --model-family fasterrcnn \\
        --weights runs/size_rfs_ms_s42__WLE__multiclass/best_model.pth \\
        --modality WLE --task multiclass --base-run size_rfs_ms_s42__WLE__multiclass

    # score a trained YOLO checkpoint
    python experiments/src/29_yolo_eval_bridge.py --model-family yolo \\
        --weights runs_yolo/yolo11n_wle_multiclass_baseline_s42/weights/best.pt \\
        --modality WLE --task multiclass --base-run size_rfs_ms_s42__WLE__multiclass \\
        --conf 0.001

Faster R-CNN's own published numbers are scored after only torchvision's
internal ``box_score_thresh=0.05`` filter -- no extra confidence cutoff -- so
``--conf`` defaults to ``0.001`` here (near-maximally permissive) rather than
Ultralytics' own ``0.25`` CLI default, which would truncate the PR curve
before AP@0.5 is computed and understate YOLO relative to Faster R-CNN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from cadsel.config import MODALITIES, TASK_CLASSES, TASK_ID_MAP
from cadsel.data import build_manifest, read_yolo_label
from cadsel.metrics import DetectionAccumulator, summarize
from yolo_common import filter_by_split


def ground_truth_xyxy(sample, task: str) -> tuple[np.ndarray, np.ndarray]:
    """Absolute xyxy boxes in the *original* image's native pixel space --
    matching where both YOLO's ``model.predict(source=path)`` (Ultralytics
    always rescales back to the source image's own size) and a
    ``torchvision``-input-mode Faster R-CNN checkpoint's predictions
    (``GeneralizedRCNNTransform`` also rescales back to whatever was fed in,
    which is the untouched native image in that mode) report boxes.
    """
    with Image.open(sample.image) as im:
        W, H = im.size
    id_map = TASK_ID_MAP[task]
    boxes, labels = [], []
    for cls, cx, cy, bw, bh in read_yolo_label(sample.label):
        x1 = (cx - bw / 2.0) * W
        y1 = (cy - bh / 2.0) * H
        x2 = (cx + bw / 2.0) * W
        y2 = (cy + bh / 2.0) * H
        x1, x2 = max(0.0, min(x1, W - 1)), max(0.0, min(x2, W - 1))
        y1, y2 = max(0.0, min(y1, H - 1)), max(0.0, min(y2, H - 1))
        if x2 - x1 < 1.0 or y2 - y1 < 1.0:
            continue
        boxes.append([x1, y1, x2, y2])
        labels.append(id_map[cls] + 1)  # +1: DetectionAccumulator's 1-based, background-reserves-0 convention
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)


def predict_yolo(model, image_path: str, conf: float, augment: bool = False):
    r = model.predict(source=image_path, conf=conf, iou=0.7, max_det=300,
                       augment=augment, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    # Ultralytics class ids are 0-based (no reserved background class); +1 to
    # match DetectionAccumulator's convention (see module docstring).
    labels = r.boxes.cls.cpu().numpy().astype(np.int64) + 1
    return boxes, scores, labels


def predict_fasterrcnn(model, image_path: str, device: torch.device):
    with Image.open(image_path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1)).to(device)
    with torch.no_grad():
        out = model([x])[0]
    # torchvision's RoIHeads already drops class-0 (background) detections,
    # so predicted labels are already 1-based -- no remap needed.
    return (out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(),
            out["labels"].cpu().numpy().astype(np.int64))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-family", choices=["yolo", "fasterrcnn"], required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--modality", choices=MODALITIES, required=True)
    p.add_argument("--task", choices=["binary", "multiclass"], required=True)
    p.add_argument("--base-run", required=True, help="whose split.json defines the val set")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--augment", action="store_true",
                    help="Ultralytics' own built-in TTA (multi-scale + flip, merged "
                         "internally) at predict time -- yolo model_family only")
    args = p.parse_args()

    samples = build_manifest(args.modality)
    _, val_samples = filter_by_split(samples, args.base_run)
    print(f"scoring {len(val_samples)} val images ({args.modality}/{args.task}) "
          f"vs runs/{args.base_run}/split.json, model_family={args.model_family}"
          f"{', augment=True (Ultralytics TTA)' if args.augment else ''}")

    class_names = TASK_CLASSES[args.task]
    acc = DetectionAccumulator()
    published = None

    if args.model_family == "yolo":
        from ultralytics import YOLO
        model = YOLO(args.weights)
        for i, s in enumerate(val_samples):
            gt_boxes, gt_labels = ground_truth_xyxy(s, args.task)
            pred_boxes, pred_scores, pred_labels = predict_yolo(model, s.image, args.conf, args.augment)
            acc.add(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(val_samples)}")
    else:
        from cadsel.models import build_detection_model, configure_transform_for_input_mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        if ck.get("input_mode", "reference") != "torchvision":
            raise SystemExit(
                f"--model-family fasterrcnn validation needs a checkpoint trained/evaluated "
                f"in --input-mode torchvision (native-resolution predictions), so its box "
                f"coordinate space matches this bridge's native-resolution ground truth; "
                f"{args.weights} used input_mode={ck.get('input_mode')!r}"
            )
        model = build_detection_model(ck["model_type"], ck["task"], pretrained=False)
        # Same eval-time scale train.py's own per-epoch evaluate() call used
        # (last/largest of multiscale_sizes in eval mode -- see
        # configure_transform_for_input_mode's docstring), so this reproduces
        # the checkpoint's *published* number, not a re-derived one at a
        # different scale.
        configure_transform_for_input_mode(
            model, "torchvision", ck.get("multiscale_sizes") or ck["img_size"]
        )
        model.load_state_dict(ck["model_state_dict"])
        model.to(device).eval()
        for i, s in enumerate(val_samples):
            gt_boxes, gt_labels = ground_truth_xyxy(s, args.task)
            pred_boxes, pred_scores, pred_labels = predict_fasterrcnn(model, s.image, device)
            acc.add(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(val_samples)}")
        published = ck.get("metrics", {}).get("AP@0.5")

    metrics = summarize(acc, class_names, iou_threshold=0.5, conf_threshold=0.5)
    print(f"\nAP@0.5 = {metrics['AP@0.5']:.4f}  P={metrics['precision']:.4f} "
          f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f}")
    for name, c in metrics["per_class"].items():
        ap = c["AP@0.5"]
        print(f"  {name:<14} AP@0.5={ap:.4f}" if ap is not None else f"  {name:<14} AP@0.5=n/a")

    if published is not None:
        delta = metrics["AP@0.5"] - published
        verdict = "PASS" if abs(delta) < 0.01 else "FAIL"
        print(f"\nbridge validation: published AP@0.5={published:.4f}  "
              f"bridge AP@0.5={metrics['AP@0.5']:.4f}  delta={delta:+.4f}  -> {verdict}")
        if verdict == "FAIL":
            print("  DO NOT TRUST any YOLO number from this bridge until this is fixed.")


if __name__ == "__main__":
    main()
