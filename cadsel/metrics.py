"""Detection metrics: AP@0.5, precision, recall and F1, as defined in the paper.

AP@0.5 is the area under the precision-recall curve using VOC all-point
interpolation; a prediction counts as correct when it overlaps a ground-truth
box by at least 0.5 IoU *and* the class label matches.

Precision / recall / F1 follow the paper's formulas and are computed at the
reference repository's confidence threshold of 0.5 by greedy IoU matching.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` xyxy arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = np.prod(np.clip(boxes_a[:, 2:] - boxes_a[:, :2], 0, None), axis=1)
    area_b = np.prod(np.clip(boxes_b[:, 2:] - boxes_b[:, :2], 0, None), axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


@dataclass
class ImageResult:
    """One image's predictions and ground truth, in absolute xyxy pixels."""

    pred_boxes: np.ndarray
    pred_scores: np.ndarray
    pred_labels: np.ndarray
    gt_boxes: np.ndarray
    gt_labels: np.ndarray


@dataclass
class DetectionAccumulator:
    results: list[ImageResult] = field(default_factory=list)

    def add(self, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels) -> None:
        self.results.append(
            ImageResult(
                np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4),
                np.asarray(pred_scores, dtype=np.float32).reshape(-1),
                np.asarray(pred_labels, dtype=np.int64).reshape(-1),
                np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4),
                np.asarray(gt_labels, dtype=np.int64).reshape(-1),
            )
        )

    # -- AP ----------------------------------------------------------------
    def average_precision(
        self,
        class_id: int,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.0,
    ) -> float | None:
        """VOC all-point-interpolation AP for one class.

        Returns ``None`` when the class has no ground-truth boxes, so absent
        classes can be excluded from the mean rather than dragging it to zero.
        """
        detections = []  # (score, image_index, box)
        n_gt = 0
        gt_by_image: dict[int, np.ndarray] = {}

        for i, r in enumerate(self.results):
            gt_mask = r.gt_labels == class_id
            gt_by_image[i] = r.gt_boxes[gt_mask]
            n_gt += int(gt_mask.sum())

            pred_mask = (r.pred_labels == class_id) & (r.pred_scores > score_threshold)
            for box, score in zip(r.pred_boxes[pred_mask], r.pred_scores[pred_mask]):
                detections.append((float(score), i, box))

        if n_gt == 0:
            return None
        if not detections:
            return 0.0

        detections.sort(key=lambda d: d[0], reverse=True)
        matched: dict[int, np.ndarray] = {
            i: np.zeros(len(boxes), dtype=bool) for i, boxes in gt_by_image.items()
        }

        tp = np.zeros(len(detections), dtype=np.float64)
        fp = np.zeros(len(detections), dtype=np.float64)

        for k, (_, image_index, box) in enumerate(detections):
            gts = gt_by_image[image_index]
            if len(gts) == 0:
                fp[k] = 1.0
                continue
            ious = box_iou(box[None, :], gts)[0]
            best = int(np.argmax(ious))
            if ious[best] >= iou_threshold and not matched[image_index][best]:
                matched[image_index][best] = True
                tp[k] = 1.0
            else:
                fp[k] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / n_gt
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

        # All-point interpolation.
        precision = np.concatenate(([1.0], precision))
        recall = np.concatenate(([0.0], recall))
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        idx = np.where(recall[1:] != recall[:-1])[0]
        return float(np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1]))

    # -- Precision / recall / F1 -------------------------------------------
    def counts_at_threshold(
        self, iou_threshold: float = 0.5, score_threshold: float = 0.5
    ) -> dict[int, dict[str, int]]:
        """Per-class TP / FP / FN by greedy score-ordered IoU matching."""
        stats: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

        for r in self.results:
            keep = r.pred_scores > score_threshold
            boxes, scores, labels = r.pred_boxes[keep], r.pred_scores[keep], r.pred_labels[keep]
            order = np.argsort(-scores)
            boxes, labels = boxes[order], labels[order]

            gt_used = np.zeros(len(r.gt_boxes), dtype=bool)
            for box, label in zip(boxes, labels):
                candidates = np.where((r.gt_labels == label) & ~gt_used)[0]
                if len(candidates):
                    ious = box_iou(box[None, :], r.gt_boxes[candidates])[0]
                    best = int(np.argmax(ious))
                    if ious[best] >= iou_threshold:
                        gt_used[candidates[best]] = True
                        stats[int(label)]["tp"] += 1
                        continue
                stats[int(label)]["fp"] += 1

            for j, label in enumerate(r.gt_labels):
                if not gt_used[j]:
                    stats[int(label)]["fn"] += 1
        return dict(stats)


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def summarize(
    accumulator: DetectionAccumulator,
    class_names: list[str],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
) -> dict:
    """Assemble the metric bundle reported in Table 4 and Table 5.

    ``AP@0.5`` is computed over all detections (the standard definition).
    ``AP@0.5 (conf>0.5)`` repeats it over the confidence-filtered detections,
    which is what the authors' ``evaluate_model`` scores.
    """
    per_class: dict[str, dict] = {}
    ap_values, ap_conf_values = [], []

    counts = accumulator.counts_at_threshold(iou_threshold, conf_threshold)

    for idx, name in enumerate(class_names):
        class_id = idx + 1  # model labels are 1-based
        ap = accumulator.average_precision(class_id, iou_threshold, score_threshold=0.0)
        ap_conf = accumulator.average_precision(class_id, iou_threshold, score_threshold=conf_threshold)
        c = counts.get(class_id, {"tp": 0, "fp": 0, "fn": 0})
        p, r, f = prf1(c["tp"], c["fp"], c["fn"])

        per_class[name] = {
            "AP@0.5": ap,
            "AP@0.5_conf": ap_conf,
            "precision": p,
            "recall": r,
            "f1": f,
            **c,
        }
        if ap is not None:
            ap_values.append(ap)
            ap_conf_values.append(ap_conf)

    total_tp = sum(c["tp"] for c in counts.values())
    total_fp = sum(c["fp"] for c in counts.values())
    total_fn = sum(c["fn"] for c in counts.values())
    micro_p, micro_r, micro_f1 = prf1(total_tp, total_fp, total_fn)

    return {
        "AP@0.5": float(np.mean(ap_values)) if ap_values else 0.0,
        "AP@0.5_conf": float(np.mean(ap_conf_values)) if ap_conf_values else 0.0,
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": per_class,
    }
