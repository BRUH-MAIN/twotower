"""Data-driven RPN anchor sizes, from the training set's own box-size distribution.

Torchvision's default ``AnchorGenerator`` sizes (32, 64, 128, 256, 512) are
tuned for COCO's object-size distribution. Faster R-CNN's anchor matching is
IoU-based, so a ground-truth box mechanically gets few or no positive anchor
matches whenever nothing in the anchor set is close to its size -- independent
of anything visual -- which is a plausible mechanical contributor to the
already-confirmed small-lesion AP gap (0.47 / 0.66 / 0.76 by size tertile,
see FINDINGS.md). This module clusters the *actual* box-size distribution
(IoU-distance k-means, as in YOLOv2's anchor clustering -- plain Euclidean
k-means favours large boxes, since their coordinates dominate squared
distance, which is backwards for covering the small end) to produce anchors
matched to what each model will really see, per modality and per training
resolution.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .data import Sample, read_yolo_label


def collect_box_dims(samples: Sequence[Sample], img_size: int) -> np.ndarray:
    """``(N, 2)`` array of ``(width, height)`` in pixels at ``img_size``.

    YOLO box width/height are already fractions of the source image, and
    ``CadSelDetectionDataset`` resizes every frame to a square ``img_size``
    before the detector sees it (directly in ``input_mode="reference"``, or
    via the internal transform in ``"torchvision"``) -- either way,
    ``frac * img_size`` is the box's size in the pixel grid the model actually
    trains on, so this needs only the label files, not the (large, slow to
    decode) source images.
    """
    dims = []
    for s in samples:
        if getattr(s, "negative", False):
            continue
        for _cls, _cx, _cy, bw, bh in read_yolo_label(s.label):
            dims.append((bw * img_size, bh * img_size))
    return np.array(dims, dtype=np.float64)


def _iou_wh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of box ``a`` (shape ``(2,)``) against boxes ``b`` (shape ``(N,2)``),
    treating both as axis-aligned and co-centred (position-independent), as in
    YOLOv2's anchor clustering."""
    inter_w = np.minimum(a[0], b[:, 0])
    inter_h = np.minimum(a[1], b[:, 1])
    inter = inter_w * inter_h
    union = a[0] * a[1] + b[:, 0] * b[:, 1] - inter
    return inter / np.maximum(union, 1e-9)


def iou_kmeans(dims: np.ndarray, k: int, seed: int = 0, max_iter: int = 100) -> np.ndarray:
    """``(k, 2)`` cluster centres over ``dims``, using ``1 - IoU`` as the
    k-means distance rather than Euclidean distance."""
    if len(dims) < k:
        raise ValueError(f"only {len(dims)} boxes available, need at least {k}")
    rng = np.random.RandomState(seed)
    centers = dims[rng.choice(len(dims), size=k, replace=False)].copy()
    for _ in range(max_iter):
        dists = np.stack([1.0 - _iou_wh(c, dims) for c in centers], axis=1)  # (N, k)
        assign = dists.argmin(axis=1)
        new_centers = np.stack(
            [
                dims[assign == i].mean(axis=0) if (assign == i).any() else centers[i]
                for i in range(k)
            ]
        )
        if np.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    order = np.argsort(centers[:, 0] * centers[:, 1])
    return centers[order]


def compute_anchor_config(
    samples: Sequence[Sample],
    img_size: int,
    n_sizes: int = 5,
    n_aspect_ratios: int = 3,
    seed: int = 0,
) -> dict:
    """Anchor ``sizes``/``aspect_ratios`` for
    ``torchvision.models.detection.anchor_utils.AnchorGenerator``, matched to
    ``samples``' actual box-size distribution at ``img_size``.

    ``sizes`` (one value per FPN level in torchvision's convention, though the
    *same* base sizes are used for every level here -- CAD-SEL has no prior
    reason to expect lesion scale to correlate with FPN level the way natural
    scenes loosely do) come from area-based IoU clusters. ``aspect_ratios``
    (shared across levels, matching torchvision's own default behaviour) come
    from tercile-binned width/height ratios.
    """
    dims = collect_box_dims(samples, img_size)
    size_clusters = iou_kmeans(dims, n_sizes, seed=seed)
    sizes = tuple(int(round(float(np.sqrt(max(w * h, 1.0))))) for w, h in size_clusters)

    ratios = dims[:, 0] / np.maximum(dims[:, 1], 1e-9)
    edges = np.quantile(ratios, np.linspace(0, 1, n_aspect_ratios + 1))
    aspect_ratios = tuple(
        round(float(np.median(ratios[(ratios >= edges[i]) & (ratios <= edges[i + 1])])), 2)
        for i in range(n_aspect_ratios)
    )
    return {"sizes": sizes, "aspect_ratios": aspect_ratios, "n_boxes": int(len(dims))}
