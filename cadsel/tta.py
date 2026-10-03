"""Multi-scale test-time augmentation with Weighted Boxes Fusion (WBF).

Standard NMS keeps one box per cluster and discards the rest; WBF instead
fuses every box in a cluster into a single confidence-weighted average box
(Solovyev, Wang & Gabruseva, 2021), which recovers more localisation
precision than NMS specifically for small objects, where a few pixels of
box disagreement between scales is a large relative error -- directly
targeting the confirmed small-lesion AP gap (FINDINGS.md: AP by size tertile
0.47 / 0.66 / 0.76). Inference-only: no retraining, fully reversible, and a
good first thing to try given it doesn't touch the already-validated training
pipeline.

Hand-rolled in NumPy rather than adding the ``ensemble-boxes`` dependency, to
match the project's existing "no dependency beyond what's already needed"
stance (see ``experiments/probes.py``).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def weighted_boxes_fusion(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse predictions from multiple passes over one image (e.g. different
    TTA scales, or different models) into a single box set.

    Each of ``boxes_list``/``scores_list``/``labels_list`` holds one array per
    pass (xyxy boxes, confidence scores, integer class labels) -- all passes
    must already share one coordinate space (true for multi-scale TTA on a
    torchvision detector: ``GeneralizedRCNNTransform`` un-resizes its own
    output boxes back to the space of whatever image tensor was actually
    passed in, so running the same original image at different internal
    resize scales already produces directly comparable boxes).

    ``conf_type``: ``"avg"`` averages scores within a cluster. WBF's own
    default, ``"box_and_model_avg"``, additionally boosts confidence by how
    many of the ``len(boxes_list)`` passes contributed a box to the cluster --
    appropriate when a real detection should plausibly show up at more than
    one scale, but not forced here since some lesions may only be resolvable
    at one particular scale.
    """
    n_passes = len(boxes_list)
    all_boxes, all_scores, all_labels, all_pass = [], [], [], []
    for i, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        keep = scores >= skip_box_thr
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
        all_labels.append(labels[keep])
        all_pass.append(np.full(int(keep.sum()), i))

    if not any(len(b) for b in all_boxes):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)

    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    passes = np.concatenate(all_pass)

    fused_boxes, fused_scores, fused_labels = [], [], []
    for label in np.unique(labels):
        mask = labels == label
        b, s, p = boxes[mask], scores[mask], passes[mask]
        order = np.argsort(-s)
        b, s, p = b[order], s[order], p[order]

        clusters: list[list[int]] = []
        cluster_boxes: list[np.ndarray] = []

        for i in range(len(b)):
            best_iou, best_c = -1.0, -1
            for c, cb in enumerate(cluster_boxes):
                iou = _iou_xyxy(b[i], cb)
                if iou > best_iou:
                    best_iou, best_c = iou, c

            if best_iou >= iou_thr:
                target = best_c
                clusters[target].append(i)
            else:
                target = len(clusters)
                clusters.append([i])
                cluster_boxes.append(b[i].copy())

            members = clusters[target]
            w = s[members]
            cluster_boxes[target] = (b[members] * w[:, None]).sum(axis=0) / w.sum()

        for members, cb in zip(clusters, cluster_boxes):
            member_scores = s[members]
            if conf_type == "box_and_model_avg":
                n_contributing = len(set(p[members].tolist()))
                conf = member_scores.sum() / max(n_passes, 1)
                conf *= min(n_contributing, n_passes) / max(n_contributing, 1)
            else:
                conf = member_scores.mean()
            fused_boxes.append(cb)
            fused_scores.append(conf)
            fused_labels.append(label)

    if not fused_boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    return np.stack(fused_boxes), np.array(fused_scores), np.array(fused_labels, dtype=np.int64)


def predict_multiscale_wbf(
    model,
    image: torch.Tensor,
    scales: list[int],
    device,
    normalize=None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.05,
    use_amp: bool | None = None,
) -> dict:
    """Runs ``model`` on one unbatched ``image`` tensor at each of ``scales``
    (by temporarily overriding ``model.transform.min_size``/``max_size``),
    then fuses the per-scale predictions with :func:`weighted_boxes_fusion`.

    Leaves the model's transform and train/eval mode exactly as found.
    ``use_amp`` defaults to autocast-on-CUDA, matching ``cadsel/engine.py``'s
    own evaluation convention -- AMP vs full precision produces small but real
    numerical differences that shift borderline detections, so leaving this
    unset would make a "single-scale" comparison inconsistent with whatever
    precision training's own validation pass used.
    """
    if use_amp is None:
        use_amp = device.type == "cuda"
    was_training = model.training
    model.eval()
    orig_min_size, orig_max_size = model.transform.min_size, model.transform.max_size
    boxes_list, scores_list, labels_list = [], [], []
    try:
        with torch.no_grad():
            for scale in scales:
                model.transform.min_size = (scale,)
                model.transform.max_size = max(scale, orig_max_size)
                img = image.to(device)
                if normalize is not None:
                    img = normalize([img])[0]
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model([img])[0]
                boxes_list.append(out["boxes"].cpu().numpy())
                scores_list.append(out["scores"].cpu().numpy())
                labels_list.append(out["labels"].cpu().numpy())
    finally:
        model.transform.min_size, model.transform.max_size = orig_min_size, orig_max_size
        if was_training:
            model.train()

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    return {"boxes": boxes, "scores": scores, "labels": labels}


def predict_at_native_scale(
    model, image_path: str, device, input_mode: str, img_size: int, use_amp: bool | None = None,
) -> dict:
    """Run ``model`` on one image, returning boxes in the *native* (raw
    on-disk) pixel coordinate space -- the common frame cross-model WBF needs,
    since different checkpoints were trained at different resolutions and
    ``input_mode`` conventions (a ``"reference"``-mode model pre-resizes to a
    fixed square ``img_size`` before its own normalisation, non-aspect-
    preserving; a ``"torchvision"``-mode model leaves the raw image alone and
    lets ``GeneralizedRCNNTransform`` resize+normalise internally, then
    un-resizes its own output boxes back to whatever space it was handed --
    see ``predict_multiscale_wbf``'s docstring for that guarantee).

    ``"reference"`` mode's square resize is inverted with independent x/y
    scale factors after inference, since the resize itself wasn't
    aspect-preserving.
    """
    if use_amp is None:
        use_amp = device.type == "cuda"
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        native_w, native_h = im.size
        arr = np.asarray(im, dtype=np.uint8)

    was_training = model.training
    model.eval()
    try:
        if input_mode == "reference":
            resized = np.asarray(
                Image.fromarray(arr).resize((img_size, img_size), Image.BILINEAR), dtype=np.float32
            ) / 255.0
            from .config import CADSEL_MEAN, CADSEL_STD
            mean, std = np.array(CADSEL_MEAN, dtype=np.float32), np.array(CADSEL_STD, dtype=np.float32)
            normed = (resized - mean) / std
            img_t = torch.from_numpy(normed.transpose(2, 0, 1)).float().to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                out = model([img_t])[0]
            boxes = out["boxes"].cpu().numpy()
            if len(boxes):
                sx, sy = native_w / img_size, native_h / img_size
                boxes = boxes * np.array([sx, sy, sx, sy], dtype=np.float32)
        else:  # "torchvision" -- pass native-resolution [0,1] pixels, let the
            # model's own transform resize+normalise; its boxes come back
            # already in this tensor's (i.e. native) coordinate space.
            img_t = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1) / 255.0).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                out = model([img_t])[0]
            boxes = out["boxes"].cpu().numpy()
    finally:
        if was_training:
            model.train()

    return {
        "boxes": boxes, "scores": out["scores"].cpu().numpy(), "labels": out["labels"].cpu().numpy(),
        "native_w": native_w, "native_h": native_h,
    }


def predict_ensemble_wbf(
    models_info: list[dict], image_path: str, device,
    iou_thr: float = 0.55, skip_box_thr: float = 0.05,
) -> dict:
    """Fuse predictions from *different trained checkpoints* (possibly
    different architectures/backbones/input_modes) on one image, via
    :func:`predict_at_native_scale` to bring every model's boxes into one
    common (native-pixel) coordinate space before calling
    :func:`weighted_boxes_fusion`. No retraining -- combines checkpoints
    already on disk.

    ``models_info``: list of ``{"model": <nn.Module>, "input_mode": str,
    "img_size": int, "weight": float (optional, default 1.0)}``. WBF supports
    per-model confidence weighting -- use it rather than treating every
    architecture as equally reliable (e.g. down-weight a single-scale ViT
    detector known to be weaker on small/extreme-scale objects).
    """
    boxes_list, scores_list, labels_list = [], [], []
    for info in models_info:
        pred = predict_at_native_scale(
            info["model"], image_path, device, info["input_mode"], info["img_size"],
        )
        weight = info.get("weight", 1.0)
        boxes_list.append(pred["boxes"])
        scores_list.append(pred["scores"] * weight)
        labels_list.append(pred["labels"])

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )
    return {"boxes": boxes, "scores": scores, "labels": labels}
