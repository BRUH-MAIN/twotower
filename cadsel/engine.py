"""Training and evaluation loops."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from tqdm import tqdm

from .config import MODALITY_NAMES
from .metrics import DetectionAccumulator, summarize
from .models import set_active_group, set_active_modality


def _target_is_usable(t) -> bool:
    """Empty targets are valid (hard negatives); degenerate boxes are not."""
    boxes = t["boxes"]
    if len(boxes) == 0:
        return True
    return bool(((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])).all())


def _batch_modality(targets) -> str | None:
    """The modality of a modality-homogeneous batch, or None when unlabelled.

    Raises on a mixed batch rather than silently conditioning on whichever sample
    happened to come first: both the FiLM layers and the per-modality normaliser
    apply one choice to the whole batch, so a mixed batch is a misconfigured
    loader, not a recoverable state.
    """
    if not targets or "modality" not in targets[0]:
        return None
    indices = {int(t["modality"].item()) for t in targets}
    if len(indices) > 1:
        raise ValueError(
            "mixed-modality batch: per-modality normalisation and conditioning "
            "both need homogeneous batches — build the loader with "
            "data.ModalityBatchSampler"
        )
    return MODALITY_NAMES[indices.pop()]


def _batch_center(targets) -> str | None:
    """The acquisition center of a center-homogeneous batch, or None when
    unlabelled. Raises on a mixed batch, same rationale as ``_batch_modality``:
    ``models.GroupDROLossState`` uses this batch's loss as its estimate of
    the whole group's loss, which only makes sense for a homogeneous batch --
    build the loader with ``data.CenterBatchSampler``.
    """
    if not targets or "center" not in targets[0]:
        return None
    centers = {t["center"] for t in targets}
    if len(centers) > 1:
        raise ValueError(
            "mixed-center batch: Group DRO's per-group loss estimate needs "
            "homogeneous batches — build the loader with data.CenterBatchSampler"
        )
    return centers.pop()


def _to_device(images, targets, device, normalize=None, modality=None):
    images = [img.to(device, non_blocking=True) for img in images]
    if normalize is not None:
        images = normalize(images, modality) if modality is not None else normalize(images)
    targets = [
        {k: v.to(device, non_blocking=True) for k, v in t.items() if torch.is_tensor(v)}
        for t in targets
    ]
    return images, targets


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
    max_norm: float = 1.0,
    epoch: int = 0,
    total_epochs: int = 0,
    progress: bool = True,
    warmup_iters: int = 0,
    base_lr: float | None = None,
    normalize=None,
    modality_aware: bool = False,
    group_dro_aware: bool = False,
    ema_model=None,
) -> dict:
    """One pass over the training split.

    A freshly initialised box predictor at lr 1e-3 can blow up in the first few
    hundred steps, so the first epoch optionally ramps the learning rate linearly
    (the usual torchvision detection-reference remedy).

    ``ema_model``, if given (a ``torch.optim.swa_utils.AveragedModel`` wrapping
    ``model``), is updated after every successful optimizer step -- not on steps
    skipped for a non-finite loss, since those never reach the step call below.
    """
    model.train()
    running = 0.0
    components: dict[str, float] = {}
    n_batches = 0
    step = 0

    # Never let the ramp outlast the epoch, or the base rate is never reached.
    warmup_active = warmup_iters > 0 and epoch == 0 and base_lr is not None
    if warmup_active:
        warmup_iters = min(warmup_iters, max(1, len(loader) - 1))

    iterator = loader
    if progress:
        iterator = tqdm(loader, desc=f"train {epoch + 1}/{total_epochs}", leave=False)

    for images, targets in iterator:
        # Declare the modality before anything touches the model: both the
        # normaliser and the FiLM/BN conditioning read it.
        modality = _batch_modality(targets) if modality_aware else None
        if modality_aware:
            set_active_modality(modality)
        if group_dro_aware:
            # Read before _to_device strips the non-tensor "center" key.
            set_active_group(_batch_center(targets))

        if warmup_active:
            if step < warmup_iters:
                factor = (step + 1) / warmup_iters
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * factor
            elif step == warmup_iters:
                for group in optimizer.param_groups:
                    group["lr"] = base_lr
        step += 1

        # An empty target is meaningful: it marks a hard negative, and
        # torchvision's RPN and ROI heads treat every anchor in such an image as
        # background. Keep those; drop only genuinely degenerate boxes.
        keep = [i for i, t in enumerate(targets) if _target_is_usable(t)]
        if not keep:
            continue
        images = [images[i] for i in keep]
        targets = [targets[i] for i in keep]
        images, targets = _to_device(images, targets, device, normalize, modality)

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

        loss_value = float(losses.detach())
        if not math.isfinite(loss_value):
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        if ema_model is not None:
            ema_model.update_parameters(model)

        running += loss_value
        for k, v in loss_dict.items():
            components[k] = components.get(k, 0.0) + float(v.detach())
        n_batches += 1

        if progress:
            iterator.set_postfix(loss=f"{running / max(n_batches, 1):.4f}")

    if warmup_active:
        for group in optimizer.param_groups:
            group["lr"] = base_lr

    n = max(n_batches, 1)
    return {"loss": running / n, **{k: v / n for k, v in components.items()}}


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    class_names,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.5,
    use_amp: bool = False,
    progress: bool = True,
    collect_predictions: bool = False,
    with_loss: bool = False,
    normalize=None,
    modality_aware: bool = False,
) -> dict:
    """Score the validation split.

    ``with_loss`` additionally returns the mean validation loss. Losses need the
    model in train mode and detections need eval mode, so that costs a second
    forward pass — but both share one pass over the data, which is the expensive
    part here (TIFF decoding dominates).

    ``modality_aware`` additionally scores each modality into its own accumulator
    and reports ``per_modality`` plus ``AP@0.5_macro``. The per-modality rows are
    what stay comparable to the paper's Table 4/5, which are per-modality; the
    pooled figures alone would hide a joint model that helps WLE by degrading EUS.
    """
    model.eval()
    accumulator = DetectionAccumulator()
    by_modality: dict[str, DetectionAccumulator] = {}
    predictions = []
    loss_total, loss_batches = 0.0, 0

    iterator = loader
    if progress:
        iterator = tqdm(loader, desc="eval", leave=False)

    for images, targets in iterator:
        modality = _batch_modality(targets) if modality_aware else None
        if modality_aware:
            set_active_modality(modality)

        images_dev = [img.to(device, non_blocking=True) for img in images]
        if normalize is not None:
            images_dev = (
                normalize(images_dev, modality) if modality is not None
                else normalize(images_dev)
            )
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images_dev)

        if with_loss:
            keep = [i for i, t in enumerate(targets) if len(t["boxes"]) > 0]
            if keep:
                loss_images = [images_dev[i] for i in keep]
                loss_targets = [
                    {k: v.to(device, non_blocking=True) for k, v in targets[i].items()
                     if torch.is_tensor(v)}
                    for i in keep
                ]
                model.train()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss_dict = model(loss_images, loss_targets)
                    total = sum(loss_dict.values())
                model.eval()
                value = float(total)
                if math.isfinite(value):
                    loss_total += value
                    loss_batches += 1

        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].float().cpu().numpy()
            pred_scores = output["scores"].float().cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()
            gt_boxes = target["boxes"].numpy()
            gt_labels = target["labels"].numpy()
            accumulator.add(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
            if modality is not None:
                by_modality.setdefault(modality, DetectionAccumulator()).add(
                    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
                )
            if collect_predictions:
                predictions.append(
                    {
                        "index": int(target["index"].item()),
                        "boxes": pred_boxes,
                        "scores": pred_scores,
                        "labels": pred_labels,
                        "gt_boxes": gt_boxes,
                        "gt_labels": gt_labels,
                    }
                )

    metrics = summarize(accumulator, class_names, iou_threshold, conf_threshold)
    if by_modality:
        metrics["per_modality"] = {
            m: summarize(acc, class_names, iou_threshold, conf_threshold)
            for m, acc in sorted(by_modality.items())
        }
        # Unweighted mean over modalities, not over images: WLE contributes 1.35x
        # the frames, and a joint model must not be allowed to buy a headline
        # number by trading EUS away.
        metrics["AP@0.5_macro"] = float(
            np.mean([s["AP@0.5"] for s in metrics["per_modality"].values()])
        )
    if with_loss:
        metrics["val_loss"] = loss_total / max(loss_batches, 1)
    if collect_predictions:
        metrics["predictions"] = predictions
    return metrics


@torch.no_grad()
def validation_loss(model, loader, device, use_amp: bool = False, normalize=None,
                    modality_aware: bool = False) -> float:
    """Mean total loss on the validation split (drives the LR scheduler)."""
    was_training = model.training
    model.train()  # torchvision detectors only return losses in train mode
    total, n = 0.0, 0
    for images, targets in loader:
        modality = _batch_modality(targets) if modality_aware else None
        if modality_aware:
            set_active_modality(modality)
        keep = [i for i, t in enumerate(targets) if _target_is_usable(t)]
        if not keep:
            continue
        images = [images[i] for i in keep]
        targets = [targets[i] for i in keep]
        images, targets = _to_device(images, targets, device, normalize, modality)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
        value = float(losses.detach())
        if math.isfinite(value):
            total += value
            n += 1
    if not was_training:
        model.eval()
    return total / max(n, 1)
