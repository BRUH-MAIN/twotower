"""Train one CAD-SEL detection baseline.

Example
-------
python -m cadsel.train --model fasterrcnn_mobilenet --modality WLE --task multiclass
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .anchors import compute_anchor_config
from .experiment_config import RunConfig, apply_to_namespace
from .config import (
    ALL_MODEL_NAMES,
    CONF_THRESHOLD,
    DINOV3_MODEL_REGISTRY,
    EFFICIENTNET_MODEL_REGISTRY,
    FCOS_MODEL_REGISTRY,
    IOU_THRESHOLD,
    MODEL_NAMES,
    MODEL_REGISTRY,
    MODALITIES,
    PROJECT_ROOT,
    TASKS,
    TASK_CLASSES,
    num_foreground_classes,
    num_model_classes,
)
from .data import (
    AUGMENT_OPS,
    DEFAULT_AUGMENT,
    build_dataloaders,
    build_joint_manifest,
    build_manifest,
    compute_class_weights,
    image_level_split,
    joint_strata_key,
    make_hard_negatives,
    make_normalizer,
    patient_level_folds,
    patient_level_split,
    save_split,
    split_summary,
)
from .engine import evaluate, train_one_epoch
from .models import (
    GroupDROLossState,
    SeesawLossState,
    build_class_weight_tensor,
    build_detection_model,
    build_dinov3_detection_model,
    build_efficientnet_detection_model,
    build_fcos_model,
    configure_transform_for_input_mode,
    install_loss_fn,
    load_backbone_weights,
    set_class_weights,
    set_groupdro_state,
    set_seesaw_state,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a CAD-SEL detection baseline")
    p.add_argument(
        "--model", choices=ALL_MODEL_NAMES, default="fasterrcnn_resnet",
        help=f"Faster R-CNN family: {MODEL_NAMES}. Anchor-free: {list(FCOS_MODEL_REGISTRY)} -- "
        "--auto-anchors/--rpn-*-iou-thresh/--box-*-iou-thresh/--loss-fn have no effect there "
        "(FCOS uses center-sampling, not IoU anchor matching, and its own internal loss).",
    )
    p.add_argument("--modality", choices=MODALITIES, default="WLE")
    p.add_argument("--task", choices=TASKS, default="multiclass")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=None, help="default: per-model (see config)")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=500,
        help="linear LR warmup over the first epoch; 0 disables",
    )
    p.add_argument("--img-size", type=int, default=None, help="default: per-model (640 / 320)")
    p.add_argument(
        "--num-workers",
        type=int,
        default=min(8, max(2, (os.cpu_count() or 4) // 2)),
        help="TIFF decoding dominates the epoch, so more workers helps up to the GPU limit",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop after N epochs with no validation AP@0.5 improvement; 0 disables",
    )
    p.add_argument(
        "--split-level",
        choices=["patient", "image"],
        default="patient",
        help="'image' leaks correlated frames across the split and exists only "
        "to quantify that effect — never for reported results",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--folds", type=int, default=0, help="0 = single split; >1 = K-fold")
    p.add_argument("--fold", type=int, default=0, help="which fold to run when --folds > 1")
    p.add_argument(
        "--joint-split",
        action="store_true",
        help="partition patients over the WLE+EUS union, then keep only --modality. "
        "Makes this run a valid denominator for a joint model: it is scored on "
        "exactly the joint run's validation patients. Without it the two use "
        "different splits and any delta between them also contains split variance.",
    )

    p.add_argument(
        "--augment",
        default=",".join(DEFAULT_AUGMENT),
        help=f"comma-separated augmentations from {AUGMENT_OPS}; "
        "'scale_crop' and 'copy_paste' are the shortcut mitigations "
        "(see experiments/README.md). Pass '' to disable all.",
    )
    p.add_argument(
        "--hard-negatives",
        type=float,
        default=0.0,
        help="add lesion-free crops as negative training images, as a fraction "
        "of the training set (e.g. 0.3). Targets the false-positive behaviour "
        "measured in experiments/08.",
    )
    p.add_argument("--input-mode", choices=["reference", "torchvision"], default="reference")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    p.add_argument("--conf-threshold", type=float, default=CONF_THRESHOLD)

    p.add_argument(
        "--auto-anchors",
        action="store_true",
        help="replace torchvision's COCO-tuned RPN anchor sizes/aspect-ratios with ones "
        "clustered (IoU k-means) from this run's own training box-size distribution "
        "at --img-size; targets the small-lesion AP gap in FINDINGS.md.",
    )
    p.add_argument("--rpn-fg-iou-thresh", type=float, default=None, help="default: torchvision's 0.7")
    p.add_argument("--rpn-bg-iou-thresh", type=float, default=None, help="default: torchvision's 0.3")
    p.add_argument("--box-fg-iou-thresh", type=float, default=None, help="default: torchvision's 0.5")
    p.add_argument("--box-bg-iou-thresh", type=float, default=None, help="default: torchvision's 0.5")
    p.add_argument(
        "--multiscale-sizes",
        default=None,
        help="comma-separated img sizes for multi-scale training (e.g. '320,416,512'); "
        "only meaningful with --input-mode torchvision, since 'reference' mode has "
        "already resized every image to one fixed size before the transform runs. "
        "Overrides --img-size for the transform when given.",
    )
    p.add_argument("--ema", action="store_true", help="track an EMA of the weights; evaluate/checkpoint those instead of the raw model")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument(
        "--rfs-threshold",
        type=float,
        default=0.0,
        help="Repeat Factor Sampling threshold (0 disables): images containing a class "
        "whose image-frequency falls below this get oversampled, complementing "
        "--no-class-weights' per-box loss weighting with per-image sampling weighting. "
        "Try ~0.1-0.3 (Detectron2's LVIS default is 0.001, but CAD-SEL's imbalance is "
        "far milder than LVIS's).",
    )
    p.add_argument(
        "--size-rfs-threshold",
        type=float,
        default=0.0,
        help="Repeat Factor Sampling threshold (0 disables), bucketed by lesion size "
        "(box_area_frac tertile) instead of class -- targets the small-lesion AP gap "
        "(FINDINGS.md: AP 0.471 -> 0.663 -> 0.755 by size tertile) directly, as an "
        "alternative axis to --rfs-threshold's class-based oversampling. Mutually "
        "exclusive with --rfs-threshold.",
    )
    p.add_argument(
        "--center-rfs-threshold",
        type=float,
        default=0.0,
        help="Repeat Factor Sampling threshold (0 disables), bucketed by acquisition "
        "center instead of class or size -- oversamples minority-center images. "
        "Directly targets a leave-one-center-out training pool's imbalance (e.g. "
        "holding out Center-3 leaves a pool that's 93.7%% Center-1 by patient count). "
        "Mutually exclusive with --rfs-threshold and --size-rfs-threshold.",
    )
    p.add_argument(
        "--centers",
        default=None,
        help="comma-separated acquisition-center filter (e.g. "
        "'Center-2,Center-3,Center-4,Center-5,Center-6' to hold out Center-1 for "
        "leave-one-center-out validation); default: no filter, all centers included",
    )
    p.add_argument(
        "--loss-fn",
        choices=["weighted_ce", "seesaw", "group_dro"],
        default="weighted_ce",
        help="'weighted_ce' (default) is the paper's static inverse-frequency class "
        "weighting; 'seesaw' (Wang et al. 2021) tracks cumulative per-class counts as "
        "training progresses and damps how much a frequent class suppresses a rarer "
        "one's probability, adapting instead of using a fixed pre-training snapshot. "
        "'group_dro' (Sagawa et al. 2020) reweights the whole loss per batch by an "
        "adversarial per-acquisition-center weight, targeting worst-group (not "
        "average) risk -- requires center-homogeneous batches, built automatically "
        "when this is selected. Mutually exclusive with --no-class-weights (only "
        "affects 'weighted_ce') and with --rfs-threshold/--size-rfs-threshold/"
        "--center-rfs-threshold (mutually exclusive batch-construction strategies).",
    )
    p.add_argument("--seesaw-p", type=float, default=0.8, help="Seesaw Loss mitigation exponent")
    p.add_argument(
        "--groupdro-step-size", type=float, default=0.01,
        help="Group DRO's adversarial-weight update step size (Sagawa et al.'s eta)",
    )
    p.add_argument(
        "--copy-paste-source",
        choices=["real", "generated"],
        default="real",
        help="'generated' draws copy_paste patches from a trained "
        "cadsel.lesion_gan generator (requires --lesion-gan-checkpoint) instead of "
        "cropping real lesions from other training patients -- the conditional "
        "generative augmentation item from the literature review (FINDINGS.md §5c). "
        "Only takes effect if 'copy_paste' is in --augment.",
    )
    p.add_argument(
        "--lesion-gan-checkpoint",
        type=Path,
        default=None,
        help="generator state_dict from cadsel.train_lesion_gan, required with "
        "--copy-paste-source generated",
    )
    p.add_argument(
        "--dinov3-trainable-backbone",
        action="store_true",
        help="fine-tune the DINOv3 backbone instead of keeping it frozen (default: frozen -- "
        "cheaper, and directly tests whether the pretrained representation itself helps, "
        "the question Lead 2's counterfactual-protocol finding raised). Only meaningful "
        "for --model fasterrcnn_dinov3.",
    )
    p.add_argument(
        "--backbone-weights",
        type=Path,
        default=None,
        help="path to a raw state_dict .pth to load into the backbone in place of (or after) "
        "the COCO-pretrained one -- for a domain-specific backbone (e.g. RadImageNet for EUS, "
        "an in-domain endoscopy self-supervised checkpoint for WLE; see FINDINGS.md §5c). "
        "No specific third-party weights are bundled; this only provides the loading mechanism.",
    )

    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="RunConfig JSON (cadsel.experiment_config.RunConfig) supplying defaults "
        "for any flag not also passed explicitly on this command line; explicit "
        "flags always win.",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="dataset root; accepts the figshare or Hugging Face layout "
        "(default: $CADSEL_DATA_ROOT or ./CAD-SEL-Dataset)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="e.g. cuda:0 / cuda:1 to pin a run to one GPU (default: cuda if available)",
    )
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--save-checkpoint", action="store_true", default=True)
    p.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")
    p.add_argument("--quiet", action="store_true", help="suppress per-epoch lines too")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="drop the tqdm bars but keep the per-epoch summary lines (good for log files)",
    )
    return p


def _write_result(
    out_dir, args, spec, run_name, img_size, batch_size,
    summary, best, best_epoch, epochs_run, elapsed,
) -> dict:
    result = {
        "run_name": run_name,
        "model": args.model,
        "paper_name": spec["paper_name"],
        "modality": args.modality,
        "task": args.task,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": args.epochs,
        "epochs_run": epochs_run,
        "learning_rate": args.learning_rate,
        "input_mode": args.input_mode,
        "augment": [o.strip() for o in getattr(args, "augment", "").split(",") if o.strip()],
        "hard_negatives": getattr(args, "hard_negatives", 0.0),
        "split_level": getattr(args, "split_level", "patient"),
        "joint_split": getattr(args, "joint_split", False),
        "class_weighting": not args.no_class_weights,
        "loss_fn": getattr(args, "loss_fn", "weighted_ce"),
        "seesaw_p": getattr(args, "seesaw_p", None) if getattr(args, "loss_fn", "weighted_ce") == "seesaw" else None,
        "groupdro_step_size": (
            getattr(args, "groupdro_step_size", None)
            if getattr(args, "loss_fn", "weighted_ce") == "group_dro" else None
        ),
        "seed": args.seed,
        "auto_anchors": getattr(args, "_anchor_config", None),
        "rpn_fg_iou_thresh": getattr(args, "rpn_fg_iou_thresh", None),
        "rpn_bg_iou_thresh": getattr(args, "rpn_bg_iou_thresh", None),
        "box_fg_iou_thresh": getattr(args, "box_fg_iou_thresh", None),
        "box_bg_iou_thresh": getattr(args, "box_bg_iou_thresh", None),
        "multiscale_sizes": getattr(args, "_multiscale_sizes", None),
        "ema": getattr(args, "ema", False),
        "ema_decay": getattr(args, "ema_decay", None) if getattr(args, "ema", False) else None,
        "rfs_threshold": getattr(args, "rfs_threshold", 0.0),
        "size_rfs_threshold": getattr(args, "size_rfs_threshold", 0.0),
        "center_rfs_threshold": getattr(args, "center_rfs_threshold", 0.0),
        "centers": getattr(args, "centers", None),
        "copy_paste_source": getattr(args, "copy_paste_source", "real"),
        "lesion_gan_checkpoint": (
            str(args.lesion_gan_checkpoint) if getattr(args, "lesion_gan_checkpoint", None) else None
        ),
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

    is_fcos = args.model in FCOS_MODEL_REGISTRY
    is_dinov3 = args.model in DINOV3_MODEL_REGISTRY
    is_efficientnet = args.model in EFFICIENTNET_MODEL_REGISTRY
    spec = (
        FCOS_MODEL_REGISTRY[args.model] if is_fcos
        else DINOV3_MODEL_REGISTRY[args.model] if is_dinov3
        else EFFICIENTNET_MODEL_REGISTRY[args.model] if is_efficientnet
        else MODEL_REGISTRY[args.model]
    )
    img_size = args.img_size or spec["default_img_size"]
    batch_size = args.batch_size or spec["batch_size"]
    class_names = TASK_CLASSES[args.task]

    if (is_dinov3 or is_efficientnet) and args.input_mode != "torchvision":
        raise SystemExit(
            f"{args.model} requires --input-mode torchvision -- it was pretrained with "
            "ImageNet normalisation (torchvision's GeneralizedRCNNTransform's own default), "
            "and 'reference' mode would feed it CADSEL_MEAN/STD-normalised pixels instead, "
            "silently degrading the pretrained representation. See cadsel/config.py's "
            "DINOV3_MODEL_REGISTRY/EFFICIENTNET_MODEL_REGISTRY docstrings."
        )

    if is_fcos:
        ignored = [
            flag for flag, val in (
                ("--auto-anchors", getattr(args, "auto_anchors", False)),
                ("--rpn-fg-iou-thresh", getattr(args, "rpn_fg_iou_thresh", None)),
                ("--rpn-bg-iou-thresh", getattr(args, "rpn_bg_iou_thresh", None)),
                ("--box-fg-iou-thresh", getattr(args, "box_fg_iou_thresh", None)),
                ("--box-bg-iou-thresh", getattr(args, "box_bg_iou_thresh", None)),
                ("--loss-fn", getattr(args, "loss_fn", "weighted_ce") != "weighted_ce"),
            ) if val
        ]
        if ignored:
            print(f"  NOTE: {args.model} is anchor-free (FCOS) -- {', '.join(ignored)} "
                  "have no effect on it and will be silently ignored below.")
        if args.learning_rate >= 5e-4:
            print(f"  WARNING: --learning-rate {args.learning_rate:g} is the Faster R-CNN "
                  "default and is too high for FCOS -- its focal-loss classification head "
                  "diverges (loss -> 0/NaN within ~1 epoch) at this LR on this dataset in "
                  "testing. --learning-rate 1e-4 --warmup-iters 0 trained stably; not changed "
                  "automatically since a different value may suit a different setup.")

    run_name = args.run_name or f"{args.model}__{args.modality}__{args.task}"
    if args.folds > 1:
        run_name = f"{run_name}__fold{args.fold}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    use_amp = (not args.no_amp) and device.type == "cuda"
    show_bars = not (args.quiet or getattr(args, "no_progress", False))

    # -- data ---------------------------------------------------------------
    centers = None
    if getattr(args, "centers", None):
        centers = [c.strip() for c in args.centers.split(",") if c.strip()]

    if getattr(args, "joint_split", False):
        # Split over the union, then keep one modality. The partition is byte-for-byte
        # the joint run's, so this run is scored on the same validation patients.
        samples = build_joint_manifest(data_root=args.data_root, centers=centers)
        strata_key = joint_strata_key
    else:
        samples = build_manifest(args.modality, data_root=args.data_root, centers=centers)
        strata_key = None

    if args.folds > 1:
        folds = patient_level_folds(
            samples, n_folds=args.folds, seed=args.seed,
            **({"strata_key": strata_key} if strata_key else {}),
        )
        train_samples, val_samples = folds[args.fold]
    elif args.split_level == "image":
        print("  WARNING: image-level split — correlated frames leak across it; "
              "diagnostic only, not a valid result")
        train_samples, val_samples = image_level_split(
            samples, val_fraction=args.val_fraction, seed=args.seed
        )
    else:
        train_samples, val_samples = patient_level_split(
            samples, val_fraction=args.val_fraction, seed=args.seed,
            **({"strata_key": strata_key} if strata_key else {}),
        )

    if getattr(args, "joint_split", False):
        train_samples = [s for s in train_samples if s.modality == args.modality]
        val_samples = [s for s in val_samples if s.modality == args.modality]
    save_split(out_dir / "split.json", train_samples, val_samples)
    summary = split_summary(train_samples, val_samples)

    augment_ops = [o.strip() for o in args.augment.split(",") if o.strip()]

    if args.hard_negatives > 0:
        negatives = make_hard_negatives(train_samples, args.hard_negatives, args.seed)
        print(f"  hard negatives: +{len(negatives)} lesion-free images "
              f"({args.hard_negatives:.0%} of {len(train_samples)})")
        train_samples = list(train_samples) + negatives

    lesion_bank = None
    if getattr(args, "copy_paste_source", "real") == "generated" and "copy_paste" in augment_ops:
        if not getattr(args, "lesion_gan_checkpoint", None):
            raise SystemExit("--copy-paste-source generated requires --lesion-gan-checkpoint")
        from .lesion_gan import GeneratedLesionBank
        lesion_bank = GeneratedLesionBank.load(
            args.lesion_gan_checkpoint, num_foreground_classes(args.task), device=device
        )
        print(f"  copy_paste source: generated lesions from {args.lesion_gan_checkpoint}")

    train_loader, val_loader = build_dataloaders(
        train_samples,
        val_samples,
        task=args.task,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=args.num_workers,
        input_mode=args.input_mode,
        seed=args.seed,
        augment_ops=augment_ops,
        rfs_threshold=getattr(args, "rfs_threshold", 0.0),
        size_rfs_threshold=getattr(args, "size_rfs_threshold", 0.0),
        center_rfs_threshold=getattr(args, "center_rfs_threshold", 0.0),
        group_dro=getattr(args, "loss_fn", "weighted_ce") == "group_dro",
        lesion_bank=lesion_bank,
    )

    # -- class weights / loss function ---------------------------------------
    fg_weights, counts = compute_class_weights(train_samples, args.task)
    loss_fn = getattr(args, "loss_fn", "weighted_ce")
    install_loss_fn(loss_fn)
    weight_tensor = None
    if loss_fn == "seesaw":
        set_groupdro_state(None)
        set_seesaw_state(SeesawLossState(num_model_classes(args.task), p=args.seesaw_p, device=device))
        if not args.quiet:
            print(f"  loss_fn=seesaw (p={args.seesaw_p}) -- overrides --no-class-weights, "
                  "which only applies to weighted_ce")
    elif loss_fn == "group_dro":
        set_seesaw_state(None)
        groups = sorted({s.center for s in train_samples})
        set_groupdro_state(
            GroupDROLossState(groups, step_size=args.groupdro_step_size, device=device)
        )
        if args.no_class_weights:
            set_class_weights(None)
        else:
            weight_tensor = build_class_weight_tensor(fg_weights).to(device)
            set_class_weights(weight_tensor)
        if not args.quiet:
            print(f"  loss_fn=group_dro (step_size={args.groupdro_step_size}) "
                  f"over {len(groups)} group(s): {groups}")
    else:
        set_seesaw_state(None)
        set_groupdro_state(None)
        if args.no_class_weights:
            set_class_weights(None)
        else:
            weight_tensor = build_class_weight_tensor(fg_weights).to(device)
            set_class_weights(weight_tensor)

    # -- model --------------------------------------------------------------
    if is_dinov3 or is_efficientnet:
        ignored = [
            flag for flag, val in (
                ("--auto-anchors", getattr(args, "auto_anchors", False)),
                ("--rpn-fg-iou-thresh", getattr(args, "rpn_fg_iou_thresh", None)),
                ("--rpn-bg-iou-thresh", getattr(args, "rpn_bg_iou_thresh", None)),
                ("--box-fg-iou-thresh", getattr(args, "box_fg_iou_thresh", None)),
                ("--box-bg-iou-thresh", getattr(args, "box_bg_iou_thresh", None)),
            ) if val
        ]
        if ignored:
            print(f"  NOTE: {args.model}'s builder doesn't wire anchor/IoU customisation "
                  f"through yet -- {', '.join(ignored)} will be silently ignored below.")

    anchor_config = None
    if getattr(args, "auto_anchors", False) and not is_fcos and not is_dinov3 and not is_efficientnet:
        anchor_config = compute_anchor_config(train_samples, img_size)
        print(f"  auto-anchors: sizes={anchor_config['sizes']} "
              f"aspect_ratios={anchor_config['aspect_ratios']} "
              f"(from {anchor_config['n_boxes']} training boxes)")
    args._anchor_config = anchor_config

    if is_fcos:
        model = build_fcos_model(
            args.model, args.task, pretrained=not args.no_pretrained,
        )
    elif is_dinov3:
        model = build_dinov3_detection_model(
            args.model, args.task, trainable_backbone=getattr(args, "dinov3_trainable_backbone", False),
        )
    elif is_efficientnet:
        model = build_efficientnet_detection_model(
            args.model, args.task, pretrained=not args.no_pretrained,
        )
    else:
        model = build_detection_model(
            args.model, args.task, pretrained=not args.no_pretrained,
            anchor_sizes=anchor_config["sizes"] if anchor_config else None,
            anchor_aspect_ratios=anchor_config["aspect_ratios"] if anchor_config else None,
            rpn_fg_iou_thresh=getattr(args, "rpn_fg_iou_thresh", None),
            rpn_bg_iou_thresh=getattr(args, "rpn_bg_iou_thresh", None),
            box_fg_iou_thresh=getattr(args, "box_fg_iou_thresh", None),
            box_bg_iou_thresh=getattr(args, "box_bg_iou_thresh", None),
        )

    multiscale_sizes = None
    if getattr(args, "multiscale_sizes", None):
        multiscale_sizes = sorted(int(s.strip()) for s in args.multiscale_sizes.split(",") if s.strip())
        print(f"  multi-scale training: {multiscale_sizes}")
    args._multiscale_sizes = multiscale_sizes
    configure_transform_for_input_mode(model, args.input_mode, multiscale_sizes or img_size)

    if getattr(args, "backbone_weights", None):
        report = load_backbone_weights(model, args.backbone_weights)
        print(f"  backbone weights from {args.backbone_weights}: "
              f"{len(report['matched'])} matched, {len(report['missing'])} missing, "
              f"{len(report['unexpected'])} unexpected key(s)")
        if not report["matched"]:
            print("    WARNING: zero keys matched -- this file's key names don't line up "
                  "with the backbone's; nothing was actually loaded")

    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    normalize = make_normalizer(args.input_mode)

    ema_model = None
    if getattr(args, "ema", False):
        ema_avg_fn = torch.optim.swa_utils.get_ema_avg_fn(getattr(args, "ema_decay", 0.999))
        ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg_fn, use_buffers=True)
        print(f"  EMA enabled: decay={getattr(args, 'ema_decay', 0.999)} "
              f"(evaluation and checkpoints use the averaged weights)")

    if not args.quiet:
        print(
            f"[{run_name}] device={device} img_size={img_size} batch_size={batch_size} "
            f"workers={args.num_workers} amp={use_amp}",
            flush=True,
        )
        print(f"  train: {summary['train']['images']} images / {summary['train']['patients']} patients")
        print(f"  val:   {summary['val']['images']} images / {summary['val']['patients']} patients")
        print(f"  box counts per class {dict(zip(class_names, counts.astype(int).tolist()))}")
        if weight_tensor is not None:
            print(f"  class weights (bg first) {weight_tensor.cpu().numpy().round(4).tolist()}")
        print(flush=True)

    # -- training loop ------------------------------------------------------
    history = []
    best = {"AP@0.5": -1.0}
    best_epoch = -1
    started = time.time()

    for epoch in range(args.epochs):
        epoch_started = time.time()
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            epoch=epoch,
            total_epochs=args.epochs,
            progress=show_bars,
            warmup_iters=args.warmup_iters,
            base_lr=args.learning_rate,
            normalize=normalize,
            group_dro_aware=(loss_fn == "group_dro"),
            ema_model=ema_model,
        )
        # With EMA enabled, the averaged weights are what gets evaluated and
        # checkpointed -- the raw `model` is training scaffolding only, the
        # standard way EMA is used (not run alongside as a second reported set
        # of numbers, which would double eval cost every epoch for a
        # comparison this repo's own multi-seed work already treats as
        # something to settle with seeds, not a single-run A/B).
        eval_model = ema_model.module if ema_model is not None else model
        metrics = evaluate(
            eval_model,
            val_loader,
            device,
            class_names,
            iou_threshold=args.iou_threshold,
            conf_threshold=args.conf_threshold,
            use_amp=use_amp,
            progress=show_bars,
            with_loss=True,
            normalize=normalize,
        )
        val_loss = metrics.pop("val_loss")
        scheduler.step(val_loss)

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_stats["loss"],
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{k: v for k, v in metrics.items() if k != "per_class"},
            "per_class": metrics["per_class"],
        }
        history.append(entry)

        if not args.quiet:
            marker = " *" if metrics["AP@0.5"] > best["AP@0.5"] else ""
            print(
                f"  epoch {epoch + 1:3d}/{args.epochs}  "
                f"train_loss={train_stats['loss']:.4f}  val_loss={val_loss:.4f}  "
                f"AP@0.5={metrics['AP@0.5']:.4f}  P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                f"({time.time() - epoch_started:.0f}s){marker}",
                flush=True,
            )

        improved = metrics["AP@0.5"] > best["AP@0.5"]
        if improved:
            best = {k: v for k, v in metrics.items()}
            best_epoch = epoch + 1
            if args.save_checkpoint:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": eval_model.state_dict(),
                        "model_type": args.model,
                        "task": args.task,
                        "modality": args.modality,
                        "img_size": img_size,
                        "multiscale_sizes": multiscale_sizes,
                        "input_mode": args.input_mode,
                        "class_names": class_names,
                        "metrics": {k: v for k, v in metrics.items() if k != "predictions"},
                    },
                    out_dir / "best_model.pth",
                )

        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        # Written every epoch, not just at the end, so a run that gets cut short
        # (Kaggle session limit, power loss) still reports its best epoch.
        _write_result(out_dir, args, spec, run_name, img_size, batch_size,
                      summary, best, best_epoch, len(history), time.time() - started)

        if args.early_stop_patience > 0:
            stale = (epoch + 1) - best_epoch
            if stale >= args.early_stop_patience:
                if not args.quiet:
                    print(
                        f"  early stop: no AP@0.5 improvement for {stale} epochs "
                        f"(best was epoch {best_epoch})",
                        flush=True,
                    )
                break

    result = _write_result(out_dir, args, spec, run_name, img_size, batch_size,
                           summary, best, best_epoch, len(history), time.time() - started)

    if not args.quiet:
        print(
            f"[{run_name}] best epoch {best_epoch}: AP@0.5={best['AP@0.5']:.4f} "
            f"P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f}"
        )
    return result


def _explicit_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """Which argparse dests were actually typed on this command line, so
    ``--config`` only fills in flags the caller left at their default."""
    dests = set()
    for tok in argv:
        if not tok.startswith("--"):
            continue
        flag = tok.split("=", 1)[0]
        for action in parser._actions:
            if flag in action.option_strings:
                dests.add(action.dest)
                break
    return dests


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.config is not None:
        explicit = _explicit_dests(parser, sys.argv[1:])
        explicit.discard("config")
        apply_to_namespace(RunConfig.from_json(args.config), args, explicit)
    run(args)


if __name__ == "__main__":
    main()
