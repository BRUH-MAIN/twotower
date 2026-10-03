"""Multi-scale test-time augmentation (Weighted Boxes Fusion) vs single-scale.

Reports AP@0.5 with and without multi-scale WBF (see ``cadsel/tta.py``) on a
trained checkpoint's own validation split, so the TTA gain can be checked
before deciding to use it -- inference-only, no retraining, fully reversible.

    python -m cadsel.evaluate_tta --checkpoint runs/fasterrcnn_mobilenet_320__WLE__binary/best_model.pth
    python -m cadsel.evaluate_tta --checkpoint ... --scales 480,640,800
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from .config import DINOV3_MODEL_REGISTRY, EFFICIENTNET_MODEL_REGISTRY, TASK_CLASSES
from .data import (
    CadSelDetectionDataset,
    build_dataloaders,
    build_manifest,
    make_normalizer,
    patient_level_split,
)
from .engine import evaluate
from .metrics import DetectionAccumulator, summarize
from .models import (
    build_detection_model,
    build_dinov3_detection_model,
    build_efficientnet_detection_model,
    configure_transform_for_input_mode,
)
from .tta import predict_multiscale_wbf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--scales", default=None,
        help="comma-separated scales for multi-scale WBF (default: 0.8x/1.0x/1.25x "
        "the checkpoint's own img_size)",
    )
    p.add_argument("--wbf-iou-thr", type=float, default=0.55, help="WBF box-clustering IoU threshold")
    p.add_argument("--skip-box-thr", type=float, default=0.05, help="drop boxes below this score before fusing")
    p.add_argument("--conf-threshold", type=float, default=0.5, help="scoring threshold for P/R/F1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_type, task, modality = ckpt["model_type"], ckpt["task"], ckpt["modality"]
    img_size = ckpt["img_size"]
    train_multiscale_sizes = ckpt.get("multiscale_sizes")
    input_mode = ckpt.get("input_mode", "reference")
    class_names = TASK_CLASSES[task]

    if input_mode != "torchvision":
        raise SystemExit(
            "multi-scale TTA needs input_mode='torchvision' (its transform actually "
            f"resizes per call); this checkpoint used input_mode={input_mode!r}, which "
            "pre-resizes every image to one fixed size before the detector ever sees "
            "it, leaving nothing for multi-scale to vary."
        )

    scales = (
        sorted(int(s) for s in args.scales.split(","))
        if args.scales
        else sorted({int(img_size * f) for f in (0.8, 1.0, 1.25)})
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type in DINOV3_MODEL_REGISTRY:
        model = build_dinov3_detection_model(model_type, task, trainable_backbone=ckpt.get("dinov3_trainable_backbone", False))
    elif model_type in EFFICIENTNET_MODEL_REGISTRY:
        model = build_efficientnet_detection_model(model_type, task, pretrained=False)
    else:
        model = build_detection_model(model_type, task, pretrained=False)
    # If this checkpoint was itself trained with multi-scale sizes, its own
    # reported "best" metrics were evaluated at the *last* (largest) of those
    # sizes -- torchvision's GeneralizedRCNNTransform deterministically picks
    # min_size[-1] in eval mode. Reconstructing with plain `img_size` alone
    # would silently evaluate at a different resolution than what training
    # reported, making the "single-scale baseline" below not actually match
    # the number in this run's result.json.
    if train_multiscale_sizes:
        configure_transform_for_input_mode(model, input_mode, train_multiscale_sizes)
        print(f"  note: checkpoint was trained with multiscale_sizes={train_multiscale_sizes}; "
              f"single-scale baseline below uses its eval-mode resolution "
              f"({train_multiscale_sizes[-1]}px), matching this run's own reported metrics")
    else:
        configure_transform_for_input_mode(model, input_mode, img_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    normalize = make_normalizer(input_mode)

    samples = build_manifest(modality)
    _, val_samples = patient_level_split(samples, args.val_fraction, args.seed)
    val_ds = CadSelDetectionDataset(val_samples, task, img_size, train=False, input_mode=input_mode)

    print(f"[{args.checkpoint.parent.name}] {len(val_samples)} val images, scales={scales}")

    # -- single-scale baseline, via the standard eval path -------------------
    _, val_loader = build_dataloaders(
        val_samples[:1], val_samples, task, img_size, batch_size=4,
        num_workers=0, input_mode=input_mode, seed=args.seed,
    )
    baseline = evaluate(
        model, val_loader, device, class_names,
        conf_threshold=args.conf_threshold, progress=False, normalize=normalize,
        use_amp=device.type == "cuda",
    )
    print(
        f"  single-scale (img_size={img_size}):   AP@0.5={baseline['AP@0.5']:.4f}  "
        f"P={baseline['precision']:.4f}  R={baseline['recall']:.4f}  F1={baseline['f1']:.4f}"
    )

    # -- multi-scale WBF, one image at a time --------------------------------
    accumulator = DetectionAccumulator()
    for idx in tqdm(range(len(val_ds)), desc="multi-scale WBF"):
        image, target = val_ds[idx]
        out = predict_multiscale_wbf(
            model, image, scales, device, normalize=normalize,
            iou_thr=args.wbf_iou_thr, skip_box_thr=args.skip_box_thr,
        )
        accumulator.add(
            out["boxes"], out["scores"], out["labels"],
            target["boxes"].numpy(), target["labels"].numpy(),
        )
    wbf_metrics = summarize(accumulator, class_names, 0.5, args.conf_threshold)
    print(
        f"  multi-scale WBF ({scales}):  AP@0.5={wbf_metrics['AP@0.5']:.4f}  "
        f"P={wbf_metrics['precision']:.4f}  R={wbf_metrics['recall']:.4f}  F1={wbf_metrics['f1']:.4f}"
    )
    print(f"\n  Delta AP@0.5: {wbf_metrics['AP@0.5'] - baseline['AP@0.5']:+.4f}")


if __name__ == "__main__":
    main()
