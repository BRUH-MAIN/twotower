"""Recommend RPN anchor sizes/aspect ratios from the training box-size distribution.

Computes one anchor config per (modality, model) since the two modalities'
lesion sizes differ substantially (EUS is narrow-field) and the two training
resolutions in ``MODEL_REGISTRY`` (320 / 640) put the same physical box at a
different pixel size.

    python -m cadsel.measure_anchors
    python -m cadsel.measure_anchors --output anchors.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchors import compute_anchor_config
from .config import MODALITIES, MODEL_REGISTRY
from .data import build_manifest, patient_level_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42, help="matches the default training split seed")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--train-only", action="store_true", default=True,
        help="cluster only the training split, not the validation split (default)",
    )
    parser.add_argument("--output", type=Path, default=None, help="also write the table as JSON")
    args = parser.parse_args()

    # de-duplicate resolutions actually used across the registry so each is only computed once
    resolutions = sorted({spec["default_img_size"] for spec in MODEL_REGISTRY.values()})

    results: dict[str, dict] = {}
    for modality in MODALITIES:
        samples = build_manifest(modality, data_root=args.data_root)
        train_s, _val_s = patient_level_split(samples, args.val_fraction, args.seed)
        pool = train_s if args.train_only else samples
        for img_size in resolutions:
            cfg = compute_anchor_config(pool, img_size)
            results[f"{modality}@{img_size}"] = cfg
            print(f"{modality} @ {img_size}px  ({cfg['n_boxes']} boxes):")
            print(f"  sizes         = {cfg['sizes']}")
            print(f"  aspect_ratios = {cfg['aspect_ratios']}")
            print(f"  torchvision default sizes were (32, 64, 128, 256, 512)")
            print()

    if args.output:
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
