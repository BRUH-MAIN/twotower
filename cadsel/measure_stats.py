"""Measure per-modality channel statistics over the full dataset.

Reproduces the ``MODALITY_STATS`` table in ``cadsel/config.py``. Uses a streaming
sum / sum-of-squares pass so memory stays flat over the 4,912 images, and reports
the channel spread alongside the mean, because the spread is what shows how far
apart the two domains actually are (WLE 0.264 vs EUS 0.0004).

    python -m cadsel.measure_stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import MODALITIES
from .data import build_manifest


def measure(modality: str, data_root: Path | None = None,
            progress_every: int = 500) -> dict:
    samples = build_manifest(modality, data_root=data_root)
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    n_px = 0

    for i, sample in enumerate(samples):
        with Image.open(sample.image) as im:
            pixels = np.asarray(im.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
        total += pixels.sum(axis=0)
        total_sq += (pixels ** 2).sum(axis=0)
        n_px += len(pixels)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {modality} {i + 1}/{len(samples)}", flush=True)

    mean = total / n_px
    # Var = E[x^2] - E[x]^2; clamped because catastrophic cancellation can push a
    # near-constant channel (EUS is grayscale) a hair below zero.
    std = np.sqrt(np.maximum(total_sq / n_px - mean ** 2, 0.0))
    return {
        "images": len(samples),
        "mean": [round(float(v), 10) for v in mean],
        "std": [round(float(v), 10) for v in std],
        "channel_spread": round(float(mean.max() - mean.min()), 10),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="also write the table as JSON")
    args = parser.parse_args()

    stats = {m: measure(m, args.data_root) for m in MODALITIES}

    print("\nMODALITY_STATS = {")
    for modality, s in stats.items():
        print(f'    "{modality}": (')
        print(f"        {[round(v, 10) for v in s['mean']]},")
        print(f"        {[round(v, 10) for v in s['std']]},")
        print("    ),")
    print("}\n")
    for modality, s in stats.items():
        print(f"{modality}: {s['images']} images, "
              f"channel spread {s['channel_spread']:.6f}")

    if args.output:
        args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
