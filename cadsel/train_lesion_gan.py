"""Train the conditional lesion GAN (see cadsel/lesion_gan.py) on real crops.

    python -m cadsel.train_lesion_gan --modality EUS --task multiclass --epochs 300

Real crops come from ``cadsel.data.LesionBank`` (the same extraction already
used for `copy_paste`), at ``lesion_gan.TILE`` resolution. Saves the trained
generator's state_dict, usable via
``cadsel.lesion_gan.GeneratedLesionBank.load(...)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT, TASKS, TASK_CLASSES, num_foreground_classes
from .data import LesionBank, build_manifest, patient_level_split
from .lesion_gan import TILE, train_lesion_gan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", default="EUS", help="generative augmentation is EUS-motivated, but not restricted to it")
    p.add_argument("--task", choices=TASKS, default="multiclass")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-per-class", type=int, default=400, help="cap on real crops per class fed to LesionBank")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    n_classes = num_foreground_classes(args.task)
    class_names = TASK_CLASSES[args.task]

    samples = build_manifest(args.modality)
    train_samples, _ = patient_level_split(samples, args.val_fraction, args.seed)

    bank = LesionBank(tile=TILE).build(train_samples, args.task, max_per_class=args.max_per_class)
    if not bank.patches:
        raise SystemExit(f"no lesion crops found for {args.modality}/{args.task}")

    patches = np.stack(bank.patches)
    labels = np.array(bank.labels, dtype=np.int64)
    counts = {class_names[c]: int((labels == c).sum()) for c in range(n_classes) if (labels == c).any()}
    print(f"training on {len(patches)} real crops at {TILE}x{TILE}: {counts}")
    if min(counts.values()) < 20:
        print(f"  WARNING: rarest class has only {min(counts.values())} crops -- a GAN this small-data "
              "is unlikely to produce diverse or clinically-plausible output for it; treat generated "
              "patches for that class with real skepticism (see cadsel/lesion_gan.py's docstring).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = train_lesion_gan(
        patches, labels, n_classes, epochs=args.epochs, batch=args.batch_size,
        lr=args.learning_rate, seed=args.seed, device=device,
    )

    out = args.output or (PROJECT_ROOT / "runs" / f"lesion_gan_{args.modality}_{args.task}.pth")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
