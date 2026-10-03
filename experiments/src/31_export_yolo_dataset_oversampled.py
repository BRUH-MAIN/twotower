"""Experiment 31 (Track B follow-up) — export a class-oversampled YOLO
training set: images containing an under-represented class (Leiomyoma is
the rarest, 112 boxes total) are physically duplicated on disk, following
``cadsel.data.RepeatFactorSampler``'s own Detectron2-style formula
(``r(c) = max(1, sqrt(threshold / f(c)))``, reused directly rather than
re-derived). The val set is exported untouched and identical to the plain
export (``28_export_yolo_dataset.py``), so this only changes what the
training loader sees.

Physical duplication, not a custom per-epoch stochastic sampler (the actual
mechanism ``RepeatFactorSampler`` uses elsewhere in this project) --
Ultralytics' CLI has no hook for a custom PyTorch ``Sampler``, so this is a
coarser, deterministic approximation of the same idea: an image with repeat
factor 2.4 gets rounded and appears round(2.4)=2 times in the exported
folder, not "2 times most epochs, 3 times some epochs" the way the real
sampler would realise it.

    python experiments/src/31_export_yolo_dataset_oversampled.py \\
        --modality WLE --task multiclass --base-run size_rfs_ms_s42__WLE__multiclass \\
        --threshold 0.2 --out experiments/yolo/wle_multiclass_ms_only_s42_oversample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from cadsel.config import MODALITIES, TASK_CLASSES, TASK_ID_MAP
from cadsel.data import RepeatFactorSampler, build_manifest, read_yolo_label
from yolo_common import filter_by_split, flat_stem


def export_split(samples, images_dir: Path, labels_dir: Path, task: str,
                  repeat_factors: np.ndarray | None = None) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    id_map = TASK_ID_MAP[task]
    n = 0
    for i, s in enumerate(samples):
        stem = flat_stem(s)
        src_image = Path(s.image).resolve()
        lines = [
            f"{id_map[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            for cls, cx, cy, bw, bh in read_yolo_label(s.label)
        ]
        label_text = "\n".join(lines) + ("\n" if lines else "")

        repeats = 1 if repeat_factors is None else max(1, int(round(repeat_factors[i])))
        for r in range(repeats):
            suffix = f"_dup{r}" if r > 0 else ""
            dst_image = images_dir / (stem + suffix + src_image.suffix)
            if dst_image.exists() or dst_image.is_symlink():
                dst_image.unlink()
            dst_image.symlink_to(src_image)
            (labels_dir / (stem + suffix + ".txt")).write_text(label_text)
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", choices=MODALITIES, required=True)
    p.add_argument("--task", choices=["binary", "multiclass"], required=True)
    p.add_argument("--base-run", required=True, help="existing runs/<name> to reuse the split.json of")
    p.add_argument("--threshold", type=float, default=0.2)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    samples = build_manifest(args.modality)
    train, val = filter_by_split(samples, args.base_run)
    print(f"{args.modality}/{args.task}: {len(train)} train / {len(val)} val "
          f"(split from runs/{args.base_run}/split.json)")

    factors = RepeatFactorSampler._compute_repeat_factors(train, args.task, args.threshold).numpy()
    n_oversampled = int((factors > 1.0).sum())
    print(f"repeat factors (threshold={args.threshold}): min={factors.min():.2f} "
          f"max={factors.max():.2f} mean={factors.mean():.2f} "
          f"({n_oversampled}/{len(factors)} images oversampled)")

    out = args.out.resolve()
    n_train = export_split(train, out / "images" / "train", out / "labels" / "train",
                            args.task, repeat_factors=factors)
    n_val = export_split(val, out / "images" / "val", out / "labels" / "val", args.task)

    class_names = TASK_CLASSES[args.task]
    data_yaml = {
        "path": str(out), "train": "images/train", "val": "images/val",
        "nc": len(class_names), "names": class_names,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"wrote {out / 'data.yaml'}: {len(train)} unique train images -> "
          f"{n_train} exported (oversampled), {n_val} val (untouched, identical to the plain export)")


if __name__ == "__main__":
    main()
