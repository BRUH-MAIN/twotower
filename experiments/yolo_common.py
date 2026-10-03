"""Shared helpers for Track B (YOLO11 detection track).

Kept separate from ``cadsel/`` because it's Ultralytics-specific glue
(dataset export, split reuse) rather than core project infrastructure other
detectors depend on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from cadsel.config import PROJECT_ROOT
from cadsel.data import Sample

RUNS_DIR = PROJECT_ROOT / "runs"


def filter_by_split(samples: Sequence[Sample], base_run: str) -> tuple[list[Sample], list[Sample]]:
    """Partition ``samples`` into ``(train, val)`` via an existing detection
    run's own ``split.json`` (``train_patients``/``val_patients``, keyed by
    ``Sample.patient_key``) rather than recomputing ``patient_level_split``.

    Pinning to an actual on-disk Faster R-CNN run's split means YOLO is
    scored against the literal same held-out patients as that run's
    published AP@0.5 -- safer against drift than re-deriving "the same"
    split from ``random``'s implementation, which could in principle change
    between environments/versions.
    """
    split_path = RUNS_DIR / base_run / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"no split.json under {split_path} -- base_run must be an existing "
            f"cadsel.train run directory under {RUNS_DIR}"
        )
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    train_keys = set(payload["train_patients"])
    val_keys = set(payload["val_patients"])

    train = [s for s in samples if s.patient_key in train_keys]
    val = [s for s in samples if s.patient_key in val_keys]
    missing = [
        s for s in samples if s.patient_key not in train_keys and s.patient_key not in val_keys
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} sample(s) belong to a patient absent from both "
            f"train_patients/val_patients in {split_path} -- base_run's own "
            f"manifest (modality/--centers filter) likely doesn't match the "
            f"samples passed in here"
        )
    return train, val


def flat_stem(sample: Sample) -> str:
    """Unique, collision-free output filename stem for one sample.

    Every patient folder restarts its own ``1.tiff, 2.tiff, ...`` numbering
    (``Sample.image``'s stem alone collides across patients), so the
    exported YOLO image/label pair is named by the full identity instead:
    ``<subtype>_<patient>_<original-stem>``.
    """
    return f"{sample.subtype}_{sample.patient}_{Path(sample.image).stem}"
