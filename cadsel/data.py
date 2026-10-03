"""Dataset indexing, patient-level splitting and the CAD-SEL torch Dataset.

Annotations ship in YOLO format (``class cx cy w h``, normalised); Faster R-CNN
wants absolute ``xyxy`` boxes and 1-based class ids (0 is background), so the
conversion happens here.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import (
    CADSEL_MEAN,
    CADSEL_STD,
    IMAGES_ROOT,
    LABELS_ROOT,
    MODALITIES,
    MODALITY_ALIASES,
    MODALITY_DIR,
    MODALITY_INDEX,
    MODALITY_STATS,
    TASK_ID_MAP,
    num_foreground_classes,
    resolve_roots,
)

IMAGE_EXTENSIONS = (".tiff", ".tif", ".png", ".jpg", ".jpeg")

# `flip`, `color` and `rotate` reproduce the reference dataloader. `scale_crop`
# and `copy_paste` are shortcut mitigations added on the evidence of the
# experiments/ audit — see experiments/README.md. `speckle_noise` and
# `ultrasound_gain` are EUS-specific (see their docstrings below): the
# literature review backing the Tier-1 performance levers (FINDINGS.md §5c)
# found that content-altering augmentations like `scale_crop`/`copy_paste`
# don't suit ultrasound the way physically-motivated noise/gain models do —
# opt these in only for `--modality EUS` runs, the same way `copy_paste` is
# opted in only where it actually helps.
AUGMENT_OPS = ("flip", "color", "rotate", "scale_crop", "copy_paste", "speckle_noise", "ultrasound_gain", "style_jitter")
DEFAULT_AUGMENT = ("flip", "color", "rotate")


def make_hard_negatives(
    samples: Sequence[Sample], fraction: float = 0.3, seed: int = 0,
) -> list[Sample]:
    """Mark a fraction of training images to be served as lesion-free negatives.

    Experiment 8 measured detectors firing on lesion-free crops at ~0.3x their
    rate on full frames, and Experiment 3 showed the label is largely recoverable
    from such crops. Nothing in the standard objective penalises that, because
    every training image contains a lesion.

    A negative sample carries the same paths but is flagged so the dataset serves
    the largest lesion-free rectangle with an *empty* target. torchvision's
    detectors accept empty boxes and treat every anchor as background, which is
    exactly the missing supervision.
    """
    rng = random.Random(seed)
    picked = rng.sample(range(len(samples)), int(round(len(samples) * fraction)))
    negatives = []
    for i in sorted(picked):
        s = samples[i]
        negatives.append(
            Sample(image=s.image, label=s.label, center=s.center,
                   modality=s.modality, subtype=s.subtype,
                   patient=s.patient, negative=True)
        )
    return negatives


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    image: str          # absolute path to the image
    label: str          # absolute path to the YOLO .txt
    center: str         # "Center-1" ... "Center-6"
    modality: str       # "WLE" | "EUS"
    subtype: str        # folder subtype, e.g. "NET_G1"
    patient: str        # pseudonym, e.g. "NET_G1-0"
    negative: bool = False   # serve the lesion-free region with an empty target

    @property
    def patient_key(self) -> str:
        """Identity of the case, for patient-level splitting.

        Deliberately excludes the centre: the pseudonyms are globally unique
        (collapsing centres reproduces the exact per-subtype patient counts of
        Table 2), and leaving the centre in would make the figshare and Hugging
        Face layouts split differently, so runs on the two would not be
        comparable.
        """
        return f"{self.subtype}/{self.patient}"


def _find_image_for_label(label_path: Path, images_root: Path, labels_root: Path) -> Path | None:
    rel = label_path.relative_to(labels_root)
    stem_dir = images_root / rel.parent
    for ext in IMAGE_EXTENSIONS:
        candidate = stem_dir / (rel.stem + ext)
        if candidate.exists():
            return candidate
    return None


def build_manifest(
    modality: str,
    images_root: Path | None = None,
    labels_root: Path | None = None,
    centers: Sequence[str] | None = None,
    data_root: Path | str | None = None,
) -> list[Sample]:
    """Index every annotated image for one modality.

    Layout-agnostic: the label tree is always ``.../<modality>/<subtype>/<patient>/<n>.txt``,
    optionally preceded by a centre directory. That covers both the figshare
    release (``Center-1/WLE-Set/...``) and the Hugging Face mirror
    (``Full/NET-WL/...``).
    """
    if images_root is None or labels_root is None:
        images_root, labels_root = resolve_roots(data_root)
    images_root, labels_root = Path(images_root), Path(labels_root)

    aliases = MODALITY_ALIASES[modality]
    samples: list[Sample] = []
    missing: list[str] = []

    for label_path in sorted(labels_root.rglob("*.txt")):
        parts = label_path.relative_to(labels_root).parts
        if len(parts) < 4:
            continue  # need <modality>/<subtype>/<patient>/<file>

        modality_idx = next(
            (i for i, p in enumerate(parts[:-3]) if p.lower() in aliases), None
        )
        if modality_idx is None:
            continue

        subtype, patient = parts[-3], parts[-2]
        center = parts[0] if modality_idx > 0 else "Full"
        if centers is not None and center not in centers:
            continue

        image_path = _find_image_for_label(label_path, images_root, labels_root)
        if image_path is None:
            missing.append(str(label_path))
            continue

        samples.append(
            Sample(
                image=str(image_path),
                label=str(label_path),
                center=center,
                modality=modality,
                subtype=subtype,
                patient=patient,
            )
        )

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} label file(s) have no matching image, e.g. {missing[:3]}"
        )
    if not samples:
        raise FileNotFoundError(
            f"no {modality} labels found under {labels_root}; "
            f"expected a folder named one of {sorted(aliases)}"
        )
    return samples


def build_joint_manifest(
    modalities: Sequence[str] = MODALITIES,
    images_root: Path | None = None,
    labels_root: Path | None = None,
    centers: Sequence[str] | None = None,
    data_root: Path | str | None = None,
) -> list[Sample]:
    """Index both modalities into one manifest for joint training.

    ``Sample.modality`` already distinguishes the two, and ``Sample.patient_key``
    deliberately omits it, so the union can be split with the same patient-level
    machinery: a patient's WLE and EUS frames land on the same side automatically.

    That property is the whole point. Splitting each modality independently — as
    the per-modality baselines do — puts 67 of 85 EUS validation patients' WLE
    frames in the *training* set, which would leak straight through any model that
    sees both.
    """
    samples: list[Sample] = []
    for modality in modalities:
        samples.extend(
            build_manifest(
                modality,
                images_root=images_root,
                labels_root=labels_root,
                centers=centers,
                data_root=data_root,
            )
        )
    return samples


def read_yolo_label(path: str | Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = parts
            rows.append((int(float(cls)), float(cx), float(cy), float(w), float(h)))
    return rows


def box_area_frac(sample: Sample) -> float:
    """Largest annotated box as a fraction of frame area, resolution-independent.

    Ported from ``experiments/src/09_size_matched_eval.py`` (kept byte-identical
    there; that file is a frozen artifact and is not imported from here to avoid
    coupling the production pipeline to the audit tree). This is the size metric
    the small-lesion AP gap (FINDINGS.md: AP 0.471 -> 0.663 -> 0.755 by size
    tertile) is measured against -- not the clinical ``Size (cm)`` metadata
    column, which is never joined to detection boxes anywhere in this codebase.
    """
    best = 0.0
    for _c, _cx, _cy, bw, bh in read_yolo_label(sample.label):
        best = max(best, float(bw * bh))
    return best


# ---------------------------------------------------------------------------
# Patient-level splitting
# ---------------------------------------------------------------------------
def _default_strata_key(items: Sequence[Sample]):
    """Stratify on histological subtype, as the unimodal baselines do."""
    return items[0].subtype


def joint_strata_key(items: Sequence[Sample]):
    """Stratify on subtype *and* whether the patient has EUS frames.

    EUS was performed on only 421 of the 641 patients, so a split stratified on
    subtype alone can leave the validation set with very few *paired* patients —
    and the paired ones are the only patients the fusion head can be evaluated on.
    Adding the flag keeps the paired/unpaired ratio stable across the split.
    """
    return (items[0].subtype, any(s.modality == "EUS" for s in items))


def patient_level_split(
    samples: Sequence[Sample],
    val_fraction: float = 0.2,
    seed: int = 42,
    strata_key=_default_strata_key,
) -> tuple[list[Sample], list[Sample]]:
    """Split patients (never images) into train/val, stratified by subtype.

    The paper is explicit that "images from the same patient were assigned
    exclusively to either the training or validation set", so the unit of
    assignment here is ``Sample.patient_key``.

    ``strata_key`` receives every sample belonging to one patient and returns the
    stratum to balance over; pass ``joint_strata_key`` for joint-modality runs.
    """
    by_patient: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_patient[s.patient_key].append(s)

    strata: dict[object, list[str]] = defaultdict(list)
    for key, items in by_patient.items():
        strata[strata_key(items)].append(key)

    rng = random.Random(seed)
    val_keys: set[str] = set()
    for stratum in sorted(strata, key=repr):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        # At least one validation patient per subtype so every class is scored.
        n_val = max(1, int(round(len(keys) * val_fraction)))
        n_val = min(n_val, len(keys) - 1) if len(keys) > 1 else len(keys)
        val_keys.update(keys[:n_val])

    train = [s for s in samples if s.patient_key not in val_keys]
    val = [s for s in samples if s.patient_key in val_keys]
    return train, val


def image_level_split(
    samples: Sequence[Sample],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """Split by image, ignoring patient identity — for diagnosis only.

    The paper's Usage Notes explicitly warn against this ("it is also essential
    to perform dataset partitioning at the patient level, rather than the image
    level, to avoid data leakage from correlated images"). It is provided purely
    so the leakage effect can be quantified: a patient contributes many
    near-duplicate frames, so an image-level validation set is largely
    already-seen data. Never use it to report results.
    """
    strata: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        strata[s.subtype].append(s)

    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    for subtype in sorted(strata):
        items = sorted(strata[subtype], key=lambda s: (s.image))
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_fraction)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def patient_level_folds(
    samples: Sequence[Sample],
    n_folds: int = 5,
    seed: int = 42,
    strata_key=_default_strata_key,
) -> list[tuple[list[Sample], list[Sample]]]:
    """Stratified patient-level K-fold, for the cross-validation protocol."""
    by_patient: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_patient[s.patient_key].append(s)

    strata: dict[object, list[str]] = defaultdict(list)
    for key, items in by_patient.items():
        strata[strata_key(items)].append(key)

    rng = random.Random(seed)
    fold_of: dict[str, int] = {}
    for stratum in sorted(strata, key=repr):
        keys = sorted(strata[stratum])
        rng.shuffle(keys)
        for i, key in enumerate(keys):
            fold_of[key] = i % n_folds

    folds = []
    for f in range(n_folds):
        train = [s for s in samples if fold_of[s.patient_key] != f]
        val = [s for s in samples if fold_of[s.patient_key] == f]
        folds.append((train, val))
    return folds


def split_summary(train: Sequence[Sample], val: Sequence[Sample]) -> dict:
    def describe(items: Sequence[Sample]) -> dict:
        out = {
            "images": len(items),
            "patients": len({s.patient_key for s in items}),
            "by_subtype": dict(sorted(Counter(s.subtype for s in items).items())),
        }
        by_modality = Counter(s.modality for s in items)
        if len(by_modality) > 1:  # joint run: the per-modality breakdown matters
            out["by_modality"] = dict(sorted(by_modality.items()))
            out["paired_patients"] = len(
                {s.patient_key for s in items if s.modality == "WLE"}
                & {s.patient_key for s in items if s.modality == "EUS"}
            )
        return out

    return {"train": describe(train), "val": describe(val)}


def save_split(path: Path, train: Sequence[Sample], val: Sequence[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": split_summary(train, val),
        "train_patients": sorted({s.patient_key for s in train}),
        "val_patients": sorted({s.patient_key for s in val}),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------
def compute_class_weights(
    samples: Iterable[Sample], task: str
) -> tuple[np.ndarray, np.ndarray]:
    """Weights inversely proportional to box frequency, per the paper.

    Normalised so the weights sum to the number of foreground classes, matching
    ``calculate_class_weights`` in the reference repository.
    """
    n_classes = num_foreground_classes(task)
    id_map = TASK_ID_MAP[task]
    counts = np.zeros(n_classes, dtype=np.float64)
    for s in samples:
        if getattr(s, "negative", False):
            continue  # negatives are served with an empty target
        for cls, *_ in read_yolo_label(s.label):
            counts[id_map[cls]] += 1

    safe = np.where(counts > 0, counts, 1.0)
    weights = 1.0 / safe
    weights = weights / weights.sum() * n_classes
    # A class absent from the training split must not attract weight.
    weights = np.where(counts > 0, weights, 0.0)
    return weights, counts


# ---------------------------------------------------------------------------
# Augmentation helpers (operate on float32 HWC images in [0, 1] + xyxy boxes)
# ---------------------------------------------------------------------------
def _hflip(image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = image[:, ::-1]
    if len(boxes):
        w = image.shape[1]
        x1 = boxes[:, 0].copy()
        boxes[:, 0] = w - boxes[:, 2]
        boxes[:, 2] = w - x1
    return np.ascontiguousarray(image), boxes


def _color_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Brightness / contrast / saturation jitter, mirroring the HSV adjustment
    of the reference dataloader."""
    brightness = rng.uniform(0.8, 1.2)
    contrast = rng.uniform(0.8, 1.2)
    saturation = rng.uniform(0.8, 1.2)

    image = image * brightness
    mean = image.mean()
    image = (image - mean) * contrast + mean
    gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    image = gray[..., None] + (image - gray[..., None]) * saturation
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _style_jitter(
    image: np.ndarray, rng: random.Random,
    channel_gain_range: tuple[float, float] = (0.6, 1.4),
    channel_bias_range: tuple[float, float] = (-0.08, 0.08),
    hue_shift_range_deg: tuple[float, float] = (-20.0, 20.0),
) -> np.ndarray:
    """EndoStyle-style per-center color-fingerprint randomisation.

    Distinct from ``_color_jitter``'s single global brightness/contrast/
    saturation triplet (which cannot change relative channel balance or hue):
    independent gain+bias per RGB channel, plus an HSV hue rotation. Targets
    the site-fingerprint shortcut directly -- centre is decodable from image
    statistics at 0.81 balanced accuracy vs 0.167 chance (FINDINGS.md §4),
    and Center-3 specifically is 45-48% darker *per channel* than Center-1,
    a per-channel signature a same-magnitude-per-channel jitter can't disrupt.
    """
    gain = np.array([rng.uniform(*channel_gain_range) for _ in range(3)], dtype=np.float32)
    bias = np.array([rng.uniform(*channel_bias_range) for _ in range(3)], dtype=np.float32)
    image = image * gain + bias
    image = np.clip(image, 0.0, 1.0).astype(np.float32)

    hue_shift = rng.uniform(*hue_shift_range_deg) / 360.0
    pil = Image.fromarray((image * 255.0).astype(np.uint8)).convert("HSV")
    h, s, v = pil.split()
    h_arr = (np.asarray(h, dtype=np.int16) + int(round(hue_shift * 255))) % 256
    pil = Image.merge("HSV", (Image.fromarray(h_arr.astype(np.uint8)), s, v)).convert("RGB")
    return np.clip(np.asarray(pil, dtype=np.float32) / 255.0, 0.0, 1.0).astype(np.float32)


def _speckle_noise(
    image: np.ndarray, rng: random.Random, sigma_range: tuple[float, float] = (0.05, 0.25)
) -> np.ndarray:
    """Multiplicative speckle noise, EUS-specific.

    Ultrasound's own physical noise model is multiplicative — speckle scales
    with local signal intensity — unlike the additive sensor noise generic
    "color jitter"-style augmentations implicitly assume for natural images.
    Applied identically across all 3 channels since EUS frames are grayscale
    replicated to RGB, so the noise stays achromatic like real speckle.
    """
    sigma = rng.uniform(*sigma_range)
    noise = np.random.RandomState(rng.randrange(2**31)).normal(0.0, sigma, size=image.shape[:2])
    out = image * (1.0 + noise[..., None])
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _ultrasound_gain(
    image: np.ndarray, rng: random.Random, strength_range: tuple[float, float] = (0.15, 0.4)
) -> np.ndarray:
    """Depth-dependent intensity gain, EUS-specific.

    Simulates operator time-gain-compensation (TGC) variation: a smooth
    gradient along image height, since ultrasound depth runs top-to-bottom in
    a B-mode frame — a physically distinct effect from the uniform
    brightness/contrast jitter `_color_jitter` already applies, which has no
    spatial structure.
    """
    h = image.shape[0]
    strength = rng.uniform(*strength_range)
    direction = 1.0 if rng.random() < 0.5 else -1.0
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)
    gain = 1.0 + direction * strength * (ramp - 0.5)
    out = image * gain[:, None, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _scale_crop(
    image: np.ndarray, boxes: np.ndarray, rng: random.Random,
    min_scale: float = 0.6, max_scale: float = 1.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Random-resized-crop that changes the lesion's *apparent* size.

    Experiment 3 showed that annotation geometry alone — box size and position —
    retains 48-189% of full-image performance, so absolute lesion scale is itself
    a shortcut. Cropping by a random factor and letting the caller resize back to
    a fixed input size decorrelates apparent size from class.

    The crop window is chosen to keep at least one ground-truth box, so this never
    silently produces an image whose label has no visible support.
    """
    h, w = image.shape[:2]
    scale = rng.uniform(min_scale, max_scale)
    cw, ch = int(round(w / scale)), int(round(h / scale))
    cw, ch = max(16, min(cw, w)), max(16, min(ch, h))

    if len(boxes):
        keep = boxes[rng.randrange(len(boxes))]
        # window must overlap the chosen box
        x0_lo = int(max(0, min(keep[2] - cw, w - cw)))
        x0_hi = int(min(max(0, keep[0]), w - cw))
        y0_lo = int(max(0, min(keep[3] - ch, h - ch)))
        y0_hi = int(min(max(0, keep[1]), h - ch))
        x0 = rng.randint(min(x0_lo, x0_hi), max(x0_lo, x0_hi)) if w > cw else 0
        y0 = rng.randint(min(y0_lo, y0_hi), max(y0_lo, y0_hi)) if h > ch else 0
    else:
        x0 = rng.randint(0, w - cw) if w > cw else 0
        y0 = rng.randint(0, h - ch) if h > ch else 0

    crop = image[y0:y0 + ch, x0:x0 + cw]
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, 0::2] -= x0
        boxes[:, 1::2] -= y0
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, cw - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, ch - 1)

    # Resize back to the input dimensions so the rest of the pipeline is
    # unaffected; this is what actually rescales the lesion.
    was_float = crop.dtype != np.uint8
    src = (np.clip(crop * 255.0, 0, 255).astype(np.uint8) if was_float else crop)
    out = np.asarray(Image.fromarray(src).resize((w, h), Image.BILINEAR))
    if was_float:
        out = out.astype(np.float32) / 255.0
    if len(boxes):
        boxes[:, 0::2] *= w / cw
        boxes[:, 1::2] *= h / ch
    return np.ascontiguousarray(out), boxes


class LesionBank:
    """Pool of cropped lesions for copy-paste augmentation.

    Built once from the training split and shared across workers. Copy-paste
    attacks the shortcut measured in Experiment 3 directly: pasting a lesion of
    class A onto an image whose context belongs to class B breaks the
    context-to-label association that lets a model score 68-95% on lesion-free
    crops. It also rebalances rare classes, which loss weighting alone cannot fix
    when there are only 112 Leiomyoma boxes.
    """

    def __init__(self, tile: int = 128) -> None:
        self.tile = tile
        self.patches: list[np.ndarray] = []
        self.labels: list[int] = []
        self.by_label: dict[int, list[int]] = defaultdict(list)

    def build(self, samples: Sequence[Sample], task: str, max_per_class: int = 400,
              verbose: bool = True) -> "LesionBank":
        id_map = TASK_ID_MAP[task]
        counts: Counter = Counter()
        for s in samples:
            if getattr(s, "negative", False):
                continue  # a negative duplicates an image already in the pool
            rows = read_yolo_label(s.label)
            if not rows:
                continue
            wanted = [r for r in rows if counts[id_map[r[0]]] < max_per_class]
            if not wanted:
                continue
            with Image.open(s.image) as im:
                im = im.convert("RGB")
                W, H = im.size
                arr = np.asarray(im, dtype=np.uint8)
            for cls, cx, cy, bw, bh in wanted:
                x1 = int(max(0, (cx - bw / 2) * W)); x2 = int(min(W, (cx + bw / 2) * W))
                y1 = int(max(0, (cy - bh / 2) * H)); y2 = int(min(H, (cy + bh / 2) * H))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                patch = Image.fromarray(arr[y1:y2, x1:x2]).resize(
                    (self.tile, self.tile), Image.BILINEAR)
                label = id_map[cls]
                self.by_label[label].append(len(self.patches))
                self.patches.append(np.asarray(patch, dtype=np.uint8))
                self.labels.append(label)
                counts[label] += 1
        if verbose:
            print(f"  lesion bank: {len(self.patches)} patches "
                  f"{ {k: len(v) for k, v in sorted(self.by_label.items())} }")
        return self

    def sample(self, rng: random.Random) -> tuple[np.ndarray, int]:
        """Draw a patch, uniformly over *classes* so rare ones are oversampled."""
        label = rng.choice(sorted(self.by_label))
        idx = rng.choice(self.by_label[label])
        return self.patches[idx], label


def _copy_paste(
    image: np.ndarray, boxes: np.ndarray, labels: np.ndarray,
    bank: LesionBank, rng: random.Random, n_paste: int = 1,
    min_frac: float = 0.08, max_frac: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paste lesions from other cases at random positions, adding their boxes."""
    h, w = image.shape[:2]
    image = image.copy()
    new_boxes, new_labels = list(boxes), list(labels)

    for _ in range(n_paste):
        patch, label = bank.sample(rng)
        side = rng.uniform(min_frac, max_frac) * min(h, w)
        pw = ph = max(12, int(side))
        if pw >= w or ph >= h:
            continue
        px, py = rng.randint(0, w - pw), rng.randint(0, h - ph)

        # avoid burying an existing annotation
        if any(not (px + pw <= b[0] or px >= b[2] or py + ph <= b[1] or py >= b[3])
               for b in new_boxes):
            continue

        resized = np.asarray(
            Image.fromarray(patch).resize((pw, ph), Image.BILINEAR), dtype=np.uint8)
        image[py:py + ph, px:px + pw] = (resized.astype(np.float32) / 255.0
                                         if image.dtype != np.uint8 else resized)
        new_boxes.append([px, py, px + pw, py + ph])
        new_labels.append(label + 1)   # +1: background occupies 0

    boxes = np.array(new_boxes, dtype=np.float32).reshape(-1, 4)
    labels = np.array(new_labels, dtype=np.int64)
    return image, boxes, labels


def _rotate(
    image: np.ndarray, boxes: np.ndarray, angle_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate about the image centre; boxes become the AABB of the rotated corners."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    pil = Image.fromarray((image * 255.0).astype(np.uint8))
    rotated = pil.rotate(angle_deg, resample=Image.BILINEAR, expand=False)
    image = np.asarray(rotated, dtype=np.float32) / 255.0

    if len(boxes):
        # PIL rotates counter-clockwise; the corresponding point mapping is the
        # clockwise rotation of the coordinates about the centre.
        corners = np.stack(
            [
                boxes[:, [0, 1]],
                boxes[:, [2, 1]],
                boxes[:, [2, 3]],
                boxes[:, [0, 3]],
            ],
            axis=1,
        )  # (N, 4, 2)
        dx = corners[..., 0] - cx
        dy = corners[..., 1] - cy
        xs = cos_t * dx + sin_t * dy + cx
        ys = -sin_t * dx + cos_t * dy + cy
        boxes = np.stack(
            [xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1
        ).astype(np.float32)
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, w - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, h - 1)
    return image, boxes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CadSelDetectionDataset(Dataset):
    """Returns ``(image_tensor, target_dict)`` ready for torchvision detectors.

    ``input_mode``
        ``"reference"``  square resize to ``img_size`` and normalisation with the
                         CAD-SEL channel statistics, as done by the reference
                         repository (the detector's own transform then runs on
                         top of it).
        ``"torchvision"`` raw [0, 1] images; the detector's ``GeneralizedRCNNTransform``
                         performs resizing and ImageNet normalisation. Cleaner,
                         and avoids normalising twice.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        task: str,
        img_size: int = 640,
        train: bool = False,
        input_mode: str = "reference",
        augment: bool = True,
        seed: int = 0,
        augment_ops: Sequence[str] | None = None,
        lesion_bank: "LesionBank | None" = None,
    ) -> None:
        self.samples = list(samples)
        self.task = task
        self.id_map = TASK_ID_MAP[task]
        self.img_size = img_size
        self.train = train
        self.input_mode = input_mode
        self.augment = augment and train
        self.seed = seed
        self.ops = set(augment_ops if augment_ops is not None else DEFAULT_AUGMENT)
        self.lesion_bank = lesion_bank
        unknown = self.ops - set(AUGMENT_OPS)
        if unknown:
            raise ValueError(f"unknown augmentation(s) {sorted(unknown)}; "
                             f"expected from {AUGMENT_OPS}")
        if "copy_paste" in self.ops and self.augment and lesion_bank is None:
            raise ValueError("copy_paste requires a LesionBank")
        self.mean = np.array(CADSEL_MEAN, dtype=np.float32)
        self.std = np.array(CADSEL_STD, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.input_mode == "reference":
                im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32) / 255.0

    def _lesion_free_crop(self, sample) -> np.ndarray | None:
        """Largest rectangle not overlapping any annotation, at img_size."""
        with Image.open(sample.image) as im:
            im = im.convert("RGB")
            W, H = im.size
            arr = np.asarray(im, dtype=np.uint8)

        boxes = []
        for _c, cx, cy, bw, bh in read_yolo_label(sample.label):
            x1 = int(max(0, (cx - bw / 2) * W)); x2 = int(min(W, (cx + bw / 2) * W))
            y1 = int(max(0, (cy - bh / 2) * H)); y2 = int(min(H, (cy + bh / 2) * H))
            if x2 - x1 > 4 and y2 - y1 > 4:
                boxes.append((x1, y1, x2, y2))
        if not boxes:
            return None

        bx1 = min(b[0] for b in boxes); bx2 = max(b[2] for b in boxes)
        by1 = min(b[1] for b in boxes); by2 = max(b[3] for b in boxes)
        cands = [(0, 0, W, by1), (0, by2, W, H), (0, 0, bx1, H), (bx2, 0, W, H)]
        cands = [c for c in cands if c[2] - c[0] >= 32 and c[3] - c[1] >= 32]
        if not cands:
            return None
        x1, y1, x2, y2 = max(cands, key=lambda c: (c[2]-c[0]) * (c[3]-c[1]))
        crop = Image.fromarray(arr[y1:y2, x1:x2]).resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        return np.asarray(crop, dtype=np.float32) / 255.0

    def __getitem__(self, index: int):
        sample = self.samples[index]

        if getattr(sample, "negative", False):
            crop = self._lesion_free_crop(sample)
            if crop is not None:
                if self.augment and random.Random(
                        (self.seed * 7_919) ^ index).random() < 0.5:
                    crop = np.ascontiguousarray(crop[:, ::-1])
                crop = np.clip(crop * 255.0, 0, 255).astype(np.uint8)
                tensor = torch.from_numpy(
                    np.ascontiguousarray(crop.transpose(2, 0, 1)))
                return tensor, {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "image_id": torch.tensor([index]),
                    "index": torch.tensor([index]),
                    "modality": torch.tensor([MODALITY_INDEX[sample.modality]]),
                    "center": sample.center,
                }
            # no usable lesion-free region: fall through and serve it normally

        image = self._load_image(sample.image)
        h, w = image.shape[:2]

        boxes, labels = [], []
        for cls, cx, cy, bw, bh in read_yolo_label(sample.label):
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
            x1, x2 = max(0.0, min(x1, w - 1)), max(0.0, min(x2, w - 1))
            y1, y2 = max(0.0, min(y1, h - 1)), max(0.0, min(y2, h - 1))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            boxes.append([x1, y1, x2, y2])
            # +1 because Faster R-CNN reserves label 0 for background.
            labels.append(self.id_map[cls] + 1)

        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels, dtype=np.int64)

        if self.augment:
            rng = random.Random((self.seed * 1_000_003) ^ index)
            if "flip" in self.ops and rng.random() < 0.5:
                image, boxes = _hflip(image, boxes)
            if "color" in self.ops and rng.random() < 0.5:
                image = _color_jitter(image, rng)
            if "rotate" in self.ops and rng.random() < 0.5:
                image, boxes = _rotate(image, boxes, rng.uniform(-15.0, 15.0))
            if "scale_crop" in self.ops and rng.random() < 0.5:
                image, boxes = _scale_crop(image, boxes, rng)
            if "copy_paste" in self.ops and rng.random() < 0.5:
                image, boxes, labels = _copy_paste(
                    image, boxes, labels, self.lesion_bank, rng)
            if "speckle_noise" in self.ops and rng.random() < 0.5:
                image = _speckle_noise(image, rng)
            if "ultrasound_gain" in self.ops and rng.random() < 0.5:
                image = _ultrasound_gain(image, rng)
            if "style_jitter" in self.ops and rng.random() < 0.5:
                image = _style_jitter(image, rng)
            if len(boxes):
                keep = (boxes[:, 2] - boxes[:, 0] >= 1.0) & (boxes[:, 3] - boxes[:, 1] >= 1.0)
                boxes, labels = boxes[keep], labels[keep]

        # Hand uint8 to the collate/IPC path, not float32: worker tensors travel
        # through shared memory, and 640x640 float32 (4.9 MB/image) exhausts it
        # on Windows (OSError 1455) and on Kaggle's small /dev/shm. Normalisation
        # happens on the GPU instead - see `make_normalizer`.
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([index]),
            "index": torch.tensor([index]),
            "modality": torch.tensor([MODALITY_INDEX[sample.modality]]),
            "center": sample.center,
        }
        return tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_normalizer(input_mode: str, per_modality: bool = False):
    """Turn the uint8 batch tensors into the floats the detector expects.

    Runs on whatever device the images are already on, so the expensive part
    never crosses the worker IPC boundary.

    ``per_modality`` selects ``MODALITY_STATS`` instead of the single global
    ``CADSEL_MEAN``/``CADSEL_STD`` pair. It is required for joint runs: the global
    statistics are effectively the WLE ones (WLE spans a 0.264 channel spread, EUS
    0.0004 and is ~2.9x darker), so reusing them centres EUS near -1.7 instead of
    zero. Callers then pass the batch's modality to ``normalize``, which is
    well-defined because ``ModalityBatchSampler`` keeps batches homogeneous.
    """
    cache: dict[tuple[torch.device, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def stats_for(device: torch.device, modality: str | None):
        key = (device, modality or "")
        if key not in cache:
            if per_modality and modality is not None:
                mean, std = MODALITY_STATS[modality]
            else:
                mean, std = CADSEL_MEAN, CADSEL_STD
            cache[key] = (
                torch.tensor(mean, device=device).view(3, 1, 1),
                torch.tensor(std, device=device).view(3, 1, 1),
            )
        return cache[key]

    def normalize(
        images: list[torch.Tensor], modality: str | None = None
    ) -> list[torch.Tensor]:
        if not images:
            return images
        out = []
        for img in images:
            # `.float()` copies, so the in-place divide never touches the uint8
            # tensor the worker shipped over shared memory.
            x = img.float().div_(255.0)
            if input_mode == "reference":
                mean, std = stats_for(x.device, modality)
                x = (x - mean) / std
            out.append(x)
        return out

    return normalize


class ModalityBatchSampler(torch.utils.data.Sampler):
    """Yields batches drawn from a single modality at a time.

    Both the per-modality normaliser and the conditioning layers in
    ``models.build_joint_detection_model`` need one well-defined modality per
    forward pass. Batches are homogeneous, but their *order* is shuffled together,
    so an epoch still interleaves the two domains rather than training all of one
    and then all of the other — which matters when the domains are as far apart as
    WLE and EUS.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.groups: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            self.groups[s.modality].append(i)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = -1

    def _batches(self, rng: random.Random | None) -> list[list[int]]:
        batches: list[list[int]] = []
        for modality in sorted(self.groups):
            idx = list(self.groups[modality])
            if rng is not None:
                rng.shuffle(idx)
            for k in range(0, len(idx), self.batch_size):
                chunk = idx[k:k + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
        if rng is not None:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        self._epoch += 1
        rng = random.Random(self.seed + self._epoch) if self.shuffle else None
        return iter(self._batches(rng))

    def __len__(self) -> int:
        return len(self._batches(None))


class CenterBatchSampler(torch.utils.data.Sampler):
    """Yields batches drawn from a single acquisition center at a time --
    ``ModalityBatchSampler``'s pattern, grouped by ``Sample.center`` instead
    of modality. Group DRO training needs this: the per-batch loss is used
    as that step's estimate of a group's loss (``models.GroupDROLossState``),
    which is only a meaningful estimate when every example in the batch
    belongs to the same group. As with ``ModalityBatchSampler``, batches are
    homogeneous but their *order* is shuffled together, so an epoch still
    interleaves centers rather than training all of one and then the next.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.groups: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(samples):
            self.groups[s.center].append(i)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = -1

    def _batches(self, rng: random.Random | None) -> list[list[int]]:
        batches: list[list[int]] = []
        for center in sorted(self.groups):
            idx = list(self.groups[center])
            if rng is not None:
                rng.shuffle(idx)
            for k in range(0, len(idx), self.batch_size):
                chunk = idx[k:k + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
        if rng is not None:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        self._epoch += 1
        rng = random.Random(self.seed + self._epoch) if self.shuffle else None
        return iter(self._batches(rng))

    def __len__(self) -> int:
        return len(self._batches(None))


class _StochasticRepeatSampler(torch.utils.data.Sampler):
    """Shared stochastic-round oversampling mechanism.

    Subclasses only need to populate ``self.repeat_factors`` (one float per
    training image) in ``__init__``; the epoch-redrawn rounding and shuffling
    below is identical regardless of what the factors are bucketed by.
    """

    seed: int
    _epoch: int
    repeat_factors: torch.Tensor

    def __iter__(self):
        self._epoch += 1
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)
        rands = torch.rand(len(self.repeat_factors), generator=g)
        counts = (self.repeat_factors + rands).floor().long()
        indices: list[int] = []
        for idx, n in enumerate(counts.tolist()):
            indices.extend([idx] * int(n))
        order = torch.randperm(len(indices), generator=g).tolist()
        return iter(indices[i] for i in order)

    def __len__(self) -> int:
        # Expected length; the realised length varies epoch to epoch by design
        # (that variation is the mechanism, not sampling noise to suppress).
        return int(self.repeat_factors.sum().item())


class RepeatFactorSampler(_StochasticRepeatSampler):
    """Oversamples images containing an under-represented class.

    Detectron2's Repeat Factor Sampling (Gupta, Dollar & Girshick, 2019): the
    image-level complement to ``compute_class_weights``' per-box loss
    weighting. Loss weighting changes how much a rare-class box's gradient
    counts once the optimiser sees it; RFS changes how *often* the optimiser
    sees an image containing one at all -- the two address different
    mechanisms and are meant to be layered together, not used as alternatives.

    ``threshold``: classes whose image-frequency (fraction of training images
    containing at least one box of that class) falls below this get a repeat
    factor > 1; images with no under-threshold class keep a factor of 1.
    Follows Detectron2's formula exactly: ``r(c) = max(1, sqrt(threshold /
    f(c)))``, ``r(image) = max over classes present of r(c)``, realised as a
    stochastic round -- redrawn every epoch -- so e.g. a 1.4x factor means
    "1x this epoch, 2x next epoch, ..." on average rather than a fixed count.
    """

    def __init__(
        self, samples: Sequence[Sample], task: str, threshold: float, seed: int = 0
    ) -> None:
        self.seed = seed
        self._epoch = -1
        self.repeat_factors = self._compute_repeat_factors(samples, task, threshold)

    @staticmethod
    def _compute_repeat_factors(
        samples: Sequence[Sample], task: str, threshold: float
    ) -> torch.Tensor:
        id_map = TASK_ID_MAP[task]
        n_images = len(samples)
        image_classes: list[set[int]] = []
        class_image_counts: Counter = Counter()
        for s in samples:
            classes: set[int] = set()
            if not getattr(s, "negative", False):
                for cls, *_ in read_yolo_label(s.label):
                    classes.add(id_map[cls])
            image_classes.append(classes)
            for c in classes:
                class_image_counts[c] += 1

        class_repeat = {
            c: max(1.0, math.sqrt(threshold / (count / n_images)))
            for c, count in class_image_counts.items()
        }
        factors = [
            max((class_repeat[c] for c in classes), default=1.0) for classes in image_classes
        ]
        return torch.tensor(factors, dtype=torch.float64)


class SizeRepeatFactorSampler(_StochasticRepeatSampler):
    """RepeatFactorSampler's mechanism, bucketed by lesion size instead of class.

    Targets the small-lesion AP gap directly (FINDINGS.md: AP 0.471 -> 0.663 ->
    0.755 by size tertile) rather than the class imbalance ``RepeatFactorSampler``
    addresses -- the two are mutually exclusive (see ``build_dataloaders``), since
    combining two independent image-level oversampling mechanisms in one run
    would confound whichever effect is being measured.

    Images are bucketed into size tertiles by ``box_area_frac``, then the same
    ``r(bucket) = max(1, sqrt(threshold / f(bucket)))`` formula applies, with
    the smallest-tertile bucket in practice always the one below threshold.
    """

    def __init__(self, samples: Sequence[Sample], threshold: float, seed: int = 0) -> None:
        self.seed = seed
        self._epoch = -1
        self.repeat_factors = self._compute_repeat_factors(samples, threshold)

    @staticmethod
    def _compute_repeat_factors(samples: Sequence[Sample], threshold: float) -> torch.Tensor:
        areas = np.array([box_area_frac(s) for s in samples])
        edges = np.quantile(areas, [0.0, 1 / 3, 2 / 3, 1.0])
        edges[-1] += 1e-9  # include the max-area image in the top bucket
        bucket = np.clip(np.digitize(areas, edges[1:-1]), 0, 2)

        n_images = len(samples)
        bucket_freq = {b: float((bucket == b).sum()) / n_images for b in range(3)}
        bucket_repeat = {
            b: max(1.0, math.sqrt(threshold / freq)) for b, freq in bucket_freq.items()
        }
        factors = [bucket_repeat[int(b)] for b in bucket]
        return torch.tensor(factors, dtype=torch.float64)


class CenterRepeatFactorSampler(_StochasticRepeatSampler):
    """RepeatFactorSampler's mechanism, bucketed by acquisition center instead
    of class or size. Center-1 supplies 80-94% of any multi-center training
    pool (worse still in a leave-one-center-out pool, where holding out a
    minor center leaves the majority center even more dominant) -- this
    directly oversamples the minority centers' images.

    ``Sample.center`` is already categorical, so no quantile binning is
    needed: the bucket is the center string itself, same ``r(bucket) =
    max(1, sqrt(threshold / f(bucket)))`` formula as the other two samplers.
    """

    def __init__(self, samples: Sequence[Sample], threshold: float, seed: int = 0) -> None:
        self.seed = seed
        self._epoch = -1
        self.repeat_factors = self._compute_repeat_factors(samples, threshold)

    @staticmethod
    def _compute_repeat_factors(samples: Sequence[Sample], threshold: float) -> torch.Tensor:
        n_images = len(samples)
        center_counts = Counter(s.center for s in samples)
        center_repeat = {
            c: max(1.0, math.sqrt(threshold / (count / n_images)))
            for c, count in center_counts.items()
        }
        factors = [center_repeat[s.center] for s in samples]
        return torch.tensor(factors, dtype=torch.float64)


def build_dataloaders(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    task: str,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
    input_mode: str = "reference",
    seed: int = 0,
    augment_ops: Sequence[str] | None = None,
    rfs_threshold: float = 0.0,
    size_rfs_threshold: float = 0.0,
    center_rfs_threshold: float = 0.0,
    group_dro: bool = False,
    lesion_bank: object | None = None,
) -> tuple[DataLoader, DataLoader]:
    """``lesion_bank``, if given, is used for ``copy_paste`` instead of building
    a fresh real-crop ``LesionBank`` -- e.g. a
    ``cadsel.lesion_gan.GeneratedLesionBank`` for synthetic-lesion copy-paste
    (see ``cadsel/train.py``'s ``--copy-paste-source generated``). Anything
    exposing ``.sample(rng) -> (patch, label)`` works, since that's the only
    interface ``_copy_paste`` calls.

    ``rfs_threshold``, ``size_rfs_threshold`` and ``center_rfs_threshold`` are
    mutually exclusive: all three are image-level oversampling, bucketed by
    class / lesion size / acquisition center respectively, and layering them
    would confound whichever effect a run is meant to measure. ``group_dro``
    (``--loss-fn group_dro``) is also mutually exclusive with all three: it
    needs ``CenterBatchSampler``'s center-homogeneous batches, which a
    per-image oversampling ``sampler`` can't be combined with in one
    ``DataLoader``.
    """
    n_rfs_active = sum(t > 0 for t in (rfs_threshold, size_rfs_threshold, center_rfs_threshold))
    if n_rfs_active > 1:
        raise ValueError(
            "rfs_threshold, size_rfs_threshold and center_rfs_threshold are "
            "mutually exclusive -- pick one oversampling axis per run"
        )
    if group_dro and n_rfs_active > 0:
        raise ValueError(
            "group_dro can't be combined with rfs_threshold/size_rfs_threshold/"
            "center_rfs_threshold -- CenterBatchSampler's batch_sampler and a "
            "per-image oversampling sampler can't both drive one DataLoader"
        )
    ops = set(augment_ops if augment_ops is not None else DEFAULT_AUGMENT)
    bank = lesion_bank
    if "copy_paste" in ops and bank is None:
        print("  building lesion bank for copy-paste ...")
        bank = LesionBank().build(train_samples, task)

    train_ds = CadSelDetectionDataset(
        train_samples, task, img_size, train=True, input_mode=input_mode, seed=seed,
        augment_ops=ops, lesion_bank=bank,
    )
    val_ds = CadSelDetectionDataset(
        val_samples, task, img_size, train=False, input_mode=input_mode, seed=seed
    )
    # Windows/macOS spawn workers rather than forking, so every worker re-imports
    # torch and costs ~600 MB RSS. Giving the validation loader the same worker
    # count as training doubles that for a pass over a much smaller split, so it
    # gets a reduced, non-persistent pool.
    val_workers = min(2, num_workers)

    sampler = (
        RepeatFactorSampler(train_samples, task, rfs_threshold, seed=seed)
        if rfs_threshold > 0
        else SizeRepeatFactorSampler(train_samples, size_rfs_threshold, seed=seed)
        if size_rfs_threshold > 0
        else CenterRepeatFactorSampler(train_samples, center_rfs_threshold, seed=seed)
        if center_rfs_threshold > 0
        else None
    )
    if group_dro:
        train_loader = DataLoader(
            train_ds,
            batch_sampler=CenterBatchSampler(train_samples, batch_size, shuffle=True, seed=seed),
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=val_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return train_loader, val_loader


def build_joint_dataloaders(
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
    task: str,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
    input_mode: str = "reference",
    seed: int = 0,
    augment_ops: Sequence[str] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Dataloaders over the WLE+EUS union, with modality-homogeneous batches."""
    ops = set(augment_ops if augment_ops is not None else DEFAULT_AUGMENT)
    if "copy_paste" in ops:
        # A shared lesion bank would paste colour WLE lesions into grayscale EUS
        # frames. Experiment 7 measured exactly that class of augmentation making
        # EUS shortcut reliance *worse* (+23 to +37%), so refuse rather than
        # silently produce a configuration the audit already ruled out.
        raise ValueError(
            "copy_paste is not supported for joint runs: pasting across the "
            "WLE/EUS domain gap degraded EUS in experiments/07. Drop it from "
            "--augment, or train per-modality."
        )

    train_ds = CadSelDetectionDataset(
        train_samples, task, img_size, train=True, input_mode=input_mode, seed=seed,
        augment_ops=ops,
    )
    val_ds = CadSelDetectionDataset(
        val_samples, task, img_size, train=False, input_mode=input_mode, seed=seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=ModalityBatchSampler(
            train_samples, batch_size, shuffle=True, seed=seed
        ),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=ModalityBatchSampler(
            val_samples, batch_size, shuffle=False
        ),
        collate_fn=collate_fn,
        num_workers=min(2, num_workers),   # see build_dataloaders
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return train_loader, val_loader
