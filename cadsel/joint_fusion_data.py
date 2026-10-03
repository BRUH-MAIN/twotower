"""Raw-image Dataset for the joint two-tower WLE+EUS model
(``cadsel.joint_fusion_model``).

One item = one patient: up to ``max_frames`` raw frames per modality, each put
through its own modality's augmentation pipeline independently -- distinct
from ``cadsel.fusion.extract_features`` / ``cadsel.classifier_fusion_features``,
both of which extract each frame's descriptor exactly once, frozen, before any
fusion training happens, so no augmentation ever touches the fusion stage
there. Reuses ``CadSelClassificationDataset``'s own transform recipe
(``RandomResizedCrop``/``HFlip``/``ColorJitter``/``RandomRotation``),
instantiated independently per modality so a WLE frame's augmentation never
mirrors its patient's EUS draw -- the two modalities aren't pixel-registered
to begin with (``cadsel/fusion.py``'s module docstring: "there is no
frame-to-frame correspondence"), so there is no spatial relationship to
preserve by coupling them.

Frame *selection* (which of a patient's raw frames get used) is deterministic
per patient index, mirroring ``CadSelClassificationDataset.__getitem__``'s own
``random.Random((seed * 1_000_003) ^ index)`` pattern -- what varies every
epoch is each selected frame's actual augmentation, drawn from torchvision's
global RNG inside ``self.transform`` exactly as it already does for the
single-modality classifier. A modality with fewer than ``max_frames`` raw
frames is sampled *with replacement* so every present modality still fills the
full ``max_frames`` slots -- zero-padding is reserved for a genuinely absent
modality (the 220 WLE-only patients' EUS side), not for ordinary frame
scarcity, so EfficientNet-B0's BatchNorm layers see all-real, non-degenerate
batches for every present-modality frame.

Two optional, mutually-exclusive per-frame content transforms (applied
before the standard augmentation pipeline): ``lesion_crop`` (always crop to
the lesion box + context padding) and ``bg_remove_prob`` (stochastically,
train-time only, erase everything outside the lesion box instead).
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .config import MODALITY_NAMES
from .data import Sample, read_yolo_label
from .fusion import patient_classes, patient_label


def _group_by_patient_modality(samples: Sequence[Sample]) -> dict[str, dict[str, list[Sample]]]:
    by_patient: dict[str, dict[str, list[Sample]]] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        by_patient[s.patient_key][s.modality].append(s)
    return by_patient


def _boxes_for(sample: Sample, W: int, H: int) -> list[tuple[int, int, int, int]]:
    """Pixel-space xyxy boxes for one sample, dropping degenerate ones.

    Byte-identical logic to ``experiments/ablation_inputs.py::boxes_for``,
    duplicated rather than imported: that module lives under ``experiments/``,
    and ``cadsel/`` doesn't otherwise depend on it -- this is the same ~10
    lines built on ``cadsel.data.read_yolo_label``, which both already share.
    """
    out = []
    for _c, cx, cy, bw, bh in read_yolo_label(sample.label):
        x1 = int(max(0, (cx - bw / 2) * W))
        x2 = int(min(W, (cx + bw / 2) * W))
        y1 = int(max(0, (cy - bh / 2) * H))
        y2 = int(min(H, (cy + bh / 2) * H))
        if x2 - x1 > 4 and y2 - y1 > 4:
            out.append((x1, y1, x2, y2))
    return out


def _lesion_crop(im: "Image.Image", sample: Sample, pad_frac: float = 0.1) -> "Image.Image":
    """Crop to the sample's largest lesion box plus ``pad_frac`` context
    padding on each side -- same box-selection and padding math as
    ``experiments/ablation_inputs.py::build_five_variants``'s ``lesion``
    variant, so a lesion-crop-trained tower sees the same kind of input that
    variant already probes at counterfactual-gate time. Falls back to the
    full frame when a sample has no box annotation (matches that function's
    own fallback), rather than cropping to nothing.
    """
    W, H = im.size
    boxes = _boxes_for(sample, W, H)
    if not boxes:
        return im
    bx1, by1, bx2, by2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    px, py = int(pad_frac * (bx2 - bx1)), int(pad_frac * (by2 - by1))
    return im.crop((max(0, bx1 - px), max(0, by1 - py), min(W, bx2 + px), min(H, by2 + py)))


def _bg_remove(im: "Image.Image", sample: Sample, pad_frac: float = 0.3) -> "Image.Image":
    """Fill everything OUTSIDE the sample's lesion box(es) -- expanded by
    ``pad_frac`` context padding on each side -- with the image's own mean
    colour, keeping only the lesion region(s) (plus padding) as real tissue.
    The inverse of ``ablation_inputs.build_five_variants``'s ``bg_mask``
    (which erases the lesion and keeps background); this erases the
    background and keeps the lesion. User-proposed augmentation, motivated
    directly by Experiment 3's finding that labels are 68-95% recoverable
    from lesion-*free* crops alone -- training on some fraction of
    background-removed frames forces the model to succeed without that
    signal for those samples, rather than only measuring reliance on it
    post-hoc the way the counterfactual gate does.

    Filled with mean colour rather than pure black for the same reason
    ``bg_mask`` makes that choice: a stark black rectangle with a hard edge
    is itself a strong, artificial cue the model could key on as a new
    shortcut, rather than genuine lesion texture. Keeps ALL annotated boxes
    (not just the largest), unlike ``_lesion_crop``'s single-box framing.
    Falls back to the untouched frame when there's no box annotation.

    ``pad_frac`` (default 0.3, larger than ``_lesion_crop``'s 0.1) exists to
    fix a real failure mode found in this campaign's first attempt at this
    augmentation (FINDINGS.md §15.9): the standard pipeline still runs
    ``RandomResizedCrop(scale=(0.7,1.0))`` *after* this transform, and at the
    box's exact boundary a large-scale crop could occasionally land almost
    entirely inside the erased region for a small lesion box, producing a
    near-featureless training example with a real label attached. A wider
    kept island makes that far less likely without changing the augmentation's
    intent (still erasing most of the background, just not razor-close to
    the box edge).
    """
    W, H = im.size
    boxes = _boxes_for(sample, W, H)
    if not boxes:
        return im
    arr = np.asarray(im, dtype=np.uint8)
    fill = arr.reshape(-1, 3).mean(0).astype(np.uint8)
    out = np.full_like(arr, fill)
    for (x1, y1, x2, y2) in boxes:
        px, py = int(pad_frac * (x2 - x1)), int(pad_frac * (y2 - y1))
        px1, py1 = max(0, x1 - px), max(0, y1 - py)
        px2, py2 = min(W, x2 + px), min(H, y2 + py)
        out[py1:py2, px1:px2] = arr[py1:py2, px1:px2]
    return Image.fromarray(out)


class JointPatientDataset(Dataset):
    """Returns ``(images, masks, label)`` for one patient.

    ``images``: ``{modality: (max_frames, C, H, W) uint8}``.
    ``masks``: ``{modality: (max_frames,) float32}``, 1.0 for a real frame.
    ``label``: ``int``, the patient-level class index for ``task``.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        task: str,
        img_size: int = 224,
        max_frames: int = 4,
        train: bool = True,
        seed: int = 0,
        lesion_crop: bool = False,
        bg_remove_prob: float = 0.0,
    ) -> None:
        self.task = task
        self.classes = patient_classes(task)
        self.img_size = img_size
        self.max_frames = max_frames
        self.train = train
        self.seed = seed
        self.lesion_crop = lesion_crop
        self.bg_remove_prob = bg_remove_prob

        by_patient = _group_by_patient_modality(samples)
        self.patient_keys = sorted(by_patient)
        self.frames = by_patient  # patient_key -> modality -> [Sample, ...]
        meta = {s.patient_key: s for s in samples}
        self.subtype = {k: meta[k].subtype for k in self.patient_keys}

        if train:
            self.transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomRotation(10),
            ])
        else:
            self.transform = T.Resize((img_size, img_size))

    def __len__(self) -> int:
        return len(self.patient_keys)

    def _sample_modality(self, patient_key: str, modality: str, rng: random.Random):
        pool = self.frames[patient_key].get(modality, [])
        imgs = np.zeros((self.max_frames, self.img_size, self.img_size, 3), dtype=np.uint8)
        mask = np.zeros(self.max_frames, dtype=np.float32)
        if not pool:
            return imgs, mask
        chosen = [pool[rng.randrange(len(pool))] for _ in range(self.max_frames)]
        for i, s in enumerate(chosen):
            with Image.open(s.image) as im0:
                im = im0.convert("RGB")
            if self.lesion_crop:
                im = _lesion_crop(im, s)
            elif self.train and self.bg_remove_prob > 0 and rng.random() < self.bg_remove_prob:
                im = _bg_remove(im, s)
            im = self.transform(im)
            imgs[i] = np.asarray(im, dtype=np.uint8)
            mask[i] = 1.0
        return imgs, mask

    def __getitem__(self, index: int):
        patient_key = self.patient_keys[index]
        rng = random.Random((self.seed * 1_000_003) ^ index)
        images, masks = {}, {}
        for modality in MODALITY_NAMES:
            imgs, mask = self._sample_modality(patient_key, modality, rng)
            images[modality] = torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
            masks[modality] = torch.from_numpy(mask)
        label = self.classes.index(patient_label(self.subtype[patient_key], self.task))
        return images, masks, label


def joint_collate(batch):
    images = {m: torch.stack([b[0][m] for b in batch]) for m in MODALITY_NAMES}
    masks = {m: torch.stack([b[1][m] for b in batch]) for m in MODALITY_NAMES}
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return images, masks, labels
