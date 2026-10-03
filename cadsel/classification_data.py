"""Image-level classification Dataset + oversampler for Track A (pure
classification accuracy campaign), reusing ``cadsel.data``'s manifest/split/
sampler infrastructure rather than duplicating it.

Distinct from ``cadsel.data.CadSelDetectionDataset``: no bounding boxes here,
so augmentation doesn't need to be box-safe -- standard image classification
transforms (``RandomResizedCrop``/``RandomHorizontalFlip``/``ColorJitter``/
``RandomRotation``) are used instead of ``cadsel.data``'s hand-rolled
box-aware ops. Label vocabulary (binary NET/non-NET vs. 5-way subtype) stays
in ``experiments/harness.py`` (``classes_for``/``label_of``/``encode_labels``)
-- this module is label-scheme-agnostic, accepting a pre-encoded ``labels``
array so it doesn't need to import "up" into ``experiments/``.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .config import CADSEL_MEAN, CADSEL_STD
from .data import Sample, _StochasticRepeatSampler, SizeRepeatFactorSampler


class CadSelClassificationDataset(Dataset):
    """Returns ``(uint8 CHW tensor, int label)``.

    Images are handed over as uint8 (not normalised float) for the same
    reason ``CadSelDetectionDataset`` does: normalisation happens on the GPU
    via ``make_classification_normalizer`` so worker IPC never has to carry a
    float32 tensor through shared memory.

    ``bg_swap_arrays``, if given, is the ``{"full", "bg_patch", "tile",
    "box"}`` dict from ``experiments.ablation_inputs.build_full_bgpatch_tile_box``
    over this same ``samples`` list (index-aligned) -- passing it turns on the
    background-swap mitigation (``--bg-swap-aug``), composed per-sample before
    the standard transform pipeline runs on top.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        labels: np.ndarray,
        img_size: int = 224,
        train: bool = False,
        augment: bool = True,
        mean: Sequence[float] = CADSEL_MEAN,
        std: Sequence[float] = CADSEL_STD,
        seed: int = 0,
        bg_swap_arrays: dict | None = None,
        bg_swap_prob: float = 0.5,
    ) -> None:
        self.samples = list(samples)
        self.labels = np.asarray(labels)
        if len(self.samples) != len(self.labels):
            raise ValueError(
                f"{len(self.samples)} samples but {len(self.labels)} labels"
            )
        self.img_size = img_size
        self.train = train
        self.augment = augment and train
        self.seed = seed
        self.mean = list(mean)
        self.std = list(std)
        self.bg_swap_arrays = bg_swap_arrays
        self.bg_swap_prob = bg_swap_prob if bg_swap_arrays is not None else 0.0

        if self.augment:
            self.transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomRotation(10),
            ])
        else:
            self.transform = T.Resize((img_size, img_size))

    def __len__(self) -> int:
        return len(self.samples)

    def _bg_swap_compose(self, index: int, rng: random.Random) -> np.ndarray:
        """Paste this sample's lesion tile onto a random other sample's
        lesion-free background. Ported from
        ``experiments/src/07_mitigation.py::compose_batch(..., bg_swap=True)``
        (that function's ``scale_jitter`` branch has no equivalent CLI flag
        here and is dropped; this is the ``bg_swap``-only path, adapted from
        batch-level to per-sample since this Dataset is DataLoader-driven
        rather than 07's own in-RAM batch loop).
        """
        D = self.bg_swap_arrays
        img_size = D["full"].shape[1]
        x1, y1, x2, y2 = D["box"][index]
        bw, bh = max(int(x2 - x1), 8), max(int(y2 - y1), 8)
        nw = int(np.clip(bw, 12, img_size))
        nh = int(np.clip(bh, 12, img_size))

        j = rng.randrange(len(D["full"]))
        canvas = D["bg_patch"][j].copy()
        patch = np.asarray(
            Image.fromarray(D["tile"][index]).resize((nw, nh), Image.BILINEAR),
            dtype=np.uint8,
        )
        px = rng.randint(0, max(img_size - nw, 0))
        py = rng.randint(0, max(img_size - nh, 0))
        canvas[py:py + nh, px:px + nw] = patch
        return canvas

    def __getitem__(self, index: int):
        rng = random.Random((self.seed * 1_000_003) ^ index)

        if self.bg_swap_arrays is not None and self.augment and rng.random() < self.bg_swap_prob:
            im = Image.fromarray(self._bg_swap_compose(index, rng))
        else:
            with Image.open(self.samples[index].image) as im0:
                im = im0.convert("RGB")

        im = self.transform(im)
        arr = np.asarray(im, dtype=np.uint8)
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return tensor, int(self.labels[index])


def make_classification_normalizer(mean: Sequence[float], std: Sequence[float]):
    """GPU-side uint8 -> normalised-float, mirroring ``cadsel.data.make_normalizer``.

    No per-modality branch here (unlike the detection version): the caller
    picks the right stats up front (``CADSEL_MEAN``/``STD``, ``MODALITY_STATS``,
    or ImageNet stats for a DINOv3 backbone) since a classifier run is always
    single-modality, single-backbone.
    """
    mean_t = torch.tensor(list(mean)).view(1, 3, 1, 1)
    std_t = torch.tensor(list(std)).view(1, 3, 1, 1)

    def normalize(batch: torch.Tensor) -> torch.Tensor:
        x = batch.float().div_(255.0)
        return (x - mean_t.to(x.device)) / std_t.to(x.device)

    return normalize


class ClassificationSizeRepeatFactorSampler(_StochasticRepeatSampler):
    """``SizeRepeatFactorSampler``'s mechanism (size-tertile bucketing via
    ``box_area_frac``, ``r(bucket) = max(1, sqrt(threshold / f(bucket)))``)
    for a plain classification ``Sample`` list -- reuses that class's own
    ``_compute_repeat_factors`` directly rather than re-deriving the same
    math, since the formula doesn't depend on anything detection-specific.
    """

    def __init__(self, samples: Sequence[Sample], threshold: float, seed: int = 0) -> None:
        self.seed = seed
        self._epoch = -1
        self.repeat_factors = SizeRepeatFactorSampler._compute_repeat_factors(samples, threshold)
