"""Patient-level two-stream fusion of WLE and EUS.

Motivation
----------
CAD-SEL is a dual-modal dataset, but the paper's baselines — and every unimodal
run in this workspace — treat the modalities as two separate problems. All 421 EUS
patients also have WLE frames, so the two views of a lesion can be combined. What
they cannot be is *paired*: there is no frame-to-frame correspondence (median 4 WLE
and 4 EUS frames per patient), so fusion happens at the level of the patient, over
two unordered sets of frames.

That makes this a different task from Table 4/5, which score per image. This module
reports patient-level balanced accuracy and macro-F1 and must not be compared
against an AP number.

Design notes
------------
*Lesion ROI features, not global pooling.* Experiment 3 found the label is 68-95%
recoverable from lesion-*free* crops, so a globally pooled frame descriptor mostly
encodes context. Taking the ROI features of the detector's highest-scoring box
forces the descriptor onto the lesion itself.

*Attention pooling over frames.* Frames of one patient vary in quality; a mean
would let a blurred or off-target frame drag the patient's descriptor. The
attention weights are learned per modality.

*Modality dropout.* EUS was performed on only 421 of 641 patients, so the model
must work when EUS is absent — both to use the 220 WLE-only patients as training
signal and because WLE-only is the common case in the clinic. Dropping a stream at
random during training is what makes the missing-modality path work rather than
merely not crash.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MODALITY_NAMES, TASK_CLASSES
from .data import Sample, make_normalizer
from .models import set_active_modality

# Patient-level label spaces. The 5-way subtype task keeps NET_G1/NET_G2 separate
# (they are separate directories); the detection label space collapses them, so
# this is strictly the harder target.
SUBTYPES = ["NET_G1", "NET_G2", "Leiomyoma", "Lipoma", "NonNeoplasm"]
BINARY = ["NET", "non-NET"]


def patient_classes(task: str) -> list[str]:
    return BINARY if task == "binary" else SUBTYPES


def patient_label(subtype: str, task: str) -> str:
    """Label a patient from its subtype directory.

    Mirrors ``experiments/harness.label_of`` so the fusion numbers sit on the same
    label convention as the shortcut diagnostics.
    """
    if task == "binary":
        return "NET" if subtype.startswith("NET") else "non-NET"
    return subtype


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@dataclass
class PatientFeatures:
    """Per-patient frame descriptors, grouped by modality."""

    patient_key: str
    subtype: str
    center: str
    # modality -> (n_frames, feature_dim)
    frames: dict[str, np.ndarray]

    def has(self, modality: str) -> bool:
        return modality in self.frames and len(self.frames[modality]) > 0


@torch.no_grad()
def extract_features(
    model,
    samples: Sequence[Sample],
    device: torch.device,
    input_mode: str = "reference",
    per_modality_norm: bool = True,
    img_size: int = 640,
    batch_size: int = 8,
    progress: bool = True,
) -> dict[str, PatientFeatures]:
    """Run the frozen detector over every frame and keep one descriptor each.

    The descriptor is the ROI ``box_head`` output of the highest-scoring detection,
    concatenated with that detection's class posterior. The posterior is included
    because the detector's own opinion is a strong, cheap signal, and the head can
    learn to override it when the two modalities disagree.

    Frames where the detector proposes nothing are represented by a zero vector, so
    "the detector found no lesion here" stays in the record rather than silently
    shrinking the frame set.

    ``model`` is either one shared detector (the Stage 1 joint/FiLM-conditioned
    model this was originally written for -- ``set_active_modality`` selects its
    conditioning per batch) or a ``{"WLE": model, "EUS": model}`` dict of
    *independent* per-modality detectors (e.g. one with an ultrasound-pretrained
    backbone via ``--backbone-weights``, the other not). In the dict case each
    batch is routed to its own modality's model instead, and
    ``set_active_modality`` is skipped since independent models have no shared
    conditioning to select. Both share one ``img_size``/``input_mode`` here
    (the dataset/transform are still built once for every frame) -- train the
    two detectors with matching values if using the dict form.
    """
    from torch.utils.data import DataLoader

    from .data import CadSelDetectionDataset, ModalityBatchSampler, collate_fn

    separate_models = isinstance(model, dict)
    models = model if separate_models else {m: model for m in MODALITY_NAMES}
    for m in models.values():
        m.eval().to(device)
    normalize = make_normalizer(input_mode, per_modality=per_modality_norm)

    dataset = CadSelDetectionDataset(
        samples, task="multiclass", img_size=img_size, train=False,
        input_mode=input_mode,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=ModalityBatchSampler(samples, batch_size, shuffle=False),
        collate_fn=collate_fn, num_workers=0,
    )

    per_frame: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_classes = len(TASK_CLASSES["multiclass"]) + 1
    n_done = 0

    try:
        for images, targets in loader:
            modality = MODALITY_NAMES[int(targets[0]["modality"].item())]
            active_model = models[modality]
            if not separate_models:
                set_active_modality(modality)
            images_dev = normalize(
                [img.to(device, non_blocking=True) for img in images], modality
            )
            outputs = active_model(images_dev)

            for i, (output, target) in enumerate(zip(outputs, targets)):
                sample = samples[int(target["index"].item())]
                scores = output["scores"]

                posterior = np.zeros(n_classes, dtype=np.float32)
                if len(scores):
                    best = int(torch.argmax(scores))
                    posterior[int(output["labels"][best])] = float(scores[best])

                roi = _roi_descriptor(active_model, images_dev[i], output, device)
                per_frame[sample.patient_key][modality].append(
                    np.concatenate([roi, posterior])
                )

            n_done += len(images)
            if progress and n_done % 400 < len(images):
                print(f"    extracted {n_done}/{len(samples)}", flush=True)
    finally:
        if not separate_models:
            set_active_modality(None)

    by_patient: dict[str, PatientFeatures] = {}
    meta = {s.patient_key: s for s in samples}
    for key, frames in per_frame.items():
        by_patient[key] = PatientFeatures(
            patient_key=key,
            subtype=meta[key].subtype,
            center=meta[key].center,
            frames={m: np.stack(v).astype(np.float32) for m, v in frames.items()},
        )
    return by_patient


@torch.no_grad()
def _roi_descriptor(model, image: torch.Tensor, output: dict, device) -> np.ndarray:
    """ROI features of the top-scoring detection for one image.

    Re-runs the backbone and box head on a single image and a single proposal. That
    is more compute than reusing the batch's pooled features, but it is
    unambiguous — the batch tensor mixes every image's proposals together with no
    index back to the source image.
    """
    if len(output["scores"]) == 0:
        dim = model.roi_heads.box_predictor.cls_score.in_features
        return np.zeros(dim, dtype=np.float32)

    best = int(torch.argmax(output["scores"]))
    box = output["boxes"][best:best + 1].to(device)

    features = model.backbone(image.unsqueeze(0))
    pooled = model.roi_heads.box_roi_pool(
        features, [box], [tuple(image.shape[-2:])]
    )
    descriptor = model.roi_heads.box_head(pooled)
    return descriptor.squeeze(0).float().cpu().numpy()


def save_features(path: Path, features: dict[str, PatientFeatures]) -> None:
    """Cache to a single npz. Extraction is the expensive part; the head trains in
    seconds, so every ablation reuses one extraction pass."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    meta: list[str] = []
    for key, pf in sorted(features.items()):
        for modality, arr in pf.frames.items():
            payload[f"{key}|{modality}"] = arr
        meta.append(f"{key}\t{pf.subtype}\t{pf.center}")
    payload["__meta__"] = np.array(meta)
    np.savez_compressed(path, **payload)


def load_features(path: Path) -> dict[str, PatientFeatures]:
    blob = np.load(path, allow_pickle=False)
    meta = {}
    for line in blob["__meta__"]:
        key, subtype, center = str(line).split("\t")
        meta[key] = (subtype, center)

    frames: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for name in blob.files:
        if name == "__meta__":
            continue
        key, modality = name.rsplit("|", 1)
        frames[key][modality] = blob[name]

    return {
        key: PatientFeatures(key, meta[key][0], meta[key][1], dict(mods))
        for key, mods in frames.items()
    }


# ---------------------------------------------------------------------------
# The fusion head
# ---------------------------------------------------------------------------
class AttentionPool(nn.Module):
    """Attention-weighted mean over a patient's frames for one modality."""

    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``frames`` (B, T, D), ``mask`` (B, T) with 1 for real frames."""
        logits = self.score(frames).squeeze(-1)              # (B, T)
        logits = logits.masked_fill(mask < 0.5, float("-inf"))
        # A patient with no frames in this modality has an all-masked row; softmax
        # would produce NaN, so give those a uniform (later zeroed) distribution.
        empty = mask.sum(dim=1, keepdim=True) < 0.5
        logits = torch.where(empty.expand_as(logits), torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (frames * weights).sum(dim=1)
        return pooled * (~empty).float()


class GatedFusion(nn.Module):
    """Gated combination of two modality descriptors.

    The gate is computed from both streams and from the availability flags, so the
    model can learn to fall back to WLE when EUS is missing rather than averaging
    against a zero vector.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim + 2, dim), nn.ReLU(), nn.Linear(dim, dim), nn.Sigmoid()
        )
        self.project = nn.Linear(2 * dim, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                avail: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([a, b], dim=1)
        gate = self.gate(torch.cat([joined, avail], dim=1))
        return gate * self.project(joined) + (1 - gate) * a


class SymmetricCrossAttentionFusion(nn.Module):
    """Modality-symmetric alternative to GatedFusion: treats the two pooled
    modality vectors as a 2-token sequence mixed by one shared self-attention
    layer, with a learned 'modality absent' embedding replacing a zero vector
    for missing modalities, so neither modality is a hardcoded fallback
    (contrast GatedFusion's (1-gate)*a, which always falls back to WLE
    specifically since MODALITY_NAMES = ["WLE", "EUS"]).

    Readout is a mean over the two post-attention tokens rather than a
    concatenation in a fixed [a, b] order: mean-pooling makes the module's
    output invariant to which physical modality is passed as ``a`` vs ``b``,
    so swapping WLE and EUS at the call site cannot change which stream is
    structurally favoured -- there simply isn't one.
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        # Stands in for a missing stream's zero vector so "absent" has its own
        # learned identity that attention can key off, rather than being
        # indistinguishable from "a real but near-zero descriptor".
        self.absent_embed = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.absent_embed, std=0.02)
        self.attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                avail: torch.Tensor) -> torch.Tensor:
        """``a``/``b`` (B, dim) pooled per-modality descriptors (already zeroed
        by ``FusionHead`` when absent); ``avail`` (B, 2), 1 where present, in
        the same order as ``MODALITY_NAMES``.

        The attention/layernorm block runs forced fp32 regardless of any
        surrounding ``torch.amp.autocast`` context: ``nn.MultiheadAttention``'s
        internal softmax can overflow to inf/nan under fp16 once training has
        moved the weights away from their init distribution (observed
        empirically training ``cadsel.joint_fusion_model.JointTwoTowerModel``
        with ``--fusion symmetric`` -- val_loss went NaN from epoch 2 onward).
        ``cadsel.train_fusion``'s own training loop never uses autocast at
        all, so this is a no-op there -- purely a robustness fix for a caller
        that does.
        """
        avail_a, avail_b = avail[:, 0:1], avail[:, 1:2]
        tok_a = a * avail_a + self.absent_embed * (1 - avail_a)
        tok_b = b * avail_b + self.absent_embed * (1 - avail_b)
        tokens = torch.stack([tok_a, tok_b], dim=1)              # (B, 2, dim)

        with torch.autocast(device_type=tokens.device.type, enabled=False):
            tokens = tokens.float()
            attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            tokens = self.norm1(tokens + attended)
            tokens = self.norm2(tokens + self.ff(tokens))
            out = self.out_proj(tokens.mean(dim=1))
        return out


class FusionHead(nn.Module):
    """Two-stream patient-level classifier.

    ``modality_dropout`` randomly zeroes a stream during training. WLE is never
    dropped when EUS is already absent, so a patient always keeps at least one
    stream and the loss stays defined.
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden: int = 256,
        modality_dropout: float = 0.3,
        dropout: float = 0.3,
        fusion: Literal["gated", "symmetric"] = "gated",
    ) -> None:
        super().__init__()
        self.modality_dropout = modality_dropout
        self.encoders = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
            for m in MODALITY_NAMES
        })
        self.pools = nn.ModuleDict({m: AttentionPool(in_dim) for m in MODALITY_NAMES})
        if fusion == "gated":
            self.fuse = GatedFusion(hidden)
        elif fusion == "symmetric":
            self.fuse = SymmetricCrossAttentionFusion(hidden)
        else:
            raise ValueError(f"unknown fusion type {fusion!r}; expected 'gated' or 'symmetric'")
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        pooled, avail = {}, {}
        for modality in MODALITY_NAMES:
            frames, mask = batch[modality], batch[f"{modality}_mask"]
            present = (mask.sum(dim=1) > 0.5).float()

            if self.training and self.modality_dropout > 0:
                drop = torch.rand_like(present) < self.modality_dropout
                # Never drop the last remaining stream.
                other = MODALITY_NAMES[1] if modality == MODALITY_NAMES[0] else MODALITY_NAMES[0]
                other_present = batch[f"{other}_mask"].sum(dim=1) > 0.5
                drop = drop & other_present
                present = present * (~drop).float()
                mask = mask * present.unsqueeze(1)

            pooled[modality] = self.encoders[modality](self.pools[modality](frames, mask))
            pooled[modality] = pooled[modality] * present.unsqueeze(1)
            avail[modality] = present

        a, b = (pooled[m] for m in MODALITY_NAMES)
        flags = torch.stack([avail[m] for m in MODALITY_NAMES], dim=1)
        return self.classifier(self.fuse(a, b, flags))


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------
def make_batch(
    patients: Sequence[PatientFeatures],
    task: str,
    device: torch.device,
    max_frames: int = 12,
    use_modalities: Sequence[str] = MODALITY_NAMES,
    rng: np.random.Generator | None = None,
) -> tuple[dict, torch.Tensor]:
    """Pad each patient's frame sets to ``max_frames`` and build the masks.

    ``use_modalities`` drives the unimodal ablations: restricting it to one
    modality gives exactly the WLE-only / EUS-only comparison, through the same
    code path as the fused model.
    """
    classes = patient_classes(task)
    index = {c: i for i, c in enumerate(classes)}
    dim = next(
        arr.shape[1] for p in patients for arr in p.frames.values() if len(arr)
    )

    batch: dict[str, torch.Tensor] = {}
    for modality in MODALITY_NAMES:
        frames = np.zeros((len(patients), max_frames, dim), dtype=np.float32)
        mask = np.zeros((len(patients), max_frames), dtype=np.float32)
        if modality in use_modalities:
            for i, patient in enumerate(patients):
                arr = patient.frames.get(modality)
                if arr is None or not len(arr):
                    continue
                if len(arr) > max_frames:
                    pick = (rng.choice(len(arr), max_frames, replace=False)
                            if rng is not None else np.arange(max_frames))
                    arr = arr[np.sort(pick)]
                frames[i, :len(arr)] = arr
                mask[i, :len(arr)] = 1.0
        batch[modality] = torch.from_numpy(frames).to(device)
        batch[f"{modality}_mask"] = torch.from_numpy(mask).to(device)

    labels = torch.tensor(
        [index[patient_label(p.subtype, task)] for p in patients],
        dtype=torch.long, device=device,
    )
    return batch, labels
