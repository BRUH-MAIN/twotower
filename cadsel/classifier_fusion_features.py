"""Classifier-backbone feature extraction for ``cadsel.fusion`` — an
alternative front-end to ``cadsel.fusion.extract_features`` (which is built
entirely around a torchvision detector's ROI/box-head API:
``model.backbone``, ``model.roi_heads.box_roi_pool``, etc., and isn't a
drop-in fit for a plain classifier checkpoint). Produces the exact same
``PatientFeatures`` structure, so everything downstream — ``FusionHead``,
``make_batch``, ``cadsel.train_fusion``'s CV loop — is reused unchanged.

**Scope: DINOv3 classifier checkpoints only** (``experiments/
finetune_classifier.py --backbone dinov3_vits16``), not the CNN backbones.
DINOv3's CLS-token pooling (``last_hidden_state[:, 0]``) gives one uniform
API for "the pooled per-frame descriptor" across any freeze mode; the CNN
backbones (`resnet50`/`convnext_tiny`/`efficientnet_b0`) would each need
their own architecture-specific hook to reach the pre-classification-head
pooled feature (``model.avgpool`` for resnet50, `model.classifier[:-1]` for
convnext/efficientnet, etc.) -- deferred rather than built speculatively,
since DINOv3 `--freeze partial` is this project's confirmed-best classifier
(FINDINGS.md §12.2) and the natural first choice to feed into fusion.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from .classification_data import make_classification_normalizer
from .data import Sample
from .fusion import PatientFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))


def load_classifier(checkpoint: Path, n_classes: int, device: torch.device):
    """Rebuild a ``finetune_classifier.py`` DINOv3 checkpoint's model, frozen."""
    from finetune_classifier import Dinov3Classifier, DINOV3_MODEL_NAME

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if blob["backbone"] != "dinov3_vits16":
        raise ValueError(
            f"{checkpoint} is a {blob['backbone']!r} checkpoint -- "
            f"classifier_fusion_features only supports dinov3_vits16 (see module docstring)"
        )
    model = Dinov3Classifier(DINOV3_MODEL_NAME, n_classes, blob["freeze"])
    model.load_state_dict(blob["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)
    return model, {"img_size": blob["img_size"], "mean": blob["mean"], "std": blob["std"]}


def _decode_batch(chunk: Sequence[Sample], img_size: int) -> np.ndarray:
    arr = np.zeros((len(chunk), img_size, img_size, 3), dtype=np.uint8)
    for i, s in enumerate(chunk):
        with Image.open(s.image) as im:
            im = im.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            arr[i] = np.asarray(im, dtype=np.uint8)
    return arr


@torch.no_grad()
def extract_classifier_features(
    models: dict[str, torch.nn.Module],
    specs: dict[str, dict],
    samples: Sequence[Sample],
    n_classes: int,
    device: torch.device,
    batch_size: int = 32,
    progress: bool = True,
) -> dict[str, PatientFeatures]:
    """Per-frame descriptor = DINOv3 CLS-token feature concatenated with the
    classifier's own softmax posterior over ``n_classes`` -- mirrors
    ``cadsel.fusion.extract_features``'s "ROI features + class posterior"
    design (the model's own opinion is a cheap, strong signal the fusion
    head can learn to trust or override), just swapping the detector's
    ROI/box-head output for the classifier's pooled CLS-token representation.

    ``models``/``specs`` are ``{"WLE": ..., "EUS": ...}`` -- two independent
    single-modality classifier checkpoints (there is no shared/conditioned
    classifier analogous to the detector's joint/FiLM model), each with its
    own ``img_size``/``mean``/``std`` from ``load_classifier``.
    """
    by_modality: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_modality[s.modality].append(s)

    per_frame: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    total = len(samples)
    n_done = 0

    for modality, mod_samples in by_modality.items():
        if modality not in models:
            continue
        model = models[modality]
        normalize = make_classification_normalizer(specs[modality]["mean"], specs[modality]["std"])
        img_size = specs[modality]["img_size"]

        for k in range(0, len(mod_samples), batch_size):
            chunk = mod_samples[k:k + batch_size]
            arr = _decode_batch(chunk, img_size)
            x = torch.from_numpy(np.ascontiguousarray(arr.transpose(0, 3, 1, 2))).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                cls = model.backbone(pixel_values=normalize(x)).last_hidden_state[:, 0]
                logits = model.head(cls)
                posterior = torch.softmax(logits.float(), dim=1)
            descriptor = torch.cat([cls.float(), posterior], dim=1).cpu().numpy()

            for s, desc in zip(chunk, descriptor):
                per_frame[s.patient_key][modality].append(desc)

            n_done += len(chunk)
            if progress and n_done % 400 < len(chunk):
                print(f"    extracted {n_done}/{total}", flush=True)

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
