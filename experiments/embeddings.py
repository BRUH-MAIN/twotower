"""Frozen pretrained embeddings for SVM classification probes.

Two backbones:

- **ResNet-50** (torchvision, ImageNet-pretrained) -- always available, no
  gating, no extra setup.
- **DINOv3** (Meta, self-supervised ViT, via ``transformers``) -- the
  ``facebook/dinov3-*`` checkpoints on Hugging Face are **gated**: request
  access on the model page (e.g.
  https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), then
  authenticate locally (``huggingface-cli login``, or set the ``HF_TOKEN``
  environment variable) before this will actually download weights --
  otherwise ``from_pretrained`` raises a 401. Not something this session can
  do on your behalf: it needs your own HF account and manual approval.

Both extractors are frozen (no fine-tuning, no gradients) and cache their
output to disk (``load_or_extract``) since extraction is the expensive part
and a classifier's hyperparameters should be free to iterate on top of it
without re-running the backbone.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from cadsel.data import Sample
from cadsel.config import PROJECT_ROOT

RESNET_IMG = 224
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_resized(path: str, size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def _resize_array(img: np.ndarray, size: int) -> np.ndarray:
    """uint8 HWC array -> float32 [0, 1] HWC array at ``size``x``size``.

    A no-op resize when the array is already the right shape (the common case:
    ``ablation_inputs.build_five_variants`` already produces ``img_size=224``
    arrays, matching ``RESNET_IMG``), so this costs nothing extra on the
    expected path and only falls back to a PIL resize otherwise.
    """
    if img.shape[0] != size or img.shape[1] != size:
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.uint8)
    return img.astype(np.float32) / 255.0


def _build_resnet_model(device: torch.device):
    """The frozen ResNet-50 backbone + ImageNet normalisation stats, shared by
    the path-based and array-based extractors below."""
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return model, mean, std


def extract_resnet_embeddings(
    samples: Sequence[Sample], batch_size: int = 32, device: torch.device | None = None,
) -> np.ndarray:
    """``(N, 2048)`` global-avg-pooled ResNet-50 features, ImageNet-pretrained, frozen."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = _build_resnet_model(device)

    feats = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            imgs = np.stack([_load_resized(s.image, RESNET_IMG) for s in batch])
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
            x = (x - mean) / std
            feats.append(model(x).cpu().numpy())
            if (i // batch_size + 1) % 10 == 0:
                print(f"    resnet embeddings: {min(i + batch_size, len(samples))}/{len(samples)}")
    return np.concatenate(feats).astype(np.float32)


def extract_resnet_embeddings_from_arrays(
    images: np.ndarray, batch_size: int = 32, device: torch.device | None = None,
) -> np.ndarray:
    """Array-based sibling of ``extract_resnet_embeddings``: same ``(N, 2048)``
    frozen ResNet-50 features, but for an in-memory uint8 ``(N, H, W, 3)``
    array -- e.g. one of ``ablation_inputs.build_five_variants``'s
    counterfactual variants, which has no ``Sample.image`` path to open."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std = _build_resnet_model(device)

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            imgs = np.stack([_resize_array(im, RESNET_IMG) for im in batch])
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)
            x = (x - mean) / std
            feats.append(model(x).cpu().numpy())
            if (i // batch_size + 1) % 10 == 0:
                print(f"    resnet embeddings (array): {min(i + batch_size, len(images))}/{len(images)}")
    return np.concatenate(feats).astype(np.float32)


def _load_hf_token() -> str | None:
    """``HF_TOKEN`` from the environment, falling back to a ``.env`` file
    (``KEY=VALUE`` per line, whitespace around ``=`` tolerated) in the project
    root -- nothing here uses ``python-dotenv``, so a ``.env`` file isn't
    auto-loaded into the process environment otherwise."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" not in line or line.strip().startswith("#"):
                continue
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                return value.strip().strip('"').strip("'")
    return None


def _build_dinov3_model(model_name: str, device: torch.device):
    """The frozen DINOv3 model + its HF image processor, shared by the
    path-based and array-based extractors below. See module docstring for the
    gating requirement -- raises a 401 from ``from_pretrained`` until access is
    granted and a token is available (via ``HF_TOKEN``, a ``.env`` file, or
    ``huggingface-cli login``)."""
    from transformers import AutoImageProcessor, AutoModel

    token = _load_hf_token()
    processor = AutoImageProcessor.from_pretrained(model_name, token=token)
    model = AutoModel.from_pretrained(model_name, token=token).eval().to(device)
    return model, processor


def extract_dinov3_embeddings(
    samples: Sequence[Sample],
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    batch_size: int = 16,
    device: torch.device | None = None,
) -> np.ndarray:
    """``(N, D)`` DINOv3 CLS-token embeddings, frozen. See module docstring for
    the Hugging Face gating requirement -- this raises a 401 from
    ``from_pretrained`` until access is granted and a token is available (via
    ``HF_TOKEN``, a ``.env`` file, or ``huggingface-cli login``)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _build_dinov3_model(model_name, device)

    feats = []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            imgs = []
            for s in batch:
                with Image.open(s.image) as im:
                    imgs.append(im.convert("RGB").copy())
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]  # CLS token
            feats.append(cls.cpu().numpy())
            if (i // batch_size + 1) % 10 == 0:
                print(f"    dinov3 embeddings: {min(i + batch_size, len(samples))}/{len(samples)}")
    return np.concatenate(feats).astype(np.float32)


def extract_dinov3_embeddings_from_arrays(
    images: np.ndarray,
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    batch_size: int = 16,
    device: torch.device | None = None,
) -> np.ndarray:
    """Array-based sibling of ``extract_dinov3_embeddings``: same ``(N, D)``
    frozen DINOv3 CLS-token features, but for an in-memory uint8
    ``(N, H, W, 3)`` array -- e.g. one of
    ``ablation_inputs.build_five_variants``'s counterfactual variants, which
    has no ``Sample.image`` path to open. Same gating requirement as the
    path-based version."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = _build_dinov3_model(model_name, device)

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            imgs = [Image.fromarray(im) for im in batch]
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]  # CLS token
            feats.append(cls.cpu().numpy())
            if (i // batch_size + 1) % 10 == 0:
                print(f"    dinov3 embeddings (array): {min(i + batch_size, len(images))}/{len(images)}")
    return np.concatenate(feats).astype(np.float32)


def cache_path(out_dir: Path, backbone: str, modality: str, variant: str | None = None) -> Path:
    """``variant`` is omitted for the path-based (whole-frame) cache to stay
    backward compatible with files already on disk from ``load_or_extract``;
    pass it for the array-based variant caches (``load_or_extract_from_arrays``)
    so e.g. a ``bg_patch`` cache never collides with the ``full`` one."""
    if variant is None:
        return out_dir / f"embeddings_{backbone}_{modality}.npz"
    return out_dir / f"embeddings_{backbone}_{modality}_{variant}.npz"


def _cached_or_none(path: Path, n: int, backbone: str) -> np.ndarray | None:
    """Shared cache-hit/miss logic for both ``load_or_extract`` functions."""
    if not path.exists():
        return None
    cached = np.load(path)
    if len(cached["X"]) == n:
        print(f"  reusing cached {backbone} embeddings: {path}")
        return cached["X"]
    print(f"  cache at {path} has {len(cached['X'])} rows, expected {n} -- re-extracting")
    return None


def load_or_extract(
    samples: Sequence[Sample], backbone: str, modality: str, out_dir: Path, **kwargs,
) -> np.ndarray:
    """Cache embeddings to disk keyed by ``(backbone, modality)``. Caller is
    responsible for ``samples`` being in a stable, repeatable order
    (``build_manifest``'s own order already is) so the cache stays valid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(out_dir, backbone, modality)
    cached = _cached_or_none(path, len(samples), backbone)
    if cached is not None:
        return cached

    print(f"  extracting {backbone} embeddings for {len(samples)} {modality} images ...")
    if backbone == "resnet":
        X = extract_resnet_embeddings(samples, **kwargs)
    elif backbone == "dinov3":
        X = extract_dinov3_embeddings(samples, **kwargs)
    else:
        raise ValueError(f"unknown backbone {backbone!r}; expected 'resnet' or 'dinov3'")
    np.savez(path, X=X)
    return X


def load_or_extract_from_arrays(
    images: np.ndarray, backbone: str, modality: str, variant: str, out_dir: Path, **kwargs,
) -> np.ndarray:
    """Array-based sibling of ``load_or_extract``, for in-memory uint8 variant
    arrays (e.g. from ``ablation_inputs.build_five_variants``) that have no
    ``Sample.image`` path to open. Cache is keyed by ``(backbone, modality,
    variant)`` -- see ``cache_path`` -- so e.g. WLE/resnet/bg_patch never
    collides with WLE/resnet/full. Caller is responsible for ``images`` being
    in a stable, repeatable row order matching the labels it will be paired
    with (``build_five_variants``' row order already matches its input
    ``samples`` list) so the cache stays valid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(out_dir, backbone, modality, variant)
    cached = _cached_or_none(path, len(images), backbone)
    if cached is not None:
        return cached

    print(f"  extracting {backbone} embeddings for {len(images)} {modality}/{variant} images ...")
    if backbone == "resnet":
        X = extract_resnet_embeddings_from_arrays(images, **kwargs)
    elif backbone == "dinov3":
        X = extract_dinov3_embeddings_from_arrays(images, **kwargs)
    else:
        raise ValueError(f"unknown backbone {backbone!r}; expected 'resnet' or 'dinov3'")
    np.savez(path, X=X)
    return X
