"""Joint end-to-end two-tower WLE+EUS model: two independently trainable
backbones (one per modality) feeding ``cadsel.fusion.FusionHead`` directly, so
the fusion loss backpropagates all the way into both backbones -- unlike
``cadsel.fusion.extract_features`` / ``cadsel.classifier_fusion_features``,
which both freeze the per-modality feature extractor beforehand and only ever
train the small head on top of a static, pre-computed feature cache.

The tower backbone is selectable (``TOWERS``). The torchvision CNN builders
mirror ``experiments/finetune_classifier.py::build_cnn_backbone``'s branches
and ``'partial'`` freeze definitions, and the DINOv3 tower mirrors its
``Dinov3Classifier`` (duplicated rather than imported: ``cadsel/`` doesn't
otherwise depend on ``experiments/``) -- keep them in sync by hand if those
change. Each tower outputs one pooled descriptor per frame, never class
logits: ``FusionHead`` supplies its own classifier on top of the fused
patient descriptor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from .classification_data import make_classification_normalizer
from .config import MODALITY_NAMES, MODALITY_STATS
from .fusion import FusionHead

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# backbone -> (pooled feature dim, input normalisation). "modality" keeps the
# two-tower's original per-modality MODALITY_STATS for the torchvision CNNs;
# the HF towers use their own pretraining convention (DINOv3/DeiT: ImageNet
# stats -- cadsel/config.py documents how CADSEL-stat inputs degrade DINOv3;
# MobileViT: raw [0, 1], its image processor does no mean/std normalisation).
TOWERS = {
    "efficientnet_b0": (1280, "modality"),
    "resnet50": (2048, "modality"),
    "convnext_tiny": (768, "modality"),
    "mobilenet_v3_large": (960, "modality"),
    "mobilevit_s": (640, "unit"),
    "deit_s": (384, "imagenet"),
    "dinov3_vits16": (384, "imagenet"),
    "dinov3_vitb16": (768, "imagenet"),
    "dinov3_vitl16": (1024, "imagenet"),
}
HF_MODEL_NAMES = {
    "mobilevit_s": "apple/mobilevit-small",
    "deit_s": "facebook/deit-small-patch16-224",
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def _freeze_all_but(model: nn.Module, trainable_modules: list[nn.Module]) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for m in trainable_modules:
        for p in m.parameters():
            p.requires_grad = True


def _build_cnn_tower(backbone: str) -> tuple[nn.Module, list[nn.Module]]:
    """ImageNet-pretrained torchvision CNN, backbone-only (pooled descriptor
    out), plus the modules ``'partial'`` leaves trainable."""
    if backbone == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        model.classifier = nn.Identity()
        partial = [model.features[-2], model.features[-1]]
    elif backbone == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()
        partial = [model.layer4]
    elif backbone == "convnext_tiny":
        model = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[-1] = nn.Identity()   # keeps the LayerNorm2d + Flatten
        partial = [model.features[-1], model.classifier]
    elif backbone == "mobilenet_v3_large":
        model = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier = nn.Identity()
        partial = [model.features[-2], model.features[-1]]
    else:
        raise ValueError(f"unknown CNN tower {backbone!r}")
    return model, partial


class _HFTower(nn.Module):
    """Hugging Face vision backbone -> one pooled descriptor per frame: the
    CLS token (``last_hidden_state[:, 0]``, already final-normed) for the
    ViTs, ``pooler_output`` for MobileViT. ``flip_bgr`` reverses the channel
    order, matching ``MobileViTImageProcessor``'s ``do_flip_channel_order``."""

    def __init__(self, model: nn.Module, pool: str, flip_bgr: bool = False) -> None:
        super().__init__()
        self.model = model
        self.pool = pool
        self.flip_bgr = flip_bgr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.flip_bgr:
            x = x.flip(1)
        out = self.model(pixel_values=x)
        return out.pooler_output if self.pool == "pooler" else out.last_hidden_state[:, 0]


def dinov3_blocks(model: nn.Module) -> nn.ModuleList:
    """DINOv3ViTModel's transformer blocks: transformers 5.x nests them as
    ``model.model.layer``, 4.x keeps them at ``model.layer``."""
    return model.model.layer if hasattr(model, "model") else model.layer


def _build_hf_tower(backbone: str, unfreeze_last_n: int) -> tuple[nn.Module, list[nn.Module]]:
    from transformers import AutoModel, MobileViTModel, ViTModel

    from .models import _load_hf_token

    name, token = HF_MODEL_NAMES[backbone], _load_hf_token()
    if backbone == "mobilevit_s":
        model = MobileViTModel.from_pretrained(name, token=token)
        return _HFTower(model, "pooler", flip_bgr=True), [model.encoder.layer[-1], model.conv_1x1_exp]
    if backbone == "deit_s":
        model = ViTModel.from_pretrained(name, token=token, add_pooling_layer=False)
        # transformers 5.x moved ViT's blocks from ``encoder.layer`` to ``layers``
        blocks = model.layers if hasattr(model, "layers") else model.encoder.layer
        return _HFTower(model, "cls"), [*blocks[-unfreeze_last_n:], model.layernorm]
    if backbone.startswith("dinov3_"):
        model = AutoModel.from_pretrained(name, token=token)
        return _HFTower(model, "cls"), [*dinov3_blocks(model)[-unfreeze_last_n:], model.norm]
    raise ValueError(f"unknown HF tower {backbone!r}")


def build_tower(backbone: str = "efficientnet_b0", freeze: str = "partial",
                unfreeze_last_n: int = 4) -> nn.Module:
    """One modality's tower. ``'partial'`` leaves only the last stage (CNNs)
    or the last ``unfreeze_last_n`` transformer blocks + final norm (ViTs)
    trainable."""
    if backbone not in TOWERS:
        raise ValueError(f"unknown tower backbone {backbone!r}; expected one of {sorted(TOWERS)}")
    if backbone in HF_MODEL_NAMES:
        model, partial = _build_hf_tower(backbone, unfreeze_last_n)
    else:
        model, partial = _build_cnn_tower(backbone)

    if freeze == "none":
        pass
    elif freeze == "partial":
        _freeze_all_but(model, partial)
    else:
        raise ValueError(f"unknown freeze mode {freeze!r} (joint towers support 'none'/'partial')")
    return model


def tower_normalizers(backbone: str = "efficientnet_b0") -> dict:
    """``{modality: uint8 -> normalised float}`` for ``backbone``'s input
    convention (see ``TOWERS``)."""
    norm = TOWERS[backbone][1]
    if norm == "modality":
        return {m: make_classification_normalizer(*MODALITY_STATS[m]) for m in MODALITY_NAMES}
    if norm == "imagenet":
        stats = (IMAGENET_MEAN, IMAGENET_STD)
    else:
        stats = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    return {m: make_classification_normalizer(*stats) for m in MODALITY_NAMES}


class JointTwoTowerModel(nn.Module):
    """Two independent towers (one per modality, same ``backbone``) + an
    unmodified ``cadsel.fusion.FusionHead``. ``backbone`` defaults to
    EfficientNet-B0, the original (and only) architecture of every
    checkpoint saved before the tower was made selectable.
    """

    def __init__(self, n_classes: int, backbone: str = "efficientnet_b0", freeze: str = "partial",
                 fusion: str = "gated", modality_dropout: float = 0.3,
                 dropout: float = 0.3, unfreeze_last_n: int = 4) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.feat_dim = TOWERS[backbone][0]
        self.backbones = nn.ModuleDict({
            m: build_tower(backbone, freeze, unfreeze_last_n) for m in MODALITY_NAMES
        })
        self.head = FusionHead(
            self.feat_dim, n_classes,
            modality_dropout=modality_dropout, fusion=fusion, dropout=dropout,
        )

    def forward(self, images: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]) -> torch.Tensor:
        """``images[modality]``: (B, T, C, H, W) float, already normalised.
        ``masks[modality]``: (B, T), 1 for a real frame, 0 for padding.

        Only real frames go through the backbone; padded/absent-modality slots
        get a zero descriptor, the same convention ``cadsel.fusion.make_batch``
        uses for a genuinely absent modality. Previously every slot went
        through, so in training the EUS tower's BatchNorm batch statistics
        (and running stats) were computed partly over the constant images of
        the 220 WLE-only patients. Eval output is unchanged by this: BatchNorm
        uses running stats per sample there, and padded features were already
        masked to zero.
        """
        batch: dict[str, torch.Tensor] = {}
        for modality in MODALITY_NAMES:
            imgs, mask = images[modality], masks[modality]
            B, T, C, H, W = imgs.shape
            real = mask.reshape(B * T) > 0.5
            feats = imgs.new_zeros(B * T, self.feat_dim)
            if real.any():
                out = self.backbones[modality](imgs.reshape(B * T, C, H, W)[real])
                feats[real] = out.to(feats.dtype)
            batch[modality] = feats.reshape(B, T, -1) * mask.unsqueeze(-1)
            batch[f"{modality}_mask"] = mask
        return self.head(batch)
