"""A ResNet-18 variant with squeeze-and-excite channel attention -- Experiment 14.

Inspired by ``Channel_Att_Bridge`` in
``/home/rohan/Ultralightunet/models/UltraLight_VM_UNet.py`` (lines 115-156),
which gates a U-Net's 5 *simultaneously available* skip connections jointly via
a shared cross-scale Conv1d before per-scale sigmoid gates. That cross-scale
mixing has no analogue here: a ResNet's stages are strictly sequential
(``layer2`` doesn't exist until ``layer1`` has run), so there is nothing to
cross-mix at gate-compute time. This is deliberately the simpler, standard
per-stage SENet block instead -- the cross-scale idea becomes relevant again
for a later FPN port (an FPN's output levels genuinely are simultaneous, like
U-Net skips; see the plan's Phase 2).

Important: plain channel attention trained with only classification loss has
no mechanism to suppress shortcut-predictive channels specifically -- CE
optimizes for accuracy, and Experiment 3 shows shortcut-only inputs already
reach 68-95% of full accuracy, so an SE gate has no incentive to prefer lesion
channels over context/geometry channels. ``build_resnet18_with_se`` is a
plain architecture change; Experiment 14 trains it both with plain CE
(expected-null control) and composed with ``experiments.adversarial``'s GRL
signal (the actually-motivated variant, where the reversed gradient gives the
gate a real reason to suppress shortcut-predictive channels).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights


class SEBlock(nn.Module):
    """Standard squeeze-and-excite: global-avg-pool -> 2-layer bottleneck MLP
    -> sigmoid gate -> per-channel rescale."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        gate = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * gate


def _stage_out_channels(stage: nn.Sequential) -> int:
    """A torchvision ResNet BasicBlock's output width is its last conv's
    out_channels (BasicBlock has no bottleneck expansion)."""
    return stage[-1].conv2.out_channels


def build_resnet18_with_se(
    stages: str = "all", reduction: int = 16, pretrained: bool = True
) -> nn.Module:
    """A full ``resnet18`` (with its original ``.fc``, so it satisfies
    ``experiments.adversarial.AdversarialResNet18``'s backbone contract
    unchanged) whose named stages are wrapped as
    ``nn.Sequential(stage, SEBlock(...))``.

    Pretrained conv weights load first; SE blocks (randomly initialized) are
    appended after, so ImageNet weights are untouched by their insertion.

    ``stages``: ``"all"`` wraps ``layer1``-``layer4`` (standard SENet);
    ``"layer4"`` wraps only the final stage (cheaper, and the direct analogue
    of a later per-FPN-level port, which gates one map per pyramid level).
    """
    if stages not in ("all", "layer4"):
        raise ValueError(f"unknown stages {stages!r}; expected 'all' or 'layer4'")

    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    targets = ("layer1", "layer2", "layer3", "layer4") if stages == "all" else ("layer4",)
    for name in targets:
        stage = getattr(model, name)
        channels = _stage_out_channels(stage)
        setattr(model, name, nn.Sequential(stage, SEBlock(channels, reduction)))
    return model
