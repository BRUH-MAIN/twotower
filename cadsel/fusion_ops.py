"""Vision-language fusion operators retrained for two image streams (WLE, EUS).

Drop-in alternatives to ``cadsel.fusion.GatedFusion`` for
``FusionHead(fusion=...)``. Each takes the two pooled, encoded descriptors
``a`` and ``b`` (B, dim), already zeroed by ``FusionHead`` when a stream is
absent or dropped, plus the availability flags ``avail`` (B, 2), and returns
one (B, dim) vector for the classifier.

  * ``concat``  -- MLP on [a, b, avail]: the plain feature-fusion baseline.
  * ``mlb``     -- low-rank bilinear pooling, P(tanh(Ua) * tanh(Vb))
                   (Kim et al., ICLR 2017).
  * ``mfb``     -- factorised bilinear pooling: k rank-1 factors per output,
                   sum-pooled, signed square root, L2 normalisation
                   (Yu et al., ICCV 2017).
  * ``mfb_res`` -- MFB plus the two unimodal descriptors, so evidence only
                   one stream carries is not lost to the product.

A product with a zeroed stream is zero, and a third of CAD-SEL patients have
no EUS. The multiplicative operators therefore replace an absent stream with
a learned per-modality "absent" vector (as ``SymmetricCrossAttentionFusion``
does), so a WLE-only patient gets a learned re-weighting of its WLE
descriptor instead of a constant.

``BiFiLM`` is different: it is not a fusion of pooled vectors but a frame-level
conditioning step (Perez et al., AAAI 2018, made bidirectional). Each stream's
unmodulated descriptor produces a feature-wise scale and shift for the other
stream's per-frame features, before attention pooling. ``FusionHead(film=True)``
applies it and then fuses with whichever operator ``fusion`` names.

The bilinear products run in fp32 with autocast off: the signed square root
has a steep gradient near zero, which fp16 does not survive.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSIONS = ("concat", "mlb", "mfb", "mfb_res")


class _AbsentFill(nn.Module):
    """A learned stand-in for an absent stream, one per modality."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embed = nn.Parameter(torch.randn(2, dim) * 0.02)

    def forward(self, a, b, avail):
        av_a, av_b = avail[:, 0:1], avail[:, 1:2]
        return (a * av_a + self.embed[0] * (1 - av_a),
                b * av_b + self.embed[1] * (1 - av_b))


class ConcatFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 2, dim), nn.ReLU())

    def forward(self, a, b, avail):
        return self.mlp(torch.cat([a, b, avail], dim=1))


class MLBFusion(nn.Module):
    def __init__(self, dim: int, rank: int | None = None) -> None:
        super().__init__()
        rank = rank or dim
        self.fill = _AbsentFill(dim)
        self.u, self.v = nn.Linear(dim, rank), nn.Linear(dim, rank)
        self.p = nn.Linear(rank, dim)

    def forward(self, a, b, avail):
        with torch.autocast(device_type=a.device.type, enabled=False):
            a, b = self.fill(a.float(), b.float(), avail.float())
            return self.p(torch.tanh(self.u(a)) * torch.tanh(self.v(b)))


class MFBFusion(nn.Module):
    """``z_i = sum_j (u_ij . a)(v_ij . b)``, i < out, j < k; then signed sqrt and
    L2 normalisation as in MFB. The unit-norm vector is scaled by sqrt(out) so
    its entries have unit RMS like the other operators' outputs (MFB feeds a
    linear classifier directly; ours starts with a hidden layer and weight decay)."""

    def __init__(self, dim: int, out: int | None = None, k: int = 5,
                 dropout: float = 0.1, residual: bool = False) -> None:
        super().__init__()
        self.out, self.k = out or dim, k
        self.fill = _AbsentFill(dim)
        self.u = nn.Linear(dim, self.out * k, bias=False)
        self.v = nn.Linear(dim, self.out * k, bias=False)
        self.drop = nn.Dropout(dropout)
        self.residual = (nn.Sequential(nn.Linear(self.out + 2 * dim + 2, dim), nn.ReLU())
                         if residual else None)
        if not residual and self.out != dim:
            raise ValueError("MFB without the residual projection needs out == dim")

    def forward(self, a, b, avail):
        with torch.autocast(device_type=a.device.type, enabled=False):
            a, b, avail = a.float(), b.float(), avail.float()
            fa, fb = self.fill(a, b, avail)
            h = self.drop(self.u(fa) * self.v(fb))
            z = h.reshape(len(h), self.out, self.k).sum(-1)
            z = z.sign() * torch.sqrt(z.abs() + 1e-6)
            z = F.normalize(z, dim=1) * self.out ** 0.5
            if self.residual is None:
                return z
            # the real (zeroed-if-absent) descriptors, not the filled ones
            return self.residual(torch.cat([z, a, b, avail], dim=1))


def make_fusion(name: str, dim: int) -> nn.Module:
    if name == "concat":
        return ConcatFusion(dim)
    if name == "mlb":
        return MLBFusion(dim)
    if name == "mfb":
        return MFBFusion(dim)
    if name == "mfb_res":
        return MFBFusion(dim, residual=True)
    raise ValueError(f"unknown fusion {name!r}; expected one of {FUSIONS}")


class BiFiLM(nn.Module):
    """Bidirectional feature-wise modulation of per-frame features.

    ``frames[m]`` (B, T, D_m) is scaled and shifted by a generator reading the
    OTHER stream's unmodulated descriptor ``context[o]`` (B, ctx_dim):
    ``frames * (1 + tanh(dgamma)) + beta``. Generators start at zero, so the
    module starts as the identity, and the bounded scale keeps a channel from
    being flipped or blown up. Where the context stream is absent (or
    dropped by modality dropout) the modulation is the identity, so a
    dropped stream cannot leak in through the generator.
    """

    def __init__(self, frame_dims: Mapping[str, int], ctx_dim: int, modalities: Sequence[str]) -> None:
        super().__init__()
        if len(modalities) != 2:
            raise ValueError("FiLM needs two modalities")
        self.modalities = list(modalities)
        self.gen = nn.ModuleDict({m: nn.Linear(ctx_dim, 2 * frame_dims[m]) for m in self.modalities})
        for layer in self.gen.values():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, frames: dict, masks: dict, context: dict, avail: dict) -> dict:
        a, b = self.modalities
        out = {}
        for m, o in ((a, b), (b, a)):
            scale, shift = self.gen[m](context[o]).chunk(2, dim=-1)
            on = avail[o].unsqueeze(1).to(scale.dtype)
            gamma = 1 + torch.tanh(scale) * on
            out[m] = (frames[m] * gamma.unsqueeze(1) + (shift * on).unsqueeze(1)) * masks[m].unsqueeze(-1)
        return out
