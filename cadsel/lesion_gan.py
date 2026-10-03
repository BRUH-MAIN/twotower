"""A small class-conditional GAN for synthetic lesion patches -- EUS-specific
conditional generative augmentation, the remaining implementable Tier-3 item
from the literature review (FINDINGS.md §5c).

Real explicit caveat, not hedging: the literature review's own assessment of
this idea was that training a *usable* generator on a few hundred lesion crops
per class is itself a small-data problem, and recommended treating it as an
exploratory prototype, not a near-term win, before committing real time. A
full diffusion model (the review's first suggestion) needs far more data and
compute than is available here to have any chance of working; this is instead
a small class-conditional DCGAN (Radford et al., 2015) on small patches
(default 64x64) -- the more tractable end of "generative augmentation",
chosen specifically because it's realistic to actually train in this setting,
not because it's expected to match diffusion-model quality.

Output usage: a trained generator wraps into ``GeneratedLesionBank``, which
exposes the same ``.sample(rng) -> (patch, label)`` interface as
``cadsel.data.LesionBank``, so it drops into the existing `copy_paste`
augmentation as an alternative *source* of pasted patches -- synthetic
lesions instead of real ones cropped from other patients.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

TILE = 64
LATENT_DIM = 64


class Generator(nn.Module):
    def __init__(self, n_classes: int, latent_dim: int = LATENT_DIM, tile: int = TILE):
        super().__init__()
        self.tile = tile
        self.label_embed = nn.Embedding(n_classes, latent_dim)
        in_dim = latent_dim * 2
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_embed(labels)
        x = torch.cat([z, c], dim=1).unsqueeze(-1).unsqueeze(-1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, n_classes: int, tile: int = TILE):
        super().__init__()
        self.tile = tile
        self.label_embed = nn.Embedding(n_classes, tile * tile)
        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 1x1
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_embed(labels).view(-1, 1, self.tile, self.tile)
        return self.net(torch.cat([x, c], dim=1)).view(-1)


def _to_tensor(patches: np.ndarray) -> torch.Tensor:
    """uint8 (N, H, W, 3) -> float32 (N, 3, H, W) in [-1, 1] (Tanh range)."""
    x = torch.from_numpy(patches).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return x


def _to_uint8(x: torch.Tensor) -> np.ndarray:
    """float32 (N, 3, H, W) in [-1, 1] -> uint8 (N, H, W, 3)."""
    x = ((x.clamp(-1, 1) + 1.0) * 127.5).round().byte()
    return x.permute(0, 2, 3, 1).cpu().numpy()


def train_lesion_gan(
    patches: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    epochs: int = 200,
    batch: int = 32,
    lr: float = 2e-4,
    seed: int = 42,
    device: torch.device | None = None,
    log_every: int = 20,
) -> Generator:
    """Trains the conditional DCGAN. ``patches``: uint8 ``(N, TILE, TILE, 3)``
    (resize to ``lesion_gan.TILE`` before calling -- this does not resize).
    ``labels``: ``(N,)`` int class ids in ``[0, n_classes)``.

    Standard DCGAN training (Adam, betas=(0.5, 0.999), BCE-with-logits,
    alternating D/G steps) -- no exotic stabilization tricks, matching the
    "small tractable prototype, not a research contribution" scope here.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    g = Generator(n_classes).to(device)
    d = Discriminator(n_classes).to(device)
    opt_g = torch.optim.Adam(g.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    x = _to_tensor(patches).to(device)
    y = torch.from_numpy(labels).long().to(device)
    n = len(x)

    rng = np.random.RandomState(seed)
    for epoch in range(epochs):
        order = rng.permutation(n)
        g_losses, d_losses = [], []
        for k in range(0, n, batch):
            idx = order[k : k + batch]
            real_x, real_y = x[idx], y[idx]
            b = len(idx)

            z = torch.randn(b, LATENT_DIM, device=device)
            fake_y = torch.randint(0, n_classes, (b,), device=device)
            fake_x = g(z, fake_y)

            opt_d.zero_grad(set_to_none=True)
            d_real = d(real_x, real_y)
            d_fake = d(fake_x.detach(), fake_y)
            d_loss = criterion(d_real, torch.ones_like(d_real)) + criterion(
                d_fake, torch.zeros_like(d_fake)
            )
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            d_fake_for_g = d(fake_x, fake_y)
            g_loss = criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))
            g_loss.backward()
            opt_g.step()

            g_losses.append(float(g_loss.detach()))
            d_losses.append(float(d_loss.detach()))

        if log_every and (epoch + 1) % log_every == 0:
            print(f"    epoch {epoch + 1}/{epochs}  g_loss={np.mean(g_losses):.3f}  d_loss={np.mean(d_losses):.3f}")

    g.eval()
    return g


class GeneratedLesionBank:
    """Drop-in replacement for ``cadsel.data.LesionBank``'s sampling
    interface: a pool of patches, sampled by plain array lookup. ``sample(rng)``
    matches ``LesionBank.sample``'s signature exactly so ``_copy_paste``
    doesn't need to know which kind of bank it has.

    Patches are generated once, up front, in ``load()`` -- *not* per-sample
    inside ``sample()`` -- because ``sample()`` runs inside
    ``CadSelDetectionDataset.__getitem__``, which executes in forked
    DataLoader worker processes when ``--num-workers > 0``, and CUDA cannot be
    re-initialised inside a forked subprocess. Pre-generating a fixed pool
    (mirroring ``LesionBank``'s own real-crop-pool design) sidesteps this
    entirely: by the time workers fork, every patch already exists as a plain
    uint8 array, exactly like a real crop.
    """

    def __init__(self, patches: list[np.ndarray], labels: list[int]):
        self.patches = patches
        self.labels = labels
        self.by_label: dict[int, list[int]] = defaultdict(list)
        for i, label in enumerate(labels):
            self.by_label[label].append(i)

    @classmethod
    def load(
        cls,
        path: str | Path,
        n_classes: int,
        pool_size_per_class: int = 200,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> "GeneratedLesionBank":
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = Generator(n_classes).to(device)
        g.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        g.eval()

        torch.manual_seed(seed)
        patches: list[np.ndarray] = []
        labels: list[int] = []
        with torch.no_grad():
            for label in range(n_classes):
                z = torch.randn(pool_size_per_class, LATENT_DIM, device=device)
                y = torch.full((pool_size_per_class,), label, device=device, dtype=torch.long)
                out = _to_uint8(g(z, y))
                patches.extend(out)
                labels.extend([label] * pool_size_per_class)
        print(f"  generated lesion pool: {len(patches)} synthetic patches "
              f"({pool_size_per_class} per class x {n_classes} classes)")
        return cls(patches, labels)

    def sample(self, rng: random.Random) -> tuple[np.ndarray, int]:
        label = rng.choice(sorted(self.by_label))
        idx = rng.choice(self.by_label[label])
        return self.patches[idx], label
