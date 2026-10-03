"""Gradient-reversal adversarial debiasing -- Experiment 12, and (composed with
channel attention) Experiment 14.

A shared ResNet-18 backbone produces embeddings for both the full image (main
classification head, ordinary loss) and that same sample's shortcut-only crop
(a discriminator head, behind a Gradient Reversal Layer, trying to predict the
TRUE class label from that crop alone). Reversed gradients push the shared
backbone toward representations from which the discriminator CANNOT predict
the label from shortcut-only pixels -- i.e. this directly minimizes the exact
quantity Experiment 3 measures as "retention".

One shared backbone, not two encoders: CNN kernels are spatially shared, so
forcing the *same* filters to fail at predicting class from bg_patch-only
pixels should also reduce how much class-predictive signal those filters
inject when processing the same kind of context pixels inside a full-frame
forward pass. Two independent encoders would not have this property.

`bg_patch` is the primary discriminator target -- a real pixel crop from the
same distribution the backbone already sees. `boxmap` (a synthetic near-black
canvas with a white rectangle) is deliberately treated as secondary/ablatable:
it is badly out-of-distribution for a natural-image backbone, and
adversarializing against it risks the backbone learning to detect and
special-case an obviously synthetic input via cheap global statistics, rather
than genuinely losing geometry sensitivity on real images.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

from harness import DEVICE

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def grl_lambda_schedule(
    epoch: int, epochs: int, lambda_max: float, warmup_epochs: int = 2
) -> float:
    """Zero during warmup -- the discriminator trains undisturbed, so it becomes
    a competent bg_patch/boxmap classifier before reversal starts fighting it --
    then a standard DANN sigmoid ramp up to ``lambda_max``."""
    if epoch < warmup_epochs:
        return 0.0
    p = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return lambda_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)


class DiscriminatorHead(nn.Module):
    """A separate small MLP -- must not reuse the main classifier head, whose
    parameters are simultaneously pushed to succeed at the real objective. A
    shared head would be pushed to win (main loss) and pushed away from the
    backbone (adversarial loss) on the same weights at once, which is not a
    meaningful adversarial setup."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def default_backbone_factory(pretrained: bool = True) -> nn.Module:
    return torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)


class AdversarialResNet18(nn.Module):
    """``backbone_factory(pretrained=...)`` must return a module exposing ``.fc``
    (an ``nn.Linear``) as its final layer -- true of
    ``torchvision.models.resnet18`` and of
    ``experiments.backbones.build_resnet18_with_se``."""

    def __init__(
        self,
        n_classes: int,
        backbone_factory: Callable[..., nn.Module] | None = None,
        discriminator_targets: tuple[str, ...] = ("bg_patch",),
        hidden: int = 128,
        pretrained: bool = True,
    ):
        super().__init__()
        backbone_factory = backbone_factory or default_backbone_factory
        self.backbone = backbone_factory(pretrained=pretrained)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.main_head = nn.Linear(feat_dim, n_classes)
        self.discriminator_targets = tuple(discriminator_targets)
        self.grl = nn.ModuleDict(
            {t: GradientReversalLayer(0.0) for t in self.discriminator_targets}
        )
        self.disc_heads = nn.ModuleDict(
            {t: DiscriminatorHead(feat_dim, n_classes, hidden) for t in self.discriminator_targets}
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        return self.main_head(self.forward_features(x))

    def forward_discriminator(self, x: torch.Tensor, target: str) -> torch.Tensor:
        feat = self.forward_features(x)
        return self.disc_heads[target](self.grl[target](feat))

    def set_lambda(self, lam: float) -> None:
        for g in self.grl.values():
            g.lambda_ = lam


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    w = w / max(w.sum(), 1e-9) * n_classes
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_eval_adversarial(
    D_tr: dict,
    ytr: np.ndarray,
    D_va: dict,
    yva: np.ndarray,
    n_classes: int,
    *,
    targets: tuple[str, ...] = (),
    lambda_max: float = 1.0,
    warmup_epochs: int = 2,
    disc_hidden: int = 128,
    disc_weight: float = 1.0,
    backbone_factory: Callable[..., nn.Module] | None = None,
    compose_fn: Callable[[dict, np.ndarray, random.Random], np.ndarray] | None = None,
    epochs: int = 15,
    batch: int = 32,
    seed: int = 42,
    tag: str = "",
    early_stop_patience: int = 0,
) -> dict:
    """Trains one model, evaluates on full/bg_patch/boxmap -- same return shape
    as ``07_mitigation.py::train_and_probe``, so results slot into the same
    table. ``targets=()`` degenerates to plain CE with zero discriminator
    heads: this is both Experiments 13/14's plain-CE training path AND the
    "does this reduce to Exp 7's baseline" sanity check for Experiment 12.

    ``D_va`` must already carry a ``"boxmap"`` array
    (``ablation_inputs.boxmap_from``); ``D_tr``/``D_va`` otherwise come
    straight from ``ablation_inputs.build_full_bgpatch_tile_box``.

    ``compose_fn(D_tr, idx, rng)``, if given, returns a *replacement* batch for
    ``D_tr["full"][idx]`` (e.g. Experiment 13's bbox-blurred images) applied
    before the main-head forward pass; the discriminator branches always read
    the un-composed ``D_tr[target][idx]``, since blurring is a training-time
    perturbation of the main task's input, not of the shortcut-only probes.

    ``early_stop_patience`` mirrors ``cadsel/train.py``'s flag of the same
    name: 0 disables it (train the full ``epochs``); otherwise stop once
    ``val_bal`` hasn't improved for that many epochs, matching
    ``cadsel/engine.py``'s early-stop convention rather than inventing a new
    one for the classifier harness.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = random.Random(seed)

    model = AdversarialResNet18(
        n_classes,
        backbone_factory=backbone_factory,
        discriminator_targets=targets,
        hidden=disc_hidden,
    ).to(DEVICE)

    crit_main = nn.CrossEntropyLoss(weight=_class_weights(ytr, n_classes))
    crit_disc = nn.CrossEntropyLoss(weight=_class_weights(ytr, n_classes))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    mean, std = _MEAN.to(DEVICE), _STD.to(DEVICE)

    def to_gpu(a: np.ndarray, flip: bool = False) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if flip:
            x = torch.flip(x, dims=[3])
        return (x - mean) / std

    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    n = len(ytr)

    def evaluate_split(arr: np.ndarray) -> float:
        model.eval()
        preds = []
        with torch.no_grad():
            for k in range(0, len(yva), 64):
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    logits = model.forward_main(to_gpu(arr[k : k + 64]))
                preds.append(logits.argmax(1).cpu().numpy())
        p = np.concatenate(preds)
        rec = [float((p[yva == c] == c).mean()) for c in range(n_classes) if (yva == c).any()]
        return float(np.mean(rec))

    best_bal, best_state, best_epoch = 0.0, None, -1
    epochs_run = 0
    for ep in range(epochs):
        epochs_run = ep + 1
        model.set_lambda(grl_lambda_schedule(ep, epochs, lambda_max, warmup_epochs))
        model.train()
        order = np.random.permutation(n)
        disc_correct, disc_total = 0, 0

        for k in range(0, n, batch):
            idx = np.sort(order[k : k + batch])
            flip = rng.random() < 0.5

            full_batch = D_tr["full"][idx] if compose_fn is None else compose_fn(D_tr, idx, rng)
            x_full = to_gpu(full_batch, flip=flip)
            y = ytr_t[torch.from_numpy(idx).to(DEVICE)]

            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = crit_main(model.forward_main(x_full), y)
                for t in targets:
                    x_aux = to_gpu(D_tr[t][idx], flip=flip)
                    logits_disc = model.forward_discriminator(x_aux, t)
                    loss = loss + disc_weight * crit_disc(logits_disc, y)
                    disc_correct += int((logits_disc.detach().argmax(1) == y).sum())
                    disc_total += len(y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        bal = evaluate_split(D_va["full"])
        if bal > best_bal:
            best_bal = bal
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = ep

        if tag and targets:
            disc_acc = disc_correct / max(disc_total, 1)
            print(
                f"    {tag} epoch {ep + 1}/{epochs}: val_bal {bal:.3f}  "
                f"disc_acc {disc_acc:.3f}  lambda {model.grl[targets[0]].lambda_:.2f}"
            )

        if early_stop_patience > 0 and (ep - best_epoch) >= early_stop_patience:
            if tag:
                print(
                    f"    {tag}: early stop at epoch {ep + 1} "
                    f"(no improvement for {early_stop_patience} epochs, best was epoch {best_epoch + 1})"
                )
            break

    model.load_state_dict(best_state)
    res = {
        "full": round(evaluate_split(D_va["full"]), 4),
        "bg_patch": round(evaluate_split(D_va["bg_patch"]), 4),
        "boxmap": round(evaluate_split(D_va["boxmap"]), 4),
        "chance": round(1 / n_classes, 4),
        "best_epoch": best_epoch + 1,
        "epochs_run": epochs_run,
    }
    denom = res["full"] - res["chance"]
    res["ret_bg_patch"] = round((res["bg_patch"] - res["chance"]) / denom, 3) if denom > 1e-6 else None
    res["ret_boxmap"] = round((res["boxmap"] - res["chance"]) / denom, 3) if denom > 1e-6 else None
    if tag:
        print(
            f"    {tag:<34} full {res['full']:.3f} | bg_patch {res['bg_patch']:.3f} "
            f"({res['ret_bg_patch']:>6.1%}) | boxmap {res['boxmap']:.3f} "
            f"({res['ret_boxmap']:>6.1%})"
        )
    return res
