"""Fast in-memory image-classification harness shared by the shortcut diagnostics.

Decodes each image once into a uint8 array and trains a ResNet-18 straight from
RAM. Classification rather than detection is used throughout the diagnostics: it
isolates "is the label predictable" from "is the lesion localisable", which is the
question the shortcut-learning literature actually asks.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision.models import ResNet18_Weights

IMG = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BINARY = ["NET", "non-NET"]
SUBTYPES = ["NET_G1", "NET_G2", "Leiomyoma", "Lipoma", "NonNeoplasm"]


def label_of(sample, task: str) -> str:
    if task == "binary":
        return "NET" if sample.subtype.startswith("NET") else "non-NET"
    return sample.subtype


def classes_for(task: str) -> list[str]:
    return BINARY if task == "binary" else SUBTYPES


def decode(samples, progress_every: int = 750) -> np.ndarray:
    """uint8 array (N, IMG, IMG, 3)."""
    out = np.zeros((len(samples), IMG, IMG, 3), dtype=np.uint8)
    for i, s in enumerate(samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB").resize((IMG, IMG), Image.BILINEAR)
            out[i] = np.asarray(im, dtype=np.uint8)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    decoded {i+1}/{len(samples)}")
    return out


def encode_labels(samples, task: str) -> np.ndarray:
    classes = classes_for(task)
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[label_of(s, task)] for s in samples])


def train_eval(
    Xtr: np.ndarray, ytr: np.ndarray,
    Xva: np.ndarray, yva: np.ndarray,
    n_classes: int, tag: str = "", epochs: int = 15, batch: int = 32, seed: int = 42,
) -> dict:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.to(DEVICE)

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    w = w / max(w.sum(), 1e-9) * n_classes
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    mean, std = _MEAN.to(DEVICE), _STD.to(DEVICE)

    def to_gpu(a, flip=False):
        x = torch.from_numpy(a).to(DEVICE, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if flip:
            x = torch.flip(x, dims=[3])
        return (x - mean) / std

    best = {"acc": 0.0, "bal": 0.0, "epoch": 0}
    n = len(ytr)
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(n)
        for k in range(0, n, batch):
            idx = np.sort(order[k:k + batch])
            x = to_gpu(Xtr[idx], flip=random.random() < 0.5)
            y = ytr_t[torch.from_numpy(idx).to(DEVICE)]
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for k in range(0, len(yva), 64):
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    preds.append(model(to_gpu(Xva[k:k + 64])).argmax(1).cpu().numpy())
        pred = np.concatenate(preds)
        acc = float((pred == yva).mean())
        recalls = [float((pred[yva == c] == c).mean())
                   for c in range(n_classes) if (yva == c).any()]
        bal = float(np.mean(recalls))
        if bal > best["bal"]:
            best = {"acc": acc, "bal": bal, "epoch": ep + 1}

    maj = float(np.bincount(yva, minlength=n_classes).max() / len(yva))
    if tag:
        print(f"    {tag:<40} acc {best['acc']:.4f}  bal {best['bal']:.4f}  "
              f"(majority {maj:.4f}, chance {1/n_classes:.3f}, ep {best['epoch']})")
    return {"acc": round(best["acc"], 4), "bal": round(best["bal"], 4),
            "majority": round(maj, 4), "chance": round(1 / n_classes, 4),
            "n_train": int(len(ytr)), "n_val": int(len(yva))}
