"""Grouped-CV probes for the shortcut diagnostics.

Uses torch rather than scikit-learn so the experiments have no dependency beyond
what the training pipeline already needs. A small MLP on tabular features is the
"auxiliary probe" of the review's Section 4.3: it asks how much of a target is
linearly/nonlinearly decodable from a feature set that should not contain it.

Every split is grouped by patient, so a high score can never be explained by
duplicate frames of the same case landing on both sides.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def grouped_folds(groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    """Deterministic group k-fold: whole groups are assigned to one fold."""
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[g]: i % n_splits for i, g in enumerate(order)}
    assign = np.array([fold_of[g] for g in groups])
    for f in range(n_splits):
        test = assign == f
        if test.any() and (~test).any():
            yield ~test, test


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.any():
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _fit_mlp(Xtr, ytr, Xte, n_classes, hidden=64, epochs=300, seed=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)

    # class-balanced loss so a majority class cannot dominate the probe
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    w = w / w.sum() * max((counts > 0).sum(), 1)
    weight = torch.tensor(w, dtype=torch.float32, device=device)

    model = nn.Sequential(
        nn.Linear(Xtr.shape[1], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_classes),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    crit = nn.CrossEntropyLoss(weight=weight)

    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = crit(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        return model(Xte_t).argmax(1).cpu().numpy()


def probe(
    X: np.ndarray,
    y_labels,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
    hidden: int = 64,
    epochs: int = 300,
) -> dict:
    """Decodability of `y_labels` from `X`, with patient-grouped CV.

    Returns accuracy and balanced accuracy alongside the majority-class and
    uniform-chance floors, plus the lift over each.
    """
    classes = sorted(set(map(str, y_labels)))
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[str(v)] for v in y_labels])
    n_classes = len(classes)

    X = np.asarray(X, dtype=np.float64)
    accs, bals, base_accs = [], [], []

    for tr, te in grouped_folds(groups, n_splits, seed):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd

        pred = _fit_mlp(Xtr, y[tr], Xte, n_classes, hidden, epochs, seed)
        accs.append(float((pred == y[te]).mean()))
        bals.append(balanced_accuracy(y[te], pred, n_classes))

        majority = np.bincount(y[tr], minlength=n_classes).argmax()
        base_accs.append(float((y[te] == majority).mean()))

    return {
        "n_classes": n_classes,
        "accuracy": round(float(np.mean(accs)), 4),
        "balanced_accuracy": round(float(np.mean(bals)), 4),
        "majority_baseline": round(float(np.mean(base_accs)), 4),
        "chance_balanced": round(1.0 / n_classes, 4),
        "lift_over_majority": round(float(np.mean(accs) - np.mean(base_accs)), 4),
        "lift_over_chance_balanced": round(float(np.mean(bals) - 1.0 / n_classes), 4),
    }


def fmt(name: str, r: dict) -> str:
    return (f"{name:<38} acc {r['accuracy']:.3f} (base {r['majority_baseline']:.3f}) | "
            f"bal {r['balanced_accuracy']:.3f} (chance {r['chance_balanced']:.3f}) | "
            f"lift {r['lift_over_chance_balanced']:+.3f}")
