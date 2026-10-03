"""Track A — trainable-backbone lesion classifier (binary NET/non-NET or 5-way
subtype), 5-fold patient-grouped cross-validation.

Fills the infrastructure gap FINDINGS.md's classification work left open:
``experiments/svm_classifier.py`` only ever fit an SVM on *frozen* embeddings.
This trains the backbone itself (fully, partially, or head-only), the
fine-tuned counterpart those frozen-embedding numbers need to beat to justify
the extra complexity (see the "fair-baseline gate" printed at the end of a
``--backbone resnet50 --freeze none --oversample none --loss ce`` run).

    python -m experiments.finetune_classifier --backbone resnet50 \\
        --modality WLE --task binary --freeze none --oversample none --loss ce

Every fold's own held-out validation balanced accuracy is what gets averaged
-- this is *not* the same 5-fold partition ``experiments/svm_classifier.py``
uses (that one calls ``experiments/probes.py::grouped_folds``; this calls
``cadsel.data.patient_level_folds`` -- same protocol class, patient-grouped
and stratified, but different RNG draw), so per-fold numbers are not
directly comparable across the two scripts, only the 5-fold means.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from cadsel.classification_data import (
    CadSelClassificationDataset,
    ClassificationSizeRepeatFactorSampler,
    make_classification_normalizer,
)
from cadsel.config import CADSEL_MEAN, CADSEL_STD, MODALITIES, MODALITY_STATS, PROJECT_ROOT
from cadsel.data import build_manifest, patient_level_folds
from cadsel.models import SeesawLossState  # noqa: F401  (see ClassificationSeesawState docstring)
from harness import classes_for, encode_labels
from probes import balanced_accuracy

OUT_DEFAULT = PROJECT_ROOT / "experiments" / "results" / "classifier_runs"
DINOV3_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Fair-baseline gate reference: experiments/svm_classifier.py's published
# frozen-embedding numbers (5-fold patient-grouped balanced accuracy). A
# fine-tuned ResNet-50 (--backbone resnet50 --freeze none --oversample none
# --loss ce) is expected to clear these; if it doesn't, that's a bug in this
# script (LR, undertraining, a data leak), not a result to record.
SVM_BASELINE = {
    ("resnet50", "WLE", "binary"): 0.6447,
    ("resnet50", "WLE", "subtype"): 0.2995,
}

BACKBONES = ("resnet50", "convnext_tiny", "efficientnet_b0", "dinov3_vits16")


# ---------------------------------------------------------------------------
# Backbone + head construction
# ---------------------------------------------------------------------------
def resolve_normalization_stats(backbone: str, modality: str, per_modality_stats: bool):
    """DINOv3 was pretrained with ImageNet normalisation -- feeding it
    CADSEL_MEAN/STD would silently degrade the pretrained representation
    exactly as documented for the detector backbone
    (cadsel/config.py::DINOV3_MODEL_REGISTRY); avoid repeating that mistake
    here for the classifier.
    """
    if backbone == "dinov3_vits16":
        return IMAGENET_MEAN, IMAGENET_STD
    if per_modality_stats:
        return MODALITY_STATS[modality]
    return CADSEL_MEAN, CADSEL_STD


def _freeze_all_but(model: nn.Module, trainable_modules: list[nn.Module]) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for m in trainable_modules:
        for p in m.parameters():
            p.requires_grad = True


def build_cnn_backbone(name: str, n_classes: int, freeze: str) -> nn.Module:
    """resnet50 / convnext_tiny / efficientnet_b0, ImageNet-pretrained, head
    swapped for ``n_classes``. ``freeze='partial'`` unfreezes each
    architecture's last stage (the block nearest the swapped head) --
    the shallower stages keep general-purpose ImageNet features frozen.
    """
    if name == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        head, partial_blocks = model.fc, [model.layer4]
    elif name == "convnext_tiny":
        model = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        head, partial_blocks = model.classifier, [model.features[-1]]
    elif name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
        head, partial_blocks = model.classifier, [model.features[-2], model.features[-1]]
    else:
        raise ValueError(f"unknown CNN backbone {name!r}")

    if freeze == "none":
        pass
    elif freeze == "head":
        _freeze_all_but(model, [head])
    elif freeze == "partial":
        _freeze_all_but(model, [*partial_blocks, head])
    else:
        raise ValueError(f"unknown freeze mode {freeze!r}")
    return model


class Dinov3Classifier(nn.Module):
    """DINOv3 ViT + linear head on the CLS token.

    Reuses ``cadsel.models``' HF-loading pattern (``AutoModel.from_pretrained``
    with the project's HF token) rather than ``cadsel.models.Dinov3Backbone``
    itself: that class's ``forward`` strips the CLS token to return a patch
    feature *map* for the detector's ROI-align head, whereas a classifier
    wants exactly the CLS token it discards. ``freeze='partial'`` unfreezes
    the last ``unfreeze_last_n`` of the ViT's 12 transformer blocks (verified
    submodule path for this checkpoint: ``model.model.layer``) plus the final
    norm and head.
    """

    def __init__(self, model_name: str, n_classes: int, freeze: str, unfreeze_last_n: int = 4) -> None:
        super().__init__()
        from transformers import AutoModel

        from cadsel.models import _load_hf_token

        self.backbone = AutoModel.from_pretrained(model_name, token=_load_hf_token())
        self.head = nn.Linear(self.backbone.config.hidden_size, n_classes)

        if freeze == "none":
            pass
        elif freeze == "head":
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif freeze == "partial":
            for p in self.backbone.parameters():
                p.requires_grad = False
            for layer in self.backbone.model.layer[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"unknown freeze mode {freeze!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.backbone(pixel_values=x).last_hidden_state[:, 0]
        return self.head(cls)


def build_model(backbone: str, n_classes: int, freeze: str, unfreeze_last_n: int = 4) -> nn.Module:
    if backbone == "dinov3_vits16":
        return Dinov3Classifier(DINOV3_MODEL_NAME, n_classes, freeze, unfreeze_last_n=unfreeze_last_n)
    return build_cnn_backbone(backbone, n_classes, freeze)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def compute_classification_class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights, ``cadsel.data.compute_class_weights``'s own
    convention (normalised to sum to ``n_classes``) -- not that function
    itself, which counts detection *boxes* by ``TASK_ID_MAP``, a different
    vocabulary than a classification label per image.
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    safe = np.where(counts > 0, counts, 1.0)
    w = 1.0 / safe
    w = w / w.sum() * n_classes
    return np.where(counts > 0, w, 0.0)


class ClassificationSeesawState:
    """Seesaw Loss (Wang et al. 2021) cumulative per-class counts + mitigation
    matrix, ported from ``cadsel.models.SeesawLossState`` for plain
    classification cross-entropy -- not reused as-is because that class is
    wired for the detector's ROI-head loss (``class 0`` hardcoded as the
    "background" label it always excludes from the mechanism). Classification
    has no background class -- every label is a real class -- so that
    exclusion is dropped here.
    """

    def __init__(self, n_classes: int, p: float = 0.8, device: torch.device | None = None) -> None:
        self.n_classes = n_classes
        self.p = p
        self.counts = torch.ones(n_classes, dtype=torch.float64, device=device)

    def update(self, labels: torch.Tensor) -> None:
        counts = torch.bincount(labels, minlength=self.n_classes).to(self.counts.device)
        self.counts += counts.double()

    def mitigation_matrix(self) -> torch.Tensor:
        n_j = self.counts.view(1, -1)
        n_i = self.counts.view(-1, 1)
        ratio = n_j / n_i
        m = torch.where(ratio > 1, ratio.pow(-self.p), torch.ones_like(ratio))
        m.fill_diagonal_(1.0)
        return m


def seesaw_ce_loss(logits: torch.Tensor, targets: torch.Tensor, state: ClassificationSeesawState) -> torch.Tensor:
    state.update(targets.detach())
    m = state.mitigation_matrix().to(logits.device, logits.dtype)
    logits = logits + torch.log(m[targets].clamp_min(1e-12))
    return F.cross_entropy(logits, targets)


# ---------------------------------------------------------------------------
# Reporting: confusion matrix, ROC/PR-AUC, loss/LR/ROC/PR plots
# ---------------------------------------------------------------------------
def compute_extra_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> dict:
    """Confusion matrix + ROC-AUC + PR-AUC (average precision) at the best
    epoch's predictions -- macro one-vs-rest for >2 classes, plain binary AUC
    for 2. Uses ``sklearn.metrics`` here (unlike the rest of this project,
    which avoids scikit-learn for its decodability probes -- see
    ``experiments/probes.py``'s docstring): these are standard report
    metrics, not a probe, and scikit-learn is already a project dependency
    (``experiments/svm_classifier.py``).
    """
    pred = probs.argmax(1)
    cm = confusion_matrix(y_true, pred, labels=list(range(n_classes)))
    roc_auc = pr_auc = None
    try:
        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_true, probs[:, 1]))
            pr_auc = float(average_precision_score(y_true, probs[:, 1]))
        else:
            roc_auc = float(roc_auc_score(
                y_true, probs, multi_class="ovr", average="macro", labels=list(range(n_classes))
            ))
            y_onehot = np.eye(n_classes)[y_true]
            pr_auc = float(average_precision_score(y_onehot, probs, average="macro"))
    except ValueError:
        pass  # e.g. a fold's val split is missing a class entirely
    return {"confusion_matrix": cm.tolist(), "roc_auc": roc_auc, "pr_auc": pr_auc}


def save_fold_plots(ckpt_dir: Path, fold_idx: int, history: list[dict],
                     y_true: np.ndarray, probs: np.ndarray, n_classes: int,
                     class_names: list[str]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    ax.plot(epochs, [h["val_loss"] for h in history], label="val_loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()
    ax.set_title(f"fold {fold_idx} loss")
    fig.tight_layout(); fig.savefig(ckpt_dir / f"fold{fold_idx}_loss_curve.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h["lr"] for h in history])
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title(f"fold {fold_idx} LR schedule")
    fig.tight_layout(); fig.savefig(ckpt_dir / f"fold{fold_idx}_lr_curve.png", dpi=110)
    plt.close(fig)

    pred = probs.argmax(1)
    cm = confusion_matrix(y_true, pred, labels=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_classes)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n_classes)); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"fold {fold_idx} confusion matrix (best epoch)")
    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(ckpt_dir / f"fold{fold_idx}_confusion_matrix.png", dpi=110)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    try:
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
            ax1.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, probs[:, 1]):.3f}")
            prec, rec, _ = precision_recall_curve(y_true, probs[:, 1])
            ax2.plot(rec, prec, label=f"AP={average_precision_score(y_true, probs[:, 1]):.3f}")
        else:
            y_onehot = np.eye(n_classes)[y_true]
            for c in range(n_classes):
                fpr, tpr, _ = roc_curve(y_onehot[:, c], probs[:, c])
                ax1.plot(fpr, tpr, label=class_names[c])
                prec, rec, _ = precision_recall_curve(y_onehot[:, c], probs[:, c])
                ax2.plot(rec, prec, label=class_names[c])
    except ValueError:
        pass  # a class missing from this fold's val split
    ax1.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC"); ax1.legend(fontsize=8)
    ax2.set_xlabel("recall"); ax2.set_ylabel("precision"); ax2.set_title("PR"); ax2.legend(fontsize=8)
    fig.suptitle(f"fold {fold_idx} ROC / PR curves (best epoch)")
    fig.tight_layout(); fig.savefig(ckpt_dir / f"fold{fold_idx}_roc_pr_curve.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
def run_fold(train_samples, val_samples, args, task: str, classes: list[str],
             device: torch.device, fold_idx: int, ckpt_dir: Path | None) -> dict:
    n_classes = len(classes)
    ytr = encode_labels(train_samples, task)
    yva = encode_labels(val_samples, task)
    mean, std = resolve_normalization_stats(args.backbone, args.modality, args.per_modality_stats)

    bg_swap_arrays = None
    if args.bg_swap_aug:
        from ablation_inputs import build_full_bgpatch_tile_box
        bg_swap_arrays = build_full_bgpatch_tile_box(
            train_samples, img_size=args.img_size, lesion_tile=min(112, args.img_size)
        )

    train_ds = CadSelClassificationDataset(
        train_samples, ytr, img_size=args.img_size, train=True,
        mean=mean, std=std, seed=args.seed, bg_swap_arrays=bg_swap_arrays,
    )
    val_ds = CadSelClassificationDataset(
        val_samples, yva, img_size=args.img_size, train=False, mean=mean, std=std,
    )

    sampler = None
    if args.oversample == "size":
        sampler = ClassificationSizeRepeatFactorSampler(
            train_samples, args.size_rfs_threshold, seed=args.seed
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(2, args.num_workers), pin_memory=device.type == "cuda",
    )

    model = build_model(args.backbone, n_classes, args.freeze, unfreeze_last_n=args.unfreeze_last_n).to(device)
    normalize = make_classification_normalizer(mean, std)

    weight_tensor = None
    seesaw_state = None
    if args.loss == "weighted_ce":
        w = compute_classification_class_weights(ytr, n_classes)
        weight_tensor = torch.tensor(w, dtype=torch.float32, device=device)
    elif args.loss == "seesaw_ce":
        seesaw_state = ClassificationSeesawState(n_classes, p=args.seesaw_p, device=device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = {"bal": -1.0, "acc": 0.0, "epoch": 0}
    best_probs = None
    history = []
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(normalize(images))
                if args.loss == "ce":
                    loss = F.cross_entropy(logits, labels)
                elif args.loss == "weighted_ce":
                    loss = F.cross_entropy(logits, labels, weight=weight_tensor)
                else:
                    loss = seesaw_ce_loss(logits, labels, seesaw_state)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * len(labels)
            train_n += len(labels)
        train_loss = train_loss_sum / max(train_n, 1)

        # Val loss is always plain (unweighted) CE, regardless of --loss --
        # a fair, comparable-across-arms number, and (for seesaw_ce) avoids
        # calling ClassificationSeesawState.update() on validation labels,
        # which would contaminate its training-derived cumulative counts.
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(normalize(images))
                    loss = F.cross_entropy(logits, labels)
                val_loss_sum += float(loss.item()) * len(labels)
                val_n += len(labels)
                all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        val_loss = val_loss_sum / max(val_n, 1)
        probs = np.concatenate(all_probs) if all_probs else np.zeros((0, n_classes))
        pred = probs.argmax(1) if len(probs) else np.zeros(0, dtype=np.int64)
        acc = float((pred == yva).mean()) if len(pred) else 0.0
        bal = balanced_accuracy(yva, pred, n_classes)
        scheduler.step(bal)
        lr = optimizer.param_groups[0]["lr"]

        improved = bal > best["bal"]
        if improved:
            best = {"bal": bal, "acc": acc, "epoch": epoch + 1}
            best_probs = probs
            if ckpt_dir is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "backbone": args.backbone,
                        "task": task,
                        "modality": args.modality,
                        "img_size": args.img_size,
                        "mean": mean,
                        "std": std,
                        "freeze": args.freeze,
                        "fold": fold_idx,
                        "bal": bal,
                    },
                    ckpt_dir / f"fold{fold_idx}_best.pt",
                )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                         "lr": lr, "acc": acc, "bal": bal})
        marker = " *" if improved else ""
        print(f"    fold {fold_idx} epoch {epoch+1:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={lr:.2e}  "
              f"acc={acc:.4f}  bal={bal:.4f}  ({time.time()-epoch_t0:.0f}s){marker}", flush=True)

        if args.early_stop_patience > 0 and (epoch + 1) - best["epoch"] >= args.early_stop_patience:
            break

    extra = compute_extra_metrics(yva, best_probs, n_classes) if best_probs is not None else {
        "confusion_matrix": None, "roc_auc": None, "pr_auc": None,
    }
    if ckpt_dir is not None and best_probs is not None:
        save_fold_plots(ckpt_dir, fold_idx, history, yva, best_probs, n_classes, classes)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"fold": fold_idx, "n_train": len(train_samples), "n_val": len(val_samples),
            "best": {**best, **extra}, "history": history}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=BACKBONES, default="resnet50")
    p.add_argument("--modality", choices=MODALITIES, default="WLE")
    p.add_argument("--task", choices=["binary", "subtype"], default="binary")
    p.add_argument("--freeze", choices=["none", "partial", "head"], default="none")
    p.add_argument("--unfreeze-last-n", type=int, default=4,
                    help="DINOv3 --freeze partial only: how many of the 12 transformer "
                         "blocks (from the end) to unfreeze")
    p.add_argument("--oversample", choices=["none", "size"], default="none")
    p.add_argument("--size-rfs-threshold", type=float, default=0.2)
    p.add_argument("--loss", choices=["ce", "weighted_ce", "seesaw_ce"], default="ce")
    p.add_argument("--seesaw-p", type=float, default=0.8)
    p.add_argument("--bg-swap-aug", action="store_true")
    p.add_argument("--per-modality-stats", action="store_true")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--save-checkpoint", action="store_true", default=True)
    p.add_argument("--no-save-checkpoint", dest="save_checkpoint", action="store_false")
    return p


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = (
        f"{args.backbone}__{args.modality}__{args.task}"
        f"__freeze-{args.freeze}__os-{args.oversample}__loss-{args.loss}"
        f"{'__bgswap' if args.bg_swap_aug else ''}"
        f"{'' if args.lr == 3e-4 else f'__lr{args.lr:g}'}"
        f"{'' if args.unfreeze_last_n == 4 else f'__unfreeze{args.unfreeze_last_n}'}__s{args.seed}"
    )
    out_path = args.output_dir / f"{run_name}.json"
    ckpt_dir = (args.output_dir / run_name) if args.save_checkpoint else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_manifest(args.modality)
    classes = classes_for(args.task)
    folds = patient_level_folds(samples, n_folds=args.n_splits, seed=args.seed)

    print(f"[{run_name}] device={device} {len(samples)} images, {args.n_splits}-fold CV")
    started = time.time()
    fold_results = []
    for k, (train_samples, val_samples) in enumerate(folds):
        t0 = time.time()
        r = run_fold(train_samples, val_samples, args, args.task, classes, device, k, ckpt_dir)
        fold_results.append(r)
        print(f"  fold {k}: bal={r['best']['bal']:.4f} acc={r['best']['acc']:.4f} "
              f"(epoch {r['best']['epoch']}, n_train={r['n_train']}, n_val={r['n_val']}, "
              f"{time.time()-t0:.0f}s)")

        bals = [f["best"]["bal"] for f in fold_results]
        accs = [f["best"]["acc"] for f in fold_results]
        roc_aucs = [f["best"]["roc_auc"] for f in fold_results if f["best"]["roc_auc"] is not None]
        pr_aucs = [f["best"]["pr_auc"] for f in fold_results if f["best"]["pr_auc"] is not None]
        cms = [f["best"]["confusion_matrix"] for f in fold_results if f["best"]["confusion_matrix"] is not None]
        confusion_matrix_sum = np.sum(cms, axis=0).tolist() if cms else None
        result = {
            "run_name": run_name, "backbone": args.backbone, "modality": args.modality,
            "task": args.task, "freeze": args.freeze, "oversample": args.oversample,
            "size_rfs_threshold": args.size_rfs_threshold if args.oversample == "size" else None,
            "loss": args.loss, "seesaw_p": args.seesaw_p if args.loss == "seesaw_ce" else None,
            "bg_swap_aug": args.bg_swap_aug, "per_modality_stats": args.per_modality_stats,
            "img_size": args.img_size, "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size, "n_splits": args.n_splits, "seed": args.seed,
            "n_images": len(samples), "chance": round(1 / len(classes), 4),
            "shortcut_suspect_pending_counterfactual": args.modality == "EUS",
            "folds_done": len(fold_results),
            "balanced_accuracy_mean": round(float(np.mean(bals)), 4),
            "balanced_accuracy_std": round(float(np.std(bals)), 4),
            "accuracy_mean": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "roc_auc_mean": round(float(np.mean(roc_aucs)), 4) if roc_aucs else None,
            "pr_auc_mean": round(float(np.mean(pr_aucs)), 4) if pr_aucs else None,
            "confusion_matrix_class_order": classes,
            "confusion_matrix_sum": confusion_matrix_sum,
            "folds": fold_results,
            "elapsed_sec": round(time.time() - started, 1),
        }
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n[{run_name}] {args.n_splits}-fold balanced accuracy "
          f"{result['balanced_accuracy_mean']:.4f} +/- {result['balanced_accuracy_std']:.4f} "
          f"(chance {result['chance']:.3f})  roc_auc={result['roc_auc_mean']}  "
          f"pr_auc={result['pr_auc_mean']}  freeze={args.freeze} "
          f"({'FROZEN backbone' if args.freeze == 'head' else 'unfrozen/fine-tuned' if args.freeze == 'none' else 'partially unfrozen'})")
    print(f"wrote {out_path}")

    gate = SVM_BASELINE.get((args.backbone, args.modality, args.task))
    if gate is not None:
        verdict = "PASS" if result["balanced_accuracy_mean"] > gate else "FAIL"
        print(f"\nfair-baseline gate ({args.backbone}/{args.modality}/{args.task} vs frozen-SVM "
              f"{gate:.4f}): fine-tuned {result['balanced_accuracy_mean']:.4f} -> {verdict}")
        if verdict == "FAIL":
            print("  FAIL means this is a bug in the pipeline (LR/undertraining/data leak) -- "
                  "do not record this as a result, fix it first.")


if __name__ == "__main__":
    main()
