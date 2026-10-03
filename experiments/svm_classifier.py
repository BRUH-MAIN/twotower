"""SVM classifiers on frozen pretrained embeddings (ResNet-50, DINOv3).

Patient-grouped CV (``probes.grouped_folds``, the same helper the audit's
decodability probes use), so a high score can't be explained by duplicate
frames of the same case landing on both sides of a fold -- matching every
other diagnostic in this audit rather than introducing a new evaluation
convention.

    python -m experiments.svm_classifier --backbone resnet --modality WLE --task binary
    python -m experiments.svm_classifier --backbone dinov3 --modality WLE --task binary

The DINOv3 path needs a Hugging Face account with access granted to the gated
``facebook/dinov3-*`` checkpoints -- see ``experiments/embeddings.py``'s
docstring.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cadsel.data import build_manifest
from embeddings import load_or_extract
from harness import classes_for, encode_labels
from probes import balanced_accuracy, grouped_folds

OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=["resnet", "dinov3"], default="resnet")
    p.add_argument("--dinov3-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--modality", choices=["WLE", "EUS"], default="WLE")
    p.add_argument("--task", choices=["binary", "subtype"], default="binary")
    p.add_argument("--kernel", choices=["linear", "rbf"], default="rbf")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    samples = build_manifest(args.modality)
    classes = classes_for(args.task)
    y = encode_labels(samples, args.task)
    groups = np.array([s.patient_key for s in samples])

    kwargs = {"model_name": args.dinov3_model} if args.backbone == "dinov3" else {}
    X = load_or_extract(samples, args.backbone, args.modality, OUT / "embeddings", **kwargs)
    print(f"  {X.shape[0]} samples, {X.shape[1]}-d embeddings")

    accs, bals, maj_accs = [], [], []
    for tr, te in grouped_folds(groups, args.n_splits, args.seed):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        clf = SVC(kernel=args.kernel, C=args.C, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        accs.append(float((pred == y[te]).mean()))
        bals.append(balanced_accuracy(y[te], pred, len(classes)))
        majority = np.bincount(y[tr], minlength=len(classes)).argmax()
        maj_accs.append(float((y[te] == majority).mean()))

    result = {
        "backbone": args.backbone,
        "dinov3_model": args.dinov3_model if args.backbone == "dinov3" else None,
        "modality": args.modality,
        "task": args.task,
        "kernel": args.kernel,
        "C": args.C,
        "accuracy": round(float(np.mean(accs)), 4),
        "balanced_accuracy": round(float(np.mean(bals)), 4),
        "majority_baseline": round(float(np.mean(maj_accs)), 4),
        "chance_balanced": round(1 / len(classes), 4),
        "n_folds": args.n_splits,
        "embedding_dim": int(X.shape[1]),
    }
    print(json.dumps(result, indent=2))

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"svm_{args.backbone}_{args.modality}_{args.task}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
