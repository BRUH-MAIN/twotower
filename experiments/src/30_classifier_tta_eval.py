"""Experiment 30 (Track A follow-up) — test-time augmentation for a
fine-tuned classifier checkpoint, compared against a plain single-view eval
computed through the identical forward-pass code (not just the number
recorded during training) so the plain-vs-TTA delta isn't confounded by any
code-path difference. Two modes:

``--mode flip`` (default) averages a horizontal-flip pair.

``--mode 5crop`` (Tier 3c follow-up) decodes each image ~15% oversized and
averages predictions over 4 corner crops + 1 center crop at the model's own
``img_size`` -- distinct from flip-TTA in that it perturbs *scale/framing*,
not just left-right orientation. Expected (per FINDINGS.md §12.4) to
largely replicate flip-TTA's null result, since `RandomResizedCrop(scale=
(0.7,1.0))` already gives scale/crop augmentation during training the same
way `RandomHorizontalFlip` already covered flip-TTA -- included for
completeness, not because a different outcome is likely.

    python experiments/src/30_classifier_tta_eval.py --run-name \\
        efficientnet_b0__WLE__subtype__freeze-none__os-none__loss-weighted_ce__s42
    python experiments/src/30_classifier_tta_eval.py --mode 5crop --run-name ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from cadsel.classification_data import make_classification_normalizer
from cadsel.data import build_manifest, patient_level_folds
from finetune_classifier import build_model
from harness import classes_for, encode_labels
from probes import balanced_accuracy

RESULTS_DIR = Path("experiments/results/classifier_runs")
OUT_DIR = Path("experiments/results")


def decode_fold(val_samples, img_size: int) -> np.ndarray:
    arrs = np.zeros((len(val_samples), img_size, img_size, 3), dtype=np.uint8)
    for i, s in enumerate(val_samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
            arrs[i] = np.asarray(im, dtype=np.uint8)
    return arrs


def decode_fold_oversized(val_samples, oversize: int) -> np.ndarray:
    arrs = np.zeros((len(val_samples), oversize, oversize, 3), dtype=np.uint8)
    for i, s in enumerate(val_samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB").resize((oversize, oversize), Image.BILINEAR)
            arrs[i] = np.asarray(im, dtype=np.uint8)
    return arrs


def five_crops(arr: np.ndarray, img_size: int) -> list[np.ndarray]:
    """4 corner crops + 1 center crop of ``img_size`` from an oversized
    ``(N, oversize, oversize, 3)`` array."""
    oversize = arr.shape[1]
    pad = oversize - img_size
    half = pad // 2
    return [
        arr[:, :img_size, :img_size, :],                        # top-left
        arr[:, :img_size, pad:pad + img_size, :],                # top-right
        arr[:, pad:pad + img_size, :img_size, :],                # bottom-left
        arr[:, pad:pad + img_size, pad:pad + img_size, :],       # bottom-right
        arr[:, half:half + img_size, half:half + img_size, :],   # center
    ]


def eval_tta_fold(ck: dict, val_samples, y: np.ndarray, n_classes: int,
                   device: torch.device, mode: str = "flip", batch: int = 32) -> tuple[float, float]:
    model = build_model(ck["backbone"], n_classes, ck["freeze"]).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])
    normalize = make_classification_normalizer(ck["mean"], ck["std"])
    img_size = ck["img_size"]

    def predict_probs(batch_arr: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(batch_arr.transpose(0, 3, 1, 2))).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(normalize(x))
        return torch.softmax(logits.float(), dim=1).cpu().numpy()

    if mode == "flip":
        arrs = decode_fold(val_samples, img_size)
        probs_plain, probs_tta = [], []
        for k in range(0, len(arrs), batch):
            chunk = arrs[k:k + batch]
            p_plain = predict_probs(chunk)
            p_flip = predict_probs(chunk[:, :, ::-1, :])
            probs_plain.append(p_plain)
            probs_tta.append((p_plain + p_flip) / 2.0)
        probs_plain = np.concatenate(probs_plain)
        probs_tta = np.concatenate(probs_tta)
    elif mode == "5crop":
        oversize = int(round(img_size * 1.15))
        big = decode_fold_oversized(val_samples, oversize)
        crops = five_crops(big, img_size)
        plain_arr = decode_fold(val_samples, img_size)  # true single-view baseline, not a crop
        probs_plain, probs_5crop = [], []
        for k in range(0, len(plain_arr), batch):
            p_plain = predict_probs(plain_arr[k:k + batch])
            probs_plain.append(p_plain)
            crop_probs = [predict_probs(c[k:k + batch]) for c in crops]
            probs_5crop.append(np.mean(crop_probs, axis=0))
        probs_plain = np.concatenate(probs_plain)
        probs_tta = np.concatenate(probs_5crop)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    bal_plain = balanced_accuracy(y, probs_plain.argmax(1), n_classes)
    bal_tta = balanced_accuracy(y, probs_tta.argmax(1), n_classes)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return bal_plain, bal_tta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True, help="matches experiments/results/classifier_runs/<run-name>.json")
    p.add_argument("--mode", choices=["flip", "5crop"], default="flip")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = json.loads((RESULTS_DIR / f"{args.run_name}.json").read_text())
    modality, task = run["modality"], run["task"]
    classes = classes_for(task)
    n_classes = len(classes)

    samples = build_manifest(modality)
    folds = patient_level_folds(samples, n_folds=run["n_splits"], seed=run["seed"])

    plains, ttas = [], []
    for f in run["folds"]:
        k = f["fold"]
        ckpt_path = RESULTS_DIR / args.run_name / f"fold{k}_best.pt"
        if not ckpt_path.exists():
            print(f"  missing {ckpt_path}, skipping fold {k}")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        _, val_samples = folds[k]
        y = encode_labels(val_samples, task)
        bal_plain, bal_tta = eval_tta_fold(ck, val_samples, y, n_classes, device, mode=args.mode)
        print(f"  fold {k}: plain={bal_plain:.4f}  tta={bal_tta:.4f}  delta={bal_tta - bal_plain:+.4f}")
        plains.append(bal_plain)
        ttas.append(bal_tta)

    plain_mean, plain_std = float(np.mean(plains)), float(np.std(plains))
    tta_mean, tta_std = float(np.mean(ttas)), float(np.std(ttas))
    delta = tta_mean - plain_mean
    print(f"\n[{args.run_name}] mode={args.mode}  plain {plain_mean:.4f}+/-{plain_std:.4f}  "
          f"tta {tta_mean:.4f}+/-{tta_std:.4f}  delta {delta:+.4f}")

    out = {
        "run_name": args.run_name, "mode": args.mode, "modality": modality, "task": task,
        "plain_mean": plain_mean, "plain_std": plain_std,
        "tta_mean": tta_mean, "tta_std": tta_std, "delta": delta,
        "per_fold": [{"fold": i, "plain": pl, "tta": tt} for i, (pl, tt) in enumerate(zip(plains, ttas))],
    }
    out_path = OUT_DIR / f"30_classifier_tta__{args.mode}__{args.run_name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
