"""Train and evaluate the patient-level WLE+EUS fusion head.

Extracts frame descriptors once from a frozen Stage 1 detector, then trains the
fusion head under 5-fold patient-level cross-validation for each ablation.
Cross-validation rather than a single split because only 421 patients have both
modalities and Leiomyoma has 14 of them — a single 20% split leaves about three
Leiomyoma patients in validation, which cannot support a claim.

The reported metric is patient-level balanced accuracy / macro-F1. It is **not**
comparable to the AP@0.5 of Table 4/5: this scores a patient, not an image.

    python -m cadsel.train_fusion --checkpoint runs/joint_film__..__multiclass/best_model.pth
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import MODALITY_NAMES, MODEL_REGISTRY, PROJECT_ROOT, TASKS
from .data import build_joint_manifest, joint_strata_key, patient_level_folds
from .fusion import (
    FusionHead,
    PatientFeatures,
    extract_features,
    load_features,
    make_batch,
    patient_classes,
    patient_label,
    save_features,
)
from .models import build_detection_model, build_joint_detection_model

# (name, modalities used, modality dropout). The unimodal rows are the controls:
# a fused number only means something next to them.
ABLATIONS = [
    ("wle_only", ["WLE"], 0.0),
    ("eus_only", ["EUS"], 0.0),
    ("fused", MODALITY_NAMES, 0.0),
    ("fused_moddrop", MODALITY_NAMES, 0.3),
]


def load_stage1(checkpoint: Path, device: torch.device):
    """Rebuild the Stage 1 detector and load its weights, frozen."""
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_type = blob["model_type"]
    task = blob["task"]
    conditioning = blob.get("conditioning")

    if conditioning:
        model = build_joint_detection_model(
            model_type, task, conditioning=conditioning, pretrained=False, verbose=False
        )
    else:
        model = build_detection_model(model_type, task, pretrained=False)

    model.load_state_dict(blob["model_state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)

    img_size = blob.get("img_size") or MODEL_REGISTRY[model_type]["default_img_size"]
    return model, {
        "model_type": model_type,
        "detector_task": task,
        "conditioning": conditioning,
        "img_size": img_size,
        "input_mode": blob.get("input_mode", "reference"),
        "per_modality_norm": blob.get("per_modality_norm", conditioning is not None),
    }


def load_stage1_separate(checkpoint_wle: Path, checkpoint_eus: Path, device: torch.device):
    """Two independent single-modality detectors -- e.g. one with an
    ultrasound-pretrained backbone for EUS (trained via
    ``cadsel.train --modality EUS --backbone-weights ...``) alongside an
    ordinary WLE detector -- instead of one shared Stage 1 joint/FiLM model.

    Both must share ``img_size``/``input_mode``: ``extract_features`` builds
    one dataset/transform for every frame regardless of which model processes
    it, so the two detectors need to agree on what that transform should do.
    """
    models: dict[str, torch.nn.Module] = {}
    specs: dict[str, dict] = {}
    for modality, checkpoint in (("WLE", checkpoint_wle), ("EUS", checkpoint_eus)):
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_type = blob["model_type"]
        task = blob["task"]
        model = build_detection_model(model_type, task, pretrained=False)
        model.load_state_dict(blob["model_state_dict"])
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval().to(device)
        models[modality] = model
        specs[modality] = {
            "model_type": model_type,
            "img_size": blob.get("img_size") or MODEL_REGISTRY[model_type]["default_img_size"],
            "input_mode": blob.get("input_mode", "reference"),
        }

    img_sizes = {s["img_size"] for s in specs.values()}
    input_modes = {s["input_mode"] for s in specs.values()}
    if len(img_sizes) > 1 or len(input_modes) > 1:
        raise SystemExit(
            f"--checkpoint-wle and --checkpoint-eus must share img_size and "
            f"input_mode for feature extraction to be well-defined "
            f"(got img_sizes={img_sizes}, input_modes={input_modes})"
        )

    combined_spec = {
        "model_type": "+".join(f"{m}={s['model_type']}" for m, s in sorted(specs.items())),
        "detector_task": None,
        "conditioning": None,
        "img_size": img_sizes.pop(),
        "input_mode": input_modes.pop(),
        # Always per-modality here: each detector already has its own
        # domain-specific backbone and should see its own modality's real
        # normalisation statistics, not a value inherited from FiLM logic
        # that doesn't apply to two independent models.
        "per_modality_norm": True,
    }
    return models, combined_spec


def train_fold(
    train_patients: list[PatientFeatures],
    val_patients: list[PatientFeatures],
    task: str,
    use_modalities: list[str],
    modality_dropout: float,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
    device: torch.device | None = None,
    fusion: str = "gated",
    capture_predictions: bool = False,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    early_stop_patience: int = 0,
    test_patients: list[PatientFeatures] | None = None,
) -> dict:
    """One fold. Trains in RAM — the features are already extracted.

    ``capture_predictions``, when true, snapshots the softmax probabilities and
    argmax predictions of the validation batch alongside the scalar metrics,
    every time a new best epoch is found (``best["pred"]``, ``best["probs"]``).
    Reuses the same forward pass already needed for ``score()`` -- no extra
    model call. The order matches ``val_patients`` (and therefore ``val_batch``,
    built from it in the same order), so callers can zip the two directly.

    ``weight_decay``/``dropout`` default to the values every prior confirmed
    result in this project was trained with (1e-4 / 0.3) -- pass higher values
    to regularize harder. ``early_stop_patience`` defaults to 0 (off, the
    original unconditional-``epochs`` behaviour); a positive value stops the
    fold once ``val_bal`` hasn't improved for that many epochs, mirroring
    ``experiments/src/34_joint_fusion_train.py``'s ``--early-stop-patience``.
    Per-epoch ``train_loss``/``val_loss``/``val_bal`` are always recorded in
    the returned ``history`` list -- previously only the best epoch's scalar
    metrics were kept, which meant no loss curve could ever be reconstructed
    after the fact without retraining.

    ``test_patients``, when given, is scored every epoch but never used for
    selection: ``val_patients`` alone picks the best epoch, and that epoch's
    test metrics are returned as ``best["test"]`` (with ``pred``/``probs``
    when ``capture_predictions``). Without it, the returned metrics are the
    max over epochs of the same set they were measured on -- an optimistic
    estimate.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    classes = patient_classes(task)
    n_classes = len(classes)
    dim = next(
        arr.shape[1] for p in train_patients for arr in p.frames.values() if len(arr)
    )

    model = FusionHead(
        dim, n_classes, modality_dropout=modality_dropout, fusion=fusion, dropout=dropout
    ).to(device)

    y_train = np.array([
        classes.index(patient_label(p.subtype, task)) for p in train_patients
    ])
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    # Same inverse-frequency weighting as experiments/harness.train_eval, so the
    # rare subtypes are not simply ignored.
    weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    weights = weights / max(weights.sum(), 1e-9) * n_classes
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    val_batch, y_val = make_batch(val_patients, task, device,
                                  use_modalities=use_modalities)
    y_val_np = y_val.cpu().numpy()
    test_batch = None
    if test_patients is not None:
        test_batch, y_test = make_batch(test_patients, task, device,
                                        use_modalities=use_modalities)
        y_test_np = y_test.cpu().numpy()

    best = {"bal": -1.0, "acc": 0.0, "macro_f1": 0.0, "epoch": 0}
    history = []
    order = np.arange(len(train_patients))

    for epoch in range(epochs):
        model.train()
        rng.shuffle(order)
        train_losses = []
        for k in range(0, len(order), batch_size):
            picked = [train_patients[i] for i in order[k:k + batch_size]]
            if len(picked) < 2:
                continue
            batch, y = make_batch(picked, task, device,
                                  use_modalities=use_modalities, rng=rng)
            loss = criterion(model(batch), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(val_batch)
            val_loss = criterion(logits, y_val).item()
        pred = logits.argmax(1).cpu().numpy()
        stats = score(pred, y_val_np, n_classes)
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_loss": val_loss,
            "val_bal": stats["bal"],
        })
        if test_batch is not None:
            with torch.no_grad():
                test_logits = model(test_batch)
            test_stats = score(test_logits.argmax(1).cpu().numpy(), y_test_np, n_classes)
            history[-1]["test_bal"] = test_stats["bal"]
        if stats["bal"] > best["bal"]:
            best = {**stats, "epoch": epoch + 1}
            if capture_predictions:
                probs = torch.softmax(logits, dim=1)
                best["pred"] = probs.argmax(1).cpu().numpy().tolist()
                best["probs"] = probs.detach().cpu().numpy().tolist()
            if test_batch is not None:
                best["test"] = dict(test_stats)
                if capture_predictions:
                    test_probs = torch.softmax(test_logits, dim=1)
                    best["test"]["pred"] = test_probs.argmax(1).cpu().numpy().tolist()
                    best["test"]["probs"] = test_probs.detach().cpu().numpy().tolist()

        if early_stop_patience > 0 and (epoch + 1) - best["epoch"] >= early_stop_patience:
            break

    best["n_train"] = len(train_patients)
    best["n_val"] = len(val_patients)
    if test_patients is not None:
        best["n_test"] = len(test_patients)
    best["history"] = history
    return best


def score(pred: np.ndarray, truth: np.ndarray, n_classes: int) -> dict:
    """Accuracy, balanced accuracy and macro-F1 over the classes present."""
    acc = float((pred == truth).mean())
    recalls, f1s = [], []
    for c in range(n_classes):
        in_class = truth == c
        if not in_class.any():
            continue
        recall = float((pred[in_class] == c).mean())
        predicted = pred == c
        precision = float((truth[predicted] == c).mean()) if predicted.any() else 0.0
        recalls.append(recall)
        f1s.append(0.0 if (precision + recall) == 0
                   else 2 * precision * recall / (precision + recall))
    return {
        "acc": round(acc, 4),
        "bal": round(float(np.mean(recalls)), 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Stage 1 joint/FiLM detector checkpoint (best_model.pth). "
                   "Mutually exclusive with --checkpoint-wle/--checkpoint-eus.")
    p.add_argument("--checkpoint-wle", type=Path, default=None,
                   help="separate WLE-only detector checkpoint, paired with "
                   "--checkpoint-eus instead of --checkpoint -- two independent "
                   "backbones instead of one shared conditioned model.")
    p.add_argument("--checkpoint-eus", type=Path, default=None,
                   help="separate EUS-only detector checkpoint (e.g. trained with "
                   "--backbone-weights for ultrasound-domain pretraining), paired "
                   "with --checkpoint-wle.")
    p.add_argument("--task", choices=TASKS + ["subtype"], default="subtype",
                   help="'subtype' is the 5-way patient-level task (NET_G1/G2 kept "
                        "separate); 'binary' is NET vs non-NET")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paired-only", action="store_true",
                   help="restrict to the 421 patients that have both modalities")
    p.add_argument("--fusion", choices=["gated", "symmetric"], default="gated",
                   help="'gated' is the original GatedFusion ((1-gate)*a always "
                        "falls back to WLE specifically); 'symmetric' is "
                        "SymmetricCrossAttentionFusion, a modality-symmetric "
                        "alternative (see cadsel/fusion.py)")
    p.add_argument("--features", type=Path, default=None,
                   help="cache path for the extracted features "
                        "(default: alongside the checkpoint)")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "results" / "fusion.json")
    p.add_argument("--predictions-output", type=Path, default=None,
                   help="where to write per-patient out-of-fold predictions "
                        "(default: alongside --output, named "
                        "'<output-stem>_predictions.json')")
    return p


def run(args) -> dict:
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    task = "multiclass" if args.task == "subtype" else args.task
    # `patient_classes` keys off the same string the CLI takes; 'subtype' and
    # 'multiclass' both mean the 5-way patient task here.
    head_task = "binary" if task == "binary" else "subtype"

    samples = build_joint_manifest(data_root=args.data_root)

    separate = bool(args.checkpoint_wle or args.checkpoint_eus)
    if args.checkpoint and separate:
        raise SystemExit(
            "pass either --checkpoint (joint model) or --checkpoint-wle/"
            "--checkpoint-eus (two separate models), not both"
        )
    if separate and not (args.checkpoint_wle and args.checkpoint_eus):
        raise SystemExit("--checkpoint-wle and --checkpoint-eus must both be given")
    if not args.checkpoint and not separate:
        raise SystemExit("pass --checkpoint, or both --checkpoint-wle and --checkpoint-eus")

    if separate:
        features_path = args.features or (
            PROJECT_ROOT / "runs"
            / f"patient_features__{args.checkpoint_wle.parent.name}__{args.checkpoint_eus.parent.name}.npz"
        )
    else:
        features_path = args.features or args.checkpoint.parent / "patient_features.npz"

    if features_path.exists() and not args.force_extract:
        print(f"loading cached features from {features_path}")
        features = load_features(features_path)
    else:
        if separate:
            model, spec = load_stage1_separate(args.checkpoint_wle, args.checkpoint_eus, device)
            print(f"extracting features with separate backbones ({spec['model_type']}, "
                  f"img_size={spec['img_size']}) over {len(samples)} frames ...")
        else:
            model, spec = load_stage1(args.checkpoint, device)
            print(f"extracting features with {spec['model_type']} "
                  f"(conditioning={spec['conditioning']}, img_size={spec['img_size']}) "
                  f"over {len(samples)} frames ...")
        started = time.time()
        features = extract_features(
            model, samples, device,
            input_mode=spec["input_mode"],
            per_modality_norm=spec["per_modality_norm"],
            img_size=spec["img_size"],
        )
        print(f"  {len(features)} patients in {time.time() - started:.0f}s")
        save_features(features_path, features)
        print(f"  cached to {features_path}")
        del model
        torch.cuda.empty_cache()

    if args.paired_only:
        features = {
            k: v for k, v in features.items()
            if all(v.has(m) for m in MODALITY_NAMES)
        }
        print(f"paired-only: {len(features)} patients")

    # Fold assignment reuses the detection splitter so the fusion protocol and the
    # detection protocol partition patients the same way.
    keep = set(features)
    folds = patient_level_folds(
        [s for s in samples if s.patient_key in keep],
        n_folds=args.folds, seed=args.seed, strata_key=joint_strata_key,
    )

    classes = patient_classes(head_task)
    results: dict[str, dict] = {}
    predictions: list[dict] = []
    for name, modalities, moddrop in ABLATIONS:
        per_fold = []
        for i, (train_samples, val_samples) in enumerate(folds):
            train_keys = {s.patient_key for s in train_samples}
            val_keys = {s.patient_key for s in val_samples}
            assert not (train_keys & val_keys), "fold leaks patients"

            train_p = [features[k] for k in sorted(train_keys) if k in features]
            val_p = [features[k] for k in sorted(val_keys) if k in features]
            if name == "eus_only":
                # A patient with no EUS frames carries no signal for this arm and
                # would only add label noise to the validation score.
                train_p = [p for p in train_p if p.has("EUS")]
                val_p = [p for p in val_p if p.has("EUS")]
            if len(val_p) < 2 or len(train_p) < 10:
                continue

            fold_result = train_fold(
                train_p, val_p, head_task, list(modalities), moddrop,
                epochs=args.epochs, seed=args.seed + i, device=device,
                fusion=args.fusion, capture_predictions=True,
            )

            # `val_p` is exactly the order `train_fold` built `val_batch` from
            # (`sorted(val_keys)`, filtered), so `pred`/`probs` line up with it
            # positionally -- zip is safe. Pop the raw per-epoch arrays out of
            # the fold dict before it goes into results/fusion.json: they only
            # belong in the dedicated predictions file below.
            pred = fold_result.pop("pred", None)
            probs_list = fold_result.pop("probs", None)
            if pred is not None and probs_list is not None:
                for patient, pred_idx, patient_probs in zip(val_p, pred, probs_list):
                    y_true = patient_label(patient.subtype, head_task)
                    y_pred = classes[pred_idx]
                    predictions.append({
                        "ablation": name,
                        "fold": i,
                        "patient_key": patient.patient_key,
                        "subtype": patient.subtype,
                        "center": patient.center,
                        "paired": all(patient.has(m) for m in MODALITY_NAMES),
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "correct": y_true == y_pred,
                        "probs": {c: patient_probs[ci] for ci, c in enumerate(classes)},
                    })

            per_fold.append(fold_result)

        agg = {
            metric: {
                "mean": round(float(np.mean([f[metric] for f in per_fold])), 4),
                "std": round(float(np.std([f[metric] for f in per_fold])), 4),
            }
            for metric in ("acc", "bal", "macro_f1")
        }
        results[name] = {"folds": per_fold, **agg}
        print(f"  {name:<16} bal {agg['bal']['mean']:.4f} +/- {agg['bal']['std']:.4f}   "
              f"macroF1 {agg['macro_f1']['mean']:.4f} +/- {agg['macro_f1']['std']:.4f}   "
              f"({len(per_fold)} folds)")

    payload = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else {
            "wle": str(args.checkpoint_wle), "eus": str(args.checkpoint_eus),
        },
        "task": args.task,
        "folds": args.folds,
        "paired_only": args.paired_only,
        "fusion": args.fusion,
        "classes": classes,
        "seed": args.seed,
        "protocol": "patient-level cross-validation; NOT comparable to Table 4/5 AP",
        "ablations": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")

    predictions_output = args.predictions_output or (
        args.output.parent / f"{args.output.stem}_predictions.json"
    )
    predictions_payload = {
        "checkpoint": payload["checkpoint"],
        "task": args.task,
        "fusion": args.fusion,
        "classes": classes,
        "seed": args.seed,
        "folds": args.folds,
        "note": (
            "One out-of-fold prediction per patient per ablation (each patient "
            "is in validation for exactly one fold). 'paired' is true for "
            "patients with both WLE and EUS frames -- restrict to those before "
            "comparing 'fused'/'fused_moddrop' against 'wle_only', since "
            "'wle_only' also includes WLE-only patients where fusion cannot "
            "structurally differ from wle_only."
        ),
        "predictions": predictions,
    }
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.write_text(json.dumps(predictions_payload, indent=2), encoding="utf-8")
    print(f"wrote {predictions_output}")
    return payload


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
