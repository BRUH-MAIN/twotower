"""Experiment 27 (Track A, Phase 6) — MANDATORY counterfactual gate for any
classification-accuracy win: does it survive lesion removal, or is it
shortcut-reading? Scores the actual fine-tuned ``finetune_classifier.py``
checkpoint (not a frozen-embedding SVM) on
``experiments/ablation_inputs.py::build_five_variants``'s 5 variants
(full/lesion/bg_mask/bg_patch/boxmap) -- the same protocol and retention
formula as ``17_svm_counterfactual.py`` / Experiment 3.

    python experiments/src/27_classifier_counterfactual.py \\
        --run-name resnet50__WLE__binary__freeze-none__os-none__loss-ce__s42

Gate (FINDINGS.md discipline, mirroring the DINOv3-EUS precedent in §5e): if
``bg_patch``/``boxmap`` retention on EUS is >= the win's own retention on
the full image (survives or grows), the gain is shortcut-driven -- do not
record it as a validated classification improvement. On WLE, only record a
win if lesion-removal retention is comparable to or lower than the plain-CE
baseline's own retention (i.e. the new lever isn't *adding* shortcut
reliance on top of whatever the baseline already has).
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

from ablation_inputs import FIVE_VARIANTS, build_five_variants
from cadsel.classification_data import make_classification_normalizer
from cadsel.data import build_manifest, patient_level_folds
from finetune_classifier import build_model
from harness import classes_for, encode_labels
from probes import balanced_accuracy

RESULTS_DIR = Path("experiments/results/classifier_runs")
OUT_DIR = Path("experiments/results")


def score_variant(model, arr: np.ndarray, y: np.ndarray, n_classes: int,
                   normalize, device, batch: int = 32) -> tuple[float, float]:
    model.eval()
    preds = []
    with torch.no_grad():
        for k in range(0, len(arr), batch):
            chunk = arr[k:k + batch]
            x = torch.from_numpy(chunk.transpose(0, 3, 1, 2)).to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(normalize(x))
            preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    acc = float((pred == y).mean())
    bal = balanced_accuracy(y, pred, n_classes)
    return acc, bal


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True, help="matches experiments/results/classifier_runs/<run-name>.json")
    p.add_argument("--baseline-run-name", default=None,
                    help="plain-CE baseline run to compare WLE retention against "
                         "(required to apply the WLE gate; omit to skip that comparison)")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_run(run_name: str) -> dict:
        path = RESULTS_DIR / f"{run_name}.json"
        if not path.exists():
            raise SystemExit(f"no such run: {path}")
        return json.loads(path.read_text())

    def counterfactual_for(run_name: str) -> dict:
        run = load_run(run_name)
        modality, task = run["modality"], run["task"]
        classes = classes_for(task)
        n_classes = len(classes)
        chance = 1 / n_classes

        samples = build_manifest(modality)
        folds = patient_level_folds(samples, n_folds=run["n_splits"], seed=run["seed"])

        per_fold = []
        for f in run["folds"]:
            k = f["fold"]
            ckpt_path = RESULTS_DIR / run_name / f"fold{k}_best.pt"
            if not ckpt_path.exists():
                print(f"  missing {ckpt_path}, skipping fold {k}")
                continue
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            _, val_samples = folds[k]
            y = encode_labels(val_samples, task)

            print(f"  fold {k}: building 5 counterfactual variants over {len(val_samples)} images ...")
            variants = build_five_variants(val_samples, img_size=ck["img_size"])

            model = build_model(ck["backbone"], n_classes, ck["freeze"]).to(device)
            model.load_state_dict(ck["model_state_dict"])
            normalize = make_classification_normalizer(ck["mean"], ck["std"])

            row = {"fold": k}
            for v in FIVE_VARIANTS:
                acc, bal = score_variant(model, variants[v], y, n_classes, normalize, device)
                row[v] = bal
            per_fold.append(row)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        mean_row = {"run_name": run_name, "modality": modality, "task": task, "chance": chance}
        for v in FIVE_VARIANTS:
            vals = [r[v] for r in per_fold]
            mean_row[v] = float(np.mean(vals))
        denom = mean_row["full"] - chance
        for v in ("lesion", "bg_mask", "bg_patch", "boxmap"):
            mean_row[f"ret_{v}"] = (mean_row[v] - chance) / denom if denom > 1e-6 else None
        mean_row["per_fold"] = per_fold
        return mean_row

    print(f"=== target run: {args.run_name} ===")
    target = counterfactual_for(args.run_name)

    baseline = None
    if args.baseline_run_name:
        print(f"\n=== baseline run: {args.baseline_run_name} ===")
        baseline = counterfactual_for(args.baseline_run_name)

    print("\n" + "=" * 100)
    print("COUNTERFACTUAL CHECK -- fine-tuned classifier, balanced accuracy (retention vs full in brackets)")
    print("=" * 100)
    hdr = (f"{'run':<55}{'chance':>7}{'full':>7}{'lesion':>16}"
           f"{'bg_mask':>16}{'bg_patch':>16}{'boxmap':>16}")
    print(hdr); print("-" * len(hdr))

    def print_row(row):
        def cell(name):
            r = row.get(f"ret_{name}")
            return f"{row[name]:.3f} ({r:>5.0%})" if name in row and r is not None else "n/a"
        print(f"{row['run_name']:<55}{row['chance']:>7.3f}{row['full']:>7.3f}"
              f"{cell('lesion'):>16}{cell('bg_mask'):>16}{cell('bg_patch'):>16}{cell('boxmap'):>16}")

    print_row(target)
    if baseline:
        print_row(baseline)

    out = {"target": target, "baseline": baseline}
    out_path = OUT_DIR / f"27_classifier_counterfactual__{args.run_name}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    modality = target["modality"]
    if modality == "EUS":
        ret_bgp, ret_box = target.get("ret_bg_patch"), target.get("ret_boxmap")
        print(f"EUS gate: bg_patch/boxmap retention must be BELOW the win's own retention "
              f"on the full image to be trusted (mirrors the DINOv3-EUS shortcut precedent).")
        print(f"  ret_bg_patch={ret_bgp}  ret_boxmap={ret_box}")
        if ret_bgp is not None and ret_bgp >= 1.0 - 1e-6:
            print("  -> SHORTCUT-DRIVEN: bg_patch retention survives/grows on the full-image "
                  "signal. Do NOT record this as a validated classification improvement.")
        elif ret_bgp is not None:
            print(f"  -> retention {ret_bgp:.0%} < 100%: consistent with genuine lesion signal, "
                  f"not purely shortcut-driven. Still weigh against the baseline comparison below.")
    else:
        print("WLE gate: lesion-removal retention should be comparable to or lower than the "
              "plain-CE baseline's own retention (the new lever must not be adding shortcut "
              "reliance on top of the baseline).")
        if baseline is not None:
            for v in ("ret_bg_patch", "ret_boxmap"):
                tv, bv = target.get(v), baseline.get(v)
                if tv is not None and bv is not None:
                    verdict = "OK (not more shortcut-reliant)" if tv <= bv + 0.05 else "WARN (more shortcut-reliant than baseline)"
                    print(f"  {v}: target={tv:.0%}  baseline={bv:.0%}  -> {verdict}")
        else:
            print("  no --baseline-run-name given -- cannot apply the WLE gate quantitatively; "
                  "inspect ret_bg_patch/ret_boxmap above manually before recording this as confirmed.")


if __name__ == "__main__":
    main()
