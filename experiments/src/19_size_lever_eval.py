# %% [markdown]
# # Experiment 19 — Does size-aware oversampling close the small-lesion AP gap?
#
# Experiment 9 found AP@0.5 by lesion-size tertile (WLE binary) is
# 0.471 -> 0.663 -> 0.755 -- small lesions are near-unusable, flagged as the
# most clinically important failure mode (small lesions are exactly what
# endoscopists miss). Lead 4 combines the one confirmed-positive detector
# lever (multiscale training, +0.028 AP at 3 seeds) with a new
# `SizeRepeatFactorSampler` (`--size-rfs-threshold`, bucketing by lesion-size
# tertile instead of class) to target the small-lesion tertile directly.
#
# This reuses Experiment 9's `box_area_frac`/`score_subset` tertile-band
# evaluation (09 itself is untouched) against the new checkpoints, reporting
# mean +/- std per arm across 3 seeds -- the deliverable is whether
# tertile-1 (small-lesion) AP improves, not just overall AP.

# %%
import sys, json
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
import torch

from cadsel.config import TASK_CLASSES
from cadsel.data import (
    CadSelDetectionDataset, build_manifest, collate_fn, make_normalizer,
    read_yolo_label,
)
from cadsel.engine import evaluate
from cadsel.models import build_detection_model, configure_transform_for_input_mode
from torch.utils.data import DataLoader

OUT = Path("experiments/results")
RUNS = Path("runs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

# (run_name, modality, task, arm) -- "arm" groups seeds for the mean +/- std verdict.
RUN_NAMES = [
    ("size_rfs_ms_s42__WLE__multiclass", "WLE", "multiclass", "size_rfs_ms"),
    ("size_rfs_ms_s123__WLE__multiclass", "WLE", "multiclass", "size_rfs_ms"),
    ("size_rfs_ms_s7__WLE__multiclass", "WLE", "multiclass", "size_rfs_ms"),
    # multiscale-alone control -- already on disk (Tier-1 confirmation grid), reused.
    ("tier1_multiscale_only_s42", "WLE", "multiclass", "ms_only"),
    ("tier1_multiscale_only_s123", "WLE", "multiclass", "ms_only"),
    ("tier1_multiscale_only_s7", "WLE", "multiclass", "ms_only"),
    ("ms_only_s42__WLE__binary", "WLE", "binary", "ms_only"),
    ("ms_only_s123__WLE__binary", "WLE", "binary", "ms_only"),
    ("ms_only_s7__WLE__binary", "WLE", "binary", "ms_only"),
    ("size_rfs_ms_s42__WLE__binary", "WLE", "binary", "size_rfs_ms"),
    ("size_rfs_ms_s123__WLE__binary", "WLE", "binary", "size_rfs_ms"),
    ("size_rfs_ms_s7__WLE__binary", "WLE", "binary", "size_rfs_ms"),
    # 2 more seeds each arm, added to settle the binary-task lever (was
    # directionally positive, didn't clear its floor at 3 seeds -- FINDINGS.md §8 item 11).
    ("ms_only_s456__WLE__binary", "WLE", "binary", "ms_only"),
    ("ms_only_s789__WLE__binary", "WLE", "binary", "ms_only"),
    ("size_rfs_ms_s456__WLE__binary", "WLE", "binary", "size_rfs_ms"),
    ("size_rfs_ms_s789__WLE__binary", "WLE", "binary", "size_rfs_ms"),
    # auto-anchors + size-RFS + multiscale, clean 3-way combo (no IoU/class-RFS
    # confounds, unlike the diluted tier1_all 5-lever stack) -- multiclass only.
    ("size_rfs_ms_anchors_s42__WLE__multiclass", "WLE", "multiclass", "size_rfs_ms_anchors"),
    ("size_rfs_ms_anchors_s123__WLE__multiclass", "WLE", "multiclass", "size_rfs_ms_anchors"),
    ("size_rfs_ms_anchors_s7__WLE__multiclass", "WLE", "multiclass", "size_rfs_ms_anchors"),
    # size-RFS threshold sweep, single seed (42) screen only -- each arm name
    # encodes its own threshold so the verdict block below can compare them
    # directly against size_rfs_ms's own threshold-0.2, seed-42 run.
    ("size_rfs_ms_thresh01_s42__WLE__multiclass", "WLE", "multiclass", "size_rfs_thresh01"),
    ("size_rfs_ms_thresh03_s42__WLE__multiclass", "WLE", "multiclass", "size_rfs_thresh03"),
    ("size_rfs_ms_thresh04_s42__WLE__multiclass", "WLE", "multiclass", "size_rfs_thresh04"),
]


# %%
def box_area_frac(sample) -> float:
    """Largest annotated box as a fraction of frame area -- resolution-independent.
    Matches cadsel.data.box_area_frac / 09_size_matched_eval.py's own copy."""
    best = 0.0
    for _c, _cx, _cy, bw, bh in read_yolo_label(sample.label):
        best = max(best, float(bw * bh))
    return best


def load_val_samples(run_name: str, modality: str) -> list:
    """Reproduce the exact validation set a run used, from its own split.json --
    each run in this grid has its own --seed, hence its own patient_level_split,
    so a single fixed SEED (09_size_matched_eval.py's approach, valid there only
    because every one of its runs used the project default seed=42) would score
    seed-123/seed-7 checkpoints against the wrong held-out patients."""
    split = json.loads((RUNS / run_name / "split.json").read_text(encoding="utf-8"))
    val_keys = set(split["val_patients"])
    samples = build_manifest(modality)
    return [s for s in samples if s.patient_key in val_keys]


def score_subset(model, samples, task, img_size, input_mode, batch=8):
    if not samples:
        return None
    ds = CadSelDetectionDataset(samples, task, img_size, train=False, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fn, num_workers=0)
    m = evaluate(model, dl, DEVICE, TASK_CLASSES[task], progress=False,
                 normalize=make_normalizer(input_mode))
    return {"AP@0.5": round(m["AP@0.5"], 4), "f1": round(m["f1"], 4), "n": len(samples)}


# %%
results = []
for run_name, modality, task, arm in RUN_NAMES:
    ckpt_path = RUNS / run_name / "best_model.pth"
    if not ckpt_path.exists():
        print(f"missing {ckpt_path} -- skipping (has Phase 3 finished?)")
        continue
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_size, input_mode = ck["img_size"], ck.get("input_mode", "reference")

    model = build_detection_model(ck["model_type"], task, pretrained=False)
    configure_transform_for_input_mode(model, input_mode, img_size)
    model.load_state_dict(ck["model_state_dict"])
    model.to(DEVICE).eval()

    val_s = load_val_samples(run_name, modality)
    areas = np.array([box_area_frac(s) for s in val_s])
    edges = np.quantile(areas, [0.0, 1 / 3, 2 / 3, 1.0])

    overall = score_subset(model, val_s, task, img_size, input_mode)
    print(f"\n=== {run_name} (arm={arm}) ===")
    print(f"  overall            AP@0.5 {overall['AP@0.5']:.4f}  (n={overall['n']})")

    bands = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = [s for s, a in zip(val_s, areas) if (a >= lo and (a < hi or i == 2))]
        r = score_subset(model, sel, task, img_size, input_mode)
        if r:
            bands[f"tertile{i+1}"] = r
            print(f"  size tertile {i+1} ({lo:.3f}-{hi:.3f})  AP@0.5 {r['AP@0.5']:.4f}  (n={r['n']})")

    results.append({
        "run": run_name, "modality": modality, "task": task, "arm": arm,
        "overall": overall["AP@0.5"], "bands": bands,
    })
    json.dump(results, open(OUT / "19_size_lever_eval.json", "w"), indent=2)

    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 19.1 Verdict -- mean +/- std per arm, tertile-1 (small-lesion) is the headline

# %%
by_task_arm = defaultdict(list)
for r in results:
    by_task_arm[(r["task"], r["arm"])].append(r)

print("=" * 92)
print("SIZE-LEVER EVAL -- AP@0.5 by size tertile, mean +/- std across seeds")
print("=" * 92)
summary = {}
for (task, arm), rows in sorted(by_task_arm.items()):
    overall = [r["overall"] for r in rows]
    tertiles = defaultdict(list)
    for r in rows:
        for k, v in r["bands"].items():
            tertiles[k].append(v["AP@0.5"])
    entry = {
        "n_seeds": len(rows),
        "overall_mean": float(np.mean(overall)), "overall_std": float(np.std(overall)),
        "tertiles": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in sorted(tertiles.items())
        },
    }
    summary[f"{task}/{arm}"] = entry
    t1 = entry["tertiles"].get("tertile1", {})
    print(f"{task:<10}{arm:<14} overall {entry['overall_mean']:.4f}+/-{entry['overall_std']:.4f}  "
          f"(n={entry['n_seeds']})  |  tertile1 (small) {t1.get('mean', float('nan')):.4f}"
          f"+/-{t1.get('std', float('nan')):.4f}")

json.dump(summary, open(OUT / "19_size_lever_eval_summary.json", "w"), indent=2)

print("\nHeadline: size_rfs_ms vs ms_only, tertile-1 (small-lesion) AP delta, per task:")
for task in ("multiclass", "binary"):
    ms = summary.get(f"{task}/ms_only", {}).get("tertiles", {}).get("tertile1")
    sr = summary.get(f"{task}/size_rfs_ms", {}).get("tertiles", {}).get("tertile1")
    if ms and sr:
        delta = sr["mean"] - ms["mean"]
        # ~ combined seed-to-seed noise floor for this specific slice (not the
        # project's overall-AP 0.023 floor, which is a different, coarser metric)
        floor = (ms["std"] ** 2 + sr["std"] ** 2) ** 0.5
        verdict = "CLEARS noise floor" if abs(delta) > floor else "within noise floor -- not yet confirmed"
        print(f"  {task:<10} ms_only {ms['mean']:.4f} -> size_rfs_ms {sr['mean']:.4f}  "
              f"(delta {delta:+.4f}, floor ~{floor:.4f}) -> {verdict}")

print("\nHeadline: size_rfs_ms vs size_rfs_ms_anchors (+auto-anchors), tertile-1 AP delta:")
sr = summary.get("multiclass/size_rfs_ms", {}).get("tertiles", {}).get("tertile1")
sra = summary.get("multiclass/size_rfs_ms_anchors", {}).get("tertiles", {}).get("tertile1")
if sr and sra:
    delta = sra["mean"] - sr["mean"]
    floor = (sr["std"] ** 2 + sra["std"] ** 2) ** 0.5
    verdict = "CLEARS noise floor" if abs(delta) > floor else "within noise floor -- not yet confirmed"
    print(f"  multiclass size_rfs_ms {sr['mean']:.4f} -> size_rfs_ms_anchors {sra['mean']:.4f}  "
          f"(delta {delta:+.4f}, floor ~{floor:.4f}) -> {verdict}")

print("\nHeadline: size-RFS threshold sweep, seed-42 only (single-seed screen, not a confirmation):")
base_run = next((r for r in results if r["run"] == "size_rfs_ms_s42__WLE__multiclass"), None)
sweep_runs = {
    "0.1": next((r for r in results if r["run"] == "size_rfs_ms_thresh01_s42__WLE__multiclass"), None),
    "0.2 (adopted)": base_run,
    "0.3": next((r for r in results if r["run"] == "size_rfs_ms_thresh03_s42__WLE__multiclass"), None),
    "0.4": next((r for r in results if r["run"] == "size_rfs_ms_thresh04_s42__WLE__multiclass"), None),
}
for label, r in sweep_runs.items():
    if r:
        t1 = r["bands"].get("tertile1", {}).get("AP@0.5", float("nan"))
        print(f"  threshold {label:<14} overall {r['overall']:.4f}  tertile1 {t1:.4f}")
