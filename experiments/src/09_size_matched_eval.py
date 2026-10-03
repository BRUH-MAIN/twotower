# %% [markdown]
# # Experiment 9 — Size-stratified and size-matched evaluation
#
# Experiment 1 found lesion size predicts class (balanced-accuracy lift +0.20),
# and Experiment 3 found a black canvas showing only the bounding rectangle
# retains 48-189% of full-image performance. Size is therefore doing real work in
# the reported metric.
#
# Rather than trying to train that away, this fixes the **measurement**. Two
# analyses, both reusing the checkpoints already trained — no GPU training:
#
# 1. **Stratified**: AP@0.5 within lesion-size bands. Shows where performance
#    actually lives.
# 2. **Size-matched**: AP@0.5 on a resampled validation subset in which every
#    class shares the same size distribution, so box size carries *no* class
#    information. Compared against random subsets of identical size, so the
#    comparison is not confounded by having fewer images.
#
# If AP falls on the size-matched subset, part of the headline number was size
# prior rather than lesion recognition.

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
    patient_level_split, read_yolo_label,
)
from cadsel.engine import evaluate
from cadsel.models import build_detection_model, configure_transform_for_input_mode
from torch.utils.data import DataLoader

OUT = Path("experiments/results")
RUNS = Path("runs")
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.RandomState(0)
print("device:", DEVICE)

RUN_NAMES = [
    ("fasterrcnn_mobilenet__WLE__multiclass", "WLE", "multiclass"),
    ("fasterrcnn_mobilenet__WLE__binary", "WLE", "binary"),
    ("fasterrcnn_mobilenet__EUS__multiclass", "EUS", "multiclass"),
    ("fasterrcnn_mobilenet__EUS__binary", "EUS", "binary"),
    ("fasterrcnn_mobilenet_320__WLE__multiclass", "WLE", "multiclass"),
    ("fasterrcnn_mobilenet_320__WLE__binary", "WLE", "binary"),
    ("fasterrcnn_mobilenet_320__EUS__multiclass", "EUS", "multiclass"),
    ("fasterrcnn_mobilenet_320__EUS__binary", "EUS", "binary"),
]

# %%
def box_area_frac(sample) -> float:
    """Largest annotated box as a fraction of frame area — resolution-independent."""
    best = 0.0
    for _c, _cx, _cy, bw, bh in read_yolo_label(sample.label):
        best = max(best, float(bw * bh))
    return best


def subtype_of(sample, task):
    if task == "binary":
        return "NET" if sample.subtype.startswith("NET") else "non-NET"
    return "NET" if sample.subtype.startswith("NET") else sample.subtype


def score_subset(model, samples, task, img_size, input_mode, batch=8):
    if not samples:
        return None
    ds = CadSelDetectionDataset(samples, task, img_size, train=False,
                                input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fn,
                    num_workers=0)
    m = evaluate(model, dl, DEVICE, TASK_CLASSES[task], progress=False,
                 normalize=make_normalizer(input_mode))
    return {"AP@0.5": round(m["AP@0.5"], 4), "f1": round(m["f1"], 4), "n": len(samples)}


def size_matched_subset(samples, task, n_bins=4, seed=0):
    """Resample so each class shares the same size distribution.

    Within every size bin, take an equal number of images per class. Size then
    carries no information about class within the returned subset.
    """
    r = np.random.RandomState(seed)
    areas = np.array([box_area_frac(s) for s in samples])
    edges = np.quantile(areas, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_of = np.clip(np.digitize(areas, edges[1:-1]), 0, n_bins - 1)

    by_bin_class = defaultdict(list)
    for i, s in enumerate(samples):
        by_bin_class[(bin_of[i], subtype_of(s, task))].append(i)

    chosen = []
    for b in range(n_bins):
        classes = [c for (bb, c) in by_bin_class if bb == b]
        if len(classes) < 2:
            continue
        k = min(len(by_bin_class[(b, c)]) for c in classes)
        if k == 0:
            continue
        for c in classes:
            idx = by_bin_class[(b, c)]
            chosen.extend(r.choice(idx, size=k, replace=False).tolist())
    return [samples[i] for i in sorted(set(chosen))]


# %%
results = []
for run_name, modality, task in RUN_NAMES:
    ckpt_path = RUNS / run_name / "best_model.pth"
    if not ckpt_path.exists():
        print(f"missing {ckpt_path}, skipping")
        continue
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_size, input_mode = ck["img_size"], ck.get("input_mode", "reference")

    model = build_detection_model(ck["model_type"], task, pretrained=False)
    configure_transform_for_input_mode(model, input_mode, img_size)
    model.load_state_dict(ck["model_state_dict"])
    model.to(DEVICE).eval()

    samples = build_manifest(modality)
    _, val_s = patient_level_split(samples, 0.2, SEED)
    areas = np.array([box_area_frac(s) for s in val_s])
    edges = np.quantile(areas, [0.0, 1/3, 2/3, 1.0])

    overall = score_subset(model, val_s, task, img_size, input_mode)
    print(f"\n=== {run_name} ===")
    print(f"  overall            AP@0.5 {overall['AP@0.5']:.4f}  (n={overall['n']})")

    bands = {}
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = [s for s, a in zip(val_s, areas)
               if (a >= lo and (a < hi or i == 2))]
        r = score_subset(model, sel, task, img_size, input_mode)
        if r:
            bands[f"tertile{i+1}"] = r
            print(f"  size tertile {i+1} ({lo:.3f}-{hi:.3f})  "
                  f"AP@0.5 {r['AP@0.5']:.4f}  (n={r['n']})")

    matched = size_matched_subset(val_s, task, seed=0)
    m_res = score_subset(model, matched, task, img_size, input_mode)

    # Random subsets of identical size, so the comparison isn't just "fewer images"
    rand_scores = []
    for s_i in range(3):
        rr = np.random.RandomState(100 + s_i)
        idx = rr.choice(len(val_s), size=len(matched), replace=False)
        rs = score_subset(model, [val_s[i] for i in sorted(idx)], task,
                          img_size, input_mode)
        rand_scores.append(rs["AP@0.5"])
    rand_mean = float(np.mean(rand_scores))

    print(f"  size-matched       AP@0.5 {m_res['AP@0.5']:.4f}  (n={m_res['n']})")
    print(f"  random same-size   AP@0.5 {rand_mean:.4f}  "
          f"(mean of 3, n={m_res['n']})")
    print(f"  -> size-prior contribution {rand_mean - m_res['AP@0.5']:+.4f}")

    results.append({
        "run": run_name, "modality": modality, "task": task,
        "overall": overall["AP@0.5"], "bands": bands,
        "size_matched": m_res["AP@0.5"], "n_matched": m_res["n"],
        "random_same_size": round(rand_mean, 4),
        "random_scores": rand_scores,
        "size_prior_contribution": round(rand_mean - m_res["AP@0.5"], 4),
    })
    json.dump(results, open(OUT / "09_size_matched.json", "w"), indent=2)

    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## 9.1 Verdict

# %%
print("=" * 96)
print("SIZE-MATCHED EVALUATION — AP@0.5")
print("=" * 96)
hdr = (f"{'run':<44}{'overall':>9}{'matched':>9}{'random':>9}{'size prior':>12}")
print(hdr); print("-" * len(hdr))
for r in results:
    print(f"{r['run']:<44}{r['overall']:>9.4f}{r['size_matched']:>9.4f}"
          f"{r['random_same_size']:>9.4f}{r['size_prior_contribution']:>+12.4f}")

vals = [r["size_prior_contribution"] for r in results]
if vals:
    print(f"\nmean size-prior contribution: {np.mean(vals):+.4f} AP@0.5")
    print("(positive = AP falls once size carries no class information,")
    print(" i.e. that much of the headline number was size prior)")

print("\nAP by lesion-size tertile (smallest -> largest):")
for r in results:
    b = r["bands"]
    row = "  ".join(f"{b[k]['AP@0.5']:.3f}" for k in sorted(b))
    print(f"  {r['run']:<44} {row}")
