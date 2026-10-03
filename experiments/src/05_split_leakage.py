# %% [markdown]
# # Experiment 5 — Patient-level vs image-level splitting
#
# Not shortcut learning in the Geirhos sense, but the failure mode most likely to
# inflate a reported number on this dataset, and the one the CAD-SEL paper itself
# warns about:
#
# > *"it is also essential to perform dataset partitioning at the patient level,
# > rather than the image level, to avoid data leakage from correlated images
# > appearing in both training and testing sets."*
#
# Each patient contributes several near-duplicate frames of the same lesion. Under
# an image-level split those frames straddle the boundary, so the validation set is
# largely already-seen data. This experiment measures exactly how large that
# inflation is, using identical models and identical split sizes — only the
# grouping changes.

# %%
import sys, json, gc
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np

from cadsel.data import build_manifest, patient_level_split, image_level_split
from harness import DEVICE, classes_for, decode, encode_labels, train_eval

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
print("device:", DEVICE)

# %%
results = []
for modality in ("WLE", "EUS"):
    samples = build_manifest(modality)

    splits = {
        "patient": patient_level_split(samples, 0.2, SEED),
        "image": image_level_split(samples, 0.2, SEED),
    }
    for name, (tr, va) in splits.items():
        shared = {s.patient_key for s in tr} & {s.patient_key for s in va}
        print(f"{modality} {name:<8} split: {len(tr)} train / {len(va)} val, "
              f"{len(shared)} of {len({s.patient_key for s in va})} val patients "
              f"also in train")

    print("  decoding (once, shared across both splits) ...")
    all_idx = {id(s): i for i, s in enumerate(samples)}
    X = decode(samples)

    for split_name, (tr, va) in splits.items():
        itr = np.array([all_idx[id(s)] for s in tr])
        iva = np.array([all_idx[id(s)] for s in va])
        for task in ("binary", "subtype"):
            classes = classes_for(task)
            ytr = encode_labels(tr, task)
            yva = encode_labels(va, task)
            r = train_eval(X[itr], ytr, X[iva], yva, len(classes),
                           f"{modality}/{task}/{split_name}-level split", seed=SEED)
            r.update({"modality": modality, "task": task, "split": split_name})
            results.append(r)
            json.dump(results, open(OUT / "05_split_leakage.json", "w"), indent=2)

    del X
    gc.collect()

# %% [markdown]
# ## 5.1 Verdict

# %%
by = {(r["modality"], r["task"], r["split"]): r for r in results}
verdict = []
for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        p, i = by[(modality, task, "patient")], by[(modality, task, "image")]
        verdict.append({
            "modality": modality, "task": task, "chance": p["chance"],
            "patient_level": p["bal"], "image_level": i["bal"],
            "inflation": round(i["bal"] - p["bal"], 4),
        })

json.dump({"runs": results, "verdict": verdict},
          open(OUT / "05_split_leakage_verdict.json", "w"), indent=2)

print("=" * 74)
print("SPLIT-LEVEL LEAKAGE — balanced accuracy")
print("=" * 74)
hdr = (f"{'modality':<9}{'task':<9}{'chance':>8}{'patient':>10}{'image':>9}"
       f"{'inflation':>11}")
print(hdr); print("-" * len(hdr))
for v in verdict:
    print(f"{v['modality']:<9}{v['task']:<9}{v['chance']:>8.3f}"
          f"{v['patient_level']:>10.3f}{v['image_level']:>9.3f}"
          f"{v['inflation']:>+11.3f}")

print()
for v in verdict:
    inf = v["inflation"]
    verdict_txt = ("substantial inflation from correlated frames" if inf > 0.10
                   else "modest inflation" if inf > 0.03
                   else "little difference")
    print(f"  {v['modality']:<4} {v['task']:<8} {inf:+.3f}  {verdict_txt}")

print("\nAn image-level split is not a valid protocol here; this run exists only")
print("to quantify what it would have bought. All reported CAD-SEL results in")
print("this project use the patient-level split.")
