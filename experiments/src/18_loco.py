# %% [markdown]
# # Experiment 18 — Classifier-level leave-one-centre-out (LOCO) validation
#
# Experiment 4 ran exactly one external-validation split: train on Center-1's own
# patients, test the same model on the pooled rest. That is asymmetric and
# Center-1-specific — it never asks whether *any other* centre, trained out of the
# same pooled majority, also collapses when held out. Experiment 2 established that
# acquisition site is strongly decodable from pixels alone (0.81 balanced accuracy
# vs 0.167 chance), so a model could in principle be keying off centre-specific
# artefacts for any of the sites, not just Center-1.
#
# This generalises Experiment 4 into a genuine leave-one-centre-out protocol:
#
# | Fold | Train on | Test on |
# | --- | --- | --- |
# | `Center-1` | Centers 2-6 | Center-1 |
# | `Center-2` | Centers 1,3-6 | Center-2 |
# | `Center-3` | Centers 1-2,4-6 | Center-3 |
# | `Centers-4-6` | Centers 1-3 | Centers 4,5,6 pooled |
#
# Every fold trains on all centres *except* the held-out one and tests on that
# held-out centre alone — symmetric in structure across Center-1/2/3, unlike
# Experiment 4's Center-1-only asymmetric split. Centers 4-6 individually have
# only 6-8 patients each (Table: 854/50/136/8/8/6 patient-label-dirs across
# WLE+EUS), too few to support a subtype-stratified test set on their own —
# several classes would likely be entirely absent — so they are pooled into one
# fold and flagged `low_confidence: true` in the output. That fold is still worth
# computing (it is the only way to say anything about those three sites at all),
# but should not be read with the same confidence as Center-1/2/3.
#
# Same protocol as Experiment 4 throughout: ResNet-18, patient-level 80/20 split
# of the training pool for the internal arm (`cadsel.data.patient_level_split`,
# seed 42), identical hyperparameters (`harness.train_eval`, 15 epochs, AdamW,
# cosine schedule, AMP), and the same retention statistic:
#
# ```
# retained = (external - chance) / (internal - chance)
# ```
#
# A large, consistent drop across the individually-reliable folds (Center-1/2/3)
# — beyond what internal train/val noise alone would predict — is the signature
# the review calls decisive for site-specific shortcut reliance.

# %%
import sys, json, gc
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np

from cadsel.data import build_manifest, patient_level_split
from harness import DEVICE, classes_for, decode, encode_labels, train_eval

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
print("device:", DEVICE)

ALL_CENTERS = ["Center-1", "Center-2", "Center-3", "Center-4", "Center-5", "Center-6"]
CENTER_FOLDS = [
    ("Center-1", ["Center-1"]),
    ("Center-2", ["Center-2"]),
    ("Center-3", ["Center-3"]),
    ("Centers-4-6", ["Center-4", "Center-5", "Center-6"]),   # low_confidence: too few patients individually
]
LOW_CONFIDENCE_FOLDS = {"Centers-4-6"}
MIN_EXTERNAL = 40   # mirrors Experiment 4's own minimum-size guard

# %%
RESULTS_PATH = OUT / "18_loco.json"
results = json.loads(RESULTS_PATH.read_text())["runs"] if RESULTS_PATH.exists() else []
done = {(r["fold_name"], r["modality"], r["task"], r["arm"]) for r in results}
if done:
    print(f"resuming: {len(done)} run(s) already recorded")

ALL_ARMS = [(t, a) for t in ("binary", "subtype") for a in ("internal", "external")]

for modality in ("WLE", "EUS"):
    for fold_name, held_out in CENTER_FOLDS:
        low_confidence = fold_name in LOW_CONFIDENCE_FOLDS

        if all((fold_name, modality, t, a) in done for t, a in ALL_ARMS):
            print(f"\n=== {modality}/{fold_name}: all runs already recorded, skipping ===")
            continue

        train_centers = [c for c in ALL_CENTERS if c not in held_out]
        train_pool = build_manifest(modality, centers=train_centers)
        test_pool = build_manifest(modality, centers=held_out)

        print(f"\n=== {modality} / {fold_name} (held out: {held_out}) "
              f"{'[LOW CONFIDENCE: pooled small-N fold]' if low_confidence else ''} ===")
        print(f"  train pool (centers {train_centers}): {len(train_pool)} images / "
              f"{len({s.patient_key for s in train_pool})} patients")
        print(f"  test pool  (centers {held_out}): {len(test_pool)} images / "
              f"{len({s.patient_key for s in test_pool})} patients")
        print("  external subtype mix:", dict(sorted(Counter(s.subtype for s in test_pool).items())))

        if len(test_pool) < MIN_EXTERNAL:
            print(f"  fewer than {MIN_EXTERNAL} external images; skipping")
            continue

        # Internal split of the training pool; the external set is the held-out centre(s).
        train_s, val_s_internal = patient_level_split(train_pool, 0.2, SEED)

        print("  decoding ...")
        Xtr = decode(train_s)
        Xin = decode(val_s_internal, progress_every=0)
        Xex = decode(test_pool, progress_every=0)

        for task in ("binary", "subtype"):
            need_internal = (fold_name, modality, task, "internal") not in done
            need_external = (fold_name, modality, task, "external") not in done
            if not need_internal and not need_external:
                continue

            classes = classes_for(task)
            ytr = encode_labels(train_s, task)
            yin = encode_labels(val_s_internal, task)
            yex = encode_labels(test_pool, task)

            print(f"  -- {task} --")
            if need_internal:
                internal = train_eval(
                    Xtr, ytr, Xin, yin, len(classes),
                    f"{modality}/{fold_name}/{task}/internal", seed=SEED,
                )
                internal.update({"modality": modality, "task": task, "fold_name": fold_name,
                                  "held_out": held_out, "arm": "internal",
                                  "low_confidence": low_confidence})
                results.append(internal)
                done.add((fold_name, modality, task, "internal"))
                json.dump({"runs": results}, open(RESULTS_PATH, "w"), indent=2)

            if need_external:
                external = train_eval(
                    Xtr, ytr, Xex, yex, len(classes),
                    f"{modality}/{fold_name}/{task}/EXTERNAL", seed=SEED,
                )
                external.update({"modality": modality, "task": task, "fold_name": fold_name,
                                  "held_out": held_out, "arm": "external",
                                  "low_confidence": low_confidence})
                results.append(external)
                done.add((fold_name, modality, task, "external"))
                json.dump({"runs": results}, open(RESULTS_PATH, "w"), indent=2)

        del Xtr, Xin, Xex
        gc.collect()

# %% [markdown]
# ## 18.1 Verdict
#
# ```
# retained = (external - chance) / (internal - chance)
# ```
#
# Read Center-1/2/3 as the individually-reliable folds. Centers-4-6 is pooled
# out of necessity (6-8 patients per site) and is reported for completeness only
# — `low_confidence: true` on every row of that fold.

# %%
by = {(r["fold_name"], r["modality"], r["task"], r["arm"]): r for r in results}
verdict = []
for modality in ("WLE", "EUS"):
    for fold_name, held_out in CENTER_FOLDS:
        low_confidence = fold_name in LOW_CONFIDENCE_FOLDS
        for task in ("binary", "subtype"):
            key_i = (fold_name, modality, task, "internal")
            key_e = (fold_name, modality, task, "external")
            if key_i not in by or key_e not in by:
                continue
            i, e = by[key_i], by[key_e]
            chance = i["chance"]
            denom = i["bal"] - chance
            retained = (e["bal"] - chance) / denom if denom > 1e-6 else None
            verdict.append({
                "modality": modality, "task": task, "fold_name": fold_name,
                "chance": chance, "internal": i["bal"], "external": e["bal"],
                "drop": round(i["bal"] - e["bal"], 4),
                "retained": None if retained is None else round(retained, 3),
                "n_external": e["n_val"],
                "low_confidence": low_confidence,
            })

json.dump({"runs": results, "verdict": verdict},
          open(OUT / "18_loco.json", "w"), indent=2)

print("=" * 100)
print("LEAVE-ONE-CENTRE-OUT VALIDATION — balanced accuracy")
print("=" * 100)
hdr = (f"{'modality':<9}{'fold':<13}{'task':<9}{'chance':>8}{'internal':>10}{'external':>10}"
       f"{'drop':>8}{'retained':>10}{'n_ext':>7}  flag")
print(hdr); print("-" * len(hdr))
for v in verdict:
    ret = v["retained"]
    flag = "LOW_CONF" if v["low_confidence"] else ""
    print(f"{v['modality']:<9}{v['fold_name']:<13}{v['task']:<9}{v['chance']:>8.3f}"
          f"{v['internal']:>10.3f}{v['external']:>10.3f}{v['drop']:>8.3f}"
          f"{'n/a' if ret is None else format(ret, '>9.1%')}{v['n_external']:>7}  {flag}")

print()
for v in verdict:
    ret = v["retained"]
    tag = " [LOW CONFIDENCE — pooled small-N fold]" if v["low_confidence"] else ""
    if ret is None:
        continue
    if ret < 0.4:
        msg = "LARGE external drop — consistent with site-specific shortcut reliance"
    elif ret < 0.75:
        msg = "moderate external drop"
    else:
        msg = "transfers well across sites"
    print(f"  {v['modality']:<4} {v['fold_name']:<12} {v['task']:<8} retained {ret:>6.1%}  {msg}{tag}")

print("\nCaveat: even the individually-reliable folds (Center-1/2/3) draw external")
print("test sets far smaller than a typical internal validation split, so part of")
print("any drop reflects small-sample noise and prevalence shift, not purely")
print("acquisition-site shortcutting. Centers-4-6 is pooled from three sites with")
print("6-8 patients each and is reported for completeness, not as equally reliable")
print("evidence as the other three folds.")
