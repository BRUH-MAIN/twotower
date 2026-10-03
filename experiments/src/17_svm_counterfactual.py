# %% [markdown]
# # Experiment 17 — SVM on frozen embeddings, run through Experiment 3's counterfactual protocol
#
# Two prior results this experiment joins:
#
# 1. **Experiment 3** (`03_lesion_vs_background.py`, frozen -- not imported here):
#    a ResNet-18 trained from scratch on five counterfactual input variants
#    (`full`, `lesion`, `bg_mask`, `bg_patch`, `boxmap`) retains 68-95% of
#    full-image balanced accuracy on `bg_patch` (honest lesion removal) and
#    48-189% on `boxmap` (annotation geometry alone) -- most of the label is
#    recoverable without the lesion actually being visible.
# 2. **`experiments/svm_classifier.py`**: an SVM on frozen ResNet-50 vs DINOv3
#    embeddings (patient-grouped 5-fold CV) found DINOv3 notably ahead of
#    ResNet-50 (e.g. WLE binary 0.735 vs 0.645 balanced accuracy). Left as an
#    open question there: is DINOv3's edge genuine better lesion-reading, or is
#    DINOv3 just a *better shortcut-reader* than ResNet-50? Nobody had run the
#    embeddings pipeline through Experiment 3's counterfactual protocol to
#    check -- this experiment does exactly that.
#
# Both backbones are frozen (no gradient training) -- extraction is 2
# modalities x 2 backbones x 5 variants = 20 forward passes total
# (label-agnostic, so shared across both tasks), each cached to disk. SVM
# fitting itself is CPU-bound and fast. Pixel variants come from
# `ablation_inputs.build_five_variants`, a verified line-for-line port of 03's
# own in-file `build_variants` (see that module's docstring, and the
# byte-identical spot-check recorded in FINDINGS.md / the run log for this
# experiment) -- so the *inputs* here are numerically identical to Experiment
# 3's, and only the classifier (frozen-embedding SVM vs from-scratch ResNet-18)
# differs.

# %%
import sys, json, time, gc
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cadsel.data import build_manifest
from ablation_inputs import build_five_variants
from embeddings import load_or_extract_from_arrays
from harness import classes_for, encode_labels
from probes import balanced_accuracy, grouped_folds

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
EMB_DIR = OUT / "embeddings"
RESULTS_PATH = OUT / "17_svm_counterfactual.json"
VERDICT_PATH = OUT / "17_svm_counterfactual_verdict.json"
EXP3_PATH = OUT / "03_lesion_vs_background.json"

VARIANTS = ("full", "lesion", "bg_mask", "bg_patch", "boxmap")
BACKBONES = ("resnet", "dinov3")
N_SPLITS = 5
SEED = 0  # matches svm_classifier.py's grouped_folds convention (probes.py default)
print("variants:", VARIANTS, " backbones:", BACKBONES)

# %%
# Resume: deterministic (fixed seed, fixed CV), so anything already recorded is
# reused. Delete 17_svm_counterfactual.json to force a full rerun.
store = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {"runs": []}
results = store["runs"]
done = {(r["modality"], r["task"], r["backbone"], r["variant"]) for r in results}
if done:
    print(f"resuming: {len(done)} run(s) already recorded")


def save():
    RESULTS_PATH.write_text(json.dumps(store, indent=2))


for modality in ("WLE", "EUS"):
    todo = [(t, b, v) for t in ("binary", "subtype") for b in BACKBONES for v in VARIANTS
            if (modality, t, b, v) not in done]
    if not todo:
        print(f"\n=== {modality}: all runs already recorded, skipping ===")
        continue

    samples = build_manifest(modality)
    groups = np.array([s.patient_key for s in samples])
    print(f"\n=== {modality}: {len(samples)} images ({len(todo)} run(s) to do) ===")

    # One decode pass per modality, shared across both tasks and both
    # backbones -- the pixel variants don't depend on the label or the
    # embedding model, only on the sample's image + box.
    print("  building five counterfactual variants (one decode pass) ...")
    variant_arrays = build_five_variants(samples)

    for task in ("binary", "subtype"):
        classes = classes_for(task)
        n_classes = len(classes)
        y = encode_labels(samples, task)
        pending = [(b, v) for b, v in [(b, v) for b in BACKBONES for v in VARIANTS]
                   if (modality, task, b, v) not in done]
        if not pending:
            continue
        print(f"  -- {task} --")
        for backbone, variant in pending:
            t0 = time.time()
            X = load_or_extract_from_arrays(
                variant_arrays[variant], backbone, modality, variant, EMB_DIR)

            accs, bals, maj_accs = [], [], []
            for tr, te in grouped_folds(groups, N_SPLITS, SEED):
                scaler = StandardScaler().fit(X[tr])
                Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
                clf = SVC(kernel="rbf", C=1.0, class_weight="balanced")
                clf.fit(Xtr, y[tr])
                pred = clf.predict(Xte)
                accs.append(float((pred == y[te]).mean()))
                bals.append(balanced_accuracy(y[te], pred, n_classes))
                majority = np.bincount(y[tr], minlength=n_classes).argmax()
                maj_accs.append(float((y[te] == majority).mean()))

            r = {
                "modality": modality, "task": task, "backbone": backbone, "variant": variant,
                "acc": round(float(np.mean(accs)), 4),
                "bal": round(float(np.mean(bals)), 4),
                "majority": round(float(np.mean(maj_accs)), 4),
                "chance": round(1 / n_classes, 4),
                "embedding_dim": int(X.shape[1]),
                "n_images": int(len(samples)),
                "n_folds": N_SPLITS,
                "minutes": round((time.time() - t0) / 60, 1),
            }
            print(f"    {modality}/{task}/{backbone}/{variant:<8} "
                  f"acc {r['acc']:.4f}  bal {r['bal']:.4f}  "
                  f"(majority {r['majority']:.4f}, chance {r['chance']:.3f}, "
                  f"{r['minutes']:.1f} min)")
            results.append(r)
            done.add((modality, task, backbone, variant))
            save()

    del variant_arrays
    gc.collect()

# %% [markdown]
# ## 17.1 Verdict
#
# Same retention formula as Experiment 3:
#
# ```
# retention = (variant_bal - chance) / (full_bal - chance)
# ```
#
# computed separately per `(modality, task, backbone)` so ResNet-50's and
# DINOv3's retention curves can be compared directly, and against Experiment
# 3's trained-classifier retention.

# %%
by = {(r["modality"], r["task"], r["backbone"], r["variant"]): r for r in results}
verdict = []
for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        for backbone in BACKBONES:
            full = by.get((modality, task, backbone, "full"))
            if full is None:
                continue
            chance = full["chance"]
            denom = full["bal"] - chance
            row = {"modality": modality, "task": task, "backbone": backbone,
                   "chance": chance, "full": full["bal"]}
            for v in ("lesion", "bg_mask", "bg_patch", "boxmap"):
                rec = by.get((modality, task, backbone, v))
                if rec is None:
                    continue
                row[v] = rec["bal"]
                row[f"ret_{v}"] = (round((row[v] - chance) / denom, 3)
                                   if denom > 1e-6 else None)
            verdict.append(row)

store["verdict"] = verdict
save()

print("=" * 100)
print("SVM ON FROZEN EMBEDDINGS — balanced accuracy (retention vs full in brackets)")
print("=" * 100)
hdr = (f"{'mod':<5}{'task':<9}{'backbone':<9}{'chance':>7}{'full':>7}{'lesion':>16}"
       f"{'bg_mask':>16}{'bg_patch':>16}{'boxmap':>16}")
print(hdr); print("-" * len(hdr))
for v in verdict:
    def cell(name):
        r = v.get("ret_" + name)
        return f"{v[name]:.3f} ({r:>5.0%})" if name in v and r is not None else "n/a"
    print(f"{v['modality']:<5}{v['task']:<9}{v['backbone']:<9}{v['chance']:>7.3f}{v['full']:>7.3f}"
          f"{cell('lesion'):>16}{cell('bg_mask'):>16}"
          f"{cell('bg_patch'):>16}{cell('boxmap'):>16}")

# %% [markdown]
# ## 17.2 Does DINOv3's advantage survive lesion removal?
#
# Joins against Experiment 3's published `bg_patch`/`boxmap` retention (the
# trained-classifier reference) and computes the ResNet-50 vs DINOv3 gap at
# `full`, `bg_patch` and `boxmap` -- if DINOv3 stays ahead of ResNet-50 by a
# *similar or larger* margin once the lesion is removed, its edge on `full`
# looks like better shortcut-reading rather than better lesion-reading; if the
# gap collapses toward zero (or DINOv3 falls behind) under ablation, that's
# evidence its edge is genuine.

# %%
exp3 = json.loads(EXP3_PATH.read_text()) if EXP3_PATH.exists() else {"verdict": []}
exp3_by = {(r["modality"], r["task"]): r for r in exp3["verdict"]}
by_verdict = {(v["modality"], v["task"], v["backbone"]): v for v in verdict}

comparison = []
for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        row = {"modality": modality, "task": task}
        e3 = exp3_by.get((modality, task))
        if e3 is not None:
            row["exp3_trained_classifier"] = {
                "full": e3["full"],
                "ret_bg_patch": e3.get("ret_bg_patch"),
                "ret_boxmap": e3.get("ret_boxmap"),
            }
        rn = by_verdict.get((modality, task, "resnet"))
        dv = by_verdict.get((modality, task, "dinov3"))
        if rn is not None:
            row["resnet_svm"] = {
                "full": rn["full"], "bg_patch": rn.get("bg_patch"), "boxmap": rn.get("boxmap"),
                "ret_bg_patch": rn.get("ret_bg_patch"), "ret_boxmap": rn.get("ret_boxmap"),
            }
        if dv is not None:
            row["dinov3_svm"] = {
                "full": dv["full"], "bg_patch": dv.get("bg_patch"), "boxmap": dv.get("boxmap"),
                "ret_bg_patch": dv.get("ret_bg_patch"), "ret_boxmap": dv.get("ret_boxmap"),
            }
        if rn is not None and dv is not None:
            def gap(key):
                a, b = dv.get(key), rn.get(key)
                return round(a - b, 4) if a is not None and b is not None else None
            row["dinov3_minus_resnet_gap"] = {
                "full": gap("full"), "bg_patch": gap("bg_patch"), "boxmap": gap("boxmap"),
            }
            row["dinov3_minus_resnet_ret_gap"] = {
                "ret_bg_patch": (round(dv.get("ret_bg_patch") - rn.get("ret_bg_patch"), 3)
                                 if dv.get("ret_bg_patch") is not None and rn.get("ret_bg_patch") is not None
                                 else None),
                "ret_boxmap": (round(dv.get("ret_boxmap") - rn.get("ret_boxmap"), 3)
                               if dv.get("ret_boxmap") is not None and rn.get("ret_boxmap") is not None
                               else None),
            }
        comparison.append(row)

json.dump({"comparison": comparison}, open(VERDICT_PATH, "w"), indent=2)

print()
print("=" * 100)
print("DINOv3 vs ResNet-50 gap (balanced accuracy) at full image vs under ablation,")
print("alongside Experiment 3's trained-classifier retention for reference")
print("=" * 100)
hdr2 = (f"{'mod':<5}{'task':<9}{'gap@full':>10}{'gap@bgpatch':>13}{'gap@boxmap':>12}"
        f"{'exp3 ret_bgp':>14}{'exp3 ret_box':>14}{'resnet ret_bgp':>16}{'dinov3 ret_bgp':>16}")
print(hdr2); print("-" * len(hdr2))
for row in comparison:
    g = row.get("dinov3_minus_resnet_gap")
    e3 = row.get("exp3_trained_classifier", {})
    rn = row.get("resnet_svm", {})
    dv = row.get("dinov3_svm", {})
    if g is None:
        continue
    def fmt(x, pct=False):
        if x is None:
            return "n/a"
        return f"{x:+.0%}" if pct else f"{x:+.3f}"
    print(f"{row['modality']:<5}{row['task']:<9}"
          f"{fmt(g['full']):>10}{fmt(g['bg_patch']):>13}{fmt(g['boxmap']):>12}"
          f"{fmt(e3.get('ret_bg_patch'), pct=True):>14}{fmt(e3.get('ret_boxmap'), pct=True):>14}"
          f"{fmt(rn.get('ret_bg_patch'), pct=True):>16}{fmt(dv.get('ret_bg_patch'), pct=True):>16}")

print()
print("Interpretation guide: if gap@bgpatch/gap@boxmap stays >= gap@full, DINOv3's edge")
print("survives lesion removal at least as well as it holds on the full image -- consistent")
print("with DINOv3 reading the SAME context/geometry shortcut more effectively, not with")
print("genuine extra lesion-reading. If the gap shrinks toward zero or flips negative under")
print("ablation, that is evidence DINOv3's advantage is concentrated in the lesion itself.")

for row in comparison:
    g = row.get("dinov3_minus_resnet_gap")
    if g is None or g["full"] is None or g["bg_patch"] is None:
        continue
    if g["full"] <= 1e-6:
        verdict_msg = "DINOv3 has no advantage on the full image here to begin with"
    elif g["bg_patch"] >= g["full"] * 0.8:
        verdict_msg = "SHORTCUT-READING — DINOv3's edge survives lesion removal almost fully"
    elif g["bg_patch"] > 0.3 * g["full"]:
        verdict_msg = "PARTIAL — DINOv3's edge shrinks but does not vanish without the lesion"
    else:
        verdict_msg = "GENUINE-LEANING — DINOv3's edge collapses once the lesion is removed"
    print(f"  {row['modality']:<4} {row['task']:<8} "
          f"gap full {g['full']:+.3f} -> bg_patch {g['bg_patch']:+.3f}   {verdict_msg}")

print(f"\nwrote {RESULTS_PATH}")
print(f"wrote {VERDICT_PATH}")
