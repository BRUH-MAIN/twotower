# %% [markdown]
# # Experiment 1 — Dataset-level confound audit
#
# Tests whether the label is predictable from variables carrying **no lesion
# morphology at all**: contributing centre, patient demographics, lesion size, and
# raw image geometry.
#
# Motivated by the review's Section 3 ("dataset construction artefacts",
# "confounding label sources in medicine") and Section 7.1 ("dataset-source
# homogeneity ... a persistent risk that a model learns 'which dataset/hospital'
# rather than 'which pathology'").
#
# No neural network on pixels here — this is dataset epidemiology. A confound
# found at this stage undermines every downstream result before a single
# convolutional weight is trained.

# %%
import sys, json, math
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
from PIL import Image

import openpyxl

from cadsel.config import METADATA_XLSX
from cadsel.data import build_manifest
from probes import probe, fmt

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)

# %%
# ---- load patient metadata -------------------------------------------------
wb = openpyxl.load_workbook(METADATA_XLSX, read_only=True)
ws = wb.active
rows = list(ws.iter_rows(values_only=True))
header = [str(h).strip() for h in rows[0]]


def col_for(prefix):
    for i, h in enumerate(header):
        if h.lower().startswith(prefix):
            return i
    return None


IDX = {k: col_for(k) for k in
       ("center", "category", "age", "sex", "size", "location", "name")}

patients = []
for r in rows[1:]:
    if r[0] is None:
        continue
    rec = {k: (r[i] if i is not None else None) for k, i in IDX.items()}
    rec["is_net"] = "NET" if str(rec["category"]).startswith("NET") else "non-NET"
    patients.append(rec)

print(f"{len(patients)} patients")
print("columns:", header)

# %% [markdown]
# ## 1.1 Centre x class contingency
#
# If subtypes are unevenly distributed across contributing centres, "which
# hospital" is statistically informative about "which pathology".

# %%
centers = sorted({str(p["center"]) for p in patients})
cats = sorted({str(p["category"]) for p in patients})
table = np.zeros((len(centers), len(cats)), dtype=int)
for p in patients:
    table[centers.index(str(p["center"])), cats.index(str(p["category"]))] += 1

w = max(len(c) for c in cats) + 2
print("Patients per centre x subtype\n")
print(" " * 12 + "".join(f"{c:>{w}}" for c in cats) + f"{'total':>9}")
for i, c in enumerate(centers):
    print(f"{c:<12}" + "".join(f"{v:>{w}}" for v in table[i]) + f"{table[i].sum():>9}")
print(f"{'total':<12}" + "".join(f"{v:>{w}}" for v in table.sum(0)))

print("\nClass mix within each centre (%)\n")
print(" " * 12 + "".join(f"{c:>{w}}" for c in cats))
for i, c in enumerate(centers):
    pct = 100 * table[i] / max(table[i].sum(), 1)
    print(f"{c:<12}" + "".join(f"{v:>{w}.1f}" for v in pct))


# %%
def cramers_v(conf: np.ndarray) -> float:
    """Cramer's V with Bergsma-Wicher bias correction."""
    n = conf.sum()
    expected = np.outer(conf.sum(1), conf.sum(0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(np.where(expected > 0, (conf - expected) ** 2 / expected, 0.0))
    phi2 = chi2 / n
    r, k = conf.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return math.sqrt(phi2corr / denom) if denom > 0 else 0.0


bin_table = np.zeros((len(centers), 2), dtype=int)
for p in patients:
    bin_table[centers.index(str(p["center"])), 0 if p["is_net"] == "NET" else 1] += 1

v_sub = cramers_v(table)
v_bin = cramers_v(bin_table)
print(f"Cramer's V  centre vs subtype    : {v_sub:.3f}")
print(f"Cramer's V  centre vs NET/non-NET: {v_bin:.3f}")
print("\n(0 = independent; >0.1 small, >0.3 moderate, >0.5 strong)")

# %% [markdown]
# ## 1.2 How much label information do non-image variables carry?
#
# MLP probes, 5-fold, grouped by patient (each patient is its own group here, so
# this is ordinary k-fold). Balanced accuracy is the headline because the class
# mix is ~15:1 — plain accuracy would be flattered by always predicting NET.

# %%
def encode(values):
    """Numeric passthrough; categorical -> one-hot."""
    vals = [v for v in values]
    if all(isinstance(v, (int, float)) and v is not None and not isinstance(v, bool)
           for v in vals):
        arr = np.array(vals, dtype=np.float64).reshape(-1, 1)
        return np.nan_to_num(arr, nan=float(np.nanmean(arr)))
    cats_ = sorted({str(v) for v in vals})
    out = np.zeros((len(vals), len(cats_)))
    for i, v in enumerate(vals):
        out[i, cats_.index(str(v))] = 1.0
    return out


def meta_probe(feature_keys, target_key, label):
    keep = [p for p in patients
            if all(p[k] is not None for k in feature_keys) and p[target_key] is not None]
    X = np.hstack([encode([p[k] for p in keep]) for k in feature_keys])
    y = [str(p[target_key]) for p in keep]
    groups = np.arange(len(keep))          # one patient per group
    r = probe(X, y, groups, n_splits=5, seed=0)
    r.update({"probe": label, "target": target_key,
              "features": "+".join(feature_keys), "n": len(keep)})
    print(fmt(f"{label} -> {target_key}", r))
    return r


meta_results = []
print("Target = 5-way subtype\n")
for keys, label in (
    (["center"], "centre only"),
    (["age", "sex"], "demographics"),
    (["size"], "lesion size"),
    (["location"], "location"),
    (["center", "age", "sex", "size", "location"], "all metadata"),
):
    meta_results.append(meta_probe(keys, "category", label))

print("\nTarget = NET vs non-NET\n")
for keys, label in (
    (["center"], "centre only"),
    (["size"], "lesion size"),
    (["center", "age", "sex", "size", "location"], "all metadata"),
):
    meta_results.append(meta_probe(keys, "is_net", label))

# %% [markdown]
# ## 1.3 Image geometry as a confound
#
# The paper states burned-in overlays were cropped away. But cropping leaves a
# trace: different devices and capture settings produce different frame
# dimensions. If the raw rectangle predicts the label, that is a shortcut a
# convolutional network can exploit for free.

# %%
img_rows = []
for modality in ("WLE", "EUS"):
    for s in build_manifest(modality):
        with Image.open(s.image) as im:
            w_, h_ = im.size
        img_rows.append({
            "modality": modality, "center": s.center, "subtype": s.subtype,
            "is_net": "NET" if s.subtype.startswith("NET") else "non-NET",
            "patient": s.patient_key,
            "width": w_, "height": h_, "aspect": w_ / h_, "pixels": w_ * h_,
        })
print(f"{len(img_rows)} images measured")

for modality in ("WLE", "EUS"):
    d = [r for r in img_rows if r["modality"] == modality]
    res = Counter((r["width"], r["height"]) for r in d)
    print(f"\n{modality}: {len(res)} distinct resolutions across {len(d)} images")
    for (w_, h_), n in res.most_common(6):
        print(f"   {w_:>5} x {h_:<5}  {n:>5} images")

# %%
# Are particular resolutions specific to particular subtypes?
for modality in ("WLE", "EUS"):
    d = [r for r in img_rows if r["modality"] == modality]
    print(f"\n=== {modality}: dominant subtype per resolution (>=40 images) ===")
    by_res = defaultdict(list)
    for r in d:
        by_res[(r["width"], r["height"])].append(r["subtype"])
    for res_, subs in sorted(by_res.items(), key=lambda kv: -len(kv[1])):
        if len(subs) < 40:
            continue
        c = Counter(subs).most_common(1)[0]
        print(f"   {res_[0]:>5} x {res_[1]:<5} n={len(subs):>5}  "
              f"dominant {c[0]:<12} {100*c[1]/len(subs):5.1f}%")

# %%
GEOM = ["width", "height", "aspect", "pixels"]
geo_results = []
for modality in ("WLE", "EUS"):
    d = [r for r in img_rows if r["modality"] == modality]
    X = np.array([[r[k] for k in GEOM] for r in d], dtype=np.float64)
    groups = np.array([r["patient"] for r in d])
    for target in ("subtype", "is_net", "center"):
        y = [r[target] for r in d]
        r_ = probe(X, y, groups, n_splits=5, seed=0)
        r_.update({"modality": modality, "target": target, "features": "geometry"})
        geo_results.append(r_)
        print(fmt(f"{modality} geometry -> {target}", r_))

# %% [markdown]
# ## 1.4 Verdict

# %%
summary = {
    "cramers_v_center_vs_subtype": round(v_sub, 4),
    "cramers_v_center_vs_binary": round(v_bin, 4),
    "metadata_probes": meta_results,
    "geometry_probes": geo_results,
}
(OUT / "01_dataset_confounds.json").write_text(json.dumps(summary, indent=2))

print("=" * 78)
print("CONFOUND AUDIT — lift is balanced accuracy above uniform chance")
print("=" * 78)
print(f"\nCramer's V, centre vs subtype: {v_sub:.3f}   "
      f"centre vs NET/non-NET: {v_bin:.3f}\n")

for r in meta_results:
    flag = "   <-- CONFOUND" if r["lift_over_chance_balanced"] > 0.10 else ""
    print(f"  {r['probe']:<16} -> {r['target']:<9} "
          f"bal {r['balanced_accuracy']:.3f} vs chance {r['chance_balanced']:.3f} "
          f"(lift {r['lift_over_chance_balanced']:+.3f}){flag}")
print()
for r in geo_results:
    flag = "   <-- CONFOUND" if r["lift_over_chance_balanced"] > 0.10 else ""
    print(f"  {r['modality']} geometry     -> {r['target']:<9} "
          f"bal {r['balanced_accuracy']:.3f} vs chance {r['chance_balanced']:.3f} "
          f"(lift {r['lift_over_chance_balanced']:+.3f}){flag}")
