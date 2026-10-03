# %% [markdown]
# # Experiment 2 — Low-level image-statistic probes
#
# A *deliberately engineered confound probe* in the sense of the review's Section
# 4.4: build predictors with **no access to lesion morphology** — only global
# colour, exposure and geometry statistics of the frame — and measure how much
# they recover.
#
# Two questions:
#
# 1. **Is the acquisition centre decodable from image statistics?** If so the
#    dataset carries a device/processor fingerprint (review §7.1) — the raw
#    material for a "learned the site, not the disease" shortcut.
# 2. **Is the lesion class decodable from those same statistics?** A tree on colour
#    histograms cannot find or characterise a subepithelial lesion. Any lift over
#    chance is label information travelling a non-clinical route.
#
# Two probe families (MLP and random forest) are run so the conclusion does not
# hinge on one model class. All cross-validation is **grouped by patient**.

# %%
import sys, json
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
from PIL import Image

from cadsel.data import build_manifest
from probes import probe, fmt, grouped_folds, balanced_accuracy

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "02_features.npz"

# %%
def features_for(path: str) -> list[float]:
    """Cheap global statistics. None of these encode lesion shape or position."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        small = np.asarray(im.resize((128, 128), Image.BILINEAR), dtype=np.float32) / 255.0

    f = [w, h, w / h, w * h]
    for i in range(3):
        ch = small[..., i]
        f += [ch.mean(), ch.std(),
              np.percentile(ch, 10), np.percentile(ch, 50), np.percentile(ch, 90)]
        hist, _ = np.histogram(ch, bins=8, range=(0, 1), density=True)
        f += list(hist)

    gray = small @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    f += [gray.mean(), gray.std()]
    f += [float((gray < 0.06).mean())]     # letterbox / field-of-view crop
    f += [float((gray > 0.94).mean())]     # specular highlight fraction
    mx, mn = small.max(axis=2), small.min(axis=2)
    f += [float((mx - mn).mean())]         # colourfulness
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
           - 4 * gray[1:-1, 1:-1])
    f += [float(lap.var())]                # sharpness / noise proxy
    return [float(v) for v in f]


NAMES = (["width", "height", "aspect", "pixels"]
         + [f"{c}_{s}" for c in "rgb"
            for s in ["mean", "std", "p10", "p50", "p90"] + [f"h{b}" for b in range(8)]]
         + ["gray_mean", "gray_std", "black_frac", "white_frac",
            "saturation", "laplacian_var"])

if CACHE.exists():
    z = np.load(CACHE, allow_pickle=True)
    X_all, meta_all = z["X"], z["meta"]
    print(f"loaded cached features {X_all.shape}")
else:
    feats, metas = [], []
    for modality in ("WLE", "EUS"):
        samples = build_manifest(modality)
        for i, s in enumerate(samples):
            feats.append(features_for(s.image))
            metas.append([modality, s.center, s.subtype,
                          "NET" if s.subtype.startswith("NET") else "non-NET",
                          s.patient_key])
            if (i + 1) % 750 == 0:
                print(f"  {modality} {i+1}/{len(samples)}")
    X_all = np.array(feats, dtype=np.float64)
    meta_all = np.array(metas, dtype=object)
    np.savez_compressed(CACHE, X=X_all, meta=meta_all)
    print(f"computed features {X_all.shape}")

print(f"{X_all.shape[1]} features per image, {len(NAMES)} names")
COLS = {n: i for i, n in enumerate(NAMES)}

# %%
from sklearn.ensemble import RandomForestClassifier

def rf_probe(X, y_labels, groups, n_splits=5, seed=0):
    classes = sorted(set(map(str, y_labels)))
    idx = {c: i for i, c in enumerate(classes)}
    y = np.array([idx[str(v)] for v in y_labels])
    accs, bals, bases = [], [], []
    for tr, te in grouped_folds(groups, n_splits, seed):
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     class_weight="balanced_subsample",
                                     random_state=seed, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(float((pred == y[te]).mean()))
        bals.append(balanced_accuracy(y[te], pred, len(classes)))
        maj = np.bincount(y[tr], minlength=len(classes)).argmax()
        bases.append(float((y[te] == maj).mean()))
    return {
        "n_classes": len(classes),
        "accuracy": round(float(np.mean(accs)), 4),
        "balanced_accuracy": round(float(np.mean(bals)), 4),
        "majority_baseline": round(float(np.mean(bases)), 4),
        "chance_balanced": round(1 / len(classes), 4),
        "lift_over_majority": round(float(np.mean(accs) - np.mean(bases)), 4),
        "lift_over_chance_balanced": round(float(np.mean(bals) - 1 / len(classes)), 4),
    }


def subset(modality):
    m = meta_all[:, 0] == modality
    return X_all[m], meta_all[m]


# %% [markdown]
# ## 2.1 Probe A — is the acquisition centre decodable?

# %%
results = []
for modality in ("WLE", "EUS"):
    X, M = subset(modality)
    groups = M[:, 4]
    for name, fn in (("MLP", probe), ("RF", rf_probe)):
        r = fn(X, M[:, 1], groups)
        r.update({"modality": modality, "probe": f"A/{name}", "target": "center",
                  "features": "all image stats"})
        results.append(r)
        print(fmt(f"{modality} {name}: centre from image stats", r))

# %% [markdown]
# ## 2.2 Probe B — is the lesion class decodable from the same statistics?
#
# The key test. These features cannot see a lesion.

# %%
for modality in ("WLE", "EUS"):
    X, M = subset(modality)
    groups = M[:, 4]
    for target, col in (("subtype", 2), ("is_net", 3)):
        for name, fn in (("MLP", probe), ("RF", rf_probe)):
            r = fn(X, M[:, col], groups)
            r.update({"modality": modality, "probe": f"B/{name}", "target": target,
                      "features": "all image stats"})
            results.append(r)
            print(fmt(f"{modality} {name}: {target} from image stats", r))

# %% [markdown]
# ## 2.3 Probe C — colour/exposure only, geometry removed
#
# Probe B mixes appearance with frame dimensions, and Experiment 1 already showed
# geometry identifies the centre. Dropping the four geometry columns isolates
# whether *appearance alone* carries label information.

# %%
APPEARANCE = [i for n, i in COLS.items()
              if n not in ("width", "height", "aspect", "pixels")]
for modality in ("WLE", "EUS"):
    X, M = subset(modality)
    Xa = X[:, APPEARANCE]
    groups = M[:, 4]
    for target, col in (("subtype", 2), ("is_net", 3), ("center", 1)):
        r = rf_probe(Xa, M[:, col], groups)
        r.update({"modality": modality, "probe": "C/RF", "target": target,
                  "features": "appearance only"})
        results.append(r)
        print(fmt(f"{modality} RF: {target} from appearance only", r))

# %% [markdown]
# ## 2.4 Which statistics carry the signal?

# %%
for modality in ("WLE", "EUS"):
    X, M = subset(modality)
    for target, col in (("center", 1), ("subtype", 2)):
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     class_weight="balanced_subsample",
                                     random_state=0, n_jobs=-1)
        clf.fit(X, M[:, col])
        order = np.argsort(clf.feature_importances_)[::-1][:8]
        top = ", ".join(f"{NAMES[i]} {100*clf.feature_importances_[i]:.1f}%"
                        for i in order)
        print(f"{modality} -> {target:<8}: {top}")

# %% [markdown]
# ## 2.5 Verdict

# %%
json.dump(results, open(OUT / "02_lowlevel_probes.json", "w"), indent=2)

print("=" * 84)
print("LOW-LEVEL PROBES — balanced accuracy vs uniform chance")
print("=" * 84)
hdr = f"{'modality':<9}{'probe':<9}{'features':<18}{'target':<9}{'bal':>7}{'chance':>8}{'lift':>8}"
print(hdr)
print("-" * len(hdr))
for r in results:
    flag = "  <-- decodable" if r["lift_over_chance_balanced"] > 0.10 else ""
    print(f"{r['modality']:<9}{r['probe']:<9}{r['features']:<18}{r['target']:<9}"
          f"{r['balanced_accuracy']:>7.3f}{r['chance_balanced']:>8.3f}"
          f"{r['lift_over_chance_balanced']:>+8.3f}{flag}")

centre = [r for r in results if r["target"] == "center"]
klass = [r for r in results if r["target"] in ("subtype", "is_net")]
print(f"\ncentre probes : max lift {max(r['lift_over_chance_balanced'] for r in centre):+.3f}")
print(f"class probes  : max lift {max(r['lift_over_chance_balanced'] for r in klass):+.3f}")
