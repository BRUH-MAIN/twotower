"""Experiment 49 -- clinical baselines for the two-tower ablation.

01 found the label predictable from metadata.xlsx, but nothing set that against
the image models. It matters: 454 of the 455 NETs are rectal, so "rectum => NET"
is already a strong binary classifier, and the open question is what images add
to what the endoscopist knows anyway.

CPU only, on the outer folds stored with 41's out-of-fold predictions, so every
clinical prediction comes from a model that never saw that patient and shares
the image model's split.

  * Clinical models: logistic regression (primary; C chosen by inner CV on the
    outer-train patients) and gradient boosting (fixed small configuration).
    Both class-balanced, like the image models' weighted CE. Size is numeric
    here; 01 one-hot encoded it, because two patients have "/" for size.
  * ``clinical_no_size`` is reported alongside ``clinical``: the dataset paper
    does not say how size was measured, and a size from the resected specimen
    would not be known before diagnosis.
  * Image vs. clinical vs. the mean of their probabilities (no fitted weights,
    as in 44's and 47's late fusion), for seed 42 and the 3-seed ensembles.
    Paired bootstrap of each difference, Holm-corrected over the backbones.
  * Three populations: all patients, patients with EUS, and rectal lesions,
    where location can no longer separate the classes. Clinical models are not
    refitted per population, so rectal balanced accuracy keeps the threshold
    learned on everyone; AUROC is the fairer reading there.

    python -m experiments.src.49_clinical_baseline
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import openpyxl
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import METADATA_XLSX, PROJECT_ROOT
from cadsel.fusion import patient_classes

exp44 = importlib.import_module("experiments.src.44_two_tower_stats")
exp42 = exp44.exp42

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "49_clinical_baseline"
SEEDS = (42, 123, 7)
TAGS = ("s42", "ens")
TASKS = ("binary", "subtype")
POPULATIONS = ("all", "eus", "rectum")
FEATURE_SETS = {
    "location": ("location",),
    "size": ("size",),
    "demographics": ("age", "sex"),
    "clinical_no_size": ("age", "sex", "location"),
    "clinical": ("age", "sex", "size", "location"),
}
COMPARED = ("clinical", "clinical_no_size")
MODELS = ("logreg", "gboost")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_metadata() -> dict:
    """patient_key -> age, sex, size (cm, NaN where recorded as "/"), location."""
    rows = list(openpyxl.load_workbook(METADATA_XLSX, read_only=True).active.iter_rows(values_only=True))
    header = [str(h).strip().lower() for h in rows[0]]
    col = {k: next(i for i, h in enumerate(header) if h.startswith(k))
           for k in ("category", "name", "age", "sex", "size", "location")}
    meta = {}
    for r in rows[1:]:
        if r[0] is None:
            continue
        size = r[col["size"]]
        meta[f"{r[col['category']]}/{r[col['name']]}"] = {
            "age": float(r[col["age"]]), "sex": float(r[col["sex"]]),
            "size": float("nan") if isinstance(size, str) else float(size),
            "location": str(r[col["location"]]).strip()}
    return meta


def seed_folds(task: str) -> dict:
    """seed -> patient_key / fold / y, rows sorted by patient. Every backbone
    must share them, or the comparison would not be on one split."""
    out = {}
    for run in exp42.load_oof():
        if run["task"] != task or run["seed"] not in SEEDS:
            continue
        order = np.argsort(run["patient_key"])
        cols = {k: run[k][order] for k in ("patient_key", "fold", "y")}
        ref = out.setdefault(run["seed"], cols)
        assert all(np.array_equal(ref[k], cols[k]) for k in cols), f"split differs in {run['run_name']}"
    assert sorted(out) == sorted(SEEDS), f"missing seeds for {task}: {sorted(out)}"
    return out


def design(meta: dict, keys: np.ndarray, features: tuple) -> np.ndarray:
    locations = sorted({m["location"] for m in meta.values()})
    cols = []
    for f in features:
        if f == "location":
            cols += [[float(meta[k]["location"] == loc) for k in keys] for loc in locations]
        else:
            cols.append([meta[k][f] for k in keys])
    return np.array(cols, dtype=np.float64).T


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def make_model(kind: str, seed: int):
    if kind == "logreg":
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegressionCV(Cs=np.logspace(-2, 2, 9).tolist(), cv=inner, max_iter=5000,
                                                  scoring="balanced_accuracy", class_weight="balanced"))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          class_weight="balanced", random_state=seed)


def oof_probs(X: np.ndarray, y: np.ndarray, fold: np.ndarray, kind: str, seed: int, n_classes: int) -> np.ndarray:
    probs = np.zeros((len(y), n_classes))
    for f in np.unique(fold):
        train, test = fold != f, fold == f
        assert len(np.unique(y[train])) == n_classes, f"fold {f} lost a class"
        probs[test] = make_model(kind, seed).fit(X[train], y[train]).predict_proba(X[test])
    return probs


def clinical_sources(meta: dict, task: str, n_classes: int) -> dict:
    """(feature set, model) -> {tag: (n, C) probabilities}, rows sorted by patient."""
    folds = seed_folds(task)
    out = {}
    for name, features in FEATURE_SETS.items():
        for kind in MODELS:
            per_seed = {s: oof_probs(design(meta, folds[s]["patient_key"], features), folds[s]["y"],
                                     folds[s]["fold"], kind, s, n_classes) for s in SEEDS}
            out[(name, kind)] = {"s42": per_seed[42], "ens": np.mean(list(per_seed.values()), axis=0)}
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def boot_bal(y: np.ndarray, pred: np.ndarray, idx: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced accuracy of every bootstrap row of ``idx`` at once."""
    yb, pb = y[idx], pred[idx]
    return np.mean([((yb == c) & (pb == c)).sum(1) / (yb == c).sum(1)
                    for c in range(n_classes) if (y == c).any()], axis=0)


def boot_auroc(y: np.ndarray, p_net: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """AUROC of the NET probability per bootstrap row, in rank form."""
    pos = y[idx] == 0
    n_pos, n_neg = pos.sum(1), (~pos).sum(1)
    return ((rankdata(p_net[idx], axis=1) * pos).sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def score_row(y: np.ndarray, probs: np.ndarray, idx: np.ndarray, n_classes: int) -> dict:
    pred = probs.argmax(1)
    lo, hi = np.percentile(boot_bal(y, pred, idx, n_classes), [2.5, 97.5])
    row = {"bal": exp44.bal_acc(y, pred, n_classes), "bal_lo": float(lo), "bal_hi": float(hi),
           "macro_f1": exp44.macro_f1(y, pred, n_classes), "auroc": exp44.auroc(y, probs),
           "sens_net": None, "spec_net": None}
    if n_classes == 2:
        row.update(sens_net=float((pred[y == 0] == 0).mean()), spec_net=float((pred[y == 1] == 1).mean()))
    return row


def delta(a: np.ndarray, b: np.ndarray) -> tuple:
    """Mean, 95% CI and two-sided bootstrap p of a - b, as in 44."""
    d = a - b
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))


def population_mask(pop: str, keys: np.ndarray, paired: np.ndarray, meta: dict) -> np.ndarray:
    if pop == "all":
        return np.ones(len(keys), bool)
    return paired if pop == "eus" else np.array([meta[k]["location"] == "rectum" for k in keys])


def baselines(task: str, sources: dict, clinical: dict, meta: dict) -> list[dict]:
    n_classes = len(patient_classes(task))
    rows = []
    for tag in TAGS:
        by_bb = sources[(task, tag)]
        any_src = next(iter(by_bb.values()))
        keys, y_all = any_src["patient_key"], any_src["y"]
        rectal = np.array([meta[k]["location"] == "rectum" for k in keys])
        for pop in POPULATIONS:
            m = population_mask(pop, keys, any_src["paired"], meta)
            y = y_all[m]
            idx = exp44.boot_indices(y, np.random.default_rng(0))
            entries = [(kind, name, probs[tag][m]) for (name, kind), probs in clinical.items()]
            entries += [("two_tower", b, by_bb[b]["probs_fused"][m]) for b in exp44.ORDER if b in by_bb]
            if n_classes == 2:
                entries.insert(0, ("rule", "rectum_means_net", np.stack([rectal, ~rectal], 1)[m].astype(float)))
            for model, inputs, probs in entries:
                rows.append({"task": task, "seeds": tag, "population": pop, "n": int(m.sum()), "model": model,
                             "inputs": inputs, **score_row(y, probs, idx, n_classes)})
    return rows


def image_vs_clinical(task: str, sources: dict, clinical: dict, meta: dict) -> list[dict]:
    n_classes = len(patient_classes(task))
    rows = []
    for tag in TAGS:
        by_bb = sources[(task, tag)]
        any_src = next(iter(by_bb.values()))
        for pop in POPULATIONS:
            m = population_mask(pop, any_src["patient_key"], any_src["paired"], meta)
            y = any_src["y"][m]
            idx = exp44.boot_indices(y, np.random.default_rng(0))
            for name in COMPARED:
                clin = clinical[(name, "logreg")][tag][m]
                panel = []
                for backbone in [b for b in exp44.ORDER if b in by_bb]:
                    image = by_bb[backbone]["probs_fused"][m]
                    both = (image + clin) / 2
                    bal = {k: boot_bal(y, p.argmax(1), idx, n_classes)
                           for k, p in (("image", image), ("clinical", clin), ("both", both))}
                    row = {"task": task, "seeds": tag, "population": pop, "n": int(m.sum()),
                           "clinical": name, "backbone": backbone}
                    for k, p in (("image", image), ("clinical", clin), ("both", both)):
                        row[f"bal_{k}"] = exp44.bal_acc(y, p.argmax(1), n_classes)
                        row[f"auroc_{k}"] = exp44.auroc(y, p)
                    for label, a, b in (("both_minus_clinical", "both", "clinical"),
                                        ("both_minus_image", "both", "image"),
                                        ("clinical_minus_image", "clinical", "image")):
                        d, lo, hi, p = delta(bal[a], bal[b])
                        row.update({f"{label}": d, f"{label}_lo": lo, f"{label}_hi": hi, f"{label}_p": p})
                    if n_classes == 2:
                        auc = {k: boot_auroc(y, p[:, 0], idx)
                               for k, p in (("image", image), ("clinical", clin), ("both", both))}
                        for label, a, b in (("auroc_both_minus_clinical", "both", "clinical"),
                                            ("auroc_clinical_minus_image", "clinical", "image")):
                            d, lo, hi, p = delta(auc[a], auc[b])
                            row.update({f"{label}": d, f"{label}_lo": lo, f"{label}_hi": hi, f"{label}_p": p})
                    panel.append(row)
                for key in [k for k in panel[0] if k.endswith("_p")]:
                    for r, adj in zip(panel, exp44.holm([r[key] for r in panel])):
                        r[f"{key}_holm"] = adj
                rows += panel
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_metadata()
    sources = exp44.load_sources()
    base, versus = [], []
    for task in TASKS:
        n_classes = len(patient_classes(task))
        keys = next(iter(sources[(task, "ens")].values()))["patient_key"]
        assert set(keys) == set(meta), "metadata and predictions cover different patients"
        clinical = clinical_sources(meta, task, n_classes)
        base += baselines(task, sources, clinical, meta)
        versus += image_vs_clinical(task, sources, clinical, meta)
    exp44.write_csv(OUT_DIR / "baselines.csv", base)
    exp44.write_csv(OUT_DIR / "image_vs_clinical.csv", versus)

    print("\n3-seed ensembles, balanced accuracy [95% CI] and AUROC")
    for r in base:
        if r["seeds"] == "ens" and (r["model"] in ("rule", "logreg") or r["inputs"] in ("dinov3_vits16", "efficientnet_b0")):
            print(f"  {r['task']:8s}{r['population']:8s}{r['n']:5d}  {r['model']:10s}{r['inputs']:19s}"
                  f"{r['bal']:.3f} [{r['bal_lo']:.3f}, {r['bal_hi']:.3f}]  auroc {r['auroc']:.3f}")
    print("\nDoes the image add to clinical data? 3-seed ensembles, balanced accuracy")
    for r in versus:
        if r["seeds"] == "ens" and r["backbone"] in ("dinov3_vits16", "efficientnet_b0"):
            print(f"  {r['task']:8s}{r['population']:8s}{r['clinical']:17s}{r['backbone']:17s}"
                  f"image {r['bal_image']:.3f} clinical {r['bal_clinical']:.3f} both {r['bal_both']:.3f}  "
                  f"both-clinical {r['both_minus_clinical']:+.3f} "
                  f"[{r['both_minus_clinical_lo']:+.3f}, {r['both_minus_clinical_hi']:+.3f}] "
                  f"{exp44.fmt_p(r['both_minus_clinical_p_holm'])}")


if __name__ == "__main__":
    main()
