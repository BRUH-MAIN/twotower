"""Experiment 50 -- what the two-tower predictions mean clinically.

CPU only, from 41's stored out-of-fold predictions and 49's clinical models.
49 showed location carries the binary task on all patients, so these analyses
ask what is left once location cannot help.

  * ``eus_value.csv``: does EUS add anything, on patients with EUS and on the
    rectal subset of them? Views of the two-tower (WLE, EUS, fused, late fusion
    of its own views) and, where they exist, the separately trained models.
  * ``subgroups.csv``: image and clinical models by lesion size, location, EUS
    availability, centre, sex and age. Cells with fewer than 5 patients of a
    class are flagged ``small``.
  * ``merged_net.csv``: the subtype task with NET G1 and G2 summed into one
    class. Grade comes from pathology and may not be visible in the images.
  * ``operating_points.csv``: thresholds chosen on the other four folds rather
    than on the patients being scored, so unlike 44's these are not optimistic.
  * ``decision_curve.csv`` / ``calibration.csv``: net benefit of treating a
    lesion as a NET. Class-balanced training distorts probabilities, so they
    are first recalibrated on the other four folds.

    python -m experiments.src.50_two_tower_subgroups
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import PROJECT_ROOT
from cadsel.fusion import patient_classes

exp49 = importlib.import_module("experiments.src.49_clinical_baseline")
exp47 = importlib.import_module("experiments.src.47_cross_dataset_comparison")
exp44 = exp49.exp44

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "50_two_tower_subgroups"
FOCUS = ("dinov3_vits16", "efficientnet_b0")
CLINICAL = ("clinical", "logreg")
SIZE_BANDS = (("size <0.5 cm", 0.0, 0.5), ("size 0.5-1 cm", 0.5, 1.0), ("size >=1 cm", 1.0, np.inf))
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
TARGETS = (("sens", 0.95), ("sens", 0.90), ("spec", 0.90))


def score(y: np.ndarray, probs: np.ndarray, idx: np.ndarray, n_classes: int) -> dict:
    lo, hi = np.percentile(exp49.boot_bal(y, probs.argmax(1), idx, n_classes), [2.5, 97.5])
    return {"bal": exp44.bal_acc(y, probs.argmax(1), n_classes), "bal_lo": float(lo), "bal_hi": float(hi),
            "auroc": exp44.auroc(y, probs)}


# ---------------------------------------------------------------------------
# What EUS adds
# ---------------------------------------------------------------------------
def single_models(groups: dict, task: str, backbone: str, tag: str, keys: np.ndarray) -> dict:
    """The separately trained WLE and EUS models on the rows of ``keys`` that
    have EUS, or {} until every seed the tag needs exists."""
    picked = {}
    for mod in ("WLE", "EUS"):
        runs = groups.get(("cadsel", task, mod, backbone), [])
        runs = [r for r in runs if r["seed"] == 42] if tag == "s42" else runs
        if len(runs) != (1 if tag == "s42" else 3):
            return {}
        probs = []
        for r in runs:
            pos = {k: i for i, k in enumerate(r["patient_key"].tolist())}
            probs.append(r["probs_fused"][[pos[k] for k in keys]])
        picked[f"{mod.lower()}_model"] = np.mean(probs, axis=0)
    picked["late_models"] = (picked["wle_model"] + picked["eus_model"]) / 2
    return picked


def eus_value(sources: dict, meta: dict) -> list[dict]:
    groups = exp47.load_runs()
    rows = []
    for task in exp49.TASKS:
        n_classes = len(patient_classes(task))
        for tag in exp49.TAGS:
            for backbone, src in sources[(task, tag)].items():
                keys = src["patient_key"]
                rectal = exp49.population_mask("rectum", keys, src["paired"], meta)
                for pop, m in (("eus", src["paired"]), ("rectum_eus", src["paired"] & rectal)):
                    y = src["y"][m]
                    idx = exp44.boot_indices(y, np.random.default_rng(0))
                    views = {"wle_view": src["probs_wle_only"][m], "eus_view": src["probs_eus_only"][m],
                             "fused": src["probs_fused"][m]}
                    views["late_views"] = (views["wle_view"] + views["eus_view"]) / 2
                    views.update(single_models(groups, task, backbone, tag, keys[m]))
                    bal = {k: exp49.boot_bal(y, p.argmax(1), idx, n_classes) for k, p in views.items()}
                    for view, probs in views.items():
                        base = "wle_model" if view in ("eus_model", "late_models") else "wle_view"
                        row = {"task": task, "seeds": tag, "population": pop, "n": int(m.sum()),
                               "backbone": backbone, "view": view, **score(y, probs, idx, n_classes),
                               "versus": "", "delta_bal": None, "delta_lo": None, "delta_hi": None, "p": None}
                        if view != base and view != "wle_model":
                            d, lo, hi, p = exp49.delta(bal[view], bal[base])
                            row.update(versus=base, delta_bal=d, delta_lo=lo, delta_hi=hi, p=p)
                        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Subgroups and the merged task
# ---------------------------------------------------------------------------
def subgroup_masks(keys: np.ndarray, paired: np.ndarray, meta: dict, centres: dict) -> dict:
    size = np.array([meta[k]["size"] for k in keys])
    age = np.array([meta[k]["age"] for k in keys])
    rectal = np.array([meta[k]["location"] == "rectum" for k in keys])
    out = {name: (size >= lo) & (size < hi) for name, lo, hi in SIZE_BANDS}
    out.update({"rectum": rectal, "colon": ~rectal, "EUS present": paired, "EUS absent": ~paired,
                "Center-1": np.array([centres[k] == "Center-1" for k in keys]),
                "Centres 2-6": np.array([centres[k] != "Center-1" for k in keys]),
                "female": np.array([meta[k]["sex"] == 0 for k in keys]),
                "male": np.array([meta[k]["sex"] == 1 for k in keys]),
                "age <50": age < 50, "age >=50": age >= 50})
    for name, lo, hi in SIZE_BANDS:
        out[f"rectum, {name}"] = rectal & out[name]
    return out


def subgroups(sources: dict, clinical: dict, meta: dict, centres: dict) -> list[dict]:
    rows = []
    for task in exp49.TASKS:
        n_classes = len(patient_classes(task))
        by_bb = sources[(task, "ens")]
        any_src = next(iter(by_bb.values()))
        keys, y_all = any_src["patient_key"], any_src["y"]
        models = {b: by_bb[b]["probs_fused"] for b in FOCUS}
        models["clinical"] = clinical[task][CLINICAL]["ens"]
        for b in FOCUS:
            models[f"{b}+clinical"] = (models[b] + models["clinical"]) / 2
        for group, m in subgroup_masks(keys, any_src["paired"], meta, centres).items():
            y = y_all[m]
            counts = np.bincount(y, minlength=n_classes)
            usable = [c for c in range(n_classes) if counts[c] > 0]
            for name, probs in models.items():
                pred = probs[m].argmax(1)
                rows.append({"task": task, "group": group, "n": int(m.sum()), "class_counts": " ".join(map(str, counts)),
                             "small": bool(counts[usable].min() < 5), "model": name,
                             "bal": exp44.bal_acc(y, pred, n_classes),
                             "auroc": exp44.auroc(y, probs[m]) if len(usable) == n_classes else None})
    return rows


def merged_net(sources: dict, clinical: dict) -> list[dict]:
    """NET G1 and G2 summed: NET, leiomyoma, lipoma, non-neoplasm."""
    def merge(p):
        return np.column_stack([p[:, :2].sum(1), p[:, 2:]])

    rows = []
    for tag, by_bb in ((t, sources[("subtype", t)]) for t in exp49.TAGS):
        y5 = next(iter(by_bb.values()))["y"]
        y = np.where(y5 <= 1, 0, y5 - 1)
        idx = exp44.boot_indices(y, np.random.default_rng(0))
        entries = {b: src["probs_fused"] for b, src in by_bb.items()}
        entries["clinical"] = clinical["subtype"][CLINICAL][tag]
        for name, probs in entries.items():
            pm = merge(probs)
            rec = [float((pm[y == c].argmax(1) == c).mean()) for c in range(4)]
            rows.append({"seeds": tag, "model": name, **score(y, pm, idx, 4), "recall_net": rec[0],
                         "recall_leiomyoma": rec[1], "recall_lipoma": rec[2], "recall_nonneoplasm": rec[3],
                         "bal_5way": exp44.bal_acc(y5, probs.argmax(1), 5)})
    return rows


# ---------------------------------------------------------------------------
# Operating points, calibration, net benefit (binary; NET is class 0)
# ---------------------------------------------------------------------------
def threshold_for(net: np.ndarray, p: np.ndarray, kind: str, target: float) -> float:
    """Largest threshold reaching a target sensitivity, or smallest reaching a
    target specificity, on the patients given."""
    if kind == "sens":
        return float(np.quantile(p[net], 1 - target, method="lower"))
    return float(np.quantile(p[~net], target, method="higher"))


def cross_fit_points(net: np.ndarray, p: np.ndarray, fold: np.ndarray, kind: str, target: float) -> dict:
    called = np.zeros(len(net), bool)
    for f in np.unique(fold):
        test = fold == f
        called[test] = p[test] >= threshold_for(net[~test], p[~test], kind, target)
    tp, fp = (called & net).sum(), (called & ~net).sum()
    fn, tn = (~called & net).sum(), (~called & ~net).sum()
    return {"sens": tp / (tp + fn), "spec": tn / (tn + fp), "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1),
            "missed_net": int(fn), "false_alarm": int(fp)}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def recalibrate(net: np.ndarray, p: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Logistic recalibration, each fold fitted on the other four."""
    out = np.zeros(len(p))
    z = logit(p).reshape(-1, 1)
    for f in np.unique(fold):
        test = fold == f
        fit = LogisticRegression(C=1e6, max_iter=1000).fit(z[~test], net[~test])
        out[test] = fit.predict_proba(z[test])[:, 1]
    return out


def net_benefit(net: np.ndarray, p: np.ndarray, t: float) -> float:
    called = p >= t
    n = len(net)
    return float((called & net).sum() / n - (called & ~net).sum() / n * t / (1 - t))


def clinical_utility(sources: dict, clinical: dict, meta: dict, fold: np.ndarray) -> tuple[list, list, list]:
    by_bb = sources[("binary", "ens")]
    any_src = next(iter(by_bb.values()))
    keys, net_all = any_src["patient_key"], any_src["y"] == 0
    rectal = exp49.population_mask("rectum", keys, any_src["paired"], meta)
    models = {b: by_bb[b]["probs_fused"][:, 0] for b in FOCUS}
    models["clinical"] = clinical["binary"][CLINICAL]["ens"][:, 0]
    for b in FOCUS:
        models[f"{b}+clinical"] = (models[b] + models["clinical"]) / 2
    points, curve, calib = [], [], []
    for pop, m in (("all", np.ones(len(keys), bool)), ("rectum", rectal)):
        net, f = net_all[m], fold[m]
        for name, p_all in models.items():
            p = p_all[m]
            for kind, target in TARGETS:
                points.append({"population": pop, "model": name, "chosen_for": f"{kind} {target:.2f}",
                               "n": int(m.sum()), **cross_fit_points(net, p, f, kind, target)})
            cal = recalibrate(net, p, f)
            fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), net)
            calib.append({"population": pop, "model": name, "prevalence": float(net.mean()),
                          "mean_p_raw": float(p.mean()), "mean_p_recal": float(cal.mean()),
                          "slope_raw": float(fit.coef_[0, 0]), "intercept_raw": float(fit.intercept_[0]),
                          "ece_raw": exp44.ece(np.column_stack([p, 1 - p]), (~net).astype(int))[0],
                          "ece_recal": exp44.ece(np.column_stack([cal, 1 - cal]), (~net).astype(int))[0]})
            curve += [{"population": pop, "model": name, "threshold": float(t), "net_benefit": net_benefit(net, cal, t)}
                      for t in THRESHOLDS]
        curve += [{"population": pop, "model": "treat_all", "threshold": float(t),
                   "net_benefit": net_benefit(net, np.ones(len(net)), t)} for t in THRESHOLDS]
    return points, curve, calib


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = exp49.load_metadata()
    sources = exp44.load_sources()
    centres = dict(np.loadtxt(PROJECT_ROOT / "experiments" / "data" / "patient_centre.csv", delimiter=",",
                              skiprows=1, dtype=str))
    clinical = {t: exp49.clinical_sources(meta, t, len(patient_classes(t))) for t in exp49.TASKS}
    fold = exp49.seed_folds("binary")[42]["fold"]

    eus = eus_value(sources, meta)
    groups = subgroups(sources, clinical, meta, centres)
    merged = merged_net(sources, clinical)
    points, curve, calib = clinical_utility(sources, clinical, meta, fold)
    for name, rows in (("eus_value", eus), ("subgroups", groups), ("merged_net", merged),
                       ("operating_points", points), ("decision_curve", curve), ("calibration", calib)):
        exp44.write_csv(OUT_DIR / f"{name}.csv", rows)

    print("\nWhat EUS adds (balanced accuracy, AUROC; delta vs the WLE counterpart)")
    for r in eus:
        if r["backbone"] in FOCUS:
            d = "" if r["delta_bal"] is None else (f"  {r['delta_bal']:+.3f} [{r['delta_lo']:+.3f}, {r['delta_hi']:+.3f}] "
                                                   f"{exp44.fmt_p(r['p'])}")
            print(f"  {r['task']:8s}{r['seeds']:4s}{r['population']:11s}{r['n']:4d} {r['backbone']:16s}{r['view']:12s}"
                  f"{r['bal']:.3f} {r['auroc']:.3f}{d}")


if __name__ == "__main__":
    main()
