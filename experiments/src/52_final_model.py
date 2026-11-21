"""Experiment 52 -- candidate final models: image, clinical, and the two combined.

CPU only, from 41's out-of-fold predictions (via 51's run loader) and 49's
clinical models.

Image candidates: the two-tower re-baseline (lesion crop), the full-frame
two-tower (no boxes needed) and the warm-started two-tower, each DINOv3 and
EfficientNet, at seed 42 and as 3-seed ensembles. Each is combined with the
clinical variables two ways:

  * ``average``: the mean of the image and clinical probabilities (no fitting).
  * ``stacked``: logistic regression on the image log-probabilities plus the
    clinical variables, cross-fitted at the outer-fold level: each test fold
    is predicted by a model fitted on the other folds' out-of-fold image
    predictions and labels. The image models behind those predictions were
    trained with the test fold's patients, so the meta-model sees them only
    through a handful of fitted weights; reported as such.

Clinical variables come with and without lesion size (its provenance is
unknown; see 49).

  * ``performance.csv``: balanced accuracy [95% CI] and AUROC on all patients,
    patients with EUS and rectal lesions, and each combination minus the
    clinical model (paired bootstrap, Holm over the combinations of a panel).
  * ``operating_points.csv``, ``decision_curve.csv``, ``calibration.csv``:
    binary only, as 50 does them (thresholds and recalibration cross-fitted
    on the other folds).

    python -m experiments.src.52_final_model
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import PROJECT_ROOT
from cadsel.fusion import patient_classes

exp51 = importlib.import_module("experiments.src.51_phase1_comparisons")
exp50 = importlib.import_module("experiments.src.50_two_tower_subgroups")
exp49, exp44 = exp51.exp49, exp51.exp44

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "52_final_model"
CANDIDATES = ("baseline", "full_frame", "warm_start")
CLINICAL_SETS = ("clinical", "clinical_no_size")


def stacked(p_img: np.ndarray, X: np.ndarray, y: np.ndarray, fold: np.ndarray, seed: int = 0) -> np.ndarray:
    """Out-of-fold probabilities of a logistic model on [image log-probs, clinical]."""
    img = np.log(np.clip(p_img, 1e-6, 1.0))
    feats = np.column_stack([img[:, :1] - img[:, 1:2] if p_img.shape[1] == 2 else img, X])
    out = np.zeros_like(p_img)
    for f in np.unique(fold):
        train, test = fold != f, fold == f
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegressionCV(Cs=np.logspace(-2, 2, 9).tolist(), cv=inner, max_iter=5000,
                                                   scoring="neg_log_loss", class_weight="balanced"))
        out[test] = model.fit(feats[train], y[train]).predict_proba(feats[test])
    return out


def candidates(groups: dict, meta: dict, task: str) -> dict:
    """(condition, backbone, tag, clinical set) -> {model name: probs}, plus
    the shared patient keys, labels, EUS flag and fold."""
    n_classes = len(patient_classes(task))
    fold42 = exp49.seed_folds(task)[42]
    clinical = {name: exp49.clinical_sources(meta, task, n_classes)[(name, "logreg")] for name in CLINICAL_SETS}
    out = {}
    for cond in CANDIDATES:
        for backbone in exp51.BACKBONES:
            runs = groups.get((cond, backbone, task))
            for tag in exp49.TAGS:
                src = exp51.pick(runs, tag) if runs else None
                if src is None:
                    continue
                keys = src["patient_key"]
                assert np.array_equal(keys, fold42["patient_key"]), "image and clinical rows differ"
                fold = src["fold"] if tag == "s42" else fold42["fold"]
                for name in CLINICAL_SETS:
                    clin = clinical[name][tag]
                    X = exp49.design(meta, keys, exp49.FEATURE_SETS[name])
                    out[(cond, backbone, tag, name)] = {
                        "keys": keys, "y": src["y"], "paired": src["paired"], "fold": fold,
                        "models": {"image": src["probs_fused"], "clinical": clin,
                                   "average": (src["probs_fused"] + clin) / 2,
                                   "stacked": stacked(src["probs_fused"], X, src["y"], fold)}}
    return out


def performance(cands: dict, meta: dict, task: str) -> list[dict]:
    n_classes = len(patient_classes(task))
    rows = []
    for (cond, backbone, tag, clin_name), c in cands.items():
        for pop in ("all", "eus", "rectum"):
            m = exp49.population_mask(pop, c["keys"], c["paired"], meta)
            y = c["y"][m]
            idx = exp44.boot_indices(y, np.random.default_rng(0))
            ref = c["models"]["clinical"][m]
            panel = []
            for model, probs in c["models"].items():
                p = probs[m]
                row = {"task": task, "condition": cond, "backbone": backbone, "seeds": tag, "clinical_set": clin_name,
                       "population": pop, "model": model, "n": int(m.sum()), **exp49.score_row(y, p, idx, n_classes)}
                if model != "clinical":
                    d, lo, hi, pv = exp49.delta(exp49.boot_bal(y, p.argmax(1), idx, n_classes),
                                                exp49.boot_bal(y, ref.argmax(1), idx, n_classes))
                    row.update(minus_clinical_bal=d, bal_lo=lo, bal_hi=hi, p_bal=pv)
                    if n_classes == 2:
                        da, alo, ahi, ap = exp49.delta(exp49.boot_auroc(y, p[:, 0], idx),
                                                       exp49.boot_auroc(y, ref[:, 0], idx))
                        row.update(minus_clinical_auroc=da, auroc_lo=alo, auroc_hi=ahi, p_auroc=ap)
                    panel.append(row)
                rows.append(row)
            for key in ("p_bal", "p_auroc"):
                if panel and key in panel[0]:
                    for r, adj in zip(panel, exp44.holm([r[key] for r in panel])):
                        r[f"{key}_holm"] = adj
    return rows


def clinical_utility(cands: dict, meta: dict) -> tuple[list, list, list]:
    points, curve, calib = [], [], []
    for (cond, backbone, tag, clin_name), c in cands.items():
        if tag != "ens":
            continue
        net_all = c["y"] == 0
        for pop in ("all", "rectum"):
            m = exp49.population_mask(pop, c["keys"], c["paired"], meta)
            net, fold = net_all[m], c["fold"][m]
            base = {"condition": cond, "backbone": backbone, "clinical_set": clin_name, "population": pop}
            for model, probs in c["models"].items():
                p = probs[m, 0]
                for kind, target in exp50.TARGETS:
                    points.append({**base, "model": model, "chosen_for": f"{kind} {target:.2f}", "n": int(m.sum()),
                                   **exp50.cross_fit_points(net, p, fold, kind, target)})
                cal = exp50.recalibrate(net, p, fold)
                calib.append({**base, "model": model, "prevalence": float(net.mean()), "mean_p_raw": float(p.mean()),
                              "ece_raw": exp44.ece(np.column_stack([p, 1 - p]), (~net).astype(int))[0],
                              "ece_recal": exp44.ece(np.column_stack([cal, 1 - cal]), (~net).astype(int))[0]})
                curve += [{**base, "model": model, "threshold": float(t), "net_benefit": exp50.net_benefit(net, cal, t)}
                          for t in exp50.THRESHOLDS]
            curve += [{**base, "model": "treat_all", "threshold": float(t),
                       "net_benefit": exp50.net_benefit(net, np.ones(len(net)), t)} for t in exp50.THRESHOLDS]
    return points, curve, calib


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = exp49.load_metadata()
    groups = exp51.load_conditions()
    perf, points, curve, calib = [], [], [], []
    for task in exp49.TASKS:
        cands = candidates(groups, meta, task)
        perf += performance(cands, meta, task)
        if task == "binary":
            points, curve, calib = clinical_utility(cands, meta)
    for name, rows in (("performance", perf), ("operating_points", points),
                       ("decision_curve", curve), ("calibration", calib)):
        exp44.write_csv(OUT_DIR / f"{name}.csv", rows)

    print("\n3-seed ensembles, all patients: balanced accuracy / AUROC, and AUROC minus clinical [95% CI] (Holm)")
    for r in perf:
        if r["seeds"] == "ens" and r["population"] in ("all", "rectum") and r["task"] == "binary":
            d = "" if "minus_clinical_auroc" not in r else (
                f"  {r['minus_clinical_auroc']:+.3f} [{r['auroc_lo']:+.3f}, {r['auroc_hi']:+.3f}] "
                f"{exp44.fmt_p(r['p_auroc_holm'])}")
            print(f"  {r['condition']:11s}{r['backbone'][:6]:7s}{r['clinical_set']:17s}{r['population']:7s}"
                  f"{r['model']:9s}{r['bal']:.3f} / {r['auroc']:.3f}{d}")


if __name__ == "__main__":
    main()
