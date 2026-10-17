"""Experiment 44 -- uncertainty, clinical operating points and fusion value
for the two-tower backbone ablation.

CPU only, from the same cached out-of-fold predictions as 42 (seed 42 for
every backbone, plus the 3-seed ensembles). All scores are pooled over the
641 (or 421 paired) outer-fold test patients.

  * Patient-level stratified bootstrap (2000 resamples) 95% CIs for balanced
    accuracy, macro-F1 and ROC AUC; paired bootstrap of the balanced-accuracy
    difference DINOv3 - other backbone and an exact McNemar test on
    per-patient correctness, Holm-corrected over the comparisons. These CIs
    cover patient sampling only, not training variance (seed spread is in
    41's bar graphs).
  * ROC curves and operating points for the binary task (NET = positive, the
    potentially malignant class): sensitivity at 90% specificity and
    specificity at 95% sensitivity, read off the pooled OOF ROC curve -- an
    optimistic operating point, since the threshold is chosen on the same
    patients it is scored on.
  * Calibration (reliability, ECE over 10 confidence bins) and selective
    prediction: balanced accuracy when the model abstains on its least
    confident patients ("refer to an expert").
  * Is the learned fusion worth it? Late fusion (mean of the WLE-only and
    EUS-only probabilities of the same model) vs. the learned fused output,
    and how many patients adding EUS fixes vs. breaks, on the patients with
    both modalities.

    python -m experiments.src.44_two_tower_stats
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import PROJECT_ROOT
from cadsel.fusion import patient_classes

exp42 = importlib.import_module("experiments.src.42_two_tower_confusion_matrices")
exp41 = exp42.exp41
BACKBONES, TASK_TITLE, VIEWS = exp41.BACKBONES, exp41.TASK_TITLE, exp41.VIEWS
SURFACE, INK, INK2, MUTED, GRID, AXIS = exp41.SURFACE, exp41.INK, exp41.INK2, exp41.MUTED, exp41.GRID, exp41.AXIS

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "44_two_tower_stats"
PLOT_DIR = PROJECT_ROOT / "plots" / "two_tower_stats"
N_BOOT = 2000
REFERENCE = "dinov3_vits16"
NAME = {b[0]: b[1] for b in BACKBONES}
SHORT = {b[0]: b[2] for b in BACKBONES}
COLOR = {b[0]: b[3] for b in BACKBONES}
ORDER = [b[0] for b in BACKBONES]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def bal_acc(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    rec = [(pred[y == c] == c).mean() for c in range(n_classes) if (y == c).any()]
    return float(np.mean(rec))


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    return exp42.cm_scores(_cm(y, pred, n_classes))["macro_f1"]


def _cm(y, pred, n):
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return cm


def auroc(y: np.ndarray, probs: np.ndarray) -> float:
    """41's convention: binary = AUC of the NET (class 0) probability;
    subtype = mean one-vs-rest AUC over the classes present."""
    from sklearn.metrics import roc_auc_score

    if probs.shape[1] == 2:
        return float(roc_auc_score(y == 0, probs[:, 0]))
    aucs = [roc_auc_score(y == c, probs[:, c]) for c in range(probs.shape[1]) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(aucs))


def boot_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(N_BOOT, n) resample indices, stratified by class so every resample
    keeps the class sizes (and so every class is present)."""
    idx = np.empty((N_BOOT, len(y)), dtype=np.int64)
    pos = 0
    for c in np.unique(y):
        members = np.flatnonzero(y == c)
        idx[:, pos:pos + len(members)] = rng.choice(members, size=(N_BOOT, len(members)), replace=True)
        pos += len(members)
    return idx


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> tuple[float, list]:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(1 / probs.shape[1], 1, n_bins + 1)
    bins, total = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
        if sel.any():
            acc, c = float((pred[sel] == y[sel]).mean()), float(conf[sel].mean())
            total += sel.mean() * abs(acc - c)
            bins.append((c, acc, int(sel.sum())))
    return float(total), bins


def selective_curve(probs: np.ndarray, y: np.ndarray, n_classes: int, coverages: np.ndarray) -> np.ndarray:
    order = np.argsort(-probs.max(1), kind="stable")
    out = []
    for cov in coverages:
        keep = order[:max(int(round(cov * len(y))), 1)]
        out.append(bal_acc(y[keep], probs[keep].argmax(1), n_classes))
    return np.array(out)


def operating_points(y: np.ndarray, p_net: np.ndarray) -> dict:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y == 0, p_net)
    spec = 1 - fpr
    return {"sens_at_spec90": float(tpr[spec >= 0.90].max()), "spec_at_sens95": float(spec[tpr >= 0.95].max()),
            "fpr": fpr, "tpr": tpr}


def mcnemar(right_a: np.ndarray, right_b: np.ndarray) -> tuple[int, int, float]:
    from scipy.stats import binomtest

    b = int((right_a & ~right_b).sum())
    c = int((~right_a & right_b).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, float(p)


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


# ---------------------------------------------------------------------------
def load_sources() -> dict:
    """(task, tag) -> {backbone: src}; tag 's42' or 'ens'."""
    runs = exp42.load_oof()
    by_bt = defaultdict(list)
    for r in runs:
        by_bt[(r["backbone"], r["task"])].append(r)
    out = defaultdict(dict)
    for (backbone, task), rs in by_bt.items():
        s42 = [r for r in rs if r["seed"] == 42]
        if s42:
            out[(task, "s42")][backbone] = s42[0]
        if len(rs) >= 3:
            out[(task, "ens")][backbone] = exp42.ensemble(rs)
    for task in ("binary", "subtype"):        # same patient order everywhere
        for tag in ("s42", "ens"):
            for backbone, src in out[(task, tag)].items():
                order = np.argsort(src["patient_key"])
                out[(task, tag)][backbone] = {k: (v[order] if isinstance(v, np.ndarray) and len(v) == len(order)
                                                  else v) for k, v in src.items()}
    return out


def bootstrap(sources: dict) -> tuple[list[dict], list[dict]]:
    ci_rows, test_rows = [], []
    for (task, tag), by_bb in sorted(sources.items()):
        if not by_bb:
            continue
        n_classes = len(patient_classes(task))
        any_src = next(iter(by_bb.values()))
        y = any_src["y"]
        idx = boot_indices(y, np.random.default_rng(0))
        boot_bal = {}
        for backbone in [b for b in ORDER if b in by_bb]:
            probs = by_bb[backbone]["probs_fused"]
            assert np.array_equal(by_bb[backbone]["y"], y)
            pred = probs.argmax(1)
            bals = np.array([bal_acc(y[i], pred[i], n_classes) for i in idx])
            f1s = np.array([macro_f1(y[i], pred[i], n_classes) for i in idx])
            aucs = np.array([auroc(y[i], probs[i]) for i in idx[:500]])     # AUC is slow; 500 resamples
            boot_bal[backbone] = bals
            for metric, point, dist in (("bal", bal_acc(y, pred, n_classes), bals),
                                        ("macro_f1", macro_f1(y, pred, n_classes), f1s),
                                        ("auroc", auroc(y, probs), aucs)):
                lo, hi = np.percentile(dist, [2.5, 97.5])
                ci_rows.append({"task": task, "seeds": tag, "backbone": backbone, "metric": metric,
                                "point": point, "ci_lo": float(lo), "ci_hi": float(hi), "n": len(y)})
        if REFERENCE not in by_bb:
            continue
        ref_right = by_bb[REFERENCE]["probs_fused"].argmax(1) == y
        rows = []
        for backbone in [b for b in ORDER if b in by_bb and b != REFERENCE]:
            diff = boot_bal[REFERENCE] - boot_bal[backbone]
            right = by_bb[backbone]["probs_fused"].argmax(1) == y
            b, c, p_mc = mcnemar(ref_right, right)
            p_boot = float(min(1.0, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))
            lo, hi = np.percentile(diff, [2.5, 97.5])
            rows.append({"task": task, "seeds": tag, "reference": REFERENCE, "other": backbone,
                         "delta_bal": float(np.mean(diff)), "ci_lo": float(lo), "ci_hi": float(hi),
                         "p_bootstrap": p_boot, "ref_only_right": b, "other_only_right": c, "p_mcnemar": p_mc})
        for key in ("p_bootstrap", "p_mcnemar"):
            for r, adj in zip(rows, holm([r[key] for r in rows])):
                r[f"{key}_holm"] = adj
        test_rows += rows
    return ci_rows, test_rows


def fusion_value(sources: dict) -> list[dict]:
    rows = []
    for (task, tag), by_bb in sorted(sources.items()):
        n_classes = len(patient_classes(task))
        for backbone in [b for b in ORDER if b in by_bb]:
            s = by_bb[backbone]
            sel = s["paired"]
            y = s["y"][sel]
            wle, eus, fused = (s[f"probs_{v}"][sel] for v in ("wle_only", "eus_only", "fused"))
            late = (wle + eus) / 2
            wle_right, fused_right = wle.argmax(1) == y, fused.argmax(1) == y
            fixes, breaks, p = mcnemar(fused_right, wle_right)
            rows.append({"task": task, "seeds": tag, "backbone": backbone, "n": int(sel.sum()),
                         "bal_wle_only": bal_acc(y, wle.argmax(1), n_classes),
                         "bal_eus_only": bal_acc(y, eus.argmax(1), n_classes),
                         "bal_late_fusion": bal_acc(y, late.argmax(1), n_classes),
                         "bal_learned_fusion": bal_acc(y, fused.argmax(1), n_classes),
                         "eus_fixes": fixes, "eus_breaks": breaks, "p_mcnemar": p})
    return rows


def clinical(sources: dict, coverages: np.ndarray) -> tuple[list[dict], dict]:
    rows, curves = [], {}
    for (task, tag), by_bb in sorted(sources.items()):
        n_classes = len(patient_classes(task))
        for backbone in [b for b in ORDER if b in by_bb]:
            s = by_bb[backbone]
            probs, y = s["probs_fused"], s["y"]
            e, bins = ece(probs, y)
            sel = selective_curve(probs, y, n_classes, coverages)
            row = {"task": task, "seeds": tag, "backbone": backbone, "ece": e,
                   "bal_full": bal_acc(y, probs.argmax(1), n_classes),
                   "bal_cov80": float(sel[np.argmin(np.abs(coverages - 0.8))]),
                   "bal_cov50": float(sel[np.argmin(np.abs(coverages - 0.5))])}
            curve = {"bins": bins, "selective": sel}
            if task == "binary":
                op = operating_points(y, probs[:, 0])
                row.update(sens_at_spec90=op["sens_at_spec90"], spec_at_sens95=op["spec_at_sens95"])
                curve.update(fpr=op["fpr"], tpr=op["tpr"])
            rows.append(row)
            curves[(task, tag, backbone)] = curve
    return rows, curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _axis(ax, xlim=None, ylim=None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def _title(fig, title: str, subtitle: str) -> None:
    h = fig.get_figheight()
    fig.text(0.04, 1 - 0.2 / h, title, fontsize=14, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.04, 1 - 0.55 / h, subtitle, fontsize=8.8, color=INK2, ha="left", va="top", linespacing=1.45)


def _save(fig, name: str) -> Path:
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / name
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_forest(ci_rows: list[dict], tests: list[dict]) -> Path:
    """Balanced accuracy with 95% CI per backbone: tasks x (seed 42, ensemble)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.83, bottom=0.07, wspace=0.42, hspace=0.42)
    for r, task in enumerate(("binary", "subtype")):
        for c, tag in enumerate(("s42", "ens")):
            ax = axes[r, c]
            rows = {x["backbone"]: x for x in ci_rows if (x["task"], x["seeds"], x["metric"]) == (task, tag, "bal")}
            sig = {t["other"]: t for t in tests if (t["task"], t["seeds"]) == (task, tag)}
            lo_all = min([x["ci_lo"] for x in rows.values()] + [1])
            hi_all = max([x["ci_hi"] for x in rows.values()] + [0])
            _axis(ax, (lo_all - 0.04, hi_all + 0.12))
            ax.grid(False, axis="y")
            for i, b in enumerate(ORDER):
                y = len(ORDER) - 1 - i
                if b not in rows:
                    ax.text(lo_all - 0.035, y, "single seed only", fontsize=7.5, color=MUTED, va="center")
                    continue
                x = rows[b]
                ax.plot([x["ci_lo"], x["ci_hi"]], [y, y], color=COLOR[b], linewidth=2, solid_capstyle="round")
                ax.scatter([x["point"]], [y], s=46, color=COLOR[b], edgecolor=SURFACE, linewidth=1.5, zorder=3)
                note = f"{x['point']:.3f}  [{x['ci_lo']:.3f}, {x['ci_hi']:.3f}]"
                if b in sig:
                    note += f"   Δ vs DINOv3 {fmt_p(sig[b]['p_bootstrap_holm'])}"
                ax.text(x["ci_hi"] + 0.006, y, note, fontsize=7.3, color=INK2, va="center")
            ax.set_yticks(range(len(ORDER)))
            ax.set_yticklabels([NAME[b] for b in reversed(ORDER)], fontsize=8.5, color=INK2)
            ax.set_title(f"{TASK_TITLE[task]} · {'seed 42' if tag == 's42' else '3-seed ensemble'}", loc="left",
                         fontsize=10, color=INK, fontweight="bold", pad=8)
            ax.set_xlabel("balanced accuracy (pooled out-of-fold)", fontsize=8, color=MUTED)
    _title(fig, "How sure are the backbone differences? Balanced accuracy with 95% bootstrap CIs",
           "Patient-level stratified bootstrap (2000 resamples) of the pooled outer-fold test predictions, fused "
           "WLE+EUS, all 641 patients. p = paired bootstrap test of DINOv3 minus that backbone, Holm-corrected over "
           "the\ncomparisons in the panel. The CIs reflect which patients happen to be in the dataset, not training "
           "randomness. Same colour = same backbone in every figure.")
    return _save(fig, "bootstrap_forest.png")


def plot_roc(curves: dict, clin: list[dict]) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.8, bottom=0.11, wspace=0.22)
    for ax, tag in zip(axes, ("s42", "ens")):
        _axis(ax, (0, 1), (0, 1.01))
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.axvline(0.10, color=GRID, linewidth=1.2)
        for b in ORDER:
            cv = curves.get(("binary", tag, b))
            if cv is None:
                continue
            row = next(r for r in clin if (r["task"], r["seeds"], r["backbone"]) == ("binary", tag, b))
            ax.plot(cv["fpr"], cv["tpr"], color=COLOR[b], linewidth=2,
                    label=f"{NAME[b]} · sens@spec90 {row['sens_at_spec90']:.2f} · spec@sens95 "
                          f"{row['spec_at_sens95']:.2f}")
        ax.set_xlabel("1 − specificity (non-NET called NET)", fontsize=8.5, color=MUTED)
        ax.set_ylabel("sensitivity (NET detected)", fontsize=8.5, color=MUTED)
        ax.set_title("seed 42" if tag == "s42" else "3-seed ensemble", loc="left", fontsize=10, color=INK,
                     fontweight="bold", pad=8)
        ax.legend(loc="lower right", frameon=False, fontsize=7.3, labelcolor=INK2, handlelength=1.4)
    _title(fig, "Binary NET vs non-NET: ROC curves and clinical operating points",
           "Pooled outer-fold test predictions, fused WLE+EUS, n=641; NET (potentially malignant) is the positive "
           "class. Grey line: 90% specificity. The operating points are read off the same patients they are "
           "scored on,\nso they are optimistic. Dashed: chance. Same colour = same backbone in every figure.")
    return _save(fig, "roc_binary.png")


def plot_calibration_selective(curves: dict, clin: list[dict], coverages: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.06, wspace=0.2, hspace=0.32)
    for c, task in enumerate(("binary", "subtype")):
        ax = axes[0, c]
        k = len(patient_classes(task))
        _axis(ax, (1 / k, 1), (0, 1.01))
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
        for b in ORDER:
            cv = curves.get((task, "s42", b))
            if cv is None:
                continue
            row = next(r for r in clin if (r["task"], r["seeds"], r["backbone"]) == (task, "s42", b))
            xs, ys = [x[0] for x in cv["bins"]], [x[1] for x in cv["bins"]]
            ax.plot(xs, ys, color=COLOR[b], linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, label=f"{NAME[b]} · ECE {row['ece']:.3f}")
        ax.set_xlabel("confidence (max class probability)", fontsize=8.5, color=MUTED)
        ax.set_ylabel("accuracy in the bin", fontsize=8.5, color=MUTED)
        ax.set_title(f"{TASK_TITLE[task]}: reliability", loc="left", fontsize=10, color=INK, fontweight="bold",
                     pad=8)
        ax.legend(loc="lower right", frameon=False, fontsize=7.5, labelcolor=INK2, handlelength=1.4)

        ax = axes[1, c]
        _axis(ax, (coverages.min(), 1.0))
        for b in ORDER:
            cv = curves.get((task, "s42", b))
            if cv is not None:
                ax.plot(coverages, cv["selective"], color=COLOR[b], linewidth=2, label=NAME[b])
        ax.axvline(0.8, color=GRID, linewidth=1.2)
        ax.invert_xaxis()
        ax.set_xlabel("coverage (share of patients the model decides; the rest are referred)", fontsize=8.5,
                      color=MUTED)
        ax.set_ylabel("balanced accuracy on the decided patients", fontsize=8.5, color=MUTED)
        ax.set_title(f"{TASK_TITLE[task]}: abstain on the least confident", loc="left", fontsize=10, color=INK,
                     fontweight="bold", pad=8)
        ax.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK2, handlelength=1.4)
    _title(fig, "Can the confidence be trusted? Calibration and selective prediction",
           "Seed-42 pooled outer-fold test predictions, fused WLE+EUS. Top: patients binned by confidence; on the "
           "dashed diagonal, confidence = accuracy (ECE = weighted gap).\nBottom: the least confident patients are "
           "referred to an expert; grey line = 80% coverage. Same colour = same backbone in every figure.")
    return _save(fig, "calibration_selective.png")


def plot_fusion_value(rows: list[dict]) -> Path:
    """Per backbone: WLE-only vs late fusion vs learned fusion (dot plot; the
    marker shape is the method, the colour the backbone), and the patients
    EUS fixes vs breaks."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.1, wspace=0.2, hspace=0.55)
    markers = [("bal_wle_only", "o", "none", "WLE only"), ("bal_late_fusion", "D", None, "late fusion (mean of WLE-only "
               "and EUS-only probabilities)"), ("bal_learned_fusion", "o", None, "learned fusion (the two-tower model)")]
    for r, task in enumerate(("binary", "subtype")):
        sub = {x["backbone"]: x for x in rows if (x["task"], x["seeds"]) == (task, "s42")}
        ax = axes[r, 0]
        vals = [x[k] for x in sub.values() for k, *_ in markers]
        _axis(ax, (-0.6, len(ORDER) - 0.4), (min(vals) - 0.03, max(vals) + 0.03))
        ax.grid(False, axis="x")
        for i, b in enumerate(ORDER):
            if b not in sub:
                continue
            ys = [sub[b][k] for k, *_ in markers]
            ax.plot([i - 0.18, i, i + 0.18], ys, color=COLOR[b], linewidth=1, alpha=0.6)
            for (k, mk, fc, _), dx in zip(markers, (-0.18, 0, 0.18)):
                ax.scatter([i + dx], [sub[b][k]], marker=mk, s=44, facecolor=COLOR[b] if fc is None else SURFACE,
                           edgecolor=COLOR[b], linewidth=1.6, zorder=3)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([SHORT[b] for b in ORDER], rotation=30, ha="right", rotation_mode="anchor",
                           fontsize=8, color=INK2)
        ax.set_ylabel("balanced accuracy (421 paired patients)", fontsize=8.5, color=MUTED)
        ax.set_title(f"{TASK_TITLE[task]}: does end-to-end fusion beat averaging?", loc="left", fontsize=10,
                     color=INK, fontweight="bold", pad=8)

        ax = axes[r, 1]
        mx = max(max(x["eus_fixes"], x["eus_breaks"]) for x in sub.values())
        _axis(ax, (-mx * 1.35, mx * 1.35), (-0.6, len(ORDER) - 0.4))
        ax.grid(False, axis="y")
        ax.axvline(0, color=INK2, linewidth=0.9)
        for i, b in enumerate(ORDER):
            if b not in sub:
                continue
            y = len(ORDER) - 1 - i
            x = sub[b]
            ax.barh([y], [x["eus_fixes"]], height=0.6, color=COLOR[b])
            ax.barh([y], [-x["eus_breaks"]], height=0.6, color=COLOR[b], alpha=0.4)
            ax.text(x["eus_fixes"] + mx * 0.04, y, f"+{x['eus_fixes']}", va="center", fontsize=7.5, color=INK2)
            ax.text(-x["eus_breaks"] - mx * 0.04, y, f"−{x['eus_breaks']}  ({fmt_p(x['p_mcnemar'])})", va="center",
                    ha="right", fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([NAME[b] for b in reversed(ORDER)], fontsize=8.5, color=INK2)
        ax.set_xticks([])
        ax.set_title(f"{TASK_TITLE[task]}: patients EUS breaks (left) vs fixes (right)", loc="left", fontsize=10,
                     color=INK, fontweight="bold", pad=8)
    fig.legend(handles=[Line2D([], [], marker=mk, linestyle="none", markerfacecolor=INK if fc is None else SURFACE,
                               markeredgecolor=INK, markersize=7, label=lab) for _, mk, fc, lab in markers],
               loc="upper left", bbox_to_anchor=(0.045, 0.905), ncols=3, frameon=False, fontsize=8.3,
               labelcolor=INK2, columnspacing=2.2)
    _title(fig, "Is the learned fusion worth it?",
           "Seed 42, the 421 patients with both modalities. Left: the same model's WLE-only output (EUS masked), a "
           "no-training late-fusion baseline, and the learned fused output. Right: patients whose prediction "
           "becomes\ncorrect (fixes) or incorrect (breaks) when EUS is added; p = exact McNemar test. Same colour = "
           "same backbone in every figure.")
    return _save(fig, "fusion_value.png")


def fmt_p(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    ci_rows, tests = bootstrap(sources)
    # bootstrap point estimates must equal 42's pooled matrices
    with open(exp42.OUT_DIR / "per_class.csv") as fh:
        pooled = {(r["task"], r["backbone"], r["seeds"]): float(r["pooled_bal"]) for r in csv.DictReader(fh)
                  if r["view"] == "fused"}
    worst = max(abs(r["point"] - pooled[(r["task"], r["backbone"], r["seeds"])])
                for r in ci_rows if r["metric"] == "bal")
    print(f"bootstrap point estimates vs 42 pooled balanced accuracy: max |diff| = {worst:.1e}")
    assert worst < 6e-5
    fusion = fusion_value(sources)
    coverages = np.linspace(0.3, 1.0, 71)
    clin, curves = clinical(sources, coverages)
    write_csv(OUT_DIR / "bootstrap_ci.csv", ci_rows)
    write_csv(OUT_DIR / "paired_tests.csv", tests)
    write_csv(OUT_DIR / "fusion_value.csv", fusion)
    write_csv(OUT_DIR / "clinical.csv", clin)
    for p in (plot_forest(ci_rows, tests), plot_roc(curves, clin),
              plot_calibration_selective(curves, clin, coverages), plot_fusion_value(fusion)):
        print(f"wrote {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
