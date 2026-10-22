"""Experiment 47 -- CAD-SEL against other colonoscopy datasets, same protocol.

CPU only. Reads the out-of-fold predictions that 41 writes (``41 --score-only``
/ ``--variants`` on whichever machine trained the run) for every run of the
shared protocol: nested 5-fold with patient/video groups, ep100/es20, gated
fusion, lesion-crop wherever the dataset has boxes. Rows:
- CAD-SEL two-tower (WLE+EUS), and CAD-SEL single-backbone WLE-only / EUS-only
  (``--modalities``), scored on the same outer folds.
- external single-backbone datasets (LIMUC, REAL-colon) and two-tower WL+NBI
  datasets (CPC-Paired, PIBAdb, PICCOLO) as their runs arrive.

The tasks differ (subepithelial-lesion subtype, polyp histology, UC grade), so
raw balanced accuracies do not form one leaderboard. The comparable quantities:
- normalised balanced accuracy (bal - chance) / (1 - chance)
- for two-tower datasets, the fusion gain: fused minus the first modality alone
  on the units that have both, with a patient-level bootstrap CI.

Scores are pooled over the outer folds; CIs are a stratified patient-level
bootstrap (2000 resamples, as in 44). LIMUC's Mayo task also gets quadratic
weighted kappa, its usual metric.

    python -m experiments.src.47_cross_dataset_comparison
"""

from __future__ import annotations

import argparse
import csv
import importlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import PROJECT_ROOT
from cadsel.external_datasets import get_spec, modality_views

exp41 = importlib.import_module("experiments.src.41_two_tower_ablation_bargraphs")
exp42 = importlib.import_module("experiments.src.42_two_tower_confusion_matrices")
exp44 = importlib.import_module("experiments.src.44_two_tower_stats")
SURFACE, INK, INK2, MUTED, GRID, AXIS = exp41.SURFACE, exp41.INK, exp41.INK2, exp41.MUTED, exp41.GRID, exp41.AXIS
COLOR = {b[0]: b[3] for b in exp41.BACKBONES + exp41.CLUSTER_BACKBONES}
NAME = {b[0]: b[1] for b in exp41.BACKBONES + exp41.CLUSTER_BACKBONES}

OOF_DIR = exp41.OUT_DIR / "oof"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "47_cross_dataset"
PLOT_DIR = PROJECT_ROOT / "plots" / "cross_dataset"
BACKBONES = ("efficientnet_b0", "dinov3_vits16")      # the two run on every dataset
DATASET_LABEL = {"cadsel": "CAD-SEL", "limuc": "LIMUC", "real_colon": "REAL-colon",
                 "cpc_paired": "CPC-Paired", "pibadb": "PIBAdb", "piccolo": "PICCOLO"}

RUN_RE = re.compile(r"^joint_(?P<backbone>.+?)__(?P<task>[a-z]+)__(?P<config>.+)__s(?P<seed>\d+)$")


def parse_run(stem: str) -> dict | None:
    """Dataset / modalities / config from a run name (34's naming), or None
    if the run is not on the shared protocol."""
    m = RUN_RE.match(stem)
    if m is None:
        return None
    cfg = m["config"]
    ds = re.search(r"__ds-([a-z_]+)", cfg)
    mod = re.search(r"__mod-([A-Za-z+]+)", cfg)
    dataset = ds.group(1) if ds else "cadsel"
    try:
        spec = get_spec(dataset)
    except SystemExit:
        return None
    on_protocol = ("fusion-gated" in cfg and "__nested0.2" in cfg and "__ep100__es20" in cfg
                   and "__ext-" not in cfg and "__paired" not in cfg
                   and ("__lesioncrop" in cfg) == spec.has_boxes)
    if not on_protocol:
        return None
    mods = mod.group(1).split("+") if mod else list(spec.modalities)
    return {"backbone": m["backbone"], "task": m["task"], "seed": int(m["seed"]), "dataset": dataset,
            "modalities": mods, "spec": spec}


def load_runs() -> dict:
    """(dataset, task, modality config, backbone) -> list of per-seed runs."""
    out = defaultdict(list)
    for path in sorted(OOF_DIR.glob("*.npz")):
        info = parse_run(path.stem)
        if info is None or info["backbone"] not in BACKBONES:
            continue
        d = np.load(path)
        order = np.argsort(d["patient_key"])         # 44's row order, so its bootstrap CIs reproduce
        info.update({k: d[k][order] for k in d.files}, run_name=path.stem)
        cfg = "+".join(info["modalities"])
        out[(info["dataset"], info["task"], cfg, info["backbone"])].append(info)
    return out


def qwk(y: np.ndarray, pred: np.ndarray, n: int) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(y, pred, labels=list(range(n)), weights="quadratic"))


def summarise(group: list[dict], rng_seed: int = 0) -> dict:
    """Pooled seed-42 scores with bootstrap CIs; the 3-seed ensemble's
    balanced accuracy when all three seeds exist."""
    run = next((r for r in group if r["seed"] == 42), group[0])
    spec, task, mods = run["spec"], run["task"], run["modalities"]
    n = len(spec.classes(task))
    probs, y = run["probs_fused"], run["y"]
    pred = probs.argmax(1)
    chance = 1 / n
    idx = exp44.boot_indices(y, np.random.default_rng(rng_seed))
    bals = np.array([exp44.bal_acc(y[i], pred[i], n) for i in idx])
    bal = exp44.bal_acc(y, pred, n)
    row = {"dataset": run["dataset"], "task": task, "modalities": "+".join(mods),
           "model": "two-tower" if len(mods) > 1 else "single-backbone", "backbone": run["backbone"],
           "units": len(y), "unit": spec.unit, "classes": n, "chance": chance,
           "imbalance": float(np.bincount(y, minlength=n).max() / max(np.bincount(y, minlength=n).min(), 1)),
           "bal": bal, "bal_lo": float(np.percentile(bals, 2.5)), "bal_hi": float(np.percentile(bals, 97.5)),
           "norm_bal": (bal - chance) / (1 - chance),
           "norm_lo": (float(np.percentile(bals, 2.5)) - chance) / (1 - chance),
           "norm_hi": (float(np.percentile(bals, 97.5)) - chance) / (1 - chance),
           "macro_f1": exp44.macro_f1(y, pred, n), "auroc": exp44.auroc(y, probs),
           "qwk": qwk(y, pred, n) if (run["dataset"], task) == ("limuc", "mayo") else float("nan"),
           "seeds": len(group), "centres": spec.centres}
    if len(group) >= 3:
        row["bal_ensemble"] = exp42.cm_scores(exp42.confusion(*exp42.view_arrays(exp42.ensemble(group), "fused"),
                                                               n))["bal"]
    if len(mods) > 1:                    # fusion gain on the units with every modality
        first = f"{mods[0].lower()}_only"
        sel = run["paired"]
        yp = y[sel]
        f_pred, s_pred = probs[sel].argmax(1), run[f"probs_{first}"][sel].argmax(1)
        idx_p = exp44.boot_indices(yp, np.random.default_rng(rng_seed))
        gains = np.array([exp44.bal_acc(yp[i], f_pred[i], n) - exp44.bal_acc(yp[i], s_pred[i], n) for i in idx_p])
        row.update(paired_units=int(sel.sum()),
                   fusion_gain=exp44.bal_acc(yp, f_pred, n) - exp44.bal_acc(yp, s_pred, n),
                   gain_lo=float(np.percentile(gains, 2.5)), gain_hi=float(np.percentile(gains, 97.5)))
    return row


def fusion_strategies(groups: dict, rng_seed: int = 0) -> list[dict]:
    """CAD-SEL, seed 42: the two-tower against separately trained
    single-modality models (same outer folds) and their late fusion (mean
    probability of the WLE-only and EUS-only models). Scored on the patients
    with both modalities, and on all patients (late fusion = the WLE model for
    patients without EUS). The bootstrap CI is for late fusion minus the two-tower."""
    rows = []
    for task in ("binary", "subtype"):
        n = len(get_spec("cadsel").classes(task))
        for bb in BACKBONES:
            pick = {cfg: next((r for r in groups.get(("cadsel", task, cfg, bb), []) if r["seed"] == 42), None)
                    for cfg in ("WLE+EUS", "WLE", "EUS")}
            if None in pick.values():
                continue
            tt, w, e = pick["WLE+EUS"], pick["WLE"], pick["EUS"]
            assert np.array_equal(tt["patient_key"], w["patient_key"]) and np.array_equal(tt["y"], w["y"])
            pos = {k: i for i, k in enumerate(tt["patient_key"].tolist())}
            ip = np.array([pos[k] for k in e["patient_key"].tolist()])        # EUS model: patients with EUS
            assert np.array_equal(tt["y"][ip], e["y"])
            y_p = e["y"]
            late_p = (w["probs_fused"][ip] + e["probs_fused"]) / 2
            preds = {"wle_single": w["probs_fused"][ip].argmax(1), "eus_single": e["probs_fused"].argmax(1),
                     "two_tower_fused": tt["probs_fused"][ip].argmax(1),
                     "two_tower_wle_view": tt["probs_wle_only"][ip].argmax(1),
                     "two_tower_eus_view": tt["probs_eus_only"][ip].argmax(1),
                     "late_fusion_singles": late_p.argmax(1)}
            idx = exp44.boot_indices(y_p, np.random.default_rng(rng_seed))
            diff = np.array([exp44.bal_acc(y_p[i], preds["late_fusion_singles"][i], n)
                             - exp44.bal_acc(y_p[i], preds["two_tower_fused"][i], n) for i in idx])
            late_all = w["probs_fused"].copy()
            late_all[ip] = late_p
            row = {"task": task, "backbone": bb, "n_paired": len(y_p), "n_all": len(tt["y"]),
                   **{f"bal_paired_{k}": exp44.bal_acc(y_p, v, n) for k, v in preds.items()},
                   "late_minus_twotower_paired": float(np.mean(diff)),
                   "ci_lo": float(np.percentile(diff, 2.5)), "ci_hi": float(np.percentile(diff, 97.5)),
                   "bal_all_wle_single": exp44.bal_acc(tt["y"], w["probs_fused"].argmax(1), n),
                   "bal_all_two_tower_fused": exp44.bal_acc(tt["y"], tt["probs_fused"].argmax(1), n),
                   "bal_all_late_fusion_singles": exp44.bal_acc(tt["y"], late_all.argmax(1), n)}
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
def _style(ax, xlim=None, ylim=None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)


def _row_label(r: dict) -> str:
    what = r["modalities"] if r["model"] == "two-tower" else f"{r['modalities']} only"
    return f"{DATASET_LABEL[r['dataset']]} · {r['task']}\n{what} ({r['model']})"


def plot_normalised(rows: list[dict]) -> Path:
    import matplotlib.pyplot as plt

    keys = list(dict.fromkeys((r["dataset"], r["task"], r["modalities"]) for r in rows))
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(keys) * len(BACKBONES) + 2.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.27, right=0.97, top=1 - 1.25 / fig.get_figheight(), bottom=0.6 / fig.get_figheight())
    _style(ax, (-0.05, 1.0))
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))
    yticks, ylabels = [], []
    for gi, key in enumerate(keys):
        base = -gi * (len(BACKBONES) + 0.8)
        for bi, bb in enumerate(BACKBONES):
            r = next((x for x in rows if (x["dataset"], x["task"], x["modalities"]) == key and x["backbone"] == bb),
                     None)
            if r is None:
                continue
            y = base - bi
            ax.barh([y], [r["norm_bal"]], height=0.72, color=COLOR[bb])
            ax.plot([r["norm_lo"], r["norm_hi"]], [y, y], color=INK2, linewidth=1)
            ax.text(max(r["norm_hi"], r["norm_bal"]) + 0.01, y, f"bal {r['bal']:.3f}", va="center", fontsize=7.3,
                    color=INK2)
        yticks.append(base - (len(BACKBONES) - 1) / 2)
        ylabels.append(_row_label(next(x for x in rows if (x["dataset"], x["task"], x["modalities"]) == key)))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.2, color=INK2)
    ax.set_xlabel("normalised balanced accuracy  (bal − chance) / (1 − chance)", fontsize=8.5, color=MUTED)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLOR[b], label=NAME[b]) for b in BACKBONES], loc="lower right",
              frameon=False, fontsize=8, labelcolor=INK2)
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.2 / h, "CAD-SEL vs other colonoscopy datasets, same protocol", fontsize=14, color=INK,
             fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.55 / h, "Patient-level nested 5-fold, seed 42, pooled outer-fold predictions; whisker = "
             "95% bootstrap CI. 0 = chance, 1 = perfect, so tasks with different class counts compare.\nTasks "
             "differ (lesion subtype, polyp histology, UC grade): read differences as task + data difficulty, "
             "not as one leaderboard.", fontsize=8.6, color=INK2, va="top", linespacing=1.45)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "normalised_balanced_accuracy.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_fusion_gain(rows: list[dict]) -> Path | None:
    import matplotlib.pyplot as plt

    two = [r for r in rows if r["model"] == "two-tower"]
    if not two:
        return None
    keys = list(dict.fromkeys((r["dataset"], r["task"]) for r in two))
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(keys) * len(BACKBONES) + 2.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.22, right=0.96, top=1 - 1.2 / fig.get_figheight(), bottom=0.6 / fig.get_figheight())
    lo = min(r["gain_lo"] for r in two)
    hi = max(r["gain_hi"] for r in two)
    _style(ax, (min(lo, -0.05) - 0.02, max(hi, 0.05) + 0.06))
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK2, linewidth=0.9)
    yticks, ylabels = [], []
    for gi, key in enumerate(keys):
        base = -gi * (len(BACKBONES) + 0.8)
        for bi, bb in enumerate(BACKBONES):
            r = next((x for x in two if (x["dataset"], x["task"]) == key and x["backbone"] == bb), None)
            if r is None:
                continue
            y = base - bi
            ax.plot([r["gain_lo"], r["gain_hi"]], [y, y], color=COLOR[bb], linewidth=2, solid_capstyle="round")
            ax.scatter([r["fusion_gain"]], [y], s=46, color=COLOR[bb], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.text(r["gain_hi"] + 0.005, y, f"{r['fusion_gain']:+.3f}  (n={r['paired_units']})", va="center",
                    fontsize=7.4, color=INK2)
        yticks.append(base - (len(BACKBONES) - 1) / 2)
        mods = next(x for x in two if (x["dataset"], x["task"]) == key)["modalities"]
        ylabels.append(f"{DATASET_LABEL[key[0]]} · {key[1]}\n{mods}")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.4, color=INK2)
    ax.set_xlabel("fusion gain = balanced accuracy (fused) − (first modality alone)", fontsize=8.5, color=MUTED)
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.2 / h, "Does the second modality help? WLE+EUS vs WL+NBI", fontsize=14, color=INK,
             fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.55 / h, "Same trained two-tower model, second modality masked vs present, on the units "
             "that have both; 95% patient-level bootstrap CI.\nSeed 42, nested 5-fold outer-fold predictions. "
             "Colour = backbone, as in every figure.", fontsize=8.6, color=INK2, va="top", linespacing=1.45)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "fusion_gain.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_fusion_strategies(rows: list[dict]) -> Path | None:
    """Dot plot per backbone: marker shape = strategy, colour = backbone."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if not rows:
        return None
    strategies = [("bal_paired_wle_single", "o", "none", "WLE-only model (separately trained)"),
                  ("bal_paired_eus_single", "s", "none", "EUS-only model (separately trained)"),
                  ("bal_paired_two_tower_fused", "o", None, "two-tower, fused (end-to-end)"),
                  ("bal_paired_late_fusion_singles", "D", None, "late fusion of the two single models")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.72, bottom=0.14, wspace=0.18)
    for ax, task in zip(axes, ("binary", "subtype")):
        sub = [r for r in rows if r["task"] == task]
        vals = [r[k] for r in sub for k, *_ in strategies]
        _style(ax, (-0.6, len(BACKBONES) - 0.4), (min(vals) - 0.03, max(vals) + 0.04))
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        offs = np.linspace(-0.27, 0.27, len(strategies))
        for i, bb in enumerate(BACKBONES):
            r = next((x for x in sub if x["backbone"] == bb), None)
            if r is None:
                continue
            for (k, mk, fc, _), dx in zip(strategies, offs):
                ax.scatter([i + dx], [r[k]], marker=mk, s=52, facecolor=COLOR[bb] if fc is None else SURFACE,
                           edgecolor=COLOR[bb], linewidth=1.6, zorder=3)
            ax.text(i, max(r[k] for k, *_ in strategies) + 0.012,
                    f"late − two-tower {r['late_minus_twotower_paired']:+.3f}\n"
                    f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]",
                    ha="center", fontsize=7.2, color=INK2)
        ax.set_xticks(range(len(BACKBONES)))
        ax.set_xticklabels([NAME[b] for b in BACKBONES], fontsize=8.5, color=INK2)
        ax.set_ylabel("balanced accuracy (421 patients with EUS)", fontsize=8.5, color=MUTED)
        ax.set_title(f"CAD-SEL {exp41.TASK_TITLE[task]}", loc="left", fontsize=10, color=INK, fontweight="bold",
                     pad=8)
    fig.legend(handles=[Line2D([], [], marker=mk, linestyle="none", markerfacecolor=INK if fc is None else SURFACE,
                               markeredgecolor=INK, markersize=7, label=lab) for _, mk, fc, lab in strategies],
               loc="upper left", bbox_to_anchor=(0.055, 0.86), ncols=4, frameon=False, fontsize=8.2, labelcolor=INK2,
               columnspacing=1.8)
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.2 / h, "End-to-end fusion vs separately trained towers on CAD-SEL", fontsize=14, color=INK,
             fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.55 / h, "Seed 42, nested 5-fold, identical outer folds for every model; the 421 patients "
             "with both WLE and EUS. Late fusion = mean of the two single models' class probabilities.\nNumbers: "
             "late fusion minus two-tower, 95% patient-level bootstrap CI. Colour = backbone, as in every figure.",
             fontsize=8.6, color=INK2, va="top", linespacing=1.45)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "cadsel_fusion_strategies.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"

    groups = load_runs()
    order = {"cadsel": 0, "limuc": 2, "real_colon": 3, "cpc_paired": 4, "pibadb": 5, "piccolo": 6}
    rows = [summarise(g) for _, g in sorted(groups.items(),
                                             key=lambda kv: (order[kv[0][0]], kv[0][1], -kv[0][2].count("+"),
                                                             kv[0][2], BACKBONES.index(kv[0][3])))]
    print(f"{len(rows)} rows from {sum(len(g) for g in groups.values())} runs")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(OUT_DIR / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(f"  {DATASET_LABEL[r['dataset']]:10s} {r['task']:9s} {r['modalities']:8s} {r['backbone']:16s} "
              f"bal {r['bal']:.3f} [{r['bal_lo']:.3f},{r['bal_hi']:.3f}]  norm {r['norm_bal']:+.3f}"
              + (f"  gain {r['fusion_gain']:+.3f}" if "fusion_gain" in r else ""))
    strat = fusion_strategies(groups)
    if strat:
        with open(OUT_DIR / "cadsel_fusion_strategies.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(strat[0]))
            w.writeheader()
            for r in strat:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        for r in strat:
            print(f"  fusion strategies CAD-SEL {r['task']:8s} {r['backbone']:16s} paired: WLE {r['bal_paired_wle_single']:.3f} "
                  f"EUS {r['bal_paired_eus_single']:.3f} two-tower {r['bal_paired_two_tower_fused']:.3f} "
                  f"late {r['bal_paired_late_fusion_singles']:.3f} (late-TT {r['late_minus_twotower_paired']:+.3f} "
                  f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]) | all-641: TT {r['bal_all_two_tower_fused']:.3f} "
                  f"late {r['bal_all_late_fusion_singles']:.3f}")
    for p in (plot_normalised(rows), plot_fusion_gain(rows), plot_fusion_strategies(strat)):
        if p is not None:
            print(f"wrote {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
