"""Experiment 42 -- confusion matrices for the two-tower backbone ablation.

CPU only. Reads the per-patient out-of-fold probabilities that
``41_two_tower_ablation_bargraphs.py`` cached for every nested-protocol run
(``experiments/results/41_two_tower_ablation/oof/*.npz``) and pools each
run's five outer test folds into one matrix per backbone x task x view,
predicted class = argmax (the rule 41's metrics use).

Matrices:
  * seed 42, every backbone (the one seed all 7 have for both tasks);
  * 3-seed ensemble (mean of the seeds' probabilities per patient) for the
    backbones run with 3 seeds -- the ensembles 35 reports;
  * fused minus WLE-only on the patients with EUS frames: where fusion
    fixes and breaks predictions.

Cells: patient count + row-normalised share (= per-class recall); colour is
the row share, so the 24-patient Leiomyoma row reads as clearly as the
411-patient NET_G1 row. Before plotting, every per-fold balanced accuracy is
checked against 41's ``metrics_per_fold.csv`` and every ensemble against 35's
JSON, so the matrices are the predictions behind the reported numbers.

    python -m experiments.src.42_two_tower_confusion_matrices
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import PROJECT_ROOT
from cadsel.fusion import patient_classes

exp41 = importlib.import_module("experiments.src.41_two_tower_ablation_bargraphs")
BACKBONES, VIEWS, fold_metrics = exp41.BACKBONES, exp41.VIEWS, exp41.fold_metrics
SURFACE, INK, INK2, MUTED, GRID, AXIS = exp41.SURFACE, exp41.INK, exp41.INK2, exp41.MUTED, exp41.GRID, exp41.AXIS
TASK_TITLE = exp41.TASK_TITLE

OOF_DIR = exp41.OUT_DIR / "oof"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "42_two_tower_confusion"
PLOT_DIR = PROJECT_ROOT / "plots" / "two_tower_confusion"
EXP35 = PROJECT_ROOT / "experiments" / "results" / "35_joint_fusion_multiseed_ensemble__{}.json"

RUN_RE = re.compile(r"^joint_(?P<backbone>.+?)__(?P<task>binary|subtype)__(?P<config>.+)__s(?P<seed>\d+)$")
# Only the ablation's own configuration; later variants (full frame,
# symmetric fusion, external-centre test) get their own figures.
ABLATION_CONFIG = re.compile(r"fusion-gated__.*__lesioncrop__nested0\.2$")
MAIN_SEED = 42
CLASS_SHORT = {"NET_G1": "NET G1", "NET_G2": "NET G2", "Leiomyoma": "Leiomyoma", "Lipoma": "Lipoma",
               "NonNeoplasm": "Non-neoplasm", "NET": "NET", "non-NET": "non-NET"}

# Sequential: the reference blue ramp, surface -> step 700. Diverging: blue
# (fusion helps) <-> red (fusion hurts) around the neutral grey midpoint.
SEQ_STEPS = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_STEPS = ["#a52b2b", "#e34948", "#f09a99", "#f0efec", "#86b6ef", "#2a78d6", "#184f95"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_oof() -> list[dict]:
    runs = []
    for path in sorted(OOF_DIR.glob("*.npz")):
        m = RUN_RE.match(path.stem)
        if m is None or not ABLATION_CONFIG.search(m["config"]):
            continue
        d = np.load(path)
        runs.append({"run_name": path.stem, "backbone": m["backbone"], "task": m["task"],
                     "seed": int(m["seed"]), "config": m["config"],
                     **{k: d[k] for k in d.files}})
    return runs


def view_arrays(run: dict, view: str) -> tuple[np.ndarray, np.ndarray]:
    ablation, paired_only, _ = VIEWS[view]
    sel = run["paired"] if paired_only else np.ones(len(run["y"]), bool)
    return run[f"probs_{ablation}"][sel], run["y"][sel]


def ensemble(runs: list[dict]) -> dict:
    """Mean of the seeds' class probabilities per patient. Fold assignment
    differs between seeds, so rows are aligned on ``patient_key``."""
    keys = sorted(runs[0]["patient_key"].tolist())
    out = {"backbone": runs[0]["backbone"], "task": runs[0]["task"], "seed": "ens",
           "seeds": sorted(r["seed"] for r in runs), "patient_key": np.array(keys)}
    for r in runs:
        order = np.argsort(r["patient_key"])
        assert r["patient_key"][order].tolist() == keys, f"patient set differs in {r['run_name']}"
        for name in ("y", "paired"):
            col = r[name][order]
            if name in out:
                assert np.array_equal(out[name], col), f"{name} differs across seeds"
            out[name] = col
        for name in ("probs_wle_only", "probs_eus_only", "probs_fused"):
            out.setdefault(name, []).append(r[name][order])
    for name in ("probs_wle_only", "probs_eus_only", "probs_fused"):
        out[name] = np.mean(out[name], axis=0)
    return out


def confusion(probs: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y, probs.argmax(1)), 1)
    return cm


def cm_scores(cm: np.ndarray) -> dict:
    """Per-class precision / recall / F1 and the pooled summary scores, over
    the classes present (the convention of 41's ``fold_metrics``)."""
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp),
                   where=(precision + recall) > 0)
    present = support > 0
    return {"precision": precision, "recall": recall, "f1": f1, "support": support,
            "bal": float(recall[present].mean()), "macro_f1": float(f1[present].mean()),
            "acc": float(tp.sum() / cm.sum()), "n": int(cm.sum())}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_against_41(runs: list[dict]) -> None:
    """Every run x fold x view balanced accuracy must equal 41's CSV (which
    rounds to 4 dp)."""
    recorded = {}
    with open(exp41.OUT_DIR / "metrics_per_fold.csv") as fh:
        for r in csv.DictReader(fh):
            recorded[(r["task"], r["backbone"], int(r["seed"]), int(r["fold"]), r["view"])] = float(r["bal"])
    if not recorded:
        raise SystemExit("41's metrics_per_fold.csv is empty -- was 41 run with the results volume unmounted?")
    worst, checked = 0.0, 0
    for run in runs:
        for view, (ablation, paired_only, _) in VIEWS.items():
            for k in sorted(set(run["fold"].tolist())):
                sel = run["fold"] == k
                if paired_only:
                    sel = sel & run["paired"]
                bal = fold_metrics(run[f"probs_{ablation}"][sel], run["y"][sel], run["task"])["bal"]
                ref = recorded[(run["task"], run["backbone"], run["seed"], k, view)]
                worst = max(worst, abs(bal - ref))
                checked += 1
    print(f"verify vs 41: {checked} fold x view balanced accuracies, max |diff| = {worst:.2e}")
    if worst > 6e-5:
        raise SystemExit("per-fold balanced accuracy does not match 41's metrics_per_fold.csv")


def verify_against_35(ensembles: list[dict]) -> None:
    for ens in ensembles:
        stem = next(iter(EXP35.parent.glob(
            f"35_joint_fusion_multiseed_ensemble__joint_{ens['backbone']}__{ens['task']}__*lesioncrop__nested0.2.json")),
            None)
        if stem is None:
            print(f"  no 35 JSON for {ens['backbone']} {ens['task']}")
            continue
        ref = json.loads(stem.read_text())
        n_classes = len(patient_classes(ens["task"]))
        got = {"ensemble_unpaired": cm_scores(confusion(*view_arrays(ens, "fused"), n_classes))["bal"],
               "ensemble_paired_only": cm_scores(confusion(*view_arrays(ens, "fused_paired"), n_classes))["bal"]}
        for key, value in got.items():
            diff = abs(value - ref[key]["bal"])
            flag = "ok" if diff <= 6e-5 else "MISMATCH"
            print(f"  35 {ens['backbone']:<20} {ens['task']:<8} {key:<21} ours {value:.4f}  "
                  f"35 {ref[key]['bal']:.4f}  {flag}")
            if diff > 6e-5:
                raise SystemExit("ensemble balanced accuracy does not match 35")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_tables(cells: dict) -> None:
    """``cells[(task, view, backbone, tag)] = cm``; tag is 's42' or 'ens'."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "cm_counts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "view", "backbone", "seeds", "true", "pred", "count"])
        for (task, view, backbone, tag), cm in sorted(cells.items()):
            classes = patient_classes(task)
            for i, t in enumerate(classes):
                for j, p in enumerate(classes):
                    w.writerow([task, view, backbone, tag, t, p, int(cm[i, j])])
    with open(OUT_DIR / "per_class.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "view", "backbone", "seeds", "class", "support", "precision", "recall", "f1",
                    "pooled_bal", "pooled_macro_f1", "pooled_acc"])
        for (task, view, backbone, tag), cm in sorted(cells.items()):
            s = cm_scores(cm)
            for i, c in enumerate(patient_classes(task)):
                w.writerow([task, view, backbone, tag, c, int(s["support"][i]), round(s["precision"][i], 4),
                            round(s["recall"][i], 4), round(s["f1"][i], 4), round(s["bal"], 4),
                            round(s["macro_f1"], 4), round(s["acc"], 4)])
    print(f"wrote {OUT_DIR / 'cm_counts.csv'}\nwrote {OUT_DIR / 'per_class.csv'}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _cmaps():
    from matplotlib.colors import LinearSegmentedColormap

    return (LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS),
            LinearSegmentedColormap.from_list("div_red_blue", DIV_STEPS))


def _identity_rule(ax, color: str) -> None:
    """Backbone colour as a short bar left of the panel title: identity is
    carried by the mark, the title text stays in ink."""
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((0.0, 1.035), 0.09, 0.022, transform=ax.transAxes, facecolor=color,
                           edgecolor="none", clip_on=False))


def _axes_frame(ax, classes: list[str], show_y: bool, show_x: bool) -> None:
    n = len(classes)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [CLASS_SHORT[c] for c in classes]
    ax.set_xticklabels(labels if show_x else [], rotation=35 if n > 2 else 0,
                       ha="right" if n > 2 else "center", rotation_mode="anchor", fontsize=7.5, color=INK2)
    ax.set_yticklabels(labels if show_y else [], fontsize=7.5, color=INK2)
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    if show_x:
        ax.set_xlabel("predicted", fontsize=8, color=MUTED, labelpad=4)
    if show_y:
        ax.set_ylabel("true", fontsize=8, color=MUTED, labelpad=4)


def draw_cm(ax, cm: np.ndarray, classes: list[str], cmap, show_y: bool, show_x: bool) -> None:
    rows = cm.sum(1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros(cm.shape), where=rows > 0)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    n = len(classes)
    big = n <= 2
    for i in range(n):
        for j in range(n):
            ink = SURFACE if share[i, j] >= 0.5 else INK
            ax.text(j, i - (0.1 if big else 0.13), f"{cm[i, j]}", ha="center", va="center",
                    fontsize=12 if big else 8.5, color=ink, fontweight="bold")
            ax.text(j, i + (0.17 if big else 0.22), f"{share[i, j]:.0%}", ha="center", va="center",
                    fontsize=8.5 if big else 6.4, color=ink)
    _axes_frame(ax, classes, show_y, show_x)


def draw_delta(ax, delta: np.ndarray, classes: list[str], cmap, vmax: float, show_y: bool, show_x: bool) -> None:
    """Fused minus WLE-only counts, coloured by benefit: more patients on the
    diagonal or fewer off it is blue, the reverse red."""
    n = len(classes)
    benefit = np.where(np.eye(n, dtype=bool), delta, -delta)
    ax.imshow(benefit, cmap=cmap, vmin=-vmax, vmax=vmax)
    for i in range(n):
        for j in range(n):
            v = delta[i, j]
            strong = abs(benefit[i, j]) >= 0.6 * vmax
            ax.text(j, i, "0" if v == 0 else f"{v:+d}", ha="center", va="center",
                    fontsize=12 if n <= 2 else 8.5, fontweight="bold" if v else "normal",
                    color=SURFACE if strong else (MUTED if v == 0 else INK))
    _axes_frame(ax, classes, show_y, show_x)


def _grid_figure(task: str):
    import matplotlib.pyplot as plt

    big = task == "binary"
    w, h = (3.0, 3.25) if big else (3.55, 3.75)
    fig, axes = plt.subplots(2, 4, figsize=(4 * w, 2 * h + 1.15), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07 if big else 0.085, right=0.985, top=0.80 if big else 0.82,
                        bottom=0.1 if big else 0.13, wspace=0.18 if big else 0.12, hspace=0.42 if big else 0.5)
    return fig, axes.ravel()


def _headers(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.965, title, fontsize=14, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.02, 0.918, subtitle, fontsize=8.8, color=INK2, ha="left", va="top", linespacing=1.45)


def _empty_slot(ax, name: str, color: str, note: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    _identity_rule(ax, color)
    ax.set_title(name, loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=12)
    ax.text(0.5, 0.5, note, transform=ax.transAxes, ha="center", va="center", fontsize=8.5, color=MUTED)


def _colorbar_slot(ax, fig, cmap, label: str, ticks, ticklabels, vmin: float, vmax: float, notes: str) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    ax.axis("off")
    cax = ax.inset_axes([0.08, 0.66, 0.84, 0.07])
    cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=7.5, colors=INK2, length=0)
    cax.set_title(label, fontsize=8.5, color=INK, loc="left", pad=5)
    ax.text(0.08, 0.52, notes, transform=ax.transAxes, fontsize=7.8, color=INK2, va="top", ha="left",
            linespacing=1.5)


def plot_grid(task: str, view: str, tag: str, cells: dict, seed_note: dict) -> Path | None:
    import matplotlib.pyplot as plt

    seq, _ = _cmaps()
    classes = patient_classes(task)
    if not any((task, view, b[0], tag) in cells for b in BACKBONES):
        return None
    fig, flat = _grid_figure(task)
    for slot, (key, name, _, color) in enumerate(BACKBONES):
        ax = flat[slot]
        cm = cells.get((task, view, key, tag))
        if cm is None:
            _empty_slot(ax, name, color, "single seed only:\nno 3-seed ensemble" if tag == "ens" else "no run")
            continue
        s = cm_scores(cm)
        draw_cm(ax, cm, classes, seq, show_y=slot % 4 == 0, show_x=True)
        _identity_rule(ax, color)
        ax.set_title(f"{name}\nbal. acc. {s['bal']:.3f}  ·  macro-F1 {s['macro_f1']:.3f}", loc="left",
                     fontsize=9.5 if task == "binary" else 9.2, color=INK, fontweight="bold", pad=12,
                     linespacing=1.35)
    n = int(next(c for k, c in cells.items() if k[0] == task and k[1] == view).sum())
    _colorbar_slot(flat[7], fig, seq, "Colour: share of the true class", [0, 0.5, 1],
                   ["0%", "50%", "100%"], 0, 1,
                   f"Number: patients · %: row share.\nDiagonal %: per-class recall;\n"
                   f"balanced accuracy = their mean.\nn = {n} patients per matrix.")
    what = "Seed 42" if tag == "s42" else "3-seed ensemble (mean probability over seeds " + seed_note.get(task, "") + ")"
    _headers(fig, f"Two-tower confusion matrices: {TASK_TITLE[task]}, {VIEWS[view][2]}",
             f"{what}. Outer-fold test predictions of the nested 5-fold protocol pooled over the 5 folds, "
             "lesion-crop input, predicted class = argmax.\nPooled scores differ slightly from the bar graphs, "
             "which average the 5 per-fold scores. Same colour = same backbone in every figure.")
    out = PLOT_DIR / f"cm_{task}_{view}_{tag}_grid.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_delta_grid(task: str, tag: str, cells: dict) -> Path | None:
    import matplotlib.pyplot as plt

    _, div = _cmaps()
    classes = patient_classes(task)
    deltas = {}
    for key, *_ in BACKBONES:
        fused, wle = cells.get((task, "fused_paired", key, tag)), cells.get((task, "wle_only_paired", key, tag))
        if fused is not None and wle is not None:
            deltas[key] = fused - wle
    if not deltas:
        return None
    vmax = max(max(int(np.abs(d).max()) for d in deltas.values()), 1)
    fig, flat = _grid_figure(task)
    for slot, (key, name, _, color) in enumerate(BACKBONES):
        ax = flat[slot]
        if key not in deltas:
            _empty_slot(ax, name, color, "no run")
            continue
        fused_s = cm_scores(cells[(task, "fused_paired", key, tag)])
        wle_s = cm_scores(cells[(task, "wle_only_paired", key, tag)])
        draw_delta(ax, deltas[key], classes, div, vmax, show_y=slot % 4 == 0, show_x=True)
        _identity_rule(ax, color)
        ax.set_title(f"{name}\nbal. acc. {wle_s['bal']:.3f} → {fused_s['bal']:.3f} "
                     f"({fused_s['bal'] - wle_s['bal']:+.3f})", loc="left", fontsize=9.2, color=INK,
                     fontweight="bold", pad=12, linespacing=1.35)
    _colorbar_slot(flat[7], fig, div, "Cell colour: effect of adding EUS", [-vmax, 0, vmax],
                   ["hurts", "no change", "helps"], -vmax, vmax,
                   "Number: fused minus WLE-only patients.\nDiagonal +: more patients correct.\n"
                   "Off-diagonal +: more patients\nmisclassified this way.\nn = 421 patients with EUS frames.")
    _headers(fig, f"What fusion changes: {TASK_TITLE[task]}, fused minus WLE-only",
             f"{'Seed 42' if tag == 's42' else '3-seed ensemble'}, the 421 patients that have EUS frames (for the "
             "other 220 the two views are the same input). Same model, EUS masked out vs. present; outer-fold "
             "test predictions,\nnested 5-fold protocol, lesion-crop input. Same colour = same backbone in every "
             "figure.")
    out = PLOT_DIR / f"cm_{task}_fusion_delta_{tag}_grid.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_single(task: str, view: str, backbone: str, tag: str, cm: np.ndarray, seeds: str) -> Path:
    """One matrix per file, for slides."""
    import matplotlib.pyplot as plt

    seq, _ = _cmaps()
    classes = patient_classes(task)
    name, color = next((b[1], b[3]) for b in BACKBONES if b[0] == backbone)
    size = (3.6, 4.1) if task == "binary" else (4.6, 5.0)
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    fig.subplots_adjust(left=0.24 if task == "subtype" else 0.2, right=0.96, top=0.78, bottom=0.2)
    draw_cm(ax, cm, classes, seq, show_y=True, show_x=True)
    s = cm_scores(cm)
    _identity_rule(ax, color)
    ax.set_title(f"{name} · {seeds}\n{VIEWS[view][2]}\nbal. acc. {s['bal']:.3f}  ·  macro-F1 {s['macro_f1']:.3f}",
                 loc="left", fontsize=8.6, color=INK, fontweight="bold", pad=12, linespacing=1.35)
    out = PLOT_DIR / "single" / f"cm_{backbone}_{task}_{view}_{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"

    runs = load_oof()
    print(f"{len(runs)} ablation runs with cached OOF predictions")
    verify_against_41(runs)

    by_bt = defaultdict(list)
    for r in runs:
        by_bt[(r["backbone"], r["task"])].append(r)
    ensembles = [ensemble(rs) for rs in by_bt.values() if len(rs) >= 3]
    verify_against_35(ensembles)

    # The extra "wle_only_paired" view (WLE-only on the patients with EUS)
    # is only for the fusion-delta matrices.
    views = dict(VIEWS)
    views["wle_only_paired"] = ("wle_only", True, "WLE only, patients with EUS")
    cells, seed_note = {}, {}
    sources = [(r, f"s{r['seed']}") for r in runs if r["seed"] == MAIN_SEED] + [(e, "ens") for e in ensembles]
    for src, tag in sources:
        n_classes = len(patient_classes(src["task"]))
        for view, (ablation, paired_only, _) in views.items():
            sel = src["paired"] if paired_only else np.ones(len(src["y"]), bool)
            cells[(src["task"], view, src["backbone"], tag)] = confusion(src[f"probs_{ablation}"][sel],
                                                                         src["y"][sel], n_classes)
        if tag == "ens":
            seed_note[src["task"]] = "/".join(str(s) for s in src["seeds"])

    for (task, view, backbone, tag), cm in cells.items():
        expected = 421 if VIEWS.get(view, views[view])[1] else 641
        assert cm.sum() == expected, (task, view, backbone, tag, cm.sum())
    write_tables({k: v for k, v in cells.items() if k[1] != "wle_only_paired"})

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for task in ("binary", "subtype"):
        for view in ("fused", "wle_only", "eus_only", "fused_paired"):
            written.append(plot_grid(task, view, "s42", cells, seed_note))
        written.append(plot_grid(task, "fused", "ens", cells, seed_note))
        written.append(plot_delta_grid(task, "s42", cells))
        written.append(plot_delta_grid(task, "ens", cells))
        for (t, view, backbone, tag), cm in sorted(cells.items()):
            if t == task and view == "fused":
                label = "seed 42" if tag == "s42" else f"3-seed ensemble ({seed_note[task]})"
                written.append(plot_single(task, view, backbone, tag, cm, label))
    for p in written:
        if p is not None:
            print(f"wrote {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
