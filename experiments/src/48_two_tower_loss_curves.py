"""Experiment 48 -- training curves for every nested two-tower backbone run.

CPU only. Reads the per-epoch history that 34 stores in each run JSON:
training loss, select-split loss and select-split balanced accuracy, plus the
epoch whose checkpoint was kept. Draws:
- ``loss_{task}_s{seed}_grid.png``: one panel per backbone; each outer fold's
  training loss (solid) and select loss (dashed); dot = kept checkpoint.
- ``selectbal_{task}_s{seed}_grid.png``: select-split balanced accuracy per
  epoch. This is what drives checkpoint choice, the LR plateau and early stopping.
- ``single/curves_{run_name}.png``: one figure per run (every seed).
- ``experiments/results/48_two_tower_curves/training_summary.csv``: per run x
  fold epochs trained, kept epoch, select bal, wall time, peak GPU memory.
  These are the training details a paper reports.

The training loss is class-weighted cross-entropy (34's ``--loss weighted_ce``);
the select loss is plain cross-entropy. The two curves are not on the same
footing, so read the trend of each, not the gap between them.

    python -m experiments.src.48_two_tower_loss_curves
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import PROJECT_ROOT

exp41 = importlib.import_module("experiments.src.41_two_tower_ablation_bargraphs")
BACKBONES, TASK_TITLE = exp41.BACKBONES, exp41.TASK_TITLE
SURFACE, INK, INK2, MUTED, GRID, AXIS = exp41.SURFACE, exp41.INK, exp41.INK2, exp41.MUTED, exp41.GRID, exp41.AXIS

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "48_two_tower_curves"
PLOT_DIR = PROJECT_ROOT / "plots" / "two_tower_loss"
NAME = {b[0]: b[1] for b in BACKBONES}
COLOR = {b[0]: b[3] for b in BACKBONES}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=0)


def draw_loss(ax, run: dict, color: str) -> None:
    for f in run["folds"]:
        h = f["history"]
        ep = [e["epoch"] for e in h]
        ax.plot(ep, [e["train_loss"] for e in h], color=color, linewidth=1.2, alpha=0.8)
        ax.plot(ep, [e["val_loss"] for e in h], color=color, linewidth=1.2, alpha=0.8, linestyle=(0, (3, 2)))
        best = f["best"]["epoch"]
        ax.plot([best], [h[best - 1]["val_loss"]], marker="o", markersize=5, color=color,
                markeredgecolor=SURFACE, markeredgewidth=1, linestyle="none", zorder=3)


def draw_selectbal(ax, run: dict, color: str, chance: float) -> None:
    for f in run["folds"]:
        h = f["history"]
        ax.plot([e["epoch"] for e in h], [e["bal"] for e in h], color=color, linewidth=1.2, alpha=0.8)
        best = f["best"]
        ax.plot([best["epoch"]], [best["bal"]], marker="o", markersize=5, color=color, markeredgecolor=SURFACE,
                markeredgewidth=1, linestyle="none", zorder=3)
    ax.axhline(chance, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)))


def _identity(ax, color: str) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((0.0, 1.03), 0.09, 0.022, transform=ax.transAxes, facecolor=color, edgecolor="none",
                           clip_on=False))


def _legend_slot(ax, kind: str, chance: float | None = None) -> None:
    from matplotlib.lines import Line2D

    ax.axis("off")
    if kind == "loss":
        handles = [Line2D([], [], color=INK2, linewidth=1.4, label="training loss (class-weighted CE)"),
                   Line2D([], [], color=INK2, linewidth=1.4, linestyle=(0, (3, 2)),
                          label="select-split loss (plain CE)"),
                   Line2D([], [], color=INK2, marker="o", markersize=6, linestyle="none",
                          markeredgecolor=SURFACE, label="kept checkpoint (best select bal)")]
    else:
        handles = [Line2D([], [], color=INK2, linewidth=1.4, label="select-split balanced accuracy"),
                   Line2D([], [], color=INK2, marker="o", markersize=6, linestyle="none",
                          markeredgecolor=SURFACE, label="kept checkpoint"),
                   Line2D([], [], color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), label=f"chance ({chance:.2f})")]
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=8, labelcolor=INK2, handlelength=2.2,
              title="One line per outer fold", title_fontsize=8.3, alignment="left")
    ax.get_legend().get_title().set_color(INK)


def plot_grid(task: str, seed: int, runs: dict, kind: str) -> Path | None:
    import matplotlib.pyplot as plt

    cells = {bb: runs.get((bb, task, seed)) for bb, *_ in BACKBONES}
    if not any(cells.values()):
        return None
    chance = 1 / (2 if task == "binary" else 5)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.0), facecolor=SURFACE)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.08, wspace=0.22, hspace=0.45)
    flat = axes.ravel()
    ymax = 0.0
    for slot, (bb, name, _, color) in enumerate(BACKBONES):
        ax = flat[slot]
        _style(ax)
        run = cells[bb]
        _identity(ax, color)
        ax.set_title(name, loc="left", fontsize=9.5, color=INK, fontweight="bold", pad=10)
        if run is None:
            ax.text(0.5, 0.5, f"no seed-{seed} run", transform=ax.transAxes, ha="center", color=MUTED, fontsize=8.5)
            continue
        if kind == "loss":
            draw_loss(ax, run, color)
            ymax = max(ymax, max(max(e["train_loss"], e["val_loss"]) for f in run["folds"] for e in f["history"]))
        else:
            draw_selectbal(ax, run, color, chance)
        ax.set_xlabel("epoch", fontsize=8, color=MUTED)
    for ax in flat[:7]:
        if kind == "loss":
            ax.set_ylim(0, min(ymax * 1.05, 3.0))
        else:
            ax.set_ylim(0, 1.0)
    _legend_slot(flat[7], kind, chance)
    what = "Loss" if kind == "loss" else "Select-split balanced accuracy"
    fig.text(0.05, 0.955, f"{what} per epoch: two-tower backbones, {TASK_TITLE[task]}, seed {seed}", fontsize=14,
             color=INK, fontweight="bold", ha="left")
    fig.text(0.05, 0.905, "Nested protocol: each outer fold trains on its fit split and picks the checkpoint, LR "
             "plateau and early stop (patience 20) on its select split; the outer test fold is never seen.\n"
             "Curves end where early stopping fired. Same colour = same backbone in every figure.",
             fontsize=8.6, color=INK2, ha="left", va="top", linespacing=1.45)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / f"{'loss' if kind == 'loss' else 'selectbal'}_{task}_s{seed}_grid.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_single(run: dict) -> Path:
    import matplotlib.pyplot as plt

    bb, task = run["backbone"], run["task"]
    color = COLOR[bb]
    chance = 1 / (2 if task == "binary" else 5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.15, wspace=0.2)
    for ax in (a1, a2):
        _style(ax)
        ax.set_xlabel("epoch", fontsize=8, color=MUTED)
    draw_loss(a1, run, color)
    a1.set_title("loss: training (solid), select split (dashed)", loc="left", fontsize=9, color=INK, pad=6)
    draw_selectbal(a2, run, color, chance)
    a2.set_ylim(0, 1)
    a2.set_title("select-split balanced accuracy (dot = kept checkpoint)", loc="left", fontsize=9, color=INK, pad=6)
    best = [f["best"]["epoch"] for f in run["folds"]]
    fig.text(0.07, 0.94, f"{NAME[bb]} · {TASK_TITLE[task]} · seed {run['seed']}", fontsize=12, color=INK,
             fontweight="bold")
    fig.text(0.07, 0.875, f"5 outer folds; kept epochs {best}; one line per fold.", fontsize=8.5, color=INK2)
    out = PLOT_DIR / "single" / f"curves_{run['run_name']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out


def write_summary(runs: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "training_summary.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backbone", "task", "seed", "fold", "epochs_trained", "kept_epoch", "select_bal",
                    "n_fit", "n_select", "n_test", "trainable_params", "elapsed_min", "peak_mem_gb"])
        for r in runs:
            for f in r["folds"]:
                w.writerow([r["backbone"], r["task"], r["seed"], f["fold"], len(f["history"]), f["best"]["epoch"],
                            round(f["best"]["bal"], 4), f["n_train"], f["n_val"], f["n_test"],
                            f.get("n_params", {}).get("trainable", ""), round(f.get("elapsed_sec", 0) / 60, 1),
                            f.get("peak_mem_gb", "")])
    return path


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"

    runs = exp41.load_runs()
    if not runs:
        raise SystemExit("no runs found -- is the results volume mounted?")
    print(f"{len(runs)} completed nested runs")
    by_key = {(r["backbone"], r["task"], r["seed"]): r for r in runs}
    written = []
    for task in ("binary", "subtype"):
        for seed in sorted({r["seed"] for r in runs}):
            for kind in ("loss", "selectbal"):
                written.append(plot_grid(task, seed, by_key, kind))
    written += [plot_single(r) for r in runs]
    written.append(write_summary(runs))
    for p in written:
        if p is not None and "single" not in str(p):
            print(f"wrote {p.relative_to(PROJECT_ROOT)}")
    print(f"wrote {len(runs)} per-run figures to {(PLOT_DIR / 'single').relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
