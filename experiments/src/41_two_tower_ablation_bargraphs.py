"""Experiment 41 -- bar graphs for the two-tower backbone ablation.

``34_joint_fusion_train.py`` only records accuracy / balanced accuracy /
macro-F1 per fold, so this re-scores every completed nested-protocol run's
``fold{k}_best.pt`` on its own outer (test) fold -- same folds, frame
selection and masked-inference ablations as 34's final pass -- and keeps the
per-patient class probabilities. From those it computes ROC AUC, accuracy,
balanced accuracy, precision, recall, specificity (binary) and F1 per fold,
and draws one bar per backbone, a fixed colour per backbone in every figure.

Bars are the mean of the 5-fold means across the seeds a backbone was run
with, whiskers +/- 1 SD over folds, dots the individual seeds. Views: WLE
only and fused over all test patients, EUS only and fused over the patients
that have EUS frames (the same denominators 34 reports).

    python -m experiments.src.41_two_tower_ablation_bargraphs            # score + plot
    python -m experiments.src.41_two_tower_ablation_bargraphs --plot-only
    python -m experiments.src.41_two_tower_ablation_bargraphs --variants   # score follow-up runs only
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
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import MODALITY_NAMES, PROJECT_ROOT
from cadsel.data import build_joint_manifest, joint_strata_key, patient_level_folds
from cadsel.fusion import SUBTYPES, patient_classes, patient_label
from cadsel.joint_fusion_data import JointPatientDataset, joint_collate
from cadsel.joint_fusion_model import JointTwoTowerModel, tower_normalizers

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "joint_fusion_runs"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "41_two_tower_ablation"
PLOT_DIR = PROJECT_ROOT / "plots" / "two_tower_ablation"
ABLATION_MODALITIES = {"wle_only": ["WLE"], "eus_only": ["EUS"], "fused": MODALITY_NAMES}

# view -> (ablation, restrict to patients with both modalities, title)
VIEWS = {
    "wle_only": ("wle_only", False, "WLE only, all test patients"),
    "eus_only": ("eus_only", True, "EUS only, patients with EUS"),
    "fused": ("fused", False, "fused WLE+EUS, all test patients"),
    "fused_paired": ("fused", True, "fused WLE+EUS, patients with EUS"),
}
VIEW_SHORT = {"wle_only": "WLE only", "eus_only": "EUS only\n(paired)", "fused": "Fused",
              "fused_paired": "Fused\n(paired)"}

# Fixed order = fixed colour: a backbone keeps its colour and slot in every
# figure, whichever runs exist. Order validated as an adjacent categorical
# palette (dataviz skill validator, light surface).
BACKBONES = [
    ("efficientnet_b0", "EfficientNet-B0", "EffNet-B0", "#2a78d6"),
    ("resnet50", "ResNet-50", "ResNet-50", "#eb6834"),
    ("convnext_tiny", "ConvNeXt-Tiny", "ConvNeXt-T", "#1baf7a"),
    ("mobilenet_v3_large", "MobileNetV3-Large", "MobNetV3-L", "#eda100"),
    ("mobilevit_s", "MobileViT-S", "MobileViT-S", "#e87ba4"),
    ("deit_s", "DeiT-S", "DeiT-S", "#008300"),
    ("dinov3_vits16", "DINOv3 ViT-S/16", "DINOv3-S", "#4a3aa7"),
]
# Colours reserved for the cluster follow-up's bigger DINOv3 towers (slot-8
# red, then a lighter violet that reads as DINOv3's family); appended to
# BACKBONES' slots once their runs exist.
CLUSTER_BACKBONES = [
    ("dinov3_vitb16", "DINOv3 ViT-B/16", "DINOv3-B", "#e34948"),
    ("dinov3_vitl16", "DINOv3 ViT-L/16", "DINOv3-L", "#9085e9"),
]
METRICS = {
    "binary": [("auroc", "ROC AUC"), ("acc", "Accuracy"), ("bal", "Balanced accuracy"),
               ("precision", "Precision (NET)"), ("recall", "Recall / sensitivity (NET)"),
               ("specificity", "Specificity (non-NET)"), ("f1", "F1 (NET)")],
    "subtype": [("auroc", "ROC AUC (macro one-vs-rest)"), ("acc", "Accuracy"),
                ("bal", "Balanced accuracy (= macro recall)"), ("precision", "Precision (macro)"),
                ("f1", "F1 (macro)")],
}
TASK_TITLE = {"binary": "binary (NET vs non-NET)", "subtype": "5-way subtype"}

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def is_ablation_config(run: dict) -> bool:
    """The ablation's own configuration; the follow-up variants (full
    frame, symmetric fusion, external-centre test) are compared elsewhere."""
    return (bool(run.get("lesion_crop")) and run.get("fusion", "gated") == "gated"
            and not run.get("external_test_centres") and not run.get("paired_only"))


def load_runs(ablation_only: bool = True) -> list[dict]:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*__nested*.json")):
        run = json.loads(path.read_text())
        if run.get("inner_val_fraction", 0) <= 0 or run["folds_done"] != run["n_splits"]:
            continue
        if ablation_only and not is_ablation_config(run):
            continue
        runs.append(run)
    return runs


def _normalize_frames(images: dict, normalizers: dict, device: torch.device) -> dict:
    out = {}
    for m in MODALITY_NAMES:
        x = images[m].to(device, non_blocking=True)
        B, T, C, H, W = x.shape
        flat = normalizers[m](x.reshape(B * T, C, H, W).float())
        out[m] = flat.reshape(B, T, C, H, W)
    return out


def build_fold_inputs(run: dict, num_workers: int) -> list[dict]:
    """Each outer fold's test patients as pre-collated uint8 batches. The
    inputs depend only on the split / frame-selection settings, not on the
    backbone or task, so every run of one seed shares them."""
    samples = build_joint_manifest()
    if run.get("paired_only"):
        paired_keys = {s.patient_key for s in samples if s.modality == "EUS"}
        samples = [s for s in samples if s.patient_key in paired_keys]
    if run.get("external_test_centres"):
        exp34 = importlib.import_module("experiments.src.34_joint_fusion_train")
        held_out = set(run["external_test_centres"].split(","))
        centre = exp34.patient_centres(samples)
        folds = [([s for s in samples if centre[s.patient_key] not in held_out],
                  [s for s in samples if centre[s.patient_key] in held_out])]
    else:
        folds = patient_level_folds(samples, n_folds=run["n_splits"], seed=run["seed"],
                                     strata_key=joint_strata_key)
    out = []
    for k, (_, test_samples) in enumerate(folds):
        ds = JointPatientDataset(test_samples, "subtype", img_size=run["img_size"],
                                  max_frames=run["max_frames"], train=False, seed=run["seed"],
                                  lesion_crop=run.get("lesion_crop", False),
                                  bg_remove_prob=run.get("bg_remove_prob", 0.0))
        loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=joint_collate,
                            num_workers=num_workers)
        batches = [(images, masks) for images, masks, _ in loader]
        out.append({"patient_keys": list(ds.patient_keys),
                    "subtypes": [ds.subtype[key] for key in ds.patient_keys], "batches": batches})
        print(f"    cached fold {k}: {len(ds)} test patients", flush=True)
    return out


@torch.no_grad()
def score_run(run: dict, fold_inputs: list[dict], device: torch.device) -> dict:
    task = run["task"]
    classes = patient_classes(task)
    normalizers = tower_normalizers(run.get("backbone", "efficientnet_b0"))
    keys, fold_of, y, paired = [], [], [], []
    probs = {name: [] for name in ABLATION_MODALITIES}
    for f in sorted(run["folds"], key=lambda f: f["fold"]):
        k = f["fold"]
        ck = torch.load(RESULTS_DIR / run["run_name"] / f"fold{k}_best.pt", map_location="cpu",
                        weights_only=False)
        model = JointTwoTowerModel(len(classes), backbone=ck.get("backbone", "efficientnet_b0"),
                                    freeze=ck["freeze"], fusion=ck["fusion"],
                                    modality_dropout=ck["modality_dropout"],
                                    unfreeze_last_n=ck.get("unfreeze_last_n", 4)).to(device).eval()
        model.load_state_dict(ck["model_state_dict"])
        inp = fold_inputs[k]
        for images, masks in inp["batches"]:
            images_norm = _normalize_frames(images, normalizers, device)
            paired.append(((masks["WLE"].sum(dim=1) > 0.5) & (masks["EUS"].sum(dim=1) > 0.5)).numpy())
            for name, use_modalities in ABLATION_MODALITIES.items():
                masks_dev = {m: (masks[m] if m in use_modalities else torch.zeros_like(masks[m])).to(device)
                             for m in MODALITY_NAMES}
                logits = model(images_norm, masks_dev)
                probs[name].append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        keys += inp["patient_keys"]
        fold_of += [k] * len(inp["patient_keys"])
        y += [classes.index(patient_label(st, task)) for st in inp["subtypes"]]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"patient_key": np.array(keys), "fold": np.array(fold_of), "y": np.array(y),
            "paired": np.concatenate(paired),
            **{f"probs_{name}": np.concatenate(p) for name, p in probs.items()}}


def fold_metrics(probs: np.ndarray, y: np.ndarray, task: str) -> dict:
    """Same conventions as ``cadsel.train_fusion.score`` (macro averages over
    the classes present in ``y``), plus ROC AUC and, for binary, the NET
    (class 0, the potentially-malignant one) precision / recall / F1."""
    from sklearn.metrics import roc_auc_score

    n_classes = probs.shape[1]
    pred = probs.argmax(1)
    prec, rec, f1, auc = [], [], [], []
    for c in range(n_classes):
        in_class = y == c
        if not in_class.any():
            continue
        predicted = pred == c
        r = float((pred[in_class] == c).mean())
        p = float((y[predicted] == c).mean()) if predicted.any() else 0.0
        prec.append(p)
        rec.append(r)
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
        if not in_class.all():
            auc.append(float(roc_auc_score(in_class, probs[:, c])))
    out = {"n": int(len(y)), "acc": float((pred == y).mean()), "bal": float(np.mean(rec)),
           "macro_f1": float(np.mean(f1)), "majority": float(np.bincount(y).max() / len(y))}
    if task == "binary":
        out.update(auroc=auc[0], precision=prec[0], recall=rec[0], specificity=rec[1], f1=f1[0])
    else:
        out.update(auroc=float(np.mean(auc)), precision=float(np.mean(prec)), f1=float(np.mean(f1)))
    return out


def collect(runs: list[dict], device: torch.device, num_workers: int) -> None:
    """Score every run that has no cached predictions yet."""
    (OUT_DIR / "oof").mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for run in runs:
        if (OUT_DIR / "oof" / f"{run['run_name']}.npz").exists():
            continue
        key = (run["seed"], run["n_splits"], run.get("paired_only", False), run["img_size"],
               run["max_frames"], run.get("lesion_crop", False), run.get("bg_remove_prob", 0.0),
               run.get("external_test_centres"))
        groups[key].append(run)
    for key, group in groups.items():
        print(f"seed {key[0]}: loading test-fold inputs for {len(group)} run(s)", flush=True)
        fold_inputs = build_fold_inputs(group[0], num_workers)
        for run in group:
            pred = score_run(run, fold_inputs, device)
            # the re-scored fused arm should reproduce what 34 recorded for each fold
            worst = 0.0
            for f in run["folds"]:
                sel = pred["fold"] == f["fold"]
                m = fold_metrics(pred["probs_fused"][sel], pred["y"][sel], run["task"])
                worst = max(worst, abs(m["bal"] - f["ablations"]["fused"]["unpaired"]["bal"]))
            flag = "" if worst < 5e-4 else "   <-- differs from the recorded fold scores"
            print(f"  {run['backbone']:20s} {run['task']:8s} s{run['seed']:<4d} "
                  f"max |bal - recorded| over folds = {worst:.4f}{flag}", flush=True)
            np.savez_compressed(OUT_DIR / "oof" / f"{run['run_name']}.npz", **pred)
        del fold_inputs


def tabulate(runs: list[dict]) -> list[dict]:
    """One row per run x fold x view."""
    rows = []
    for run in runs:
        path = OUT_DIR / "oof" / f"{run['run_name']}.npz"
        if not path.exists():
            print(f"  no predictions for {run['run_name']}, skipping")
            continue
        pred = np.load(path)
        for view, (ablation, paired_only, _) in VIEWS.items():
            for k in sorted(set(pred["fold"].tolist())):
                sel = pred["fold"] == k
                if paired_only:
                    sel = sel & pred["paired"]
                m = fold_metrics(pred[f"probs_{ablation}"][sel], pred["y"][sel], run["task"])
                rows.append({"task": run["task"], "backbone": run["backbone"], "seed": run["seed"],
                             "fold": k, "view": view, **m})
    return rows


def summarise(rows: list[dict]) -> dict:
    """(task, view, backbone, metric) -> mean of per-seed fold means, fold SD,
    and the per-seed values."""
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for metric, value in r.items():
            if metric in ("task", "backbone", "seed", "fold", "view", "n"):
                continue
            cells[(r["task"], r["view"], r["backbone"], metric)][r["seed"]].append(value)
    out = {}
    for key, by_seed in cells.items():
        seed_means = {s: float(np.mean(v)) for s, v in sorted(by_seed.items())}
        every_fold = [x for v in by_seed.values() for x in v]
        out[key] = {"mean": float(np.mean(list(seed_means.values()))), "fold_sd": float(np.std(every_fold)),
                    "seeds": seed_means, "n_folds": len(every_fold)}
    return out


def write_tables(rows: list[dict], summary: dict) -> None:
    fields = ["task", "backbone", "seed", "fold", "view", "n", "auroc", "acc", "bal", "precision",
              "recall", "specificity", "f1", "macro_f1", "majority"]
    with open(OUT_DIR / "metrics_per_fold.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(OUT_DIR / "metrics_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task", "view", "backbone", "metric", "mean", "fold_sd", "n_seeds", "per_seed"])
        for (task, view, backbone, metric), c in sorted(summary.items()):
            if metric == "majority":
                continue
            w.writerow([task, view, backbone, metric, round(c["mean"], 4), round(c["fold_sd"], 4),
                        len(c["seeds"]), " ".join(f"s{s}={v:.4f}" for s, v in c["seeds"].items())])
    print(f"wrote {OUT_DIR / 'metrics_per_fold.csv'}\nwrote {OUT_DIR / 'metrics_summary.csv'}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", colors=INK2, labelsize=7.5, length=0)


def _bar(ax, x: float, width: float, height: float, color: str, radius_pt: float = 2.5) -> None:
    """Column with a rounded data end, square at the baseline. Needs the
    axis limits and position to be final (the corner is sized in pixels)."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    (ox, oy), (ux, uy) = ax.transData.transform([(0, 0), (1, 1)])
    r_px = radius_pt * ax.figure.dpi / 72
    rx = min(r_px / abs(ux - ox), width / 2)
    ry = min(r_px / abs(uy - oy), max(height, 0))
    x0, x1 = x - width / 2, x + width / 2
    verts = [(x0, 0), (x0, height - ry), (x0, height), (x0 + rx, height), (x1 - rx, height),
             (x1, height), (x1, height - ry), (x1, 0), (x0, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO,
             MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", zorder=2))


def _reference(ax, metric: str, task: str, view: str, summary: dict) -> None:
    """Dashed baseline behind the bars: chance for ROC AUC / balanced
    accuracy, the always-predict-the-majority-class rate for accuracy."""
    if metric == "auroc":
        y = 0.5
    elif metric == "bal":
        y = 1 / len(patient_classes(task))
    elif metric == "acc":
        maj = [c["mean"] for (t, v, _, m), c in summary.items() if (t, v, m) == (task, view, "majority")]
        if not maj:
            return
        y = float(np.mean(maj))
    else:
        return
    ax.axhline(y, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1.5)


def _legend(ax, task: str, view: str, summary: dict) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    for key, name, _, color in BACKBONES:
        cell = summary.get((task, view, key, "bal"))
        if cell is None:
            continue
        seeds = sorted(cell["seeds"])
        handles.append(Patch(facecolor=color, edgecolor="none",
                             label=f"{name}  ·  {len(seeds)} seed{'s' if len(seeds) > 1 else ''} "
                                   f"({', '.join(str(s) for s in seeds)})"))
    handles += [
        Line2D([], [], color=INK2, linewidth=1.0, label="whisker: ± 1 SD over folds"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=INK, markeredgecolor=SURFACE,
               markersize=5, label="dot: one seed's 5-fold mean"),
        Line2D([], [], color=INK2, linewidth=0.9, linestyle=(0, (4, 3)),
               label="dashed: chance (accuracy: always\npredicting the majority class)"),
    ]
    ax.axis("off")
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=8.5, labelcolor=INK2,
              handlelength=1.4, borderaxespad=0, labelspacing=0.75,
              title="Bar: mean over seeds of the 5-fold mean", title_fontsize=8.5, alignment="left")
    ax.get_legend().get_title().set_color(INK)


def _draw_cell(ax, x: float, width: float, cell: dict, color: str, detail: bool) -> None:
    """Bar + fold-SD whisker; ``detail`` adds the per-seed dots and the value."""
    _bar(ax, x, width, cell["mean"], color)
    lo, hi = max(cell["mean"] - cell["fold_sd"], 0), min(cell["mean"] + cell["fold_sd"], 1.0)
    ax.plot([x, x], [lo, hi], color=INK2, linewidth=0.9, zorder=3, solid_capstyle="butt")
    top = hi
    if detail and len(cell["seeds"]) > 1:
        vals = list(cell["seeds"].values())
        offsets = np.linspace(-0.22, 0.22, len(vals)) * width
        ax.scatter(x + offsets, vals, s=13, color=INK, edgecolor=SURFACE, linewidth=0.6, zorder=4)
        top = max(top, max(vals))
    if detail:
        # surface-coloured pad so a reference line never strikes through the number
        ax.text(x, top + 0.014, f"{cell['mean']:.3f}", ha="center", va="bottom", fontsize=6.8,
                color=INK2, zorder=5, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6})


def plot_view(task: str, view: str, summary: dict) -> Path | None:
    """One panel per metric, one bar per backbone."""
    import matplotlib.pyplot as plt

    metrics = METRICS[task]
    if not any((task, view, b[0], "bal") in summary for b in BACKBONES):
        return None
    ncols = (len(metrics) + 2) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(4.3 * ncols, 8.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.11, wspace=0.2, hspace=0.5)
    flat = axes.ravel()
    n = len(BACKBONES)
    for ax, (metric, title) in zip(flat, metrics):
        _style_axis(ax)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8, fontweight="bold")
        for i, (key, _, short, color) in enumerate(BACKBONES):
            cell = summary.get((task, view, key, metric))
            if cell is not None:
                _draw_cell(ax, i, 0.62, cell, color, detail=True)
        _reference(ax, metric, task, view, summary)
        ax.set_xticks(range(n))
        ax.set_xticklabels([b[2] for b in BACKBONES], rotation=38, ha="right", rotation_mode="anchor")
    for ax in flat[len(metrics):-1]:
        ax.axis("off")
    _legend(flat[-1], task, view, summary)
    fig.text(0.045, 0.95, f"Two-tower backbone ablation: {TASK_TITLE[task]}, {VIEWS[view][2]}",
             fontsize=14, color=INK, fontweight="bold", ha="left")
    fig.text(0.045, 0.915, "Outer-fold test scores, nested 5-fold protocol (checkpoint chosen on an inner "
             "select split), lesion-crop input. Same colour = same backbone in every figure.",
             fontsize=9, color=INK2, ha="left")
    out = PLOT_DIR / f"two_tower_ablation_{task}_{view}.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_modalities(task: str, summary: dict) -> Path:
    """Headline metrics with the four views side by side, bars grouped by view."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    wanted = ["bal", "auroc", "f1"]
    metrics = [(m, t) for m, t in METRICS[task] if m in wanted]
    metrics.sort(key=lambda mt: wanted.index(mt[0]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 5.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.74, bottom=0.12, wspace=0.14)
    n, views = len(BACKBONES), list(VIEWS)
    step = 0.115
    for ax, (metric, title) in zip(axes, metrics):
        _style_axis(ax)
        ax.set_xlim(-0.55, len(views) - 0.45)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8, fontweight="bold")
        for g, view in enumerate(views):
            for i, (key, _, _, color) in enumerate(BACKBONES):
                cell = summary.get((task, view, key, metric))
                if cell is not None:
                    _draw_cell(ax, g + (i - (n - 1) / 2) * step, step * 0.8, cell, color, detail=False)
        _reference(ax, metric, task, "fused", summary)
        ax.set_xticks(range(len(views)))
        ax.set_xticklabels([VIEW_SHORT[v] for v in views])
        ax.tick_params(axis="x", labelsize=9)
    handles = [Patch(facecolor=c, edgecolor="none", label=name) for _, name, _, c in BACKBONES]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.036, 0.875), ncols=len(handles),
               frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.3, columnspacing=1.6)
    fig.text(0.04, 0.945, f"Two-tower backbone ablation: {TASK_TITLE[task]}, by input modality",
             fontsize=14, color=INK, fontweight="bold", ha="left")
    fig.text(0.04, 0.9, "Outer-fold test scores, nested 5-fold protocol, lesion-crop input. Bar: mean over "
             "seeds of the 5-fold mean; whisker: ± 1 SD over folds; dashed: chance. "
             "'(paired)' = the patients that have EUS frames.", fontsize=9, color=INK2, ha="left")
    out = PLOT_DIR / f"two_tower_ablation_{task}_by_modality.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot-only", action="store_true", help="reuse cached predictions, no inference")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--only", default=None, help="regex filter on run_name (smoke tests)")
    p.add_argument("--variants", action="store_true",
                   help="score the follow-up variant runs (full frame, symmetric fusion, external centres) "
                        "into oof/ and stop; the tables and bar graphs stay ablation-only")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"

    runs = load_runs(ablation_only=not args.variants)
    if args.variants:
        runs = [r for r in runs if not is_ablation_config(r)]
    if args.only:
        runs = [r for r in runs if re.search(args.only, r["run_name"])]
    print(f"{len(runs)} completed nested run(s)")
    if not runs:
        # writing the tables now would overwrite them with headers only
        raise SystemExit(f"no matching runs under {RESULTS_DIR} (is the results volume mounted?)")
    if args.variants:
        collect(runs, torch.device(args.device), args.num_workers)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        collect(runs, torch.device(args.device), args.num_workers)

    rows = tabulate(runs)
    summary = summarise(rows)
    write_tables(rows, summary)
    for task in ("binary", "subtype"):
        if not any(r["task"] == task for r in rows):
            continue
        for view in VIEWS:
            out = plot_view(task, view, summary)
            if out is not None:
                print(f"wrote {out}")
        print(f"wrote {plot_modalities(task, summary)}")


if __name__ == "__main__":
    main()
