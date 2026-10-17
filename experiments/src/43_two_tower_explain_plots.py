"""Figures for experiment 43 (``43_two_tower_explain.py``).

Reads only the cached results in ``experiments/results/43_two_tower_explain/``
(plus 41's OOF predictions and 36's counterfactual JSONs), so it runs without
a GPU or the results volume:

    python -m experiments.src.43_two_tower_explain --plot-only
"""

from __future__ import annotations

import csv
import importlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from cadsel.config import MODALITY_NAMES, PROJECT_ROOT
from cadsel.fusion import patient_classes

exp41 = importlib.import_module("experiments.src.41_two_tower_ablation_bargraphs")
BACKBONES, TASK_TITLE = exp41.BACKBONES, exp41.TASK_TITLE
SURFACE, INK, INK2, MUTED, GRID, AXIS = exp41.SURFACE, exp41.INK, exp41.INK2, exp41.MUTED, exp41.GRID, exp41.AXIS

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "43_two_tower_explain"
PLOT_DIR = PROJECT_ROOT / "plots" / "two_tower_explain"
OOF_DIR = exp41.OUT_DIR / "oof"
SEED = 42
CLASS_SHORT = {"NET_G1": "NET G1", "NET_G2": "NET G2", "Leiomyoma": "Leiomyoma", "Lipoma": "Lipoma",
               "NonNeoplasm": "Non-neoplasm", "NET": "NET", "non-NET": "non-NET"}
NAME = {b[0]: b[1] for b in BACKBONES}
COLOR = {b[0]: b[3] for b in BACKBONES}
# CAM overlay: the reference blue ramp from light to step 700, with opacity
# rising with saliency, so low-saliency tissue stays visible underneath.
CAM_STEPS = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CF_VARIANTS = [("lesion", "Lesion crop (as trained)"), ("full", "Full frame"),
               ("bg_mask", "Lesion filled with mean colour"), ("bg_patch", "Lesion-free background patch"),
               ("boxmap", "Box outline only (white on black)")]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _run_file(backbone: str, task: str, suffix: str) -> Path | None:
    hits = sorted(OUT_DIR.glob(f"joint_{backbone}__{task}__*__s{SEED}__{suffix}"))
    return hits[0] if hits else None


def _read_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(v: str) -> float:
    return float(v) if v not in ("", "nan") else float("nan")


def load_gallery(task: str) -> tuple[list[dict], dict, dict]:
    """(gallery patients, shared inputs, per-backbone gallery outputs)."""
    path = OUT_DIR / f"gallery_inputs_{task}.npz"
    if not path.exists():
        return [], {}, {}
    inp = np.load(path)
    gallery = [json.loads(g) for g in inp["gallery"]]
    outs = {}
    for key, *_ in BACKBONES:
        f = _run_file(key, task, "gallery.npz")
        if f is not None:
            outs[key] = np.load(f)
    return gallery, inp, outs


def unique_frames(inp, p: str, m: str) -> list[tuple[int, list[int]]]:
    """(first slot, all slots) per distinct image of patient ``p``'s
    modality ``m`` -- frames are drawn with replacement."""
    key = f"{p}|path_{m}"
    if key not in inp.files:
        return []
    groups: dict[str, list[int]] = {}
    for slot, path in enumerate(inp[key].tolist()):
        groups.setdefault(path, []).append(slot)
    return [(slots[0], slots) for slots in groups.values()]


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def _cam_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("cam_blue", CAM_STEPS)


def _up(cam: np.ndarray, size: int = 224) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(cam, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def draw_overlay(ax, img: np.ndarray, cam: np.ndarray | None, box, label: str | None = None) -> None:
    """Crop with the saliency map on top (opacity ~ normalised saliency) and
    the GT lesion box as a dashed outline."""
    from matplotlib.patches import Rectangle

    ax.imshow(img.transpose(1, 2, 0))
    if cam is not None:
        c = _up(cam)
        c = c / c.max() if c.max() > 0 else c
        rgba = _cam_cmap()(c)
        rgba[..., 3] = 0.78 * c ** 1.2
        ax.imshow(rgba)
    if box is not None and np.all(np.isfinite(box)):
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=SURFACE, linewidth=0.9,
                               linestyle=(0, (3, 2))))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if label:
        ax.set_title(label, fontsize=6.6, color=INK2, pad=2.5)


def _identity_bar(ax, color: str) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((0.0, 0.0), 0.05, 1.0, transform=ax.transAxes, facecolor=color, edgecolor="none",
                           clip_on=False))


def _headers(fig, title: str, subtitle: str) -> None:
    """Title and subtitle pinned to fixed distances (in inches) from the top."""
    h = fig.get_figheight()
    fig.text(0.015, 1 - 0.18 / h, title, fontsize=13, color=INK, fontweight="bold", ha="left", va="top")
    fig.text(0.015, 1 - 0.5 / h, subtitle, fontsize=8.3, color=INK2, ha="left", va="top", linespacing=1.45)


def _save(fig, name: str) -> Path:
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / name
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return out


def _patient_text(task: str, g: dict, out) -> str:
    classes = patient_classes(task)
    p = g["patient_key"]
    pred = int(out[f"{p}|pred"])
    probs = out[f"{p}|probs"]
    mark = "correct" if pred == g["y"] else "wrong"
    return (f"{p.split('/')[1]}\ntrue: {CLASS_SHORT[classes[g['y']]]}\n"
            f"pred: {CLASS_SHORT[classes[pred]]} ({probs[pred]:.2f})\n{mark}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_gallery(task: str, backbone: str, gallery: list[dict], inp, out) -> Path:
    """One row per gallery patient: every distinct WLE then EUS frame with the
    predicted-class Grad-CAM and the frame's attention-pool weight."""
    import matplotlib.pyplot as plt

    cols = 8
    fig = plt.figure(figsize=(1.55 * cols + 1.9, 1.72 * len(gallery) + 1.3), facecolor=SURFACE)
    top = 1 - 1.05 / fig.get_figheight()
    gs = fig.add_gridspec(len(gallery), cols + 1, left=0.01, right=0.995, top=top, bottom=0.01,
                          width_ratios=[1.25] + [1] * cols, wspace=0.06, hspace=0.32)
    for r, g in enumerate(gallery):
        p = g["patient_key"]
        tax = fig.add_subplot(gs[r, 0])
        tax.axis("off")
        tax.text(0.0, 0.5, _patient_text(task, g, out), fontsize=7.4, color=INK, va="center", linespacing=1.45)
        c = 1
        for m in MODALITY_NAMES:
            frames = unique_frames(inp, p, m)[:4]
            pool = out[f"{p}|pool_{m}"] if f"{p}|pool_{m}" in out.files else None
            for first, slots in frames:
                if c > cols:
                    break
                ax = fig.add_subplot(gs[r, c])
                w = float(sum(pool[s] for s in slots)) if pool is not None else float("nan")
                draw_overlay(ax, inp[f"{p}|img_{m}"][first], out[f"{p}|cam_pred_{m}"][first],
                             inp[f"{p}|box_{m}"][first], f"{m} · pool weight {w:.2f}")
                c += 1
    _headers(fig, f"Grad-CAM, {NAME[backbone]} two-tower: {TASK_TITLE[task]}",
             "Each row is one test patient explained by the fold model that never saw it. Blue = Grad-CAM of the "
             "predicted class's fused logit (opacity rises with saliency, normalised per frame); dashed box = GT "
             "lesion box\ninside the lesion crop. Pool weight = the attention-pooling weight of that frame within "
             "its modality. Rows: per class, the most confident correct and the most confident wrong patient.")
    return _save(fig, f"gallery_{backbone}_{task}.png")


def plot_cross_backbone(task: str, g: dict, inp, outs: dict) -> Path:
    """One patient, every tower: does the saliency agree across backbones?"""
    import matplotlib.pyplot as plt

    p = g["patient_key"]
    frames = [(m, f, s) for m in MODALITY_NAMES for f, s in unique_frames(inp, p, m)[:4]]
    rows = [b for b in BACKBONES if b[0] in outs]
    classes = patient_classes(task)
    fig = plt.figure(figsize=(1.5 * len(frames) + 2.2, 1.62 * (len(rows) + 1) + 1.2), facecolor=SURFACE)
    top = 1 - 1.15 / fig.get_figheight()
    gs = fig.add_gridspec(len(rows) + 1, len(frames) + 1, left=0.01, right=0.995, top=top, bottom=0.01,
                          width_ratios=[1.45] + [1] * len(frames), wspace=0.06, hspace=0.3)
    lab = fig.add_subplot(gs[0, 0])
    lab.axis("off")
    lab.text(0.0, 0.5, f"input\ntrue: {CLASS_SHORT[classes[g['y']]]}", fontsize=7.6, color=INK, va="center")
    for c, (m, first, _) in enumerate(frames, start=1):
        draw_overlay(fig.add_subplot(gs[0, c]), inp[f"{p}|img_{m}"][first], None, inp[f"{p}|box_{m}"][first], m)
    for r, (key, name, _, color) in enumerate(rows, start=1):
        out = outs[key]
        lab = fig.add_subplot(gs[r, 0])
        lab.axis("off")
        _identity_bar(lab, color)
        pred = int(out[f"{p}|pred"])
        lab.text(0.1, 0.5, f"{name}\npred: {CLASS_SHORT[classes[pred]]} ({out[f'{p}|probs'][pred]:.2f})",
                 fontsize=7.4, color=INK, va="center", linespacing=1.45)
        for c, (m, first, _) in enumerate(frames, start=1):
            draw_overlay(fig.add_subplot(gs[r, c]), inp[f"{p}|img_{m}"][first], out[f"{p}|cam_pred_{m}"][first],
                         inp[f"{p}|box_{m}"][first])
    _headers(fig, f"One patient, seven towers: {p} ({TASK_TITLE[task]})",
             "Grad-CAM of each backbone's predicted class (seed-42 fold model that held this patient out).\n"
             "Dashed box = GT lesion box. Colour bar = backbone, as in every figure.")
    safe = p.replace("/", "_")
    return _save(fig, f"cross_backbone_{task}_{safe}.png")


def plot_vit_attention(task: str, gallery: list[dict], inp, outs: dict) -> Path | None:
    """Grad-CAM vs last-layer CLS attention for the two ViT towers, on each
    gallery patient's highest-weighted WLE and EUS frame."""
    import matplotlib.pyplot as plt

    vits = [k for k in ("deit_s", "dinov3_vits16") if k in outs]
    if not vits:
        return None
    ref = outs[vits[-1]]
    panels = [("image", None, None)] + [(k, kind, m) for k in vits for kind in ("cam", "attn") for m in [None]]
    ncol = len(panels) * 2
    fig = plt.figure(figsize=(1.3 * ncol + 1.6, 1.5 * len(gallery) + 1.5), facecolor=SURFACE)
    top = 1 - 1.3 / fig.get_figheight()
    gs = fig.add_gridspec(len(gallery), ncol + 1, left=0.01, right=0.995, top=top, bottom=0.01,
                          width_ratios=[1.3] + [1] * ncol, wspace=0.05, hspace=0.3)
    classes = patient_classes(task)
    for r, g in enumerate(gallery):
        p = g["patient_key"]
        lab = fig.add_subplot(gs[r, 0])
        lab.axis("off")
        lab.text(0.0, 0.5, f"{p.split('/')[1]}\ntrue: {CLASS_SHORT[classes[g['y']]]}", fontsize=7.2, color=INK,
                 va="center", linespacing=1.4)
        c = 1
        for m in MODALITY_NAMES:
            frames = unique_frames(inp, p, m)
            if not frames or f"{p}|pool_{m}" not in ref.files:
                c += len(panels)
                continue
            pool = ref[f"{p}|pool_{m}"]
            first, _ = max(frames, key=lambda fs: sum(pool[s] for s in fs[1]))
            img, box = inp[f"{p}|img_{m}"][first], inp[f"{p}|box_{m}"][first]
            for key, kind, _ in panels:
                ax = fig.add_subplot(gs[r, c])
                if kind is None:
                    draw_overlay(ax, img, None, box, f"{m}" if r == 0 else None)
                else:
                    arr = outs[key][f"{p}|{'cam_pred' if kind == 'cam' else 'attn'}_{m}"][first]
                    short = "DeiT-S" if key == "deit_s" else "DINOv3-S"
                    draw_overlay(ax, img, arr, box,
                                 f"{short} {'Grad-CAM' if kind == 'cam' else 'CLS attn.'}" if r == 0 else None)
                c += 1
    _headers(fig, f"ViT towers: Grad-CAM vs CLS attention ({TASK_TITLE[task]})",
             "Per patient, the WLE frame and the EUS frame with the highest attention-pool weight. CLS attention = "
             "last-layer attention from the CLS token to each patch, mean over heads (class-agnostic);\nGrad-CAM "
             "is class-specific. Dashed box = GT lesion box.")
    return _save(fig, f"vit_cls_attention_{task}.png")


def _style_axis(ax, ylim=(0, 1.1)) -> None:
    exp41._style_axis(ax)
    ax.set_ylim(*ylim)


def plot_box_energy() -> Path | None:
    """Per backbone: share of Grad-CAM mass inside the GT box and how often
    the CAM peak is inside it, against a uniform map's expectation."""
    import matplotlib.pyplot as plt

    stats = {}
    for task in ("binary", "subtype"):
        for key, *_ in BACKBONES:
            f = _run_file(key, task, "frames.csv")
            if f is None:
                continue
            for m in MODALITY_NAMES:
                rows = [r for r in _read_csv(f) if r["modality"] == m and _f(r["box_share"]) == _f(r["box_share"])]
                stats[(task, key, m)] = {
                    "share": np.nanmean([_f(r["box_share"]) for r in rows]),
                    "area": np.mean([_f(r["box_area"]) for r in rows]),
                    "peak": np.mean([int(r["peak_in_box"]) for r in rows]), "n": len(rows)}
    if not stats:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.0), facecolor=SURFACE)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.79, bottom=0.1, wspace=0.18, hspace=0.55)
    n = len(BACKBONES)
    for r, task in enumerate(("binary", "subtype")):
        for c, (metric, m) in enumerate([("share", "WLE"), ("share", "EUS"), ("peak", "WLE"), ("peak", "EUS")]):
            ax = axes[r, c]
            _style_axis(ax)
            ax.set_xlim(-0.6, n - 0.4)
            title = "CAM mass inside the box" if metric == "share" else "CAM peak inside the box"
            ax.set_title(f"{TASK_TITLE[task]} · {m}\n{title}", loc="left", fontsize=9.5, color=INK, pad=8,
                         fontweight="bold")
            areas = []
            for i, (key, _, short, color) in enumerate(BACKBONES):
                s = stats.get((task, key, m))
                if s is None:
                    continue
                exp41._bar(ax, i, 0.62, s[metric], color)
                ax.text(i, s[metric] + 0.015, f"{s[metric]:.2f}", ha="center", va="bottom", fontsize=6.8,
                        color=INK2, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.6})
                areas.append(s["area"])
            ax.axhline(np.mean(areas), color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1.5)
            ax.set_xticks(range(n))
            ax.set_xticklabels([b[2] for b in BACKBONES], rotation=38, ha="right", rotation_mode="anchor")
    fig.text(0.045, 0.965, "Does the saliency sit on the lesion? Grad-CAM vs. the GT lesion box",
             fontsize=14, color=INK, fontweight="bold", ha="left")
    fig.text(0.045, 0.92, "Seed-42 fold models on their own test patients, every distinct frame (predicted-class "
             "Grad-CAM). Dashed: what a uniform map would score = the box's share of the crop (~0.7, because the "
             "lesion crop\nkeeps a 10% margin around the box). Above the line = saliency concentrated on the lesion "
             "rather than the margin. Same colour = same backbone in every figure.", fontsize=8.8, color=INK2,
             ha="left", va="top", linespacing=1.45)
    return _save(fig, "box_energy.png")


def plot_gate_and_eus(task: str) -> Path | None:
    """Left: mean GatedFusion gate per backbone on patients with EUS.
    Right: how adding EUS moves the probability of the true class, by class."""
    import matplotlib.pyplot as plt

    classes = patient_classes(task)
    gates, deltas = {}, {}
    for key, *_ in BACKBONES:
        f = _run_file(key, task, "patients.csv")
        if f is not None:
            gates[key] = np.array([_f(r["gate_mean"]) for r in _read_csv(f) if r["paired"] == "1"])
        hits = sorted(OOF_DIR.glob(f"joint_{key}__{task}__*fusion-gated*lesioncrop__nested0.2__s{SEED}.npz"))
        if hits:
            d = np.load(hits[0])
            sel = d["paired"]
            y = d["y"][sel]
            dp = d["probs_fused"][sel][np.arange(sel.sum()), y] - d["probs_wle_only"][sel][np.arange(sel.sum()), y]
            deltas[key] = {c: dp[y == c] for c in range(len(classes))}
    if not deltas:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), facecolor=SURFACE,
                                 gridspec_kw={"width_ratios": [1, 1.6]})
    fig.subplots_adjust(left=0.05, right=0.99, top=0.77, bottom=0.25, wspace=0.16)
    n = len(BACKBONES)
    _style_axis(a1, (0, 1.0))
    a1.set_xlim(-0.6, n - 0.4)
    a1.set_title("Mean fusion gate (patients with EUS)", loc="left", fontsize=10, color=INK, fontweight="bold",
                 pad=8)
    for i, (key, _, short, color) in enumerate(BACKBONES):
        if key not in gates or not len(gates[key]):
            continue
        g = gates[key]
        exp41._bar(a1, i, 0.62, float(np.mean(g)), color)
        lo, hi = np.percentile(g, [25, 75])
        a1.plot([i, i], [lo, hi], color=INK2, linewidth=0.9, zorder=3)
        a1.text(i, hi + 0.015, f"{np.mean(g):.2f}", ha="center", va="bottom", fontsize=6.8, color=INK2)
    a1.set_xticks(range(n))
    a1.set_xticklabels([b[2] for b in BACKBONES], rotation=38, ha="right", rotation_mode="anchor")

    a2.set_facecolor(SURFACE)
    for side in ("top", "right"):
        a2.spines[side].set_visible(False)
    a2.spines["left"].set_color(AXIS)
    a2.spines["bottom"].set_color(AXIS)
    a2.axhline(0, color=INK2, linewidth=0.9, zorder=1)
    a2.yaxis.grid(True, color=GRID, linewidth=0.8)
    a2.set_axisbelow(True)
    step = 0.1
    for i, (key, name, _, color) in enumerate(BACKBONES):
        if key not in deltas:
            continue
        xs = np.arange(len(classes)) + (i - (n - 1) / 2) * step
        ys = [float(np.mean(deltas[key][c])) if len(deltas[key][c]) else np.nan for c in range(len(classes))]
        a2.scatter(xs, ys, s=34, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=name)
    counts = [len(next(iter(deltas.values()))[c]) for c in range(len(classes))]
    a2.set_xticks(range(len(classes)))
    a2.set_xticklabels([f"{CLASS_SHORT[c]}\n(n={k})" for c, k in zip(classes, counts)], fontsize=8.5, color=INK2)
    a2.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    a2.tick_params(axis="x", length=0)
    a2.set_title("Change in the true class's probability when EUS is added (fused − WLE-only)", loc="left",
                 fontsize=10, color=INK, fontweight="bold", pad=8)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=4, frameon=False, fontsize=8,
              labelcolor=INK2, handletextpad=0.3, columnspacing=1.2)
    fig.text(0.05, 0.955, f"How much does EUS matter? {TASK_TITLE[task]}", fontsize=14, color=INK,
             fontweight="bold", ha="left")
    fig.text(0.05, 0.9, "Seed-42 outer-fold test patients with both modalities (n=421). Gate: fused vector = g·"
             "proj([WLE, EUS]) + (1−g)·WLE, so g near 0 = WLE only; bar = mean over patients and the 256 gate "
             "units, whisker = interquartile range.\nRight: mean over the class's patients; above 0 = EUS makes "
             "the model more confident in the right answer. Same colour = same backbone in every figure.",
             fontsize=8.8, color=INK2, ha="left", va="top", linespacing=1.45)
    return _save(fig, f"gate_and_eus_contribution_{task}.png")


def plot_sanity() -> Path | None:
    import matplotlib.pyplot as plt

    path = OUT_DIR / "sanity_randomisation.csv"
    if not path.exists():
        return None
    rows = _read_csv(path)
    groups = sorted({(r["backbone"], r["task"]) for r in rows},
                    key=lambda bt: ([b[0] for b in BACKBONES].index(bt[0]), bt[1]))
    examples = [OUT_DIR / f"sanity_example_{b}_{t}.npz" for b, t in groups]
    examples = [e for e in examples if e.exists()][:2]
    fig = plt.figure(figsize=(14, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2 + 3, left=0.05, right=0.99, top=0.78, bottom=0.12, wspace=0.12, hspace=0.3,
                          width_ratios=[1.6, 1.6, 0.9, 0.9, 0.9])
    ax = fig.add_subplot(gs[:, :2])
    _style_axis(ax, (-0.6, 1.0))
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticklabels(["−0.5", "0", "0.5", "1.0"])
    rng = np.random.default_rng(0)
    for i, (b, t) in enumerate(groups):
        vals = np.array([_f(r["spearman"]) for r in rows if r["backbone"] == b and r["task"] == t])
        vals = vals[np.isfinite(vals)]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=7, color=COLOR[b], alpha=0.55,
                   edgecolor="none", zorder=2)
        med = float(np.median(vals))
        ax.plot([i - 0.26, i + 0.26], [med, med], color=INK, linewidth=1.6, zorder=3)
        ax.text(i + 0.3, med, f"median {med:+.2f}", fontsize=7.5, color=INK2, va="center")
    ax.axhline(0, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1.5)
    ax.set_xlim(-0.5, len(groups) - 0.2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{NAME[b]}\n{t}" for b, t in groups], rotation=0, ha="center", fontsize=8)
    ax.set_title("Spearman ρ, trained vs. tower-randomised Grad-CAM (one dot per frame)", loc="left",
                 fontsize=10, color=INK, fontweight="bold", pad=8)
    for r, e in enumerate(examples):
        d = np.load(e)
        b = e.stem.replace("sanity_example_", "").rsplit("_", 1)[0]
        for c, (arr, lab) in enumerate([(None, "input"), (d["trained"], "trained"), (d["randomised"],
                                                                                    "randomised towers")]):
            draw_overlay(fig.add_subplot(gs[r, 2 + c]), d["image"], arr, None,
                         f"{NAME[b]} · {lab}" if c == 0 else lab)
    fig.text(0.05, 0.955, "Sanity check: do the maps depend on what the model learned?", fontsize=14, color=INK,
             fontweight="bold", ha="left")
    fig.text(0.05, 0.9, "Model-parameter randomisation test (Adebayo et al., 2018): both towers re-initialised, "
             "fusion head kept, same target class. Maps that survive randomisation would show image structure, "
             "not the model's\nreasoning; correlations near 0 mean the Grad-CAMs depend on the trained weights. "
             "Fold-0 test patients of the seed-42 runs.", fontsize=8.8, color=INK2, ha="left", va="top",
             linespacing=1.45)
    return _save(fig, "sanity_randomisation.png")


def plot_counterfactual() -> Path | None:
    """36's counterfactual gate re-plotted: balanced accuracy of each
    lesion-crop checkpoint when the input is swapped for a variant."""
    import matplotlib.pyplot as plt

    results = {}
    for f in sorted((PROJECT_ROOT / "experiments" / "results").glob(
            "36_joint_fusion_counterfactual__joint_*__lesioncrop__nested0.2__s42.json")):
        d = json.loads(f.read_text())
        key = d["run_name"].split("__")[0].removeprefix("joint_")
        results[(d["task"], key)] = d
    if not results:
        return None
    fig, axes = plt.subplots(2, len(CF_VARIANTS), figsize=(18, 9.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.76, bottom=0.1, wspace=0.16, hspace=0.6)
    n = len(BACKBONES)
    for r, task in enumerate(("binary", "subtype")):
        chance = 1 / len(patient_classes(task))
        for c, (variant, title) in enumerate(CF_VARIANTS):
            ax = axes[r, c]
            _style_axis(ax, (0, 1.0))
            ax.set_xlim(-0.6, n - 0.4)
            ax.set_title(f"{TASK_TITLE[task]}\n{title}", loc="left", fontsize=9.3, color=INK, fontweight="bold",
                         pad=8)
            for i, (key, _, short, color) in enumerate(BACKBONES):
                d = results.get((task, key))
                if d is None:
                    continue
                v = d[variant]
                exp41._bar(ax, i, 0.62, v, color)
                note = f"{v:.2f}" if variant == "lesion" else f"{v:.2f}\n{d.get('ret_' + variant, 0):+.0%}"
                ax.text(i, v + 0.015, note, ha="center", va="bottom", fontsize=6.3, color=INK2, linespacing=1.1)
            ax.axhline(chance, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1.5)
            ax.set_xticks(range(n))
            ax.set_xticklabels([b[2] for b in BACKBONES], rotation=38, ha="right", rotation_mode="anchor")
    fig.text(0.04, 0.965, "Causal check: what the models need to see (counterfactual inputs, exp. 36)",
             fontsize=14, color=INK, fontweight="bold", ha="left")
    fig.text(0.04, 0.92, "Balanced accuracy of each seed-42 lesion-crop checkpoint on its own test patients when "
             "the input is replaced; second number = retention, (variant − chance) / (lesion crop − chance). "
             "Dashed: chance.\nRemoving the lesion (mean fill, background patch) or keeping only the box outline "
             "drops every model to about chance: the predictions come from lesion pixels, not context or box "
             "geometry.\nFull frames also fail: the models are specialised to the crop, so a deployable pipeline "
             "needs a lesion box (see the full-frame cluster runs). Empty slot = backbone not gated in exp. 36. "
             "Same colour = same backbone.",
             fontsize=8.6, color=INK2, ha="left", va="top", linespacing=1.45)
    return _save(fig, "counterfactual_retention.png")


def knn_decodability() -> Path | None:
    """How well the fused embedding separates the classes and the centres:
    leave-one-out 5-NN balanced accuracy, within each fold model's own test
    patients (embedding spaces differ between fold models)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    rows = []
    for task in ("binary", "subtype"):
        for key, *_ in BACKBONES:
            fe, fp = _run_file(key, task, "embed.npz"), _run_file(key, task, "patients.csv")
            if fe is None or fp is None:
                continue
            e = np.load(fe)
            pts = {r["patient_key"]: r for r in _read_csv(fp)}
            keys = e["patient_key"].tolist()
            X = e["embed"]
            fold = np.array([int(pts[k]["fold"]) for k in keys])
            y = np.array([int(pts[k]["y"]) for k in keys])
            site = np.array([pts[k]["centre"] != "Center-1" for k in keys]).astype(int)
            scores = {"class": [], "centre": []}
            for k in sorted(set(fold)):
                sel = fold == k
                Xk = X[sel] / (np.linalg.norm(X[sel], axis=1, keepdims=True) + 1e-8)
                for name, target in (("class", y[sel]), ("centre", site[sel])):
                    knn = KNeighborsClassifier(n_neighbors=6).fit(Xk, target)
                    # leave-one-out: drop each point's own (first) neighbour
                    idx = knn.kneighbors(Xk, return_distance=False)[:, 1:]
                    pred = np.array([np.bincount(target[i], minlength=target.max() + 1).argmax() for i in idx])
                    scores[name].append(balanced_accuracy_score(target, pred))
            rows.append({"task": task, "backbone": key, "class_bal": float(np.mean(scores["class"])),
                         "centre_bal": float(np.mean(scores["centre"]))})
    if not rows:
        return None
    with open(OUT_DIR / "knn_decodability.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.9), facecolor=SURFACE)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.72, bottom=0.2, wspace=0.18)
    n = len(BACKBONES)
    panels = [("binary", "class_bal", "class (NET vs non-NET)"), ("subtype", "class_bal", "class (5 subtypes)"),
              ("binary", "centre_bal", "centre (Center-1 vs 2–6)"), ("subtype", "centre_bal",
                                                                       "centre (Center-1 vs 2–6)")]
    for ax, (task, metric, what) in zip(axes, panels):
        _style_axis(ax, (0, 1.0))
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_title(f"{TASK_TITLE[task]} model\n5-NN decodes {what}", loc="left", fontsize=9.3, color=INK,
                     fontweight="bold", pad=8)
        for i, (key, _, short, color) in enumerate(BACKBONES):
            r = next((r for r in rows if r["task"] == task and r["backbone"] == key), None)
            if r is None:
                continue
            exp41._bar(ax, i, 0.62, r[metric], color)
            ax.text(i, r[metric] + 0.015, f"{r[metric]:.2f}", ha="center", va="bottom", fontsize=6.8, color=INK2)
        chance = 0.5 if metric == "centre_bal" or task == "binary" else 0.2
        ax.axhline(chance, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels([b[2] for b in BACKBONES], rotation=38, ha="right", rotation_mode="anchor")
    fig.text(0.04, 0.955, "What the fused embedding encodes: diagnosis vs. hospital", fontsize=14, color=INK,
             fontweight="bold", ha="left")
    fig.text(0.04, 0.9, "Leave-one-out 5-nearest-neighbour balanced accuracy in the 256-d fused embedding "
             "(classifier input), computed within each seed-42 fold model's own test patients and averaged over "
             "the 5 folds.\nHigh centre decodability with low class decodability would flag a site shortcut. "
             "Dashed: chance. Same colour = same backbone in every figure.", fontsize=8.8, color=INK2, ha="left",
             va="top", linespacing=1.45)
    return _save(fig, "embedding_knn_decodability.png")


def summary_table() -> None:
    """One CSV row per run: box energy, gate, peak-in-box, pool-weight
    entropy -- the numbers quoted in the write-up."""
    rows = []
    for task in ("binary", "subtype"):
        for key, *_ in BACKBONES:
            ff, fp = _run_file(key, task, "frames.csv"), _run_file(key, task, "patients.csv")
            if ff is None or fp is None:
                continue
            fr, pt = _read_csv(ff), _read_csv(fp)
            row = {"task": task, "backbone": key}
            for m in MODALITY_NAMES:
                sub = [r for r in fr if r["modality"] == m and _f(r["box_share"]) == _f(r["box_share"])]
                row[f"{m}_box_share"] = float(np.mean([_f(r["box_share"]) for r in sub]))
                row[f"{m}_box_area"] = float(np.mean([_f(r["box_area"]) for r in sub]))
                row[f"{m}_peak_in_box"] = float(np.mean([int(r["peak_in_box"]) for r in sub]))
                by_patient = defaultdict(list)
                for r in fr:
                    if r["modality"] == m:
                        by_patient[r["patient_key"]].append(_f(r["pool_weight"]))
                ent = [-(np.array(w) * np.log(np.clip(w, 1e-9, 1))).sum() / np.log(4) for w in by_patient.values()]
                row[f"{m}_pool_entropy"] = float(np.mean(ent))
            row["gate_mean_paired"] = float(np.nanmean([_f(r["gate_mean"]) for r in pt if r["paired"] == "1"]))
            rows.append(row)
    if rows:
        with open(OUT_DIR / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print(f"wrote {OUT_DIR / 'summary.csv'}")


def plot_all() -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"

    written = []
    for task in ("binary", "subtype"):
        gallery, inp, outs = load_gallery(task)
        for key, out in outs.items():
            written.append(plot_gallery(task, key, gallery, inp, out))
        if len(outs) > 1:
            for g in gallery:
                written.append(plot_cross_backbone(task, g, inp, outs))
        written.append(plot_vit_attention(task, gallery, inp, outs))
        written.append(plot_gate_and_eus(task))
    written += [plot_box_energy(), plot_sanity(), plot_counterfactual(), knn_decodability()]
    summary_table()
    for p in written:
        if p is not None:
            print(f"wrote {p.relative_to(PROJECT_ROOT)}")
