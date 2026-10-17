"""Experiment 43 -- explainability for the two-tower backbone ablation.

For every backbone's seed-42 nested run (both tasks), each patient is
explained by the fold model whose outer test fold contains it, so every map
is on a patient that model never trained or selected on. Within a seed the
split is the same for every backbone, so a patient's maps compare across
towers.

What is computed:
  * Grad-CAM for every real frame of both towers, from the fused logit of
    the predicted class (and of the true class for misclassified gallery
    patients). One method for all 7 towers: the last conv stage for the CNNs
    and MobileViT; for the ViTs the token sequence *entering* the last block
    (its output is only read at CLS, so patch-token gradients there are 0).
  * Last-layer CLS->patch attention (mean over heads) for DeiT-S and DINOv3,
    with the HF towers switched to eager attention (SDPA returns no weights).
  * Model-native signals: each frame's attention-pool weight, the mean
    GatedFusion gate (``g * proj([wle, eus]) + (1 - g) * wle``: near 0 =
    falls back to WLE), and the fused 256-d embedding.
  * Box-energy: the share of each frame's Grad-CAM mass inside the GT lesion
    box (the crop keeps a 10% context margin around it), against the box's
    share of the crop area, and whether the CAM peak lands in the box.
  * Model-randomisation sanity check (Adebayo et al. 2018): the same CAMs
    after re-initialising both towers; faithful maps should decorrelate.

Before anything is written, every patient's recomputed fused probabilities
are checked against the OOF predictions 41 cached, proving the right
checkpoint, fold and frames are used. Saliency is supporting evidence only
(experiments/README.md); the causal evidence is the counterfactual gate (36),
re-plotted here.

    python -m experiments.src.43_two_tower_explain                 # compute (GPU) + plot
    python -m experiments.src.43_two_tower_explain --plot-only
    python -m experiments.src.43_two_tower_explain --only dinov3 --task subtype
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import MODALITY_NAMES, PROJECT_ROOT
from cadsel.data import build_joint_manifest, joint_strata_key, patient_level_folds
from cadsel.fusion import patient_classes, patient_label
from cadsel.joint_fusion_data import JointPatientDataset, _boxes_for, joint_collate
from cadsel.joint_fusion_model import JointTwoTowerModel, tower_normalizers

exp41 = importlib.import_module("experiments.src.41_two_tower_ablation_bargraphs")
BACKBONES, TASK_TITLE = exp41.BACKBONES, exp41.TASK_TITLE
SURFACE, INK, INK2, MUTED, GRID, AXIS = exp41.SURFACE, exp41.INK, exp41.INK2, exp41.MUTED, exp41.GRID, exp41.AXIS
_normalize_frames = exp41._normalize_frames

RESULTS_DIR = exp41.RESULTS_DIR
OOF_DIR = exp41.OUT_DIR / "oof"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "43_two_tower_explain"
PLOT_DIR = PROJECT_ROOT / "plots" / "two_tower_explain"
SEED = 42
GRID_VIT = 14                       # 224 / 16
PAD_FRAC = 0.1                      # _lesion_crop's context margin
VITS = ("deit_s", "dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16")
SANITY_BACKBONES = ("efficientnet_b0", "dinov3_vits16")
SANITY_PATIENTS = 48
SEL_REFERENCE = "dinov3_vits16"
VERIFY: dict = {}     # gallery patients are picked from this run's OOF predictions
CLASS_SHORT = {"NET_G1": "NET G1", "NET_G2": "NET G2", "Leiomyoma": "Leiomyoma", "Lipoma": "Lipoma",
               "NonNeoplasm": "Non-neoplasm", "NET": "NET", "non-NET": "non-NET"}
SEQ_STEPS = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ---------------------------------------------------------------------------
# Runs, folds, frames
# ---------------------------------------------------------------------------
def seed42_runs(only: str | None, task: str | None) -> list[dict]:
    runs = [r for r in exp41.load_runs()
            if r["seed"] == SEED and r.get("lesion_crop") and r.get("fusion", "gated") == "gated"
            and not r.get("paired_only")]
    if only:
        runs = [r for r in runs if re.search(only, r["run_name"])]
    if task:
        runs = [r for r in runs if r["task"] == task]
    order = [b[0] for b in BACKBONES]
    return sorted(runs, key=lambda r: (r["task"], order.index(r["backbone"])))


def oof_path(backbone: str, task: str) -> Path:
    hits = sorted(OOF_DIR.glob(f"joint_{backbone}__{task}__*fusion-gated*__lesioncrop__nested0.2__s{SEED}.npz"))
    assert len(hits) == 1, (backbone, task, hits)
    return hits[0]


def crop_box(sample, img_size: int) -> tuple[float, float, float, float] | None:
    """The GT box ``_lesion_crop`` cropped around, in the resized crop's
    pixel coordinates (same box choice and padding math)."""
    from PIL import Image

    with Image.open(sample.image) as im:
        W, H = im.size
    boxes = _boxes_for(sample, W, H)
    if not boxes:
        return None
    bx1, by1, bx2, by2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    px, py = int(PAD_FRAC * (bx2 - bx1)), int(PAD_FRAC * (by2 - by1))
    cx1, cy1, cx2, cy2 = max(0, bx1 - px), max(0, by1 - py), min(W, bx2 + px), min(H, by2 + py)
    sx, sy = img_size / (cx2 - cx1), img_size / (cy2 - cy1)
    return ((bx1 - cx1) * sx, (by1 - cy1) * sy, (bx2 - cx1) * sx, (by2 - cy1) * sy)


def frame_meta(ds: JointPatientDataset, index: int) -> dict:
    """modality -> per-slot (image path, crop-space box), reproducing
    ``JointPatientDataset._sample_modality``'s rng draws exactly."""
    rng = random.Random((ds.seed * 1_000_003) ^ index)
    out = {}
    for m in MODALITY_NAMES:
        pool = ds.frames[ds.patient_keys[index]].get(m, [])
        chosen = [pool[rng.randrange(len(pool))] for _ in range(ds.max_frames)] if pool else []
        out[m] = [(str(s.image), crop_box(s, ds.img_size)) for s in chosen]
    return out


def build_folds(run: dict, num_workers: int) -> list[dict]:
    """Each outer fold's test patients: uint8 batches + per-frame metadata.
    Depends only on the seed / frame-selection settings, so it is shared by
    every backbone and task."""
    samples = build_joint_manifest()
    centre = {s.patient_key: s.center for s in samples}
    folds = patient_level_folds(samples, n_folds=run["n_splits"], seed=SEED, strata_key=joint_strata_key)
    out = []
    for k, (_, test) in enumerate(folds):
        ds = JointPatientDataset(test, "subtype", img_size=run["img_size"], max_frames=run["max_frames"],
                                 train=False, seed=SEED, lesion_crop=True)
        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=joint_collate, num_workers=num_workers)
        out.append({"keys": list(ds.patient_keys), "subtypes": [ds.subtype[p] for p in ds.patient_keys],
                    "centre": [centre[p] for p in ds.patient_keys],
                    "meta": [frame_meta(ds, i) for i in range(len(ds))],
                    "batches": [(im, ma) for im, ma, _ in loader]})
        print(f"    cached fold {k}: {len(ds)} test patients", flush=True)
    return out


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
def cam_layer(tower, backbone: str):
    if backbone in ("efficientnet_b0", "mobilenet_v3_large", "convnext_tiny"):
        return tower.features[-1], "conv"
    if backbone == "resnet50":
        return tower.layer4, "conv"
    if backbone == "mobilevit_s":
        return tower.model.conv_1x1_exp, "conv"
    if backbone == "deit_s":
        return tower.model.layers[-1], "vit"
    if backbone.startswith("dinov3_"):
        return tower.model.model.layer[-1], "vit"
    raise ValueError(backbone)


class Probe:
    """Forward hooks that keep, per forward pass: each tower's CAM-layer
    activations (with graph), ViT last-layer attention, attention-pool
    weights, the mean gate and the fused embedding."""

    def __init__(self, model: JointTwoTowerModel, backbone: str) -> None:
        self.kind = None
        self.handles = []
        self.clear()
        for m in MODALITY_NAMES:
            layer, self.kind = cam_layer(model.backbones[m], backbone)
            if self.kind == "conv":
                self.handles.append(layer.register_forward_hook(
                    lambda mod, args, out, m=m: self.acts.__setitem__(m, out)))
            else:
                self.handles.append(layer.register_forward_pre_hook(
                    lambda mod, args, m=m: self.acts.__setitem__(m, args[0])))
                self.handles.append(layer.attention.register_forward_hook(
                    lambda mod, args, out, m=m: self.attn.__setitem__(m, out[1].detach())))
            self.handles.append(model.head.pools[m].register_forward_hook(self._pool_hook(m)))
        self.handles.append(model.head.fuse.register_forward_hook(self._gate_hook))
        self.handles.append(model.head.classifier.register_forward_pre_hook(
            lambda mod, args: self.__setattr__("embed", args[0].detach())))

    def clear(self) -> None:
        self.acts, self.attn, self.pool_w = {}, {}, {}
        self.gate, self.embed = None, None

    def _pool_hook(self, m: str):
        def hook(mod, args, out):
            frames, mask = args
            with torch.no_grad():
                logits = mod.score(frames).squeeze(-1).masked_fill(mask < 0.5, float("-inf"))
                empty = mask.sum(dim=1, keepdim=True) < 0.5
                logits = torch.where(empty.expand_as(logits), torch.zeros_like(logits), logits)
                self.pool_w[m] = torch.softmax(logits, dim=1) * (~empty).float()
        return hook

    def _gate_hook(self, mod, args, out):
        a, b, avail = args
        with torch.no_grad():
            self.gate = mod.gate(torch.cat([a, b, avail], dim=1)).mean(dim=1)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


def grad_cam(act: torch.Tensor, grad: torch.Tensor, kind: str) -> torch.Tensor:
    """(N, h, w) non-negative Grad-CAM from one layer's activations and the
    gradients of the target logit."""
    if kind == "conv":
        return (grad.mean(dim=(2, 3), keepdim=True) * act).sum(1).clamp(min=0)
    start = act.shape[1] - GRID_VIT * GRID_VIT          # CLS (+ registers) first
    a, g = act[:, start:], grad[:, start:]
    cam = (g.mean(dim=1, keepdim=True) * a).sum(-1).clamp(min=0)
    return cam.reshape(-1, GRID_VIT, GRID_VIT)


def cls_attention(attn: torch.Tensor) -> torch.Tensor:
    """(N, heads, L, L) -> (N, 14, 14) CLS->patch attention, mean over heads."""
    start = attn.shape[-1] - GRID_VIT * GRID_VIT
    return attn[:, :, 0, start:].mean(1).reshape(-1, GRID_VIT, GRID_VIT)


def slots_of(mask: torch.Tensor) -> list[tuple[int, int]]:
    """Row order of a tower's (N_real, ...) outputs -> (patient, slot)."""
    B, T = mask.shape
    idx = torch.nonzero(mask.reshape(B * T) > 0.5).flatten().tolist()
    return [(i // T, i % T) for i in idx]


def load_model(run: dict, k: int, device: torch.device) -> JointTwoTowerModel:
    ck = torch.load(RESULTS_DIR / run["run_name"] / f"fold{k}_best.pt", map_location="cpu", weights_only=False)
    model = JointTwoTowerModel(len(patient_classes(run["task"])), backbone=ck.get("backbone", "efficientnet_b0"),
                               freeze=ck["freeze"], fusion=ck["fusion"], modality_dropout=ck["modality_dropout"],
                               unfreeze_last_n=ck.get("unfreeze_last_n", 4))
    model.load_state_dict(ck["model_state_dict"])
    if run["backbone"] in VITS:
        for m in MODALITY_NAMES:
            model.backbones[m].model.set_attn_implementation("eager")
    return model.to(device).eval()


def forward_cams(model, probe: Probe, images: dict, masks: dict, targets: list[torch.Tensor | None]):
    """One forward pass; Grad-CAMs for each target vector (None = predicted
    class). Returns logits and ``cams[i][m]`` = (N_real, h, w) per target."""
    probe.clear()
    with torch.enable_grad():
        logits = model(images, masks)
        pred = logits.argmax(1)
        mods = [m for m in MODALITY_NAMES if m in probe.acts]
        cams = []
        for i, tgt in enumerate(targets):
            tgt = pred if tgt is None else tgt
            score = logits.gather(1, tgt[:, None]).sum()
            grads = torch.autograd.grad(score, [probe.acts[m] for m in mods], retain_graph=i < len(targets) - 1)
            cams.append({m: grad_cam(probe.acts[m].detach(), g.detach(), probe.kind) for m, g in zip(mods, grads)})
    return logits.detach(), cams


def upsample(cam: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(cam[:, None].float(), size=(size, size), mode="bilinear", align_corners=False)[:, 0]


def box_stats(cam224: np.ndarray, box) -> tuple[float, float, int]:
    """(share of CAM mass in the box, box share of the crop, CAM peak in box)."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    inside = np.zeros_like(cam224, dtype=bool)
    inside[max(y1, 0):min(y2, cam224.shape[0]), max(x1, 0):min(x2, cam224.shape[1])] = True
    total = float(cam224.sum())
    peak = np.unravel_index(int(cam224.argmax()), cam224.shape)
    share = float(cam224[inside].sum() / total) if total > 0 else float("nan")
    return share, float(inside.mean()), int(inside[peak])


# ---------------------------------------------------------------------------
# Gallery patients
# ---------------------------------------------------------------------------
def pick_gallery(task: str) -> list[dict]:
    """Per true class: the most confident correct and the most confident
    wrong patient (wrong = that class's most common mistake), preferring
    patients with EUS frames, from the reference backbone's OOF predictions."""
    d = np.load(oof_path(SEL_REFERENCE, task))
    probs, y, paired, keys = d["probs_fused"], d["y"], d["paired"], d["patient_key"]
    pred = probs.argmax(1)
    classes = patient_classes(task)
    picks = []
    for c in range(len(classes)):
        for kind in ("correct", "wrong"):
            cand = (y == c) & ((pred == c) if kind == "correct" else (pred != c))
            if not cand.any():
                continue
            if kind == "wrong":
                common = np.bincount(pred[cand], minlength=len(classes)).argmax()
                cand &= pred == common
            pool = cand & paired if (cand & paired).any() else cand
            conf = probs[np.arange(len(y)), pred]
            i = int(np.flatnonzero(pool)[np.argmax(conf[pool])])
            picks.append({"patient_key": str(keys[i]), "y": int(y[i]), "kind": kind})
    return picks


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
def explain_run(run: dict, folds: list[dict], gallery: list[dict], device: torch.device) -> None:
    task, backbone = run["task"], run["backbone"]
    classes = patient_classes(task)
    normalizers = tower_normalizers(backbone)
    gallery_keys = {g["patient_key"]: g for g in gallery}
    ref = np.load(oof_path(backbone, task))
    ref_probs = dict(zip(ref["patient_key"].tolist(), ref["probs_fused"]))

    frame_rows, patient_rows, embeds, embed_keys = [], [], [], []
    gal = {}
    worst, flips = 0.0, 0
    for k, fold in enumerate(folds):
        model = load_model(run, k, device)
        probe = Probe(model, backbone)
        start = 0
        for images, masks in fold["batches"]:
            B = masks["WLE"].shape[0]
            keys = fold["keys"][start:start + B]
            metas = fold["meta"][start:start + B]
            subtypes = fold["subtypes"][start:start + B]
            centres = fold["centre"][start:start + B]
            start += B
            y = torch.tensor([classes.index(patient_label(st, task)) for st in subtypes], device=device)
            x = _normalize_frames(images, normalizers, device)
            md = {m: masks[m].to(device) for m in MODALITY_NAMES}
            need_true = any(p in gallery_keys for p in keys)
            logits, cams = forward_cams(model, probe, x, md, [None, y] if need_true else [None])
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            pred = probs.argmax(1)
            for b, p in enumerate(keys):
                worst = max(worst, float(np.abs(probs[b] - ref_probs[p]).max()))
                flips += int(probs[b].argmax() != ref_probs[p].argmax())
            paired = ((masks["WLE"].sum(1) > 0.5) & (masks["EUS"].sum(1) > 0.5)).numpy()
            gate = probe.gate.cpu().numpy()
            embeds.append(probe.embed.cpu().numpy())
            embed_keys += keys
            for b, p in enumerate(keys):
                patient_rows.append({"patient_key": p, "fold": k, "centre": centres[b], "y": int(y[b]),
                                     "pred": int(pred[b]), "p_pred": float(probs[b, pred[b]]),
                                     "p_true": float(probs[b, int(y[b])]), "paired": int(paired[b]),
                                     "gate_mean": float(gate[b]) if paired[b] else float("nan")})

            for m in cams[0]:
                rows = slots_of(md[m].cpu())
                cam224 = upsample(cams[0][m], 224).cpu().numpy()
                weights = probe.pool_w[m].cpu().numpy()
                seen = {}
                for r, (b, t) in enumerate(rows):
                    path, box = metas[b][m][t]
                    key = (keys[b], path)
                    if key in seen:                 # a frame drawn twice: same CAM, add its weight
                        frame_rows[seen[key]]["pool_weight"] += float(weights[b, t])
                        continue
                    share, area, hit = box_stats(cam224[r], box) if box is not None else (np.nan, np.nan, -1)
                    seen[key] = len(frame_rows)
                    frame_rows.append({"patient_key": keys[b], "fold": k, "modality": m, "slot": t,
                                       "image": _rel(path),
                                       "pool_weight": float(weights[b, t]), "box_share": share,
                                       "box_area": area, "peak_in_box": hit})

            for b, p in enumerate(keys):
                if p not in gallery_keys:
                    continue
                entry = {"y": int(y[b]), "pred": int(pred[b]), "probs": probs[b]}
                for m in cams[0]:
                    rows = slots_of(md[m].cpu())
                    sel = [r for r, (bb, _) in enumerate(rows) if bb == b]
                    if not sel:
                        continue
                    entry[f"cam_pred_{m}"] = cams[0][m][sel].cpu().numpy()
                    if len(cams) > 1:
                        entry[f"cam_true_{m}"] = cams[1][m][sel].cpu().numpy()
                    entry[f"pool_{m}"] = probe.pool_w[m][b].cpu().numpy()
                    if m in probe.attn:
                        entry[f"attn_{m}"] = cls_attention(probe.attn[m][sel]).cpu().numpy()
                gal[p] = entry
        probe.remove()
        del model, probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 34 evaluated under fp16 autocast and 41 re-scored in fp32 (+ default
    # TF32); batch size changes cuDNN kernels too, so CNN probabilities drift
    # by ~1e-3 and near-ties can flip. A wrong checkpoint, fold or frame
    # selection moves them by tenths, which is what this guards against.
    print(f"    max |p - 41's OOF p| = {worst:.2e}, predicted class differs for {flips} patient(s)", flush=True)
    VERIFY[run["run_name"]] = {"max_abs_prob_diff": worst, "pred_flips": flips, "n": len(ref_probs)}
    if worst > 5e-2 or flips > 3:
        raise SystemExit(f"{run['run_name']}: recomputed predictions do not match 41's OOF predictions")

    stem = OUT_DIR / run["run_name"]
    _write_csv(stem.with_name(stem.name + "__frames.csv"), frame_rows)
    _write_csv(stem.with_name(stem.name + "__patients.csv"), patient_rows)
    np.savez_compressed(stem.with_name(stem.name + "__embed.npz"), patient_key=np.array(embed_keys),
                        embed=np.concatenate(embeds).astype(np.float32))
    flat = {f"{p}|{name}": v for p, e in gal.items() for name, v in e.items()}
    np.savez_compressed(stem.with_name(stem.name + "__gallery.npz"), **flat)


def save_gallery_inputs(task: str, folds: list[dict], gallery: list[dict]) -> None:
    """The gallery patients' crops, masks and boxes; identical for every
    backbone (deterministic lesion-crop eval), so stored once per task."""
    want = {g["patient_key"]: g for g in gallery}
    out = {}
    for fold in folds:
        start = 0
        for images, masks in fold["batches"]:
            B = masks["WLE"].shape[0]
            for b in range(B):
                p = fold["keys"][start + b]
                if p not in want:
                    continue
                for m in MODALITY_NAMES:
                    real = masks[m][b] > 0.5
                    out[f"{p}|img_{m}"] = images[m][b][real].numpy()
                    meta = fold["meta"][start + b][m]
                    out[f"{p}|box_{m}"] = np.array([bx if bx is not None else (np.nan,) * 4 for _, bx in meta],
                                                   dtype=np.float32).reshape(-1, 4)
                    out[f"{p}|path_{m}"] = np.array([pth for pth, _ in meta])
            start += B
    out["gallery"] = np.array([json.dumps(g) for g in gallery])
    np.savez_compressed(OUT_DIR / f"gallery_inputs_{task}.npz", **out)


def randomise_towers(model: JointTwoTowerModel) -> None:
    for m in MODALITY_NAMES:
        for mod in model.backbones[m].modules():
            if hasattr(mod, "reset_parameters") and mod is not model.backbones[m]:
                mod.reset_parameters()


def sanity_check(run: dict, folds: list[dict], device: torch.device) -> list[dict]:
    """Spearman correlation between trained and tower-randomised Grad-CAMs
    (same target class: the trained model's prediction) for the first
    ``SANITY_PATIENTS`` test patients of fold 0."""
    from scipy.stats import spearmanr

    normalizers = tower_normalizers(run["backbone"])
    batches, n = [], 0
    for images, masks in folds[0]["batches"]:
        if n >= SANITY_PATIENTS:
            break
        batches.append((images, masks))
        n += masks["WLE"].shape[0]
    cams, targets = {}, []
    for state in ("trained", "randomised"):
        model = load_model(run, 0, device)
        if state == "randomised":
            torch.manual_seed(0)
            randomise_towers(model)
        probe = Probe(model, run["backbone"])
        cams[state] = []
        for i, (images, masks) in enumerate(batches):
            x = _normalize_frames(images, normalizers, device)
            md = {m: masks[m].to(device) for m in MODALITY_NAMES}
            logits, c = forward_cams(model, probe, x, md, [None if state == "trained" else targets[i]])
            if state == "trained":
                targets.append(logits.argmax(1))
            cams[state].append({m: c[0][m].cpu().numpy() for m in c[0]})
        probe.remove()
        del model, probe
    rows = []
    for a, b in zip(cams["trained"], cams["randomised"]):
        for m in a:
            for i in range(len(a[m])):
                rho = spearmanr(a[m][i].ravel(), b[m][i].ravel()).statistic
                rows.append({"backbone": run["backbone"], "task": run["task"], "modality": m,
                             "spearman": float(rho) if np.isfinite(rho) else float("nan")})
    np.savez_compressed(OUT_DIR / f"sanity_example_{run['backbone']}_{run['task']}.npz",
                        trained=cams["trained"][0]["WLE"][0], randomised=cams["randomised"][0]["WLE"][0],
                        image=batches[0][0]["WLE"][0][0].numpy())
    return rows


def _rel(path: str) -> str:
    root = os.environ.get("CADSEL_DATA_ROOT")
    try:
        return str(Path(path).relative_to(root)) if root else path
    except ValueError:
        return path


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


def compute(runs: list[dict], device: torch.device, num_workers: int, sanity: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    folds = build_folds(runs[0], num_workers)
    sanity_rows = []
    for task in sorted({r["task"] for r in runs}):
        gallery = pick_gallery(task)
        save_gallery_inputs(task, folds, gallery)
        print(f"  {task}: {len(gallery)} gallery patients", flush=True)
        for run in [r for r in runs if r["task"] == task]:
            print(f"  explaining {run['run_name']}", flush=True)
            explain_run(run, folds, gallery, device)
            if sanity and run["backbone"] in SANITY_BACKBONES:
                sanity_rows += sanity_check(run, folds, device)
    if sanity_rows:
        _write_csv(OUT_DIR / "sanity_randomisation.csv", sanity_rows)
    (OUT_DIR / "verification.json").write_text(json.dumps(VERIFY, indent=1))


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plot-only", action="store_true", help="reuse cached results, no GPU")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--only", default=None, help="regex filter on run_name")
    p.add_argument("--task", choices=["binary", "subtype"], default=None)
    p.add_argument("--no-sanity", action="store_true", help="skip the randomisation sanity check")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "sans-serif"

    if not args.plot_only:
        runs = seed42_runs(args.only, args.task)
        print(f"{len(runs)} seed-{SEED} run(s)")
        if not runs:
            raise SystemExit("no runs found -- is the results volume mounted?")
        compute(runs, torch.device(args.device), args.num_workers, sanity=not args.no_sanity)
    plots = importlib.import_module("experiments.src.43_two_tower_explain_plots")
    plots.plot_all()


if __name__ == "__main__":
    main()
