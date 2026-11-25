"""Experiment 54 -- screen of vision-language fusion operators on frozen features.

Do the image-text fusion operators of ``WLE_EUS_Fusion_Research_Report.md``
(MLB, MFB, bidirectional FiLM, their combinations) beat the two-tower's gated
fusion, or late fusion, when every operator sees exactly the same features?
The report's Phase 2: a cheap screen on frozen embeddings before any operator
earns a full fine-tuned run in 34.

Features: the PRETRAINED towers, untouched (as in 53): DINOv3 ViT-S/16 for
WLE, USFM ViT-B/16 for EUS, on the lesion crops. Every frame is encoded once
under the eval transform (view 0) and ``--aug-views`` times under 34's
training augmentation (views 1..V). Cached in ``external_data/`` (git-ignored).

Heads: ``cadsel.fusion.FusionHead`` itself, the class 34 trains, so attention
pooling over frames, modality dropout 0.3, dropout 0.3 and the encoders are
the same. Training mirrors 34's nested protocol: CAD-SEL's patient-level
5-fold split per seed (42, 123, 7), a 20% select split of each outer-train
fold (34's ``patient_level_split``), AdamW 1e-4 / wd 1e-4, class-weighted CE, batch 8,
ReduceLROnPlateau on select balanced accuracy, up to 100 epochs, early stop
after 20, best select epoch scored on the outer fold. Training draws 4
frames per modality with replacement, each from a random augmented view;
evaluation reads every frame of a patient, eval view.

What differs from 34: the towers are frozen at their pretrained weights, so
absolute numbers sit below the fine-tuned models'. The screen compares
operators with each other on identical inputs, nothing more.

Conditions: unimodal heads (``wle_only``, ``eus_only``), their late fusion
(``late_avg``: the mean of the two where a patient has EUS, else WLE alone),
and two-stream heads with ``fusion`` in gated / symmetric / concat / mlb /
mfb / mfb_res, with and without frame-level bidirectional FiLM.

Steps (default ``all``):
  * ``extract``: build the feature cache.
  * ``train``:   one OOF file per condition x task x seed in ``oof/``; existing
                 files are skipped, so an interrupted run resumes.
  * ``report``:  ``scores.csv`` (every condition, seed 42 / mean of seeds /
                 3-seed ensemble, on all patients, patients with EUS and
                 rectal patients) and ``versus.csv`` (3-seed ensembles minus
                 the gated head and minus the WLE model on the patients both
                 cover, paired patient bootstrap, Holm over conditions per
                 task x population x reference).

    python -m experiments.src.54_fusion_operator_screen
    python -m experiments.src.54_fusion_operator_screen --step train --conditions mfb,film_mfb --seeds 42
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import PROJECT_ROOT
from cadsel.data import patient_level_folds, patient_level_split
from cadsel.external_datasets import get_spec
from cadsel.fusion import FusionHead
from cadsel.joint_fusion_data import _lesion_crop
from cadsel.joint_fusion_model import build_tower, model_normalizers
from finetune_classifier import compute_classification_class_weights

exp44 = importlib.import_module("experiments.src.44_two_tower_stats")
exp49 = importlib.import_module("experiments.src.49_clinical_baseline")

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "54_fusion_screen"
CACHE_DIR = PROJECT_ROOT / "external_data" / "54_fusion_screen"
TOWERS = {"WLE": "dinov3_vits16", "EUS": "usfm_vitb16"}
SEEDS, N_FOLDS, INNER = (42, 123, 7), 5, 0.2
EPOCHS, PATIENCE, LR, WD, BATCH, TRAIN_FRAMES = 100, 20, 1e-4, 1e-4, 8, 4
BOTH = ["WLE", "EUS"]
# name: (modalities, fusion, film)
CONDITIONS = {
    "wle_only": (["WLE"], "gated", False),
    "eus_only": (["EUS"], "gated", False),
    **{f: (BOTH, f, False) for f in ("gated", "symmetric", "concat", "mlb", "mfb", "mfb_res")},
    **{f"film_{f}": (BOTH, f, True) for f in ("concat", "gated", "mfb", "mfb_res")},
}
LATE = "late_avg"


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------
class FrameViews(Dataset):
    """Every frame of one modality: the eval view, then ``n_aug`` views under
    34's training augmentation (``JointPatientDataset``'s transforms)."""

    def __init__(self, samples, n_aug: int, img_size: int = 224) -> None:
        self.samples, self.n_aug = samples, n_aug
        self.eval_tf = T.Resize((img_size, img_size))
        self.train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomRotation(10),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> torch.Tensor:
        s = self.samples[i]
        with Image.open(s.image) as im0:
            im = _lesion_crop(im0.convert("RGB"), s)
        views = [self.eval_tf(im)]
        for v in range(self.n_aug):
            torch.manual_seed(i * 101 + v)
            views.append(self.train_tf(im))
        arr = np.stack([np.asarray(x, dtype=np.uint8) for x in views])   # (V, H, W, 3)
        return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


def cache_path(n_aug: int) -> Path:
    return CACHE_DIR / f"features__{TOWERS['WLE']}__{TOWERS['EUS']}__lesioncrop__v{n_aug}.npz"


@torch.no_grad()
def extract(n_aug: int, device: torch.device) -> Path:
    path = cache_path(n_aug)
    if path.exists():
        print(f"cache exists: {path}")
        return path
    spec = get_spec("cadsel")
    samples = spec.manifest()
    norms = model_normalizers(TOWERS, spec.modality_stats())
    out = {}
    for m, bb in TOWERS.items():
        frames = sorted((s for s in samples if s.modality == m), key=lambda s: s.image)
        tower = build_tower(bb, "partial").to(device).eval()
        loader = DataLoader(FrameViews(frames, n_aug), batch_size=16, num_workers=4)
        feats, t0 = [], time.time()
        for x in loader:
            B, V = x.shape[:2]
            flat = norms[m](x.reshape(B * V, *x.shape[2:]).to(device).float())
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                f = tower(flat)
            feats.append(f.float().reshape(B, V, -1).cpu().numpy().astype(np.float16))
        out[f"{m}_feats"] = np.concatenate(feats)
        out[f"{m}_patient"] = np.array([s.patient_key for s in frames])
        out[f"{m}_image"] = np.array([Path(s.image).name for s in frames])
        print(f"  {m}: {len(frames)} frames x {1 + n_aug} views -> {out[f'{m}_feats'].shape} "
              f"({time.time() - t0:.0f}s)", flush=True)
        del tower
        torch.cuda.empty_cache()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Heads on cached features
# ---------------------------------------------------------------------------
class Features:
    def __init__(self, path: Path, device: torch.device) -> None:
        d = np.load(path)
        self.feats = {m: torch.from_numpy(d[f"{m}_feats"].astype(np.float32)).to(device) for m in BOTH}
        self.dims = {m: self.feats[m].shape[-1] for m in BOTH}
        self.n_views = self.feats["WLE"].shape[1]
        self.frames: dict[str, dict[str, np.ndarray]] = {m: {} for m in BOTH}
        for m in BOTH:
            for i, k in enumerate(d[f"{m}_patient"]):
                self.frames[m].setdefault(str(k), []).append(i)
        self.frames = {m: {k: np.array(v) for k, v in fm.items()} for m, fm in self.frames.items()}
        self.device = device

    def batch(self, keys, mods, train: bool, rng: np.random.Generator) -> dict:
        out = {}
        for m in mods:
            pools = [self.frames[m].get(k, np.zeros(0, int)) for k in keys]
            T_ = TRAIN_FRAMES if train else max(1, max(len(p) for p in pools))
            idx = np.zeros((len(keys), T_), dtype=np.int64)
            view = np.zeros((len(keys), T_), dtype=np.int64)
            mask = np.zeros((len(keys), T_), dtype=np.float32)
            for i, p in enumerate(pools):
                if not len(p):
                    continue
                if train:   # 34: max_frames draws with replacement, augmented
                    idx[i] = rng.choice(p, TRAIN_FRAMES, replace=True)
                    view[i] = rng.integers(1, self.n_views, TRAIN_FRAMES) if self.n_views > 1 else 0
                    mask[i] = 1.0
                else:       # every frame, eval view
                    idx[i, :len(p)] = p
                    mask[i, :len(p)] = 1.0
            idx_t, view_t = torch.from_numpy(idx).to(self.device), torch.from_numpy(view).to(self.device)
            mask_t = torch.from_numpy(mask).to(self.device)
            out[m] = self.feats[m][idx_t, view_t] * mask_t.unsqueeze(-1)
            out[f"{m}_mask"] = mask_t
        return out


def bal_acc(y: np.ndarray, pred: np.ndarray, n: int) -> float:
    return float(np.mean([(pred[y == c] == c).mean() for c in range(n) if (y == c).any()]))


def train_head(feats: Features, cond: str, fit, sel, test, labels: dict, n_classes: int,
               seed: int, fold: int) -> tuple[np.ndarray, dict]:
    mods, fusion, film = CONDITIONS[cond]
    torch.manual_seed(seed * 100 + fold)
    rng = np.random.default_rng(seed * 100 + fold)
    head = FusionHead({m: feats.dims[m] for m in mods}, n_classes, fusion=fusion, film=film,
                      modalities=mods).to(feats.device)
    ytr = np.array([labels[k] for k in fit])
    weight = torch.tensor(compute_classification_class_weights(ytr, n_classes),
                          dtype=torch.float32, device=feats.device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5)
    ysel = np.array([labels[k] for k in sel])
    sel_batch = feats.batch(sel, mods, False, rng)
    best = {"bal": -1.0, "epoch": 0}
    best_state = None
    fit = np.array(fit)
    for epoch in range(EPOCHS):
        head.train()
        keys = fit[rng.permutation(len(fit))]
        for start in range(0, len(fit), BATCH):
            chunk = keys[start:start + BATCH]
            y = torch.tensor([labels[k] for k in chunk], device=feats.device)
            loss = F.cross_entropy(head(feats.batch(chunk, mods, True, rng)), y, weight=weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(sel_batch).argmax(1).cpu().numpy()
        bal = bal_acc(ysel, pred, n_classes)
        sched.step(bal)
        if bal > best["bal"]:
            best = {"bal": bal, "epoch": epoch + 1}
            best_state = copy.deepcopy(head.state_dict())
        if (epoch + 1) - best["epoch"] >= PATIENCE:
            break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        probs = torch.softmax(head(feats.batch(test, mods, False, rng)).float(), 1).cpu().numpy()
    return probs, {**best, "epochs_run": epoch + 1}


def train(conditions, tasks, seeds, n_aug: int, device: torch.device) -> None:
    spec = get_spec("cadsel")
    samples = spec.manifest()
    feats = Features(cache_path(n_aug), device)
    subtype = {s.patient_key: s.subtype for s in samples}
    (OUT_DIR / "oof").mkdir(parents=True, exist_ok=True)
    for task in tasks:
        classes = spec.classes(task)
        labels = {k: classes.index(spec.label_of(st, task)) for k, st in subtype.items()}
        for seed in seeds:
            folds = patient_level_folds(samples, n_folds=N_FOLDS, seed=seed, strata_key=spec.strata_key)
            splits = []
            for k, (outer_train, test) in enumerate(folds):
                # 34's nested select split
                fit, sel = patient_level_split(outer_train, val_fraction=INNER, seed=seed + 1000 + k,
                                               strata_key=spec.strata_key)
                splits.append([sorted({s.patient_key for s in part}) for part in (fit, sel, test)])
            for cond in conditions:
                path = OUT_DIR / "oof" / f"{cond}__{task}__s{seed}.npz"
                if path.exists():
                    continue
                mods = CONDITIONS[cond][0]
                has = lambda k: any(k in feats.frames[m] for m in mods)   # noqa: E731
                t0, rows, info = time.time(), [], []
                for k, (fit, sel, test) in enumerate(splits):
                    fit, sel, test = ([p for p in part if has(p)] for part in (fit, sel, test))
                    probs, best = train_head(feats, cond, fit, sel, test, labels, len(classes), seed, k)
                    rows += [(p, k, labels[p], pr) for p, pr in zip(test, probs)]
                    info.append({"fold": k, **best})
                keys = np.array([r[0] for r in rows])
                np.savez(path, patient_key=keys, fold=np.array([r[1] for r in rows]),
                         y=np.array([r[2] for r in rows]), probs=np.stack([r[3] for r in rows]),
                         paired=np.array([all(p in feats.frames[m] for m in BOTH) for p in keys]))
                path.with_suffix(".json").write_text(json.dumps(info))
                y, pr = np.array([r[2] for r in rows]), np.stack([r[3] for r in rows])
                print(f"[{task} s{seed}] {cond:13s} bal={bal_acc(y, pr.argmax(1), len(classes)):.4f} "
                      f"epochs={[i['epochs_run'] for i in info]} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def load_oof(cond: str, task: str, seed: int) -> dict | None:
    path = OUT_DIR / "oof" / f"{cond}__{task}__s{seed}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    order = np.argsort(d["patient_key"])
    return {k: d[k][order] for k in ("patient_key", "y", "probs", "paired")}


def late_avg(wle: dict, eus: dict) -> dict:
    pos = {k: i for i, k in enumerate(eus["patient_key"])}
    probs = wle["probs"].copy()
    for i, k in enumerate(wle["patient_key"]):
        if k in pos:
            probs[i] = (probs[i] + eus["probs"][pos[k]]) / 2
    return {**wle, "probs": probs}


def gather(task: str) -> dict:
    """{cond: {"s42": oof, ..., "ens": oof}} for the conditions with all seeds."""
    out = {}
    for cond in [*CONDITIONS, LATE]:
        per_seed = {}
        for seed in SEEDS:
            if cond == LATE:
                w, e = load_oof("wle_only", task, seed), load_oof("eus_only", task, seed)
                r = late_avg(w, e) if w is not None and e is not None else None
            else:
                r = load_oof(cond, task, seed)
            if r is not None:
                per_seed[f"s{seed}"] = r
        if len(per_seed) == len(SEEDS):
            first = per_seed[f"s{SEEDS[0]}"]
            assert all((r["patient_key"] == first["patient_key"]).all() for r in per_seed.values())
            per_seed["ens"] = {**first, "probs": np.mean([r["probs"] for r in per_seed.values()], axis=0)}
        if per_seed:
            out[cond] = per_seed
    return out


def report() -> None:
    meta = exp49.load_metadata()
    spec = get_spec("cadsel")
    score_rows, vs_rows = [], []
    for task in ("binary", "subtype"):
        n_classes = len(spec.classes(task))
        groups = gather(task)
        for cond, per_seed in groups.items():
            for tag, r in per_seed.items():
                for pop in ("all", "eus", "rectum"):
                    m = exp49.population_mask(pop, r["patient_key"], r["paired"], meta)
                    if pop == "rectum" and cond == "eus_only":
                        m &= r["paired"]
                    y, pr = r["y"][m], r["probs"][m]
                    idx = exp44.boot_indices(y, np.random.default_rng(0))
                    row = exp49.score_row(y, pr, idx, n_classes)
                    score_rows.append({"task": task, "condition": cond, "seeds": tag, "population": pop,
                                       "n": int(m.sum()), **{k: row[k] for k in ("bal", "bal_lo", "bal_hi",
                                                                                 "macro_f1", "auroc")}})
            # mean over single seeds
            singles = [s for s in per_seed if s != "ens"]
            if len(singles) == len(SEEDS):
                for pop in ("all", "eus", "rectum"):
                    vals = [r for r in score_rows if r["task"] == task and r["condition"] == cond
                            and r["population"] == pop and r["seeds"] in singles]
                    score_rows.append({"task": task, "condition": cond, "seeds": "mean3", "population": pop,
                                       "n": vals[0]["n"],
                                       **{k: float(np.mean([v[k] for v in vals])) for k in ("bal", "macro_f1", "auroc")},
                                       "bal_lo": None, "bal_hi": None})
        # 3-seed ensembles versus the gated head and versus the WLE model
        for ref in ("gated", "wle_only"):
            if ref not in groups or "ens" not in groups[ref]:
                continue
            for pop in ("all", "eus", "rectum"):
                panel = []
                for cond, per_seed in groups.items():
                    if cond == ref or "ens" not in per_seed:
                        continue
                    a, b = per_seed["ens"], groups[ref]["ens"]
                    common = np.intersect1d(a["patient_key"], b["patient_key"])
                    ia, ib = (np.searchsorted(x["patient_key"], common) for x in (a, b))
                    paired = a["paired"][ia]
                    m = exp49.population_mask(pop, common, paired, meta)
                    if pop == "rectum" and (cond == "eus_only" or ref == "eus_only"):
                        m &= paired
                    y, pa, pb = a["y"][ia][m], a["probs"][ia][m], b["probs"][ib][m]
                    idx = exp44.boot_indices(y, np.random.default_rng(0))
                    d, lo, hi, p = exp49.delta(exp49.boot_bal(y, pa.argmax(1), idx, n_classes),
                                               exp49.boot_bal(y, pb.argmax(1), idx, n_classes))
                    row = {"task": task, "reference": ref, "population": pop, "condition": cond,
                           "n": int(m.sum()), "bal": exp44.bal_acc(y, pa.argmax(1), n_classes),
                           "bal_ref": exp44.bal_acc(y, pb.argmax(1), n_classes),
                           "delta_bal": d, "bal_lo": lo, "bal_hi": hi, "p_bal": p,
                           "auroc": exp44.auroc(y, pa), "auroc_ref": exp44.auroc(y, pb),
                           "delta_auroc": None, "auroc_lo": None, "auroc_hi": None, "p_auroc": None}
                    if n_classes == 2:
                        da, alo, ahi, ap = exp49.delta(exp49.boot_auroc(y, pa[:, 0], idx),
                                                       exp49.boot_auroc(y, pb[:, 0], idx))
                        row.update(delta_auroc=da, auroc_lo=alo, auroc_hi=ahi, p_auroc=ap)
                    panel.append(row)
                for key in ("p_bal", "p_auroc"):
                    ps = [r[key] for r in panel]
                    if panel and all(p is not None for p in ps):
                        for r, adj in zip(panel, exp44.holm(ps)):
                            r[f"{key}_holm"] = adj
                    else:
                        for r in panel:
                            r[f"{key}_holm"] = None
                vs_rows += panel
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("scores.csv", score_rows), ("versus.csv", vs_rows)):
        if not rows:
            continue
        with open(OUT_DIR / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
        print(f"wrote {OUT_DIR / name} ({len(rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step", choices=["extract", "train", "report", "all"], default="all")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--tasks", default="binary,subtype")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--aug-views", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    conds = args.conditions.split(",")
    unknown = set(conds) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions {sorted(unknown)}; expected {list(CONDITIONS)}")
    if args.step in ("extract", "all"):
        extract(args.aug_views, device)
    if args.step in ("train", "all"):
        train(conds, args.tasks.split(","), [int(s) for s in args.seeds.split(",")], args.aug_views, device)
    if args.step in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
