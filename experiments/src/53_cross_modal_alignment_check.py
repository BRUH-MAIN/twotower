"""Experiment 53 -- is there patient-specific signal shared by WLE and EUS?

A go / no-go check before training with a cross-modal alignment loss
(34 ``--align-weight``). Uses the PRETRAINED encoders, untouched: DINOv3
ViT-S/16 for WLE and USFM ViT-B/16 for EUS (``--eus-backbone`` to change it),
on the lesion crops, eval-mode frame selection (4 frames), mean over frames.

Per outer fold of the seed-42 patient split (patients with both modalities):
  * standardise + PCA (fitted on the training patients) for each modality,
  * CCA between the two (fitted on the training patients),
  * on the test patients, cosine similarity in the CCA space and, for each
    patient's WLE, the rank of its own EUS among the candidates.

Candidates are all test patients ("any"), or only those with the same label
("same_binary", "same_subtype"), the same centre ("same_centre"), or both
("same_subtype_centre"). Above-chance retrieval among same-label
patients means the two modalities share something about the patient beyond
the diagnosis, which an alignment loss could use. Chance and p-values come
from shuffling the WLE-EUS pairing within the candidate groups.

Writes ``experiments/results/53_alignment_check/retrieval.csv``.

    python -m experiments.src.53_cross_modal_alignment_check
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cadsel.config import PROJECT_ROOT
from cadsel.data import patient_level_folds
from cadsel.external_datasets import get_spec
from cadsel.joint_fusion_data import JointPatientDataset, joint_collate
from cadsel.joint_fusion_model import build_tower, tower_normalizers

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "53_alignment_check"
N_PERM = 2000


@torch.no_grad()
def patient_features(backbones: dict, device: torch.device, max_frames: int = 4) -> dict:
    """{patient_key: {"WLE": vec, "EUS": vec, "subtype": str}} for patients with both."""
    spec = get_spec("cadsel")
    samples = spec.manifest()
    mods = list(backbones)
    ds = JointPatientDataset(samples, "subtype", img_size=224, max_frames=max_frames, train=False, seed=42,
                             lesion_crop=True, modalities=mods, classes=spec.classes("subtype"),
                             label_of=spec.label_of)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=joint_collate, num_workers=4)
    towers = {m: build_tower(b, "partial").to(device).eval() for m, b in backbones.items()}
    norms = {m: tower_normalizers(b, [m])[m] for m, b in backbones.items()}
    feats = {m: [] for m in mods}
    masks_all = {m: [] for m in mods}
    for images, masks, _ in loader:
        for m in mods:
            x = images[m].to(device)
            B, T, C, H, W = x.shape
            flat = norms[m](x.reshape(B * T, C, H, W).float())
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                f = towers[m](flat).float().reshape(B, T, -1).cpu()
            mk = masks[m].float()
            feats[m].append((f * mk.unsqueeze(-1)).sum(1) / mk.sum(1, keepdim=True).clamp(min=1))
            masks_all[m].append(mk.sum(1) > 0.5)
    feats = {m: torch.cat(v).numpy() for m, v in feats.items()}
    present = {m: torch.cat(v).numpy() for m, v in masks_all.items()}
    both = present[mods[0]] & present[mods[1]]
    return {k: {"WLE": feats["WLE"][i], "EUS": feats["EUS"][i], "subtype": ds.subtype[k]}
            for i, k in enumerate(ds.patient_keys) if both[i]}


def ranks(sim: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Rank (1 = best) of each row's own column among the columns in its group."""
    out = np.empty(len(sim))
    for i in range(len(sim)):
        cand = np.flatnonzero(groups == groups[i])
        out[i] = 1 + int((sim[i, cand] > sim[i, i]).sum())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eus-backbone", default="usfm_vitb16")
    p.add_argument("--pca", type=int, default=64)
    p.add_argument("--cca", type=int, default=16)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = get_spec("cadsel")

    data = patient_features({"WLE": "dinov3_vits16", "EUS": args.eus_backbone}, device)
    keys = sorted(data)
    W = np.stack([data[k]["WLE"] for k in keys])
    E = np.stack([data[k]["EUS"] for k in keys])
    subtype = np.array([data[k]["subtype"] for k in keys])
    binary = np.array([spec.label_of(s, "binary") for s in subtype])
    print(f"{len(keys)} patients with both modalities; WLE {W.shape[1]}-d, EUS {E.shape[1]}-d")

    folds = patient_level_folds(spec.manifest(), n_folds=5, seed=42, strata_key=spec.strata_key)
    fold_of = {}
    for f, (_, test) in enumerate(folds):
        for s in test:
            fold_of[s.patient_key] = f
    fold = np.array([fold_of[k] for k in keys])
    exp34 = importlib.import_module("experiments.src.34_joint_fusion_train")
    centre_of = exp34.patient_centres(spec.manifest())
    centre = np.array([centre_of[k] for k in keys])

    rows, rng = [], np.random.default_rng(0)
    # same centre too: the two modalities come from one hospital's equipment, so
    # retrieval could be matching the site rather than the patient
    groupings = {"any": np.zeros(len(keys), int), "same_binary": binary, "same_subtype": subtype,
                 "same_centre": centre,
                 "same_subtype_centre": np.array([f"{a}|{b}" for a, b in zip(subtype, centre)])}
    per_group = {g: [] for g in groupings}
    perm_stats = {g: np.zeros(N_PERM) for g in groupings}
    for f in range(5):
        tr, te = fold != f, fold == f
        zw = PCA(args.pca, random_state=0).fit(StandardScaler().fit(W[tr]).transform(W[tr]))
        ze = PCA(args.pca, random_state=0).fit(StandardScaler().fit(E[tr]).transform(E[tr]))
        sw, se = StandardScaler().fit(W[tr]), StandardScaler().fit(E[tr])
        cca = CCA(args.cca, max_iter=2000).fit(zw.transform(sw.transform(W[tr])), ze.transform(se.transform(E[tr])))
        a, b = cca.transform(zw.transform(sw.transform(W[te])), ze.transform(se.transform(E[te])))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        sim = a @ b.T
        for g, labels in groupings.items():
            lab = labels[te]
            r = ranks(sim, lab)
            sizes = np.array([(lab == lab[i]).sum() for i in range(len(lab))])
            per_group[g].append((r, sizes))
            # null: shuffle which EUS belongs to which WLE, within each candidate group
            for n in range(N_PERM):
                perm = np.arange(len(lab))
                for v in np.unique(lab):
                    idx = np.flatnonzero(lab == v)
                    perm[idx] = rng.permutation(idx)
                perm_stats[g][n] += (1.0 / ranks(sim[:, perm], lab)).sum()

    for g in groupings:
        r = np.concatenate([x[0] for x in per_group[g]])
        sizes = np.concatenate([x[1] for x in per_group[g]])
        mrr = float((1.0 / r).mean())
        null = perm_stats[g] / len(r)
        rows.append({"candidates": g, "eus_backbone": args.eus_backbone, "n": len(r),
                     "mean_candidates": float(sizes.mean()), "mrr": mrr, "mrr_chance": float(null.mean()),
                     "top1": float((r == 1).mean()), "top1_chance": float((1.0 / sizes).mean()),
                     "median_rank_percentile": float(np.median((r - 1) / np.maximum(sizes - 1, 1))),
                     "p_perm": float((1 + (null >= mrr).sum()) / (1 + N_PERM))})
        print(f"  {g:13s} n={len(r)}  candidates~{sizes.mean():5.1f}  MRR {mrr:.3f} (chance {null.mean():.3f})  "
              f"top-1 {rows[-1]['top1']:.3f} (chance {rows[-1]['top1_chance']:.3f})  "
              f"median rank pct {rows[-1]['median_rank_percentile']:.2f}  p={rows[-1]['p_perm']:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import csv
    path = OUT_DIR / f"retrieval_{args.eus_backbone}.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
