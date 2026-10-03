# %% [markdown]
# # Experiment 24 — JTT diagnostic: is there a hard group nobody's hypothesized?
#
# JTT (Liu et al., ICML 2021, "Just Train Twice") needs no group labels: train
# a model with vanilla ERM, identify which *training images* it gets wrong,
# retrain a second model with those images upweighted. Its whole appeal over
# Group DRO/DFR/`CenterRepeatFactorSampler` (all tried this session, group=
# acquisition-center, mixed-to-negative results) is that it could surface an
# unnamed interaction -- e.g. "small NET lesions in Center-3" -- that nobody
# has hand-specified as a group.
#
# Per the scoping pass done before writing any code: build only the
# identification half first, as a diagnostic. If the resulting hard set is
# mostly explained by attributes already probed (center, subtype, lesion
# size), a second JTT training run has low expected value -- those three
# cheaper levers already covered that territory. Only build the full
# sampler + second-training-run half if the hard set looks genuinely
# orthogonal to all three.
#
# Per-image loss (not per-ROI: proposals aren't stable across two separate
# training runs, so "an example" for this detection task is the image, the
# same unit `RepeatFactorSampler`'s whole family already uses) is scored at
# batch_size=1 -- `cadsel.engine.evaluate`'s own `with_loss=True` path
# (model.train() -> forward with targets -> sum(loss_dict.values()) ->
# model.eval()) computes a *batch* total, which only equals one image's loss
# when the batch size is 1.

# %%
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from cadsel.data import (
    CadSelDetectionDataset, Sample, box_area_frac, build_manifest, collate_fn, make_normalizer,
)
from cadsel.models import build_detection_model, configure_transform_for_input_mode

OUT = Path("experiments/results")
RUNS = Path("runs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Plain ERM baseline, no oversampling/lever active -- JTT's identification
# pass is meant to run against a vanilla-ERM model, not one already using a
# mitigation (using an already-levered checkpoint would identify what THAT
# lever still gets wrong, a different, less canonical question).
CHECKPOINT = RUNS / "ms_baseline_s42__WLE__multiclass" / "best_model.pth"
HARD_FRACTION = 0.2  # top 20% by loss, following JTT's usual small-hard-group framing


# %%
def per_image_losses(model, samples: list[Sample], task: str, img_size: int, input_mode: str) -> list[float]:
    """One forward+loss pass per image, batch_size=1 so the batch total IS
    that image's loss. Dataset built train=False (no augmentation) -- the
    identification pass should reflect genuine difficulty, not
    augmentation-induced noise."""
    ds = CadSelDetectionDataset(samples, task, img_size, train=False, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)
    normalize = make_normalizer(input_mode)
    losses = []
    model.train()  # needed for torchvision detectors to return a loss_dict
    with torch.no_grad():
        for images, targets in dl:
            if len(targets[0]["boxes"]) == 0:
                losses.append(float("nan"))  # hard-negative-style empty target; skip from ranking
                continue
            images_dev = [img.to(DEVICE) for img in images]
            images_dev = normalize(images_dev)
            targets_dev = [{k: v.to(DEVICE) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
            loss_dict = model(images_dev, targets_dev)
            losses.append(float(sum(loss_dict.values())))
    model.eval()
    return losses


# %%
ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model_type, task, modality = ck["model_type"], ck["task"], ck["modality"]
img_size, input_mode = ck["img_size"], ck.get("input_mode", "reference")

model = build_detection_model(model_type, task, pretrained=False)
configure_transform_for_input_mode(model, input_mode, ck.get("multiscale_sizes") or img_size)
model.load_state_dict(ck["model_state_dict"])
model.to(DEVICE)

split = json.loads((CHECKPOINT.parent / "split.json").read_text(encoding="utf-8"))
train_keys = set(split["train_patients"])
all_samples = build_manifest(modality)
train_samples = [s for s in all_samples if s.patient_key in train_keys]
print(f"scoring {len(train_samples)} training images from {CHECKPOINT.parent.name}")

losses = per_image_losses(model, train_samples, task, img_size, input_mode)

records = []
for s, loss in zip(train_samples, losses):
    if not np.isfinite(loss):
        continue
    records.append({
        "image": s.image, "center": s.center, "subtype": s.subtype,
        "area_frac": box_area_frac(s), "loss": loss,
    })

records.sort(key=lambda r: -r["loss"])
n_hard = int(round(len(records) * HARD_FRACTION))
for i, r in enumerate(records):
    r["hard"] = i < n_hard

json.dump(records, open(OUT / "24_jtt_diagnostic.json", "w"), indent=2)
print(f"scored {len(records)} images (excluding hard-negatives/empty targets); "
      f"top {n_hard} ({HARD_FRACTION:.0%}) flagged hard")

# %% [markdown]
# ## 24.1 Is the hard set explained by center / subtype / lesion size?

# %%
hard = [r for r in records if r["hard"]]
easy = [r for r in records if not r["hard"]]

print("=" * 78)
print("Hard set vs. overall pool -- center distribution")
print("=" * 78)
overall_centers = Counter(r["center"] for r in records)
hard_centers = Counter(r["center"] for r in hard)
for c in sorted(overall_centers):
    overall_frac = overall_centers[c] / len(records)
    hard_frac = hard_centers.get(c, 0) / len(hard) if hard else 0.0
    print(f"  {c:<12} overall {overall_frac:.1%}  |  hard-set {hard_frac:.1%}")

print("\nHard set vs. overall pool -- subtype distribution")
overall_subtypes = Counter(r["subtype"] for r in records)
hard_subtypes = Counter(r["subtype"] for r in hard)
for s in sorted(overall_subtypes):
    overall_frac = overall_subtypes[s] / len(records)
    hard_frac = hard_subtypes.get(s, 0) / len(hard) if hard else 0.0
    print(f"  {s:<14} overall {overall_frac:.1%}  |  hard-set {hard_frac:.1%}")

print("\nLesion size (box_area_frac) -- hard vs. easy")
hard_areas = [r["area_frac"] for r in hard]
easy_areas = [r["area_frac"] for r in easy]
print(f"  hard: mean={np.mean(hard_areas):.4f} median={np.median(hard_areas):.4f}")
print(f"  easy: mean={np.mean(easy_areas):.4f} median={np.median(easy_areas):.4f}")

summary = {
    "checkpoint": str(CHECKPOINT), "n_scored": len(records), "n_hard": len(hard),
    "hard_center_dist": dict(hard_centers), "overall_center_dist": dict(overall_centers),
    "hard_subtype_dist": dict(hard_subtypes), "overall_subtype_dist": dict(overall_subtypes),
    "hard_area_mean": float(np.mean(hard_areas)), "easy_area_mean": float(np.mean(easy_areas)),
}
json.dump(summary, open(OUT / "24_jtt_diagnostic_summary.json", "w"), indent=2)

# %% [markdown]
# ## 24.2 Verdict

# %%
print("\nInterpretation guide:")
print("- If the hard set's center/subtype/size distribution closely tracks the")
print("  overall pool's (no large over/under-representation), the hard set is")
print("  NOT well explained by any of the three already-probed axes -- build")
print("  the full JTT sampler + second training run (JTTRepeatFactorSampler).")
print("- If one axis is heavily over-represented in the hard set (e.g. small")
print("  lesions, or one center), JTT would mostly rediscover what size-RFS/")
print("  center-RFS/Group DRO already tested directly -- lower expected value")
print("  for the more expensive two-stage retrain.")
