# %% [markdown]
# # Experiment 25 — Cross-model WBF ensemble: free signal from checkpoints already trained
#
# This project already has multiple independently-trained WLE-multiclass
# checkpoints sharing the identical seed-42 patient-level validation split
# (verified: byte-identical `val_patients` across all candidates below) --
# different backbone architectures (MobileNetV3-FPN, ResNet50-FPN,
# ResNet50-FPN-v2, a frozen DINOv3 ViT) and different levers (plain baseline
# vs. the confirmed multiscale+size-RFS win). Weighted Boxes Fusion (Solovyev
# et al. 2021, already implemented in `cadsel/tta.py` for multi-scale TTA
# within one model) is a standard, low-risk way to fuse predictions across
# *different* models at inference time -- no retraining, purely combining
# checkpoints that already exist. New `cadsel/tta.py::predict_ensemble_wbf`/
# `predict_at_native_scale` bring every model's boxes into one common
# (native-pixel) coordinate space first, since different checkpoints use
# different `img_size`/`input_mode` conventions.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import json
import numpy as np
import torch

from cadsel.config import TASK_CLASSES
from cadsel.data import build_manifest, patient_level_split, read_yolo_label
from cadsel.metrics import DetectionAccumulator, summarize
from cadsel.models import build_detection_model, build_dinov3_detection_model, configure_transform_for_input_mode
from cadsel.tta import predict_at_native_scale, predict_ensemble_wbf

OUT = Path("experiments/results")
RUNS = Path("runs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODALITY, TASK = "WLE", "multiclass"
SEED = 42  # every candidate below shares this exact split -- verified before writing this script

# (run_name, weight) -- weight 1.0 unless flagged otherwise. DINOv3 is a
# single-scale detector, weaker on small/extreme-scale objects per the WBF
# literature's own caution about mixing architectures with different
# scale-handling -- down-weighted rather than equal-weighted.
import os
_VARIANT = os.environ.get("ENSEMBLE_VARIANT", "all5")
if _VARIANT == "top2":
    CANDIDATES = [
        ("size_rfs_ms_s42__WLE__multiclass", 1.0),
        ("dinov3_wle_multiclass_s42", 1.0),
    ]
elif _VARIANT == "top3":
    CANDIDATES = [
        ("size_rfs_ms_s42__WLE__multiclass", 1.0),
        ("fasterrcnn_mobilenet_320__WLE__multiclass", 1.0),
        ("dinov3_wle_multiclass_s42", 1.0),
    ]
else:
    CANDIDATES = [
        ("fasterrcnn_mobilenet_320__WLE__multiclass", 1.0),
        ("fasterrcnn_resnet__WLE__multiclass", 1.0),
        ("fasterrcnn_resnet50_fpn_v2__WLE__multiclass", 1.0),
        ("size_rfs_ms_s42__WLE__multiclass", 1.0),
        ("dinov3_wle_multiclass_s42", 0.5),
    ]


# %%
def load_model_info(run_name: str, weight: float) -> dict:
    ck = torch.load(RUNS / run_name / "best_model.pth", map_location="cpu", weights_only=False)
    model_type, input_mode = ck["model_type"], ck.get("input_mode", "reference")
    img_size = ck.get("multiscale_sizes") or ck["img_size"]
    if model_type == "fasterrcnn_dinov3":
        model = build_dinov3_detection_model(model_type, ck["task"])
    else:
        model = build_detection_model(model_type, ck["task"], pretrained=False)
    configure_transform_for_input_mode(model, input_mode, img_size)
    model.load_state_dict(ck["model_state_dict"])
    model.to(DEVICE).eval()
    return {
        "run_name": run_name, "model": model, "input_mode": input_mode,
        "img_size": ck["img_size"], "weight": weight, "solo_ap": ck["metrics"]["AP@0.5"],
    }


def gt_for(sample) -> tuple[np.ndarray, np.ndarray]:
    with __import__("PIL.Image", fromlist=["Image"]).open(sample.image) as im:
        w, h = im.size
    boxes, labels = [], []
    from cadsel.config import TASK_ID_MAP
    id_map = TASK_ID_MAP[TASK]
    for cls, cx, cy, bw, bh in read_yolo_label(sample.label):
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        boxes.append([x1, y1, x2, y2])
        labels.append(id_map[cls] + 1)
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)


# %%
samples = build_manifest(MODALITY)
_, val_s = patient_level_split(samples, 0.2, SEED)
print(f"validation set: {len(val_s)} images")

infos = [load_model_info(name, w) for name, w in CANDIDATES]
for info in infos:
    print(f"  loaded {info['run_name']:<45} solo AP@0.5={info['solo_ap']:.4f} weight={info['weight']}")

# Solo accumulators (re-derived here rather than trusted from each
# checkpoint's own result.json, so solo and ensemble numbers are computed by
# the exact same code path -- an apples-to-apples comparison) + the ensemble.
solo_accs = {info["run_name"]: DetectionAccumulator() for info in infos}
ensemble_acc = DetectionAccumulator()

for i, sample in enumerate(val_s):
    gt_boxes, gt_labels = gt_for(sample)
    preds_for_ensemble = []
    for info in infos:
        pred = predict_at_native_scale(
            info["model"], sample.image, DEVICE, info["input_mode"], info["img_size"],
        )
        solo_accs[info["run_name"]].add(pred["boxes"], pred["scores"], pred["labels"], gt_boxes, gt_labels)
        preds_for_ensemble.append(pred)

    ens = predict_ensemble_wbf(infos, sample.image, DEVICE)
    ensemble_acc.add(ens["boxes"], ens["scores"], ens["labels"], gt_boxes, gt_labels)

    if (i + 1) % 50 == 0:
        print(f"  scored {i + 1}/{len(val_s)}")

# %% [markdown]
# ## 25.1 Verdict

# %%
class_names = TASK_CLASSES[TASK]
results = {"n_val": len(val_s), "candidates": []}
print("=" * 78)
print("CROSS-MODEL WBF ENSEMBLE — WLE multiclass, shared seed-42 val split")
print("=" * 78)
for info in infos:
    ap = summarize(solo_accs[info["run_name"]], class_names)["AP@0.5"]
    results["candidates"].append({"run": info["run_name"], "weight": info["weight"], "AP@0.5": round(ap, 4)})
    print(f"  solo: {info['run_name']:<45} AP@0.5={ap:.4f}")

ensemble_ap = summarize(ensemble_acc, class_names)["AP@0.5"]
results["ensemble_AP@0.5"] = round(ensemble_ap, 4)
best_solo = max(r["AP@0.5"] for r in results["candidates"])
print(f"\n  ENSEMBLE (WBF, {len(infos)} models):        AP@0.5={ensemble_ap:.4f}")
print(f"  best single model:                          AP@0.5={best_solo:.4f}")
print(f"  ensemble delta vs. best single model: {ensemble_ap - best_solo:+.4f}")

json.dump(results, open(OUT / f"25_cross_model_ensemble_{_VARIANT}.json", "w"), indent=2)
