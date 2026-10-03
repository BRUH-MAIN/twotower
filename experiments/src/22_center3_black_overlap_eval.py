# %% [markdown]
# # Experiment 22 — Is Center-3's LOCO collapse a labeling artifact?
#
# Leave-one-center-out validation found Center-3 collapses catastrophically
# when held out (internal AP@0.5 0.6532 -> external 0.1330), unlike Center-1
# (no drop at all). Investigation found two compounding causes: the
# Center-3-holdout training pool is 93.7% Center-1 by patient count, and
# Center-3's images are a near-completely-unseen raw-canvas capture format
# (100% of Center-3 vs ~1.5% of its training pool).
#
# A further wrinkle: decoding Center-3's YOLO boxes against the raw 1920x1080
# canvas (the pipeline's own existing normalization) shows a majority
# substantially overlap black letterbox/sidebar background -- 58.6% of boxes
# >10% black, 23.3% >50%, several 100% (floating with no visible lesion
# nearby). A naive "crop to content rect, remap boxes" fix was tried on one
# example and did not resolve this, so it isn't yet safe to assume a crop
# fixes anything.
#
# This script settles the more tractable question first, with no retraining:
# does the external AP collapse concentrate in the "dirty" (likely
# mislabeled) images, or does even the "clean" subset collapse? That
# determines whether crop-and-retrain is worth pursuing at all, and gives a
# cleaner yardstick for reading every other Center-3-LOCO lever tried
# afterward (center-RFS, DFR-by-center, style-jitter augmentation).

# %%
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from cadsel.config import TASK_CLASSES
from cadsel.data import CadSelDetectionDataset, build_manifest, collate_fn, make_normalizer, read_yolo_label
from cadsel.engine import evaluate
from cadsel.models import build_detection_model, configure_transform_for_input_mode

OUT = Path("experiments/results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIRTY_THRESHOLD = 0.10  # matches the "58.6% of boxes have >10% black overlap" statistic


# %%
def box_black_overlap_frac(image_arr: np.ndarray, cx: float, cy: float, bw: float, bh: float,
                            thresh: int = 10) -> float:
    """Fraction of a decoded YOLO box's pixels with per-channel mean brightness
    below `thresh` (near-black letterbox/sidebar background). Box decode is
    byte-identical to CadSelDetectionDataset.__getitem__'s own pixel conversion
    (cadsel/data.py) -- normalized cx/cy/w/h against the image's own raw H, W."""
    H, W = image_arr.shape[:2]
    x1 = int(max(0, (cx - bw / 2) * W)); x2 = int(min(W, (cx + bw / 2) * W))
    y1 = int(max(0, (cy - bh / 2) * H)); y2 = int(min(H, (cy + bh / 2) * H))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image_arr[y1:y2, x1:x2]
    return float((crop.mean(axis=2) < thresh).mean())


def image_max_black_frac(sample) -> float:
    """Bucket at the max-over-boxes level: one badly-misaligned box is enough
    to call the whole image 'dirty', since evaluate() scores per-image."""
    with Image.open(sample.image) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    rows = read_yolo_label(sample.label)
    if not rows:
        return 0.0
    return max(box_black_overlap_frac(arr, cx, cy, bw, bh) for _c, cx, cy, bw, bh in rows)


def score_subset(model, samples, task, img_size, input_mode, batch=8):
    if not samples:
        return None
    ds = CadSelDetectionDataset(samples, task, img_size, train=False, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fn, num_workers=0)
    m = evaluate(model, dl, DEVICE, TASK_CLASSES[task], progress=False,
                 normalize=make_normalizer(input_mode))
    return {"AP@0.5": round(m["AP@0.5"], 4), "f1": round(m["f1"], 4), "n": len(samples)}


def run(checkpoint: Path, held_out_centers: list[str]) -> dict:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_type, task, modality = ck["model_type"], ck["task"], ck["modality"]
    img_size, input_mode = ck["img_size"], ck.get("input_mode", "reference")

    model = build_detection_model(model_type, task, pretrained=False)
    configure_transform_for_input_mode(model, input_mode, ck.get("multiscale_sizes") or img_size)
    model.load_state_dict(ck["model_state_dict"])
    model.to(DEVICE).eval()

    samples = build_manifest(modality, centers=held_out_centers)
    black_fracs = [image_max_black_frac(s) for s in samples]
    clean = [s for s, f in zip(samples, black_fracs) if f < DIRTY_THRESHOLD]
    dirty = [s for s, f in zip(samples, black_fracs) if f >= DIRTY_THRESHOLD]

    overall = score_subset(model, samples, task, img_size, input_mode)
    clean_r = score_subset(model, clean, task, img_size, input_mode)
    dirty_r = score_subset(model, dirty, task, img_size, input_mode)

    result = {
        "checkpoint": str(checkpoint), "modality": modality, "task": task,
        "held_out_centers": held_out_centers, "dirty_threshold": DIRTY_THRESHOLD,
        "overall": overall, "clean": clean_r, "dirty": dirty_r,
    }
    print(f"{checkpoint}: overall AP@0.5={overall['AP@0.5']:.4f} (n={overall['n']})")
    print(f"  clean (<={DIRTY_THRESHOLD:.0%} black): AP@0.5={clean_r['AP@0.5'] if clean_r else float('nan'):.4f}"
          f" (n={clean_r['n'] if clean_r else 0})")
    print(f"  dirty (>{DIRTY_THRESHOLD:.0%} black):  AP@0.5={dirty_r['AP@0.5'] if dirty_r else float('nan'):.4f}"
          f" (n={dirty_r['n'] if dirty_r else 0})")
    return result


# %%
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--held-out-centers", required=True,
                    help="comma-separated centers this checkpoint was NOT trained on")
    p.add_argument("--output", type=Path, default=None,
                    help="default: experiments/results/22_center3_black_overlap_eval.json "
                    "(appends; keyed by checkpoint path)")
    args = p.parse_args()

    held_out = [c.strip() for c in args.held_out_centers.split(",") if c.strip()]
    result = run(args.checkpoint, held_out)

    out_path = args.output or (OUT / "22_center3_black_overlap_eval.json")
    existing = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    existing[str(args.checkpoint)] = result
    out_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
