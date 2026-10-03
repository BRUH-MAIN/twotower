# %% [markdown]
# # Experiment 23 — why does the "dirty" bucket score *higher* than "clean"?
#
# Exp 22 (`22_center3_black_overlap_eval.py`) bucketed Center-3's held-out
# images by black-overlap fraction (a box floating partly/wholly in the
# letterbox/sidebar region, from decoding YOLO boxes against the raw
# 1920x1080 canvas) into "clean" (<10% black) vs "dirty" (>=10%), expecting
# clean (well-labeled) to score higher if the LOCO collapse were a labeling
# artifact. It found the opposite -- dirty scores *higher* -- and this
# pattern replicated across all 12 checkpoints tested this session (§5l/m/n),
# never once reversing. FINDINGS.md §8 item 12 flagged this as unexplained,
# with a leading hypothesis: black-overlap fraction may be an accidental
# proxy for capture *format* (the "wide" layout has more letterbox by
# construction: x=[670,1912],y=[0,1079] vs "standard"'s x=[698,1859],
# y=[36,1043], per the original Center-3 image-geometry investigation) or
# for lesion visibility/size, rather than for label quality.
#
# This tests that directly: classify every Center-3 image's capture layout,
# then check whether layout mode alone -- independent of black-overlap
# bucketing -- predicts AP against the same checkpoint, and cross-tabulate
# layout/black-bucket/subtype/lesion-size to see what's actually correlated
# with what.

# %%
import json
import sys
from collections import Counter, defaultdict
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
CHECKPOINT = Path("runs/loco_holdout_center3_binary/best_model.pth")
DIRTY_THRESHOLD = 0.10

# Fixed content rectangles, from the original Center-3 image-geometry
# investigation (94.0% "standard" + 4.9% "wide" of Center-3's 368 images;
# a rare ~1.1% "dual-view" layout fits neither and is left "ambiguous" here).
STANDARD_RECT = (698, 36, 1859, 1043)
WIDE_RECT = (670, 0, 1912, 1079)


# %%
def box_black_overlap_frac(image_arr: np.ndarray, cx, cy, bw, bh, thresh: int = 10) -> float:
    """Byte-identical to 22_center3_black_overlap_eval.py's own copy."""
    H, W = image_arr.shape[:2]
    x1 = int(max(0, (cx - bw / 2) * W)); x2 = int(min(W, (cx + bw / 2) * W))
    y1 = int(max(0, (cy - bh / 2) * H)); y2 = int(min(H, (cy + bh / 2) * H))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = image_arr[y1:y2, x1:x2]
    return float((crop.mean(axis=2) < thresh).mean())


def classify_layout(image_arr: np.ndarray, thresh: int = 15) -> str:
    """Two diagnostic pixel probes exploiting where the two known rectangles
    disagree: (680, 500) is inside 'wide's content area but in 'standard's
    left margin; (1000, 10) is inside 'wide's content area but above
    'standard's top margin. Bright at both -> wide; dark at both ->
    standard; disagreement -> a layout that fits neither (e.g. dual-view)."""
    H, W = image_arr.shape[:2]
    if W < 1913 or H < 1080:
        return "not_raw_canvas"
    p1 = image_arr[500, 680].mean() > thresh   # bright => wide-only region is lit
    p2 = image_arr[10, 1000].mean() > thresh   # bright => wide-only region is lit
    if p1 and p2:
        return "wide"
    if not p1 and not p2:
        return "standard"
    return "ambiguous"


def image_max_black_frac(sample) -> float:
    with Image.open(sample.image) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    rows = read_yolo_label(sample.label)
    if not rows:
        return 0.0
    return max(box_black_overlap_frac(arr, cx, cy, bw, bh) for _c, cx, cy, bw, bh in rows)


def box_area_frac(sample) -> float:
    best = 0.0
    for _c, _cx, _cy, bw, bh in read_yolo_label(sample.label):
        best = max(best, float(bw * bh))
    return best


def image_layout(sample) -> str:
    with Image.open(sample.image) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    return classify_layout(arr)


def score_subset(model, samples, task, img_size, input_mode, batch=8):
    if not samples:
        return None
    ds = CadSelDetectionDataset(samples, task, img_size, train=False, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fn, num_workers=0)
    m = evaluate(model, dl, DEVICE, TASK_CLASSES[task], progress=False,
                 normalize=make_normalizer(input_mode))
    return {"AP@0.5": round(m["AP@0.5"], 4), "n": len(samples)}


# %%
samples = build_manifest("WLE", centers=["Center-3"])
print(f"Center-3: {len(samples)} images")

records = []
for s in samples:
    records.append({
        "image": s.image,
        "subtype": s.subtype,
        "black_frac": image_max_black_frac(s),
        "layout": image_layout(s),
        "area_frac": box_area_frac(s),
    })

for r in records:
    r["bucket"] = "dirty" if r["black_frac"] >= DIRTY_THRESHOLD else "clean"

json.dump(records, open(OUT / "23_dirty_clean_records.json", "w"), indent=2)

# %% [markdown]
# ## 23.1 Cross-tabulations

# %%
print("=" * 78)
print("Layout mode counts")
print("=" * 78)
print(Counter(r["layout"] for r in records))

print("\nBucket (clean/dirty) x layout mode:")
cross = defaultdict(Counter)
for r in records:
    cross[r["bucket"]][r["layout"]] += 1
for bucket, counts in cross.items():
    print(f"  {bucket}: {dict(counts)}")

print("\nMean box_area_frac (lesion size) by bucket:")
for bucket in ("clean", "dirty"):
    areas = [r["area_frac"] for r in records if r["bucket"] == bucket]
    print(f"  {bucket}: mean={np.mean(areas):.4f} median={np.median(areas):.4f} n={len(areas)}")

print("\nMean box_area_frac by layout:")
for layout in sorted(set(r["layout"] for r in records)):
    areas = [r["area_frac"] for r in records if r["layout"] == layout]
    print(f"  {layout}: mean={np.mean(areas):.4f} n={len(areas)}")

print("\nSubtype distribution by bucket:")
for bucket in ("clean", "dirty"):
    subtypes = Counter(r["subtype"] for r in records if r["bucket"] == bucket)
    print(f"  {bucket}: {dict(subtypes)}")

# %% [markdown]
# ## 23.2 Does layout mode alone predict AP? (independent of black-overlap bucket)

# %%
ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
model = build_detection_model(ck["model_type"], ck["task"], pretrained=False)
configure_transform_for_input_mode(model, ck.get("input_mode", "reference"), ck["img_size"])
model.load_state_dict(ck["model_state_dict"])
model.to(DEVICE).eval()

by_layout = defaultdict(list)
for s, r in zip(samples, records):
    by_layout[r["layout"]].append(s)

layout_results = {}
print("=" * 78)
print("AP@0.5 by layout mode (independent of black-overlap bucket)")
print("=" * 78)
for layout, subset in sorted(by_layout.items()):
    res = score_subset(model, subset, ck["task"], ck["img_size"], ck.get("input_mode", "reference"))
    layout_results[layout] = res
    print(f"  {layout:<14} AP@0.5={res['AP@0.5'] if res else float('nan'):.4f}  (n={res['n'] if res else 0})")

json.dump(layout_results, open(OUT / "23_layout_ap.json", "w"), indent=2)

# %% [markdown]
# ## 23.3 Verdict

# %%
print("\nInterpretation guide:")
print("- If 'wide' layout's AP is markedly higher than 'standard's, and 'wide'")
print("  is over-represented in the 'dirty' bucket (more letterbox by construction")
print("  -> higher black-overlap fraction), that would explain dirty>clean as a")
print("  capture-format confound, not a real 'worse labels score better' effect.")
print("- If lesion size (area_frac) differs meaningfully between clean/dirty,")
print("  that's an alternative/additional confound (size already independently")
print("  affects AP throughout this project, per Exp 9).")
print("- If neither shows a meaningful pattern, the dirty>clean effect remains")
print("  genuinely unexplained.")
