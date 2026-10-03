# %% [markdown]
# # Experiment 8 — Does the mitigation change the *detector's* behaviour?
#
# Experiment 7 showed the mitigations reduce background reliance for a
# classifier. This checks the thing that actually ships: the Faster R-CNN
# detector, baseline versus `scale_crop + copy_paste`.
#
# AP cannot be computed on a lesion-free crop — there is no ground truth to match
# against. So reliance is measured directly instead:
#
# > **Hallucination rate** — how often the detector fires above its 0.5
# > confidence threshold on an image crop that provably contains no annotated
# > lesion.
#
# A lesion-dependent detector should stay quiet on lesion-free tissue. A
# context-driven one keeps predicting, because the surrounding mucosa or
# ultrasound field is what it was really keying on. Lower is better, and it is
# measured on exactly the `bg_patch` crops from Experiment 3.

# %%
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
import torch
from PIL import Image

from cadsel.config import TASK_CLASSES
from cadsel.data import build_manifest, make_normalizer, patient_level_split, read_yolo_label
from cadsel.models import build_detection_model, configure_transform_for_input_mode

OUT = Path("experiments/results")
RUNS = Path("runs")
SEED, CONF = 42, 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

PAIRS = [
    ("WLE", "multiclass",
     "fasterrcnn_mobilenet_320__WLE__multiclass",
     "mobilenet_320__WLE__multiclass__mitigated"),
    ("WLE", "binary",
     "fasterrcnn_mobilenet_320__WLE__binary",
     "mobilenet_320__WLE__binary__mitigated"),
]

# %%
def boxes_for(sample, W, H):
    out = []
    for _c, cx, cy, bw, bh in read_yolo_label(sample.label):
        x1 = int(max(0, (cx - bw / 2) * W)); x2 = int(min(W, (cx + bw / 2) * W))
        y1 = int(max(0, (cy - bh / 2) * H)); y2 = int(min(H, (cy + bh / 2) * H))
        if x2 - x1 > 4 and y2 - y1 > 4:
            out.append((x1, y1, x2, y2))
    return out


def largest_free_rect(boxes, W, H):
    if not boxes:
        return (0, 0, W, H)
    x1 = min(b[0] for b in boxes); x2 = max(b[2] for b in boxes)
    y1 = min(b[1] for b in boxes); y2 = max(b[3] for b in boxes)
    cands = [(0, 0, W, y1), (0, y2, W, H), (0, 0, x1, H), (x2, 0, W, H)]
    cands = [c for c in cands if c[2] - c[0] >= 24 and c[3] - c[1] >= 24]
    return max(cands, key=lambda c: (c[2]-c[0]) * (c[3]-c[1])) if cands else None


def make_inputs(samples, img_size):
    """full frame and lesion-free crop, both at img_size, uint8."""
    full = np.zeros((len(samples), img_size, img_size, 3), dtype=np.uint8)
    bgp = np.zeros_like(full)
    has_bg = np.zeros(len(samples), dtype=bool)
    for i, s in enumerate(samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB")
            W, H = im.size
            arr = np.asarray(im, dtype=np.uint8)
        full[i] = np.asarray(Image.fromarray(arr).resize((img_size, img_size),
                                                         Image.BILINEAR))
        free = largest_free_rect(boxes_for(s, W, H), W, H)
        if free is not None:
            x1, y1, x2, y2 = free
            bgp[i] = np.asarray(Image.fromarray(arr[y1:y2, x1:x2])
                                .resize((img_size, img_size), Image.BILINEAR))
            has_bg[i] = True
    return full, bgp, has_bg


@torch.no_grad()
def detections_per_image(model, arr, normalize, batch=16):
    """Mean count of detections above CONF, and mean top score, per image."""
    counts, tops = [], []
    for k in range(0, len(arr), batch):
        # `make_normalizer` expects uint8 and does the /255 itself — pass it
        # uint8, not pre-scaled floats, or the images are divided by 255 twice.
        chunk = torch.from_numpy(arr[k:k + batch]).to(DEVICE).permute(0, 3, 1, 2)
        imgs = normalize(list(chunk))
        for out in model(imgs):
            s = out["scores"].float().cpu().numpy()
            counts.append(int((s > CONF).sum()))
            tops.append(float(s.max()) if len(s) else 0.0)
    return float(np.mean(counts)), float(np.mean(tops))


# %%
results = []
for modality, task, base_run, mit_run in PAIRS:
    samples = build_manifest(modality)
    _, val_s = patient_level_split(samples, 0.2, SEED)

    for label, run_name in (("baseline", base_run), ("mitigated", mit_run)):
        ckpt_path = RUNS / run_name / "best_model.pth"
        if not ckpt_path.exists():
            print(f"  missing checkpoint {ckpt_path} — skipping {label}")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        img_size, input_mode = ck["img_size"], ck.get("input_mode", "reference")

        model = build_detection_model(ck["model_type"], task, pretrained=False)
        configure_transform_for_input_mode(model, input_mode, img_size)
        model.load_state_dict(ck["model_state_dict"])
        model.to(DEVICE).eval()
        normalize = make_normalizer(input_mode)

        full, bgp, has_bg = make_inputs(val_s, img_size)
        n_full, s_full = detections_per_image(model, full, normalize)
        n_bg, s_bg = detections_per_image(model, bgp[has_bg], normalize)

        r = {
            "modality": modality, "task": task, "arm": label, "run": run_name,
            "best_epoch": ck["epoch"],
            "val_AP50": round(float(ck["metrics"]["AP@0.5"]), 4),
            "det_per_full_image": round(n_full, 3),
            "det_per_lesion_free_crop": round(n_bg, 3),
            "mean_top_score_full": round(s_full, 3),
            "mean_top_score_lesion_free": round(s_bg, 3),
            "hallucination_ratio": round(n_bg / n_full, 3) if n_full > 0 else None,
            "n_val": len(val_s), "n_lesion_free": int(has_bg.sum()),
        }
        results.append(r)
        print(f"{modality}/{task}/{label:<10} AP@0.5 {r['val_AP50']:.4f} | "
              f"det/full {n_full:.2f} | det/lesion-free {n_bg:.2f} | "
              f"ratio {r['hallucination_ratio']}")

json.dump(results, open(OUT / "08_detector_reliance.json", "w"), indent=2)

# %% [markdown]
# ## 8.1 Verdict

# %%
by = {(r["modality"], r["task"], r["arm"]): r for r in results}
print("=" * 92)
print("DETECTOR RELIANCE — firing rate on lesion-free crops")
print("=" * 92)
hdr = (f"{'mod':<5}{'task':<11}{'arm':<11}{'AP@0.5':>9}{'det/full':>10}"
       f"{'det/free':>10}{'ratio':>8}{'score/free':>12}")
print(hdr); print("-" * len(hdr))
for modality, task, _b, _m in PAIRS:
    for arm in ("baseline", "mitigated"):
        r = by.get((modality, task, arm))
        if r is None:
            continue
        ratio = ("  n/a" if r["hallucination_ratio"] is None
                 else f"{r['hallucination_ratio']:>8.2f}")
        print(f"{modality:<5}{task:<11}{arm:<11}{r['val_AP50']:>9.4f}"
              f"{r['det_per_full_image']:>10.2f}{r['det_per_lesion_free_crop']:>10.2f}"
              f"{ratio:>8}{r['mean_top_score_lesion_free']:>12.3f}")
    print()

print("Change from mitigation:")
for modality, task, _b, _m in PAIRS:
    b, m = by.get((modality, task, "baseline")), by.get((modality, task, "mitigated"))
    if not (b and m):
        continue
    d_ap = m["val_AP50"] - b["val_AP50"]
    if b["hallucination_ratio"] is None or m["hallucination_ratio"] is None:
        print(f"  {modality}/{task:<11} AP {d_ap:+.4f} | firing ratio n/a "
              f"(a model produced no detections above {CONF})")
        continue
    d_ratio = m["hallucination_ratio"] - b["hallucination_ratio"]
    verdict = ("less context reliance" if d_ratio < -0.05
               else "more context reliance" if d_ratio > 0.05 else "unchanged")
    print(f"  {modality}/{task:<11} AP {d_ap:+.4f} | "
          f"lesion-free firing ratio {d_ratio:+.2f}  -> {verdict}")
