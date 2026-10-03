# %% [markdown]
# # Experiment 20 — Does synthetic (GAN) copy-paste avoid EUS's real-copy-paste backfire?
#
# Experiment 7 found `copy_paste` with real cropped lesions *increases* EUS
# shortcut reliance by 23-37%, attributed to pasted color/WLE-textured real
# lesions looking implausible on grayscale, narrow-field ultrasound. Lead 5a
# trains a class-conditional DCGAN generator on EUS lesion crops
# (`cadsel.train_lesion_gan`, `runs/lesion_gan_EUS_multiclass.pth`) and feeds
# its synthetic patches through the same `copy_paste` augmentation path via
# `--copy-paste-source generated`, to test whether a same-domain synthetic
# lesion avoids the backfire while keeping copy-paste's usual AP.
#
# This reuses Experiment 8's hallucination-rate protocol (a detector has no AP
# on a lesion-free crop — there's no ground truth to match against — so
# reliance is measured as firing rate on `bg_patch` crops instead), extended
# to 3 arms x 3 seeds instead of one baseline/mitigated pair.

# %%
import sys, json
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
import torch
from PIL import Image

from cadsel.data import build_manifest, make_normalizer, read_yolo_label
from cadsel.models import build_detection_model, configure_transform_for_input_mode

OUT = Path("experiments/results")
RUNS = Path("runs")
CONF = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

MODALITY, TASK = "EUS", "multiclass"
ARMS = {
    "plain_control": ["eus_plain_control_s42", "eus_plain_control_s123", "eus_plain_control_s7"],
    # seed 42 predates this grid -- it's the existing real-copy-paste backfire
    # run Experiment 7 already measured, reused rather than retrained.
    "real_copypaste": ["fasterrcnn_mobilenet_320__EUS__multiclass", "eus_realcp_s123", "eus_realcp_s7"],
    "gan_copypaste": ["eus_gancp_s42", "eus_gancp_s123", "eus_gancp_s7"],
}

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


def load_val_samples(run_name: str, modality: str) -> list:
    """Reproduce the exact validation set a run used, from its own split.json --
    each run in this grid has its own --seed, hence its own patient_level_split,
    so scoring every checkpoint against one shared split would evaluate the
    seed-123/seed-7 checkpoints on the wrong held-out patients."""
    split = json.loads((RUNS / run_name / "split.json").read_text(encoding="utf-8"))
    val_keys = set(split["val_patients"])
    samples = build_manifest(modality)
    return [s for s in samples if s.patient_key in val_keys]


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
        chunk = torch.from_numpy(arr[k:k + batch]).to(DEVICE).permute(0, 3, 1, 2)
        imgs = normalize(list(chunk))
        for out in model(imgs):
            s = out["scores"].float().cpu().numpy()
            counts.append(int((s > CONF).sum()))
            tops.append(float(s.max()) if len(s) else 0.0)
    return float(np.mean(counts)), float(np.mean(tops))


# %%
results = []
for arm, run_names in ARMS.items():
    for run_name in run_names:
        ckpt_path = RUNS / run_name / "best_model.pth"
        if not ckpt_path.exists():
            print(f"  missing checkpoint {ckpt_path} -- skipping (has Phase 3 finished?)")
            continue
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        img_size, input_mode = ck["img_size"], ck.get("input_mode", "reference")

        model = build_detection_model(ck["model_type"], TASK, pretrained=False)
        configure_transform_for_input_mode(model, input_mode, img_size)
        model.load_state_dict(ck["model_state_dict"])
        model.to(DEVICE).eval()
        normalize = make_normalizer(input_mode)

        val_s = load_val_samples(run_name, MODALITY)
        full_i, bgp_i, has_bg_i = make_inputs(val_s, img_size)
        n_full, s_full = detections_per_image(model, full_i, normalize)
        n_bg, s_bg = detections_per_image(model, bgp_i[has_bg_i], normalize)

        r = {
            "arm": arm, "run": run_name,
            "val_AP50": round(float(ck["metrics"]["AP@0.5"]), 4),
            "det_per_full_image": round(n_full, 3),
            "det_per_lesion_free_crop": round(n_bg, 3),
            "mean_top_score_lesion_free": round(s_bg, 3),
            "hallucination_ratio": round(n_bg / n_full, 3) if n_full > 0 else None,
        }
        results.append(r)
        print(f"{arm:<16}{run_name:<28} AP@0.5 {r['val_AP50']:.4f} | "
              f"ratio {r['hallucination_ratio']}")
        del model
        torch.cuda.empty_cache()

json.dump(results, open(OUT / "20_gan_copypaste_reliance.json", "w"), indent=2)

# %% [markdown]
# ## 20.1 Verdict — mean +/- std per arm

# %%
by_arm = defaultdict(list)
for r in results:
    by_arm[r["arm"]].append(r)

print("=" * 78)
print("EUS COPY-PASTE SOURCE — AP and hallucination ratio, mean +/- std across seeds")
print("=" * 78)
summary = {}
for arm in ARMS:
    rows = by_arm.get(arm, [])
    if not rows:
        continue
    aps = [r["val_AP50"] for r in rows]
    ratios = [r["hallucination_ratio"] for r in rows if r["hallucination_ratio"] is not None]
    ap_mean, ap_std = float(np.mean(aps)), float(np.std(aps))
    ratio_mean = float(np.mean(ratios)) if ratios else None
    ratio_std = float(np.std(ratios)) if ratios else None
    summary[arm] = {"ap_mean": ap_mean, "ap_std": ap_std, "n_seeds": len(rows),
                     "ratio_mean": ratio_mean, "ratio_std": ratio_std}
    ratio_str = f"{ratio_mean:.3f} +/- {ratio_std:.3f}" if ratio_mean is not None else "n/a"
    print(f"{arm:<16} AP@0.5 {ap_mean:.4f} +/- {ap_std:.4f}  (n={len(rows)})  |  "
          f"hallucination ratio {ratio_str}")

json.dump(summary, open(OUT / "20_gan_copypaste_reliance_summary.json", "w"), indent=2)

if "plain_control" in summary and "real_copypaste" in summary and "gan_copypaste" in summary:
    pc, rc, gc = summary["plain_control"], summary["real_copypaste"], summary["gan_copypaste"]
    print("\nHeadline question: does synthetic copy-paste avoid the real-copy-paste backfire?")
    if rc["ratio_mean"] is not None and gc["ratio_mean"] is not None and pc["ratio_mean"] is not None:
        print(f"  plain control ratio:  {pc['ratio_mean']:.3f}")
        print(f"  real copy-paste ratio: {rc['ratio_mean']:.3f}  (delta vs control {rc['ratio_mean'] - pc['ratio_mean']:+.3f})")
        print(f"  GAN copy-paste ratio:  {gc['ratio_mean']:.3f}  (delta vs control {gc['ratio_mean'] - pc['ratio_mean']:+.3f})")
        print(f"  AP: control {pc['ap_mean']:.4f}  real-cp {rc['ap_mean']:.4f}  gan-cp {gc['ap_mean']:.4f}")
