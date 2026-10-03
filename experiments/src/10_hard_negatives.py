# %% [markdown]
# # Experiment 10 — Hard-negative training
#
# Experiment 8 measured detectors firing on lesion-free crops at roughly 0.3x
# their rate on full frames, and Experiment 3 showed the label is largely
# recoverable from such crops. Nothing in the standard objective penalises that,
# because **every** training image contains a lesion — the model is never shown a
# frame and told "there is nothing here".
#
# The review calls for exactly this (§7.1, §9.4): mine hard negatives and add them
# to training. Implementation: a fraction of training images is duplicated as a
# *negative*, served as the largest lesion-free rectangle with an **empty**
# target. torchvision's RPN and ROI heads treat every anchor in such an image as
# background, which is the missing supervision.
#
# Scored on the Experiment 8 metric — the hallucination ratio — plus AP@0.5, so we
# can see both whether it works and what it costs.

# %%
import sys, json, subprocess, time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable

ARMS = [
    # (run_name, task, extra args)
    ("hn00__WLE__multiclass", "multiclass", []),
    ("hn30__WLE__multiclass", "multiclass", ["--hard-negatives", "0.3"]),
    ("hn30__WLE__binary", "binary", ["--hard-negatives", "0.3"]),
]
BASE = [
    PY, "-u", "-m", "cadsel.train",
    "--model", "fasterrcnn_mobilenet_320", "--modality", "WLE",
    "--epochs", "100", "--early-stop-patience", "15",
    "--num-workers", "3", "--no-progress",
]

# %%
# The `hn00` arm is a no-op rerun of the baseline config; it exists only so the
# comparison is against an identically-seeded run rather than an older one.
for run_name, task, extra in ARMS:
    result = Path("runs") / run_name / "result.json"
    if result.exists():
        print(f"{run_name}: already trained, skipping")
        continue
    cmd = BASE + ["--task", task, "--run-name", run_name] + extra
    print(f"\n=== {run_name} ===")
    print("  " + " ".join(cmd[3:]))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
    tail = [ln for ln in r.stdout.splitlines()
            if "best epoch" in ln or "early stop" in ln or "hard negatives" in ln]
    for ln in tail:
        print("  " + ln.strip())
    if r.returncode != 0:
        print("  FAILED\n", r.stdout[-2000:], r.stderr[-2000:])
    print(f"  {(time.time()-t0)/60:.1f} min")

# %% [markdown]
# ## 10.1 Reliance measurement
#
# Same protocol as Experiment 8: detections above confidence 0.5 on lesion-free
# crops, relative to full frames.

# %%
import numpy as np
import torch
from PIL import Image

from cadsel.data import build_manifest, make_normalizer, patient_level_split, read_yolo_label
from cadsel.models import build_detection_model, configure_transform_for_input_mode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, CONF = 42, 0.5


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
    c = [(0, 0, W, y1), (0, y2, W, H), (0, 0, x1, H), (x2, 0, W, H)]
    c = [q for q in c if q[2] - q[0] >= 24 and q[3] - q[1] >= 24]
    return max(c, key=lambda q: (q[2]-q[0]) * (q[3]-q[1])) if c else None


def make_inputs(samples, img_size):
    full = np.zeros((len(samples), img_size, img_size, 3), dtype=np.uint8)
    bgp = np.zeros_like(full)
    ok = np.zeros(len(samples), dtype=bool)
    for i, s in enumerate(samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB"); W, H = im.size
            arr = np.asarray(im, dtype=np.uint8)
        full[i] = np.asarray(Image.fromarray(arr).resize((img_size, img_size),
                                                         Image.BILINEAR))
        free = largest_free_rect(boxes_for(s, W, H), W, H)
        if free is not None:
            x1, y1, x2, y2 = free
            bgp[i] = np.asarray(Image.fromarray(arr[y1:y2, x1:x2])
                                .resize((img_size, img_size), Image.BILINEAR))
            ok[i] = True
    return full, bgp, ok


@torch.no_grad()
def fire_rate(model, arr, normalize, batch=16):
    counts, tops = [], []
    for k in range(0, len(arr), batch):
        chunk = torch.from_numpy(arr[k:k + batch]).to(DEVICE).permute(0, 3, 1, 2)
        for out in model(normalize(list(chunk))):
            s = out["scores"].float().cpu().numpy()
            counts.append(int((s > CONF).sum()))
            tops.append(float(s.max()) if len(s) else 0.0)
    return float(np.mean(counts)), float(np.mean(tops))


COMPARE = [
    ("multiclass", "hn00__WLE__multiclass", "hn30__WLE__multiclass"),
    ("binary", "fasterrcnn_mobilenet_320__WLE__binary", "hn30__WLE__binary"),
]

results = []
for task, base_run, hn_run in COMPARE:
    for arm, run_name in (("baseline", base_run), ("hard-neg", hn_run)):
        ck_path = Path("runs") / run_name / "best_model.pth"
        if not ck_path.exists():
            print(f"missing {ck_path}"); continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        img_size, input_mode = ck["img_size"], ck.get("input_mode", "reference")
        model = build_detection_model(ck["model_type"], task, pretrained=False)
        configure_transform_for_input_mode(model, input_mode, img_size)
        model.load_state_dict(ck["model_state_dict"])
        model.to(DEVICE).eval()

        _, val_s = patient_level_split(build_manifest("WLE"), 0.2, SEED)
        full, bgp, ok = make_inputs(val_s, img_size)
        n_full, _ = fire_rate(model, full, make_normalizer(input_mode))
        n_bg, s_bg = fire_rate(model, bgp[ok], make_normalizer(input_mode))

        r = {"task": task, "arm": arm, "run": run_name,
             "AP50": round(float(ck["metrics"]["AP@0.5"]), 4),
             "f1": round(float(ck["metrics"]["f1"]), 4),
             "det_full": round(n_full, 3), "det_free": round(n_bg, 3),
             "ratio": round(n_bg / n_full, 3) if n_full > 0 else None,
             "score_free": round(s_bg, 3)}
        results.append(r)
        print(f"{task}/{arm:<9} AP {r['AP50']:.4f} F1 {r['f1']:.4f} | "
              f"det/full {n_full:.2f} det/free {n_bg:.2f} ratio {r['ratio']}")
        del model
        torch.cuda.empty_cache()

json.dump(results, open(OUT / "10_hard_negatives.json", "w"), indent=2)

# %% [markdown]
# ## 10.2 Verdict

# %%
by = {(r["task"], r["arm"]): r for r in results}
print("=" * 84)
print("HARD NEGATIVES")
print("=" * 84)
hdr = f"{'task':<12}{'arm':<11}{'AP@0.5':>9}{'F1':>9}{'det/full':>10}{'det/free':>10}{'ratio':>8}"
print(hdr); print("-" * len(hdr))
for task, _b, _h in COMPARE:
    for arm in ("baseline", "hard-neg"):
        r = by.get((task, arm))
        if r:
            print(f"{task:<12}{arm:<11}{r['AP50']:>9.4f}{r['f1']:>9.4f}"
                  f"{r['det_full']:>10.2f}{r['det_free']:>10.2f}{r['ratio']:>8.2f}")
    print()

for task, _b, _h in COMPARE:
    b, h = by.get((task, "baseline")), by.get((task, "hard-neg"))
    if not (b and h):
        continue
    d_ap, d_f1 = h["AP50"] - b["AP50"], h["f1"] - b["f1"]
    d_ratio = h["ratio"] - b["ratio"]
    msg = ("fewer false alarms on lesion-free tissue" if d_ratio < -0.05
           else "more false alarms" if d_ratio > 0.05 else "unchanged")
    print(f"  {task:<12} AP {d_ap:+.4f}  F1 {d_f1:+.4f}  ratio {d_ratio:+.2f}  -> {msg}")
