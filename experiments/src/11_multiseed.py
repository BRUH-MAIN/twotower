# %% [markdown]
# # Experiment 11 — Multi-seed A/B
#
# Experiment 8 compared one baseline detector against one mitigated detector and
# got an ambiguous answer (+0.05 reliance on multiclass, −0.09 on binary). With a
# single run per arm there is no way to tell a real effect from run-to-run
# variation — and the review's Section 2 warns precisely about this:
# underspecification means *"which one a given training run lands on is highly
# sensitive to arbitrary choices (initialisation seed ... order of minibatches)"*.
#
# This runs **three seeds per arm** across three conditions so the spread can be
# measured rather than assumed:
#
# * `baseline` — flip/colour/rotate
# * `augment` — + `scale_crop` + `copy_paste`
# * `hardneg` — baseline augmentation + 30% lesion-free negatives
#
# Note the seed controls both the weight initialisation of the new box predictor
# and the augmentation RNG, but **not** the train/val split, which is fixed at
# seed 42 throughout so all arms are scored on identical validation patients.

# %%
import sys, json, subprocess, time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable

SEEDS = [42, 7, 123]
ARMS = {
    "baseline": [],
    "augment": ["--augment", "flip,color,rotate,scale_crop,copy_paste"],
    "hardneg": ["--hard-negatives", "0.3"],
}
TASK = "multiclass"

BASE = [
    PY, "-u", "-m", "cadsel.train",
    "--model", "fasterrcnn_mobilenet_320", "--modality", "WLE", "--task", TASK,
    "--epochs", "100", "--early-stop-patience", "15",
    "--num-workers", "3", "--no-progress",
]

# %%
for arm, extra in ARMS.items():
    for seed in SEEDS:
        run_name = f"ms_{arm}_s{seed}__WLE__{TASK}"
        if (Path("runs") / run_name / "result.json").exists():
            print(f"{run_name}: already trained")
            continue
        cmd = BASE + ["--seed", str(seed), "--run-name", run_name] + extra
        print(f"\n=== {run_name} ===", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path.cwd()))
        for ln in r.stdout.splitlines():
            if "best epoch" in ln or "hard negatives" in ln:
                print("  " + ln.strip())
        if r.returncode != 0:
            print("  FAILED\n", r.stdout[-1500:], r.stderr[-1500:])
        print(f"  {(time.time()-t0)/60:.1f} min", flush=True)

# %% [markdown]
# ## 11.1 Reliance + AP across seeds

# %%
import numpy as np
import torch
from PIL import Image

from cadsel.data import build_manifest, make_normalizer, patient_level_split, read_yolo_label
from cadsel.models import build_detection_model, configure_transform_for_input_mode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF = 0.5


def boxes_for(sample, W, H):
    out = []
    for _c, cx, cy, bw, bh in read_yolo_label(sample.label):
        x1 = int(max(0, (cx - bw/2) * W)); x2 = int(min(W, (cx + bw/2) * W))
        y1 = int(max(0, (cy - bh/2) * H)); y2 = int(min(H, (cy + bh/2) * H))
        if x2 - x1 > 4 and y2 - y1 > 4:
            out.append((x1, y1, x2, y2))
    return out


def largest_free_rect(boxes, W, H):
    if not boxes:
        return (0, 0, W, H)
    x1 = min(b[0] for b in boxes); x2 = max(b[2] for b in boxes)
    y1 = min(b[1] for b in boxes); y2 = max(b[3] for b in boxes)
    c = [(0, 0, W, y1), (0, y2, W, H), (0, 0, x1, H), (x2, 0, W, H)]
    c = [q for q in c if q[2]-q[0] >= 24 and q[3]-q[1] >= 24]
    return max(c, key=lambda q: (q[2]-q[0]) * (q[3]-q[1])) if c else None


_, VAL_S = patient_level_split(build_manifest("WLE"), 0.2, 42)


def make_inputs(img_size):
    full = np.zeros((len(VAL_S), img_size, img_size, 3), dtype=np.uint8)
    bgp = np.zeros_like(full); ok = np.zeros(len(VAL_S), dtype=bool)
    for i, s in enumerate(VAL_S):
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
    counts = []
    for k in range(0, len(arr), batch):
        chunk = torch.from_numpy(arr[k:k+batch]).to(DEVICE).permute(0, 3, 1, 2)
        for out in model(normalize(list(chunk))):
            s = out["scores"].float().cpu().numpy()
            counts.append(int((s > CONF).sum()))
    return float(np.mean(counts))


CACHE = {}
rows = []
for arm in ARMS:
    for seed in SEEDS:
        run_name = f"ms_{arm}_s{seed}__WLE__{TASK}"
        ck_path = Path("runs") / run_name / "best_model.pth"
        if not ck_path.exists():
            print(f"missing {run_name}"); continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        img_size, input_mode = ck["img_size"], ck.get("input_mode", "reference")
        if img_size not in CACHE:
            CACHE[img_size] = make_inputs(img_size)
        full, bgp, ok = CACHE[img_size]

        model = build_detection_model(ck["model_type"], TASK, pretrained=False)
        configure_transform_for_input_mode(model, input_mode, img_size)
        model.load_state_dict(ck["model_state_dict"])
        model.to(DEVICE).eval()
        norm = make_normalizer(input_mode)

        n_full = fire_rate(model, full, norm)
        n_free = fire_rate(model, bgp[ok], norm)
        rows.append({
            "arm": arm, "seed": seed,
            "AP50": round(float(ck["metrics"]["AP@0.5"]), 4),
            "f1": round(float(ck["metrics"]["f1"]), 4),
            "ratio": round(n_free / n_full, 4) if n_full > 0 else None,
            "best_epoch": ck["epoch"],
        })
        print(f"{arm:<9} seed {seed:<4} AP {rows[-1]['AP50']:.4f} "
              f"F1 {rows[-1]['f1']:.4f} ratio {rows[-1]['ratio']}")
        del model
        torch.cuda.empty_cache()

json.dump(rows, open(OUT / "11_multiseed.json", "w"), indent=2)

# %% [markdown]
# ## 11.2 Verdict
#
# With three seeds we can report mean +- spread, and ask whether an arm's effect
# exceeds the seed-to-seed variation of the baseline.

# %%
def agg(arm, key):
    vals = [r[key] for r in rows if r["arm"] == arm and r[key] is not None]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (None, None, 0)


print("=" * 78)
print(f"MULTI-SEED ({len(SEEDS)} seeds) — WLE / {TASK}")
print("=" * 78)
hdr = f"{'arm':<11}{'AP@0.5':>18}{'F1':>18}{'lesion-free ratio':>22}"
print(hdr); print("-" * len(hdr))
for arm in ARMS:
    ap_m, ap_s, n = agg(arm, "AP50")
    f1_m, f1_s, _ = agg(arm, "f1")
    r_m, r_s, _ = agg(arm, "ratio")
    if n == 0:
        continue
    print(f"{arm:<11}{ap_m:>10.4f} ±{ap_s:<6.4f}{f1_m:>10.4f} ±{f1_s:<6.4f}"
          f"{r_m:>14.4f} ±{r_s:<6.4f}")

base_ap, base_ap_sd, _ = agg("baseline", "AP50")
base_r, base_r_sd, _ = agg("baseline", "ratio")
print(f"\nBaseline seed-to-seed spread: AP ±{base_ap_sd:.4f}, ratio ±{base_r_sd:.4f}")
print("An arm only counts as a real effect if its shift exceeds that spread.\n")

for arm in ARMS:
    if arm == "baseline":
        continue
    ap_m, ap_s, n = agg(arm, "AP50")
    r_m, r_s, _ = agg(arm, "ratio")
    if n == 0:
        continue
    d_ap, d_r = ap_m - base_ap, r_m - base_r
    pooled = float(np.hypot(base_r_sd, r_s)) or 1e-9
    verdict = ("REAL reduction in false alarms" if d_r < -2 * pooled
               else "within seed noise")
    print(f"  {arm:<9} dAP {d_ap:+.4f} | d_ratio {d_r:+.4f} "
          f"(pooled sd {pooled:.4f})  -> {verdict}")
