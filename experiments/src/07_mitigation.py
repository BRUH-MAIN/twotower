# %% [markdown]
# # Experiment 7 — Do the Tier-1 mitigations actually reduce shortcut reliance?
#
# Experiments 1-5 established *what* the shortcut is on this dataset:
#
# * the label is recoverable from lesion-free pixels (68-95% retention), and
# * the annotation box geometry alone is diagnostic (48-189% retention).
#
# This experiment tests two targeted mitigations against exactly those two
# measurements. Crucially the protocol differs from Experiment 3: there we trained
# a separate model per input variant, which asks *"is the label learnable from
# this input?"*. Here we train **one** model on ordinary full frames and then
# ablate at **inference** time, which asks the question that matters for a
# mitigation: *"does this trained model rely on the lesion?"*
#
# | Condition | Training augmentation |
# | --- | --- |
# | `baseline` | horizontal flip only |
# | `scale` | + random rescale of the lesion (0.5-1.6x) |
# | `bgswap` | + lesion pasted onto another patient's lesion-free background |
# | `both` | + both |
#
# **Background swap** decouples the lesion from its own surrounding context: the
# canvas is a lesion-free crop taken from a *different* patient, so context can no
# longer predict the label. **Scale jitter** decouples apparent lesion size from
# class, which is what `boxmap` showed the model exploiting.
#
# Success is a *fall* in retention. A mitigated model should lose more accuracy
# when the lesion is taken away — even if its headline number also drops slightly.
# That is the trade the shortcut literature says to make.

# %%
import sys, json, time, random, gc
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision.models import ResNet18_Weights

from cadsel.data import build_manifest, patient_level_split, read_yolo_label
from harness import DEVICE, IMG, classes_for, encode_labels

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
EPOCHS, BATCH = 15, 32
LESION_TILE = 112
print("device:", DEVICE)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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


def rs(a, size):
    if a is None or a.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR),
                      dtype=np.uint8)


def build(samples, progress_every=750):
    """full224, bg_patch224, lesion tile, and the lesion box in 224-space."""
    n = len(samples)
    full = np.zeros((n, IMG, IMG, 3), dtype=np.uint8)
    bgp = np.zeros((n, IMG, IMG, 3), dtype=np.uint8)
    tile = np.zeros((n, LESION_TILE, LESION_TILE, 3), dtype=np.uint8)
    box224 = np.zeros((n, 4), dtype=np.int32)

    for i, s in enumerate(samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB")
            W, H = im.size
            arr = np.asarray(im, dtype=np.uint8)
        bs = boxes_for(s, W, H)
        full[i] = rs(arr, IMG)

        if bs:
            x1, y1, x2, y2 = max(bs, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            tile[i] = rs(arr[y1:y2, x1:x2], LESION_TILE)
            box224[i] = [int(x1 * IMG / W), int(y1 * IMG / H),
                         int(x2 * IMG / W), int(y2 * IMG / H)]
            free = largest_free_rect(bs, W, H)
            bgp[i] = (rs(arr[free[1]:free[3], free[0]:free[2]], IMG)
                      if free is not None else np.zeros((IMG, IMG, 3), np.uint8))
        else:
            tile[i] = rs(arr, LESION_TILE)
            box224[i] = [0, 0, IMG, IMG]
            bgp[i] = full[i]

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    built {i+1}/{n}")
    return {"full": full, "bg_patch": bgp, "tile": tile, "box": box224}


def boxmap_from(box224):
    n = len(box224)
    out = np.zeros((n, IMG, IMG, 3), dtype=np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(box224):
        out[i, y1:y2, x1:x2] = 255
    return out


# %%
def compose_batch(D, idx, rng, scale_jitter: bool, bg_swap: bool):
    """Build one training batch, applying the enabled mitigations."""
    imgs = D["full"][idx].copy()
    if not (scale_jitter or bg_swap):
        return imgs

    n_pool = len(D["full"])
    for k, i in enumerate(idx):
        x1, y1, x2, y2 = D["box"][i]
        bw, bh = max(x2 - x1, 8), max(y2 - y1, 8)

        if bg_swap:
            # canvas from a different sample's lesion-free region
            j = rng.randrange(n_pool)
            canvas = D["bg_patch"][j].copy()
        else:
            canvas = imgs[k]

        if scale_jitter:
            f = rng.uniform(0.5, 1.6)
            nw, nh = int(np.clip(bw * f, 12, IMG)), int(np.clip(bh * f, 12, IMG))
        else:
            nw, nh = int(np.clip(bw, 12, IMG)), int(np.clip(bh, 12, IMG))

        if not bg_swap and not scale_jitter:
            continue

        patch = np.asarray(
            Image.fromarray(D["tile"][i]).resize((nw, nh), Image.BILINEAR),
            dtype=np.uint8)

        if bg_swap:
            px = rng.randint(0, max(IMG - nw, 0))
            py = rng.randint(0, max(IMG - nh, 0))
        else:
            # keep it roughly where it was, but allow the size to change
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            px = int(np.clip(cx - nw // 2, 0, IMG - nw))
            py = int(np.clip(cy - nh // 2, 0, IMG - nh))
            canvas = canvas.copy()

        canvas[py:py + nh, px:px + nw] = patch
        imgs[k] = canvas
    return imgs


def train_and_probe(D_tr, ytr, D_va, yva, n_classes, condition, tag):
    scale_jitter = condition in ("scale", "both")
    bg_swap = condition in ("bgswap", "both")

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    rng = random.Random(SEED)

    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.to(DEVICE)

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.maximum(counts, 1), 0.0)
    w = w / max(w.sum(), 1e-9) * n_classes
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    mean, std = _MEAN.to(DEVICE), _STD.to(DEVICE)

    def to_gpu(a, flip=False):
        x = torch.from_numpy(np.ascontiguousarray(a)).to(DEVICE, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        if flip:
            x = torch.flip(x, dims=[3])
        return (x - mean) / std

    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)
    n = len(ytr)

    def evaluate(arr):
        model.eval()
        preds = []
        with torch.no_grad():
            for k in range(0, len(yva), 64):
                with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                    preds.append(model(to_gpu(arr[k:k+64])).argmax(1).cpu().numpy())
        p = np.concatenate(preds)
        rec = [float((p[yva == c] == c).mean())
               for c in range(n_classes) if (yva == c).any()]
        return float(np.mean(rec))

    best_bal, best_state = 0.0, None
    for ep in range(EPOCHS):
        model.train()
        order = np.random.permutation(n)
        for k in range(0, n, BATCH):
            idx = np.sort(order[k:k + BATCH])
            batch = compose_batch(D_tr, idx, rng, scale_jitter, bg_swap)
            x = to_gpu(batch, flip=rng.random() < 0.5)
            y = ytr_t[torch.from_numpy(idx).to(DEVICE)]
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        sched.step()

        bal = evaluate(D_va["full"])
        if bal > best_bal:
            best_bal = bal
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    res = {
        "full": round(evaluate(D_va["full"]), 4),
        "bg_patch": round(evaluate(D_va["bg_patch"]), 4),
        "boxmap": round(evaluate(D_va["boxmap"]), 4),
        "chance": round(1 / n_classes, 4),
    }
    denom = res["full"] - res["chance"]
    res["ret_bg_patch"] = round((res["bg_patch"] - res["chance"]) / denom, 3) if denom > 1e-6 else None
    res["ret_boxmap"] = round((res["boxmap"] - res["chance"]) / denom, 3) if denom > 1e-6 else None
    print(f"    {tag:<34} full {res['full']:.3f} | bg_patch {res['bg_patch']:.3f} "
          f"({res['ret_bg_patch']:>6.1%}) | boxmap {res['boxmap']:.3f} "
          f"({res['ret_boxmap']:>6.1%})")
    return res


# %%
RESULTS_PATH = OUT / "07_mitigation.json"
results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
done = {(r["modality"], r["task"], r["condition"]) for r in results}
if done:
    print(f"resuming: {len(done)} run(s) recorded")

CONDITIONS = ("baseline", "scale", "bgswap", "both")

for modality in ("WLE", "EUS"):
    todo = [(t, c) for t in ("binary", "subtype") for c in CONDITIONS
            if (modality, t, c) not in done]
    if not todo:
        print(f"\n=== {modality}: already complete ===")
        continue

    samples = build_manifest(modality)
    train_s, val_s = patient_level_split(samples, 0.2, SEED)
    print(f"\n=== {modality}: {len(train_s)} train / {len(val_s)} val "
          f"({len(todo)} run(s) to do) ===")

    print("  building train arrays ...")
    D_tr = build(train_s)
    print("  building val arrays ...")
    D_va = build(val_s, progress_every=0)
    D_va["boxmap"] = boxmap_from(D_va["box"])

    for task in ("binary", "subtype"):
        classes = classes_for(task)
        ytr, yva = encode_labels(train_s, task), encode_labels(val_s, task)
        pending = [c for c in CONDITIONS if (modality, task, c) not in done]
        if not pending:
            continue
        print(f"  -- {task} --")
        for cond in pending:
            t0 = time.time()
            r = train_and_probe(D_tr, ytr, D_va, yva, len(classes), cond,
                                f"{modality}/{task}/{cond}")
            r.update({"modality": modality, "task": task, "condition": cond,
                      "minutes": round((time.time() - t0) / 60, 1)})
            results.append(r)
            done.add((modality, task, cond))
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    del D_tr, D_va
    gc.collect()

# %% [markdown]
# ## 7.1 Verdict
#
# Read down the `bg_patch` retention column. A mitigation works if that number
# **falls** relative to `baseline` — the model has become more dependent on the
# lesion. A modest drop in the `full` column alongside it is the expected and
# acceptable cost.

# %%
by = {(r["modality"], r["task"], r["condition"]): r for r in results}
print("=" * 92)
print("MITIGATION EFFECT — balanced accuracy, and retention when the lesion is removed")
print("=" * 92)
hdr = (f"{'mod':<5}{'task':<9}{'condition':<11}{'full':>8}{'bg_patch':>11}"
       f"{'ret_bg':>9}{'boxmap':>9}{'ret_box':>9}")
print(hdr); print("-" * len(hdr))

summary = []
for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        base = by.get((modality, task, "baseline"))
        if base is None:
            continue
        for cond in CONDITIONS:
            r = by.get((modality, task, cond))
            if r is None:
                continue
            print(f"{modality:<5}{task:<9}{cond:<11}{r['full']:>8.3f}"
                  f"{r['bg_patch']:>11.3f}{r['ret_bg_patch']:>9.1%}"
                  f"{r['boxmap']:>9.3f}{r['ret_boxmap']:>9.1%}")
            if cond != "baseline":
                summary.append({
                    "modality": modality, "task": task, "condition": cond,
                    "d_full": round(r["full"] - base["full"], 4),
                    "d_ret_bg": round(r["ret_bg_patch"] - base["ret_bg_patch"], 3),
                    "d_ret_box": round(r["ret_boxmap"] - base["ret_boxmap"], 3),
                })
        print()

json.dump({"runs": results, "summary": summary},
          open(OUT / "07_mitigation_verdict.json", "w"), indent=2)

print("Change vs baseline (negative retention change = less shortcut reliance)")
print(f"{'mod':<5}{'task':<9}{'condition':<11}{'d_full':>9}{'d_ret_bg':>10}{'d_ret_box':>11}")
print("-" * 55)
for s in summary:
    print(f"{s['modality']:<5}{s['task']:<9}{s['condition']:<11}"
          f"{s['d_full']:>+9.3f}{s['d_ret_bg']:>+10.1%}{s['d_ret_box']:>+11.1%}")

if summary:
    best = min(summary, key=lambda s: s["d_ret_bg"])
    print(f"\nLargest reduction in background reliance: {best['condition']} on "
          f"{best['modality']}/{best['task']} ({best['d_ret_bg']:+.1%} retention, "
          f"{best['d_full']:+.3f} balanced accuracy)")
