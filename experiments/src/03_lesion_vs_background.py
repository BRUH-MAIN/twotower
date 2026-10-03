# %% [markdown]
# # Experiment 3 — Lesion-vs-background counterfactual ablation
#
# The review's Section 4.2: *"identify a candidate shortcut region, remove it ...
# and check whether predictions change"*.
#
# Here the *lesion* is the region that ought to matter. The same classifier is
# trained on five inputs built from identical images:
#
# | Variant | Input | Purpose |
# | --- | --- | --- |
# | `full` | whole frame | reference |
# | `lesion` | annotated box, cropped (+10% context) | does the lesion carry the signal? |
# | `bg_mask` | whole frame, boxes filled with mean colour | naive ablation |
# | `bg_patch` | largest lesion-free rectangle, cropped | **clean** ablation |
# | `boxmap` | black canvas, boxes drawn white | **control**: annotation geometry only |
#
# `bg_mask` and `boxmap` exist because of a trap. Filling a box with flat colour
# leaves a visible rectangle whose *size and position still encode the
# annotation* — and Experiment 1 showed lesion size predicts class. So `bg_mask`
# is not a clean lesion removal, and `boxmap` measures exactly how much the
# geometry alone gives away. `bg_patch` carries no such marker: it is ordinary
# mucosa or ultrasound field cropped from a region the annotation never touched.
#
# Classification rather than detection is used deliberately: it isolates "is the
# label predictable" from "is the lesion localisable".

# %%
import sys, json, time, random, gc
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np
from PIL import Image

from cadsel.data import build_manifest, patient_level_split, read_yolo_label
from harness import DEVICE, IMG, classes_for, encode_labels, train_eval

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
VARIANTS = ("full", "lesion", "bg_mask", "bg_patch", "boxmap")
print("device:", DEVICE)

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
    """Biggest strip above/below/left/right of the box union — no lesion, no marker."""
    if not boxes:
        return (0, 0, W, H)
    x1 = min(b[0] for b in boxes); x2 = max(b[2] for b in boxes)
    y1 = min(b[1] for b in boxes); y2 = max(b[3] for b in boxes)
    cands = [
        (0, 0, W, y1),      # above
        (0, y2, W, H),      # below
        (0, 0, x1, H),      # left
        (x2, 0, W, H),      # right
    ]
    cands = [c for c in cands if c[2] - c[0] >= 24 and c[3] - c[1] >= 24]
    if not cands:
        return None
    return max(cands, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))


def resize_u8(a):
    if a is None or a.size == 0:
        return np.zeros((IMG, IMG, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(a).resize((IMG, IMG), Image.BILINEAR),
                      dtype=np.uint8)


def build_variants(samples, progress_every=750):
    n = len(samples)
    store = {v: np.zeros((n, IMG, IMG, 3), dtype=np.uint8) for v in VARIANTS}
    no_free = 0

    for i, s in enumerate(samples):
        with Image.open(s.image) as im:
            im = im.convert("RGB")
            W, H = im.size
            arr = np.asarray(im, dtype=np.uint8)
        boxes = boxes_for(s, W, H)

        store["full"][i] = resize_u8(arr)

        if boxes:
            bx1, by1, bx2, by2 = max(boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            px, py = int(0.1 * (bx2 - bx1)), int(0.1 * (by2 - by1))
            store["lesion"][i] = resize_u8(
                arr[max(0, by1-py):min(H, by2+py), max(0, bx1-px):min(W, bx2+px)])

            masked = arr.copy()
            fill = arr.reshape(-1, 3).mean(0).astype(np.uint8)
            for (a, b, c, d) in boxes:
                masked[b:d, a:c] = fill
            store["bg_mask"][i] = resize_u8(masked)

            free = largest_free_rect(boxes, W, H)
            if free is None:
                no_free += 1
                store["bg_patch"][i] = np.zeros((IMG, IMG, 3), dtype=np.uint8)
            else:
                fx1, fy1, fx2, fy2 = free
                store["bg_patch"][i] = resize_u8(arr[fy1:fy2, fx1:fx2])

            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            for (a, b, c, d) in boxes:
                canvas[b:d, a:c] = 255
            store["boxmap"][i] = resize_u8(canvas)
        else:
            for v in VARIANTS:
                store[v][i] = store["full"][i]

        if progress_every and (i + 1) % progress_every == 0:
            print(f"    decoded {i+1}/{n}")

    if no_free:
        print(f"    note: {no_free}/{n} images had no lesion-free strip >=24px")
    return store


# %%
# Resume: these runs are deterministic (fixed seed), so anything already recorded
# is reused rather than recomputed. Delete 03_ablation.json to force a full rerun.
RESULTS_PATH = OUT / "03_ablation.json"
results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
done = {(r["modality"], r["task"], r["variant"]) for r in results}
if done:
    print(f"resuming: {len(done)} run(s) already recorded")

for modality in ("WLE", "EUS"):
    todo = [(t, v) for t in ("binary", "subtype") for v in VARIANTS
            if (modality, t, v) not in done]
    if not todo:
        print(f"\n=== {modality}: all runs already recorded, skipping ===")
        continue

    samples = build_manifest(modality)
    train_s, val_s = patient_level_split(samples, 0.2, SEED)
    print(f"\n=== {modality}: {len(train_s)} train / {len(val_s)} val images "
          f"({len(todo)} run(s) to do) ===")

    print("  building variants (train) ...")
    Xtr = build_variants(train_s)
    print("  building variants (val) ...")
    Xva = build_variants(val_s, progress_every=0)

    for task in ("binary", "subtype"):
        classes = classes_for(task)
        ytr, yva = encode_labels(train_s, task), encode_labels(val_s, task)
        pending = [v for v in VARIANTS if (modality, task, v) not in done]
        if not pending:
            continue
        print(f"  -- {task} --")
        for variant in pending:
            t0 = time.time()
            r = train_eval(Xtr[variant], ytr, Xva[variant], yva, len(classes),
                           f"{modality}/{task}/{variant}", seed=SEED)
            r.update({"modality": modality, "task": task, "variant": variant,
                      "minutes": round((time.time() - t0) / 60, 1)})
            results.append(r)
            done.add((modality, task, variant))
            RESULTS_PATH.write_text(json.dumps(results, indent=2))

    del Xtr, Xva
    gc.collect()

# %% [markdown]
# ## 3.1 Verdict
#
# Retention measures how much of the full-image signal survives an ablation,
# against the chance floor:
#
# ```
# retention = (ablated - chance) / (full - chance)
# ```
#
# Read `bg_patch` as the honest lesion-removal result, and `boxmap` as the amount
# obtainable from annotation geometry alone. If `boxmap` is high, then `bg_mask`
# is contaminated and should be disregarded.

# %%
by = {(r["modality"], r["task"], r["variant"]): r for r in results}
verdict = []
for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        full = by[(modality, task, "full")]
        chance = full["chance"]
        denom = full["bal"] - chance
        row = {"modality": modality, "task": task, "chance": chance,
               "full": full["bal"]}
        for v in ("lesion", "bg_mask", "bg_patch", "boxmap"):
            row[v] = by[(modality, task, v)]["bal"]
            row[f"ret_{v}"] = (round((row[v] - chance) / denom, 3)
                               if denom > 1e-6 else None)
        verdict.append(row)

json.dump({"runs": results, "verdict": verdict},
          open(OUT / "03_lesion_vs_background.json", "w"), indent=2)

print("=" * 96)
print("COUNTERFACTUAL ABLATION — balanced accuracy (retention vs full in brackets)")
print("=" * 96)
hdr = (f"{'mod':<5}{'task':<9}{'chance':>7}{'full':>7}{'lesion':>16}"
       f"{'bg_mask':>16}{'bg_patch':>16}{'boxmap':>16}")
print(hdr); print("-" * len(hdr))
for v in verdict:
    def cell(name):
        return f"{v[name]:.3f} ({v['ret_'+name]:>5.0%})"
    print(f"{v['modality']:<5}{v['task']:<9}{v['chance']:>7.3f}{v['full']:>7.3f}"
          f"{cell('lesion'):>16}{cell('bg_mask'):>16}"
          f"{cell('bg_patch'):>16}{cell('boxmap'):>16}")

print()
for v in verdict:
    r_patch, r_box = v["ret_bg_patch"], v["ret_boxmap"]
    if r_patch is None:
        continue
    if r_patch > 0.6:
        msg = "SHORTCUT — most signal survives honest lesion removal"
    elif r_patch > 0.3:
        msg = "PARTIAL shortcut — real signal outside the lesion"
    else:
        msg = "largely lesion-dependent"
    print(f"  {v['modality']:<4} {v['task']:<8} bg_patch retains {r_patch:>6.1%}  {msg}")
    if r_box is not None and r_box > 0.5:
        print(f"       (annotation geometry alone retains {r_box:.1%} — "
              f"bg_mask is contaminated by the drawn rectangle)")
