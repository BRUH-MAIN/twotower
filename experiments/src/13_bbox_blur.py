# %% [markdown]
# # Experiment 13 — Blur non-lesion pixels during training
#
# Experiment 3's own `boxmap` result (48-189% retention) proves box geometry
# *alone* — the sharp shape/position of the annotated region — is diagnostic.
# So a naive "blur everything outside the box" augmentation risks recreating
# the exact trap Experiment 3 already found for `bg_mask`: a hard blur/sharp
# boundary whose shape and position still encode the annotation, teaching the
# model to read the boundary instead of losing interest in non-lesion pixels.
#
# `experiments/bbox_blur.py` guards against this with a heavily feathered mask
# (no hard edge), generous padding before any jitter (the lesion is never
# adjacent to the transition), and independent per-sample/per-epoch
# randomisation of on/off, strength, padding and jitter. Full design rationale
# is in that module's docstring.
#
# Protocol matches Experiment 7/12: **one** model is trained (plain
# cross-entropy — no discriminator here, just the augmentation), then ablated
# at inference on the standard *unblurred* `full`/`bg_patch`/`boxmap` inputs.
# The verdict below explicitly re-checks `ret_boxmap`: if it *rises* beyond
# the noise floor, that is direct evidence of the predicted failure mode and
# the setting should be rejected, not adopted.

# %%
import sys, json, time, gc
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np

from cadsel.data import build_manifest, patient_level_split
from ablation_inputs import build_full_bgpatch_tile_box, boxmap_from
from adversarial import train_eval_adversarial
from bbox_blur import make_blur_compose_fn
from harness import DEVICE, classes_for, encode_labels

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT / "13_bbox_blur.json"
SEED = 42
SEEDS = (42, 7, 123)
EPOCHS, BATCH = 15, 32
print("device:", DEVICE)

# %%
# Reuse Experiment 12's baseline rows instead of retraining them: "baseline"
# there is the identical procedure (plain CE ResNet-18, no discriminator, no
# blur) at the same seeds, so its numbers *are* this experiment's baseline.
ADV_RESULTS_PATH = OUT / "12_adversarial.json"
if not ADV_RESULTS_PATH.exists():
    raise SystemExit(
        "Experiment 12 must run first -- its baseline rows and noise floor "
        "are reused here rather than recomputed."
    )
adv_store = json.loads(ADV_RESULTS_PATH.read_text())
baseline_rows = {
    (r["modality"], r["task"], r["seed"]): r
    for r in adv_store["runs"] if r["condition"] == "baseline"
}
print(f"reusing {len(baseline_rows)} baseline row(s) and noise floor from Experiment 12: "
      f"{adv_store.get('baseline_noise_floor', {})}")

# %%
store = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {"runs": []}
results = store["runs"]
done = {(r["modality"], r["task"], r["condition"], r["seed"]) for r in results}
if done:
    print(f"resuming: {len(done)} run(s) already recorded")


def save():
    RESULTS_PATH.write_text(json.dumps(store, indent=2))


def run_one(D_tr, ytr, D_va, yva, n_classes, modality, task, condition, seed, compose_fn):
    key = (modality, task, condition, seed)
    if key in done:
        return
    t0 = time.time()
    r = train_eval_adversarial(
        D_tr, ytr, D_va, yva, n_classes, seed=seed, epochs=EPOCHS, batch=BATCH,
        targets=(), compose_fn=compose_fn,
        tag=f"{modality}/{task}/{condition}/s{seed}",
    )
    r.update({"modality": modality, "task": task, "condition": condition, "seed": seed,
              "minutes": round((time.time() - t0) / 60, 1)})
    results.append(r)
    done.add(key)
    save()


def build_split(modality: str):
    samples = build_manifest(modality)
    train_s, val_s = patient_level_split(samples, 0.2, SEED)
    print(f"  building arrays for {modality}: {len(train_s)} train / {len(val_s)} val ...")
    D_tr = build_full_bgpatch_tile_box(train_s)
    D_va = build_full_bgpatch_tile_box(val_s, progress_every=0)
    D_va["boxmap"] = boxmap_from(D_va["box"])
    return D_tr, D_va, train_s, val_s

# %% [markdown]
# ## 13.1 Presets
#
# `blur_light`/`blur_heavy` differ only in how often the augmentation fires
# and how strong the blur is; the feathering/padding/jitter safety margins are
# identical and non-negotiable in both (see `experiments/bbox_blur.py`).

# %%
COMPOSE_FNS = {
    "blur_light": make_blur_compose_fn(p_apply=0.3, sigma_range=(4.0, 8.0)),
    "blur_heavy": make_blur_compose_fn(p_apply=0.7, sigma_range=(8.0, 16.0)),
}

for modality in ("WLE", "EUS"):
    todo = [(t, c, s) for t in ("binary", "subtype") for c in COMPOSE_FNS for s in SEEDS
            if (modality, t, c, s) not in done]
    if not todo:
        print(f"\n=== {modality}: already complete ===")
        continue

    D_tr, D_va, train_s, val_s = build_split(modality)
    print(f"\n=== {modality}: {len(todo)} run(s) to do ===")

    for task in ("binary", "subtype"):
        n_classes = len(classes_for(task))
        ytr = encode_labels(train_s, task)
        yva = encode_labels(val_s, task)
        for cond, compose_fn in COMPOSE_FNS.items():
            for seed in SEEDS:
                if (modality, task, cond, seed) in done:
                    continue
                run_one(D_tr, ytr, D_va, yva, n_classes, modality, task, cond, seed, compose_fn)

    del D_tr, D_va
    gc.collect()

# %% [markdown]
# ## 13.2 Verdict
#
# `ret_bg_patch` falling below Experiment 12's baseline is the mitigation
# signal. `ret_boxmap` **rising** beyond the noise floor is the specific
# failure mode this experiment was designed to catch — flagged explicitly
# below rather than silently averaged away.

# %%
def pooled_sd(sd_a: float, sd_b: float) -> float:
    return float(np.sqrt((sd_a ** 2 + sd_b ** 2) / 2))


by = {}
for r in results:
    by.setdefault((r["modality"], r["task"], r["condition"]), []).append(r)

summary_rows = []
print("=" * 100)
print("BBOX BLUR — mean±std over 3 seeds, vs Experiment 12's baseline")
print("=" * 100)
hdr = f"{'mod':<5}{'task':<9}{'condition':<12}{'full':>16}{'ret_bg_patch':>18}{'ret_boxmap':>16}"
print(hdr); print("-" * len(hdr))

for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        base_seed_rows = [baseline_rows.get((modality, task, s)) for s in SEEDS]
        if any(r is None for r in base_seed_rows):
            continue
        base_full = np.array([r["full"] for r in base_seed_rows])
        base_ret = np.array([r["ret_bg_patch"] for r in base_seed_rows])
        base_ret_box = np.array([r["ret_boxmap"] for r in base_seed_rows])
        print(f"{modality:<5}{task:<9}{'baseline':<12}"
              f"{base_full.mean():>8.3f}±{base_full.std():<7.3f}"
              f"{base_ret.mean():>10.1%}±{base_ret.std():<7.1%}"
              f"{base_ret_box.mean():>8.1%}±{base_ret_box.std():<7.1%}")

        for cond in COMPOSE_FNS:
            rows = by.get((modality, task, cond), [])
            if len(rows) < len(SEEDS):
                continue
            full = np.array([r["full"] for r in rows])
            ret = np.array([r["ret_bg_patch"] for r in rows])
            ret_box = np.array([r["ret_boxmap"] for r in rows])
            print(f"{modality:<5}{task:<9}{cond:<12}"
                  f"{full.mean():>8.3f}±{full.std():<7.3f}"
                  f"{ret.mean():>10.1%}±{ret.std():<7.1%}"
                  f"{ret_box.mean():>8.1%}±{ret_box.std():<7.1%}")

            d_ret = float(ret.mean() - base_ret.mean())
            d_ret_box = float(ret_box.mean() - base_ret_box.mean())
            sd = pooled_sd(float(base_ret.std()), float(ret.std()))
            sd_box = pooled_sd(float(base_ret_box.std()), float(ret_box.std()))
            real = abs(d_ret) > 2 * sd if sd > 1e-9 else abs(d_ret) > 0
            boundary_leak = d_ret_box > 2 * sd_box if sd_box > 1e-9 else d_ret_box > 0
            summary_rows.append({
                "modality": modality, "task": task, "condition": cond,
                "d_full": round(float(full.mean() - base_full.mean()), 4),
                "d_ret_bg_patch": round(d_ret, 3),
                "d_ret_boxmap": round(d_ret_box, 3),
                "real_effect": bool(real),
                "boundary_leak_warning": bool(boundary_leak),
            })
        print()

store["summary"] = summary_rows
save()

for s in summary_rows:
    flag = (" *** WARNING: model may be reading the blur boundary as geometry ***"
            if s["boundary_leak_warning"] else "")
    print(f"{s['modality']:<5}{s['task']:<9}{s['condition']:<12}"
          f"d_full={s['d_full']:+.3f}  d_ret_bg={s['d_ret_bg_patch']:+.1%}  "
          f"real={s['real_effect']}  d_ret_boxmap={s['d_ret_boxmap']:+.1%}{flag}")

real_wins = [s for s in summary_rows
             if s["real_effect"] and s["d_ret_bg_patch"] < 0 and not s["boundary_leak_warning"]]
if real_wins:
    best = min(real_wins, key=lambda s: s["d_ret_bg_patch"])
    print(f"\nBest clean mitigation: {best['condition']} on {best['modality']}/{best['task']} "
          f"({best['d_ret_bg_patch']:+.1%} retention, {best['d_full']:+.3f} balanced accuracy, "
          f"no boundary-leak warning)")
else:
    print("\nNo condition both cleared the noise floor and avoided the boundary-leak warning.")
