# %% [markdown]
# # Experiment 12 — Adversarial discriminator (Gradient Reversal)
#
# Experiments 1-5 established *what* the shortcut is on this dataset: the label
# is recoverable from lesion-free pixels (68-95% retention) and from annotation
# box geometry alone (48-189% retention). Experiment 7 showed two hand-designed
# augmentations can reduce reliance on WLE but backfire on EUS.
#
# This experiment tries a different mechanism: instead of hoping an
# augmentation decorrelates the shortcut, explicitly train the backbone to be
# *unable* to predict the label from shortcut-only pixels. A shared ResNet-18
# produces embeddings for both the full image (main head, ordinary
# cross-entropy) and that same sample's `bg_patch` crop (a small discriminator
# head, behind a Gradient Reversal Layer, trying to predict the *true* label
# from the crop alone). Reversed gradients push the backbone toward
# representations from which the discriminator cannot predict the label from
# background-only pixels — this directly minimizes the exact quantity
# Experiment 3 calls `ret_bg_patch`.
#
# Protocol matches Experiment 7, not Experiment 3: **one** model is trained,
# then ablated at inference on `full`/`bg_patch`/`boxmap` — this asks "does
# this trained model rely on the lesion", the question that matters for a
# mitigation, not "is the label learnable from this input".
#
# `bg_patch` is the primary discriminator target. `boxmap` (a synthetic
# near-black canvas) is secondary/ablatable — it is badly out-of-distribution
# for a natural-image backbone, so adversarializing against it risks the
# backbone learning to detect and special-case an obviously synthetic input
# rather than losing geometry sensitivity on real images. See
# `experiments/adversarial.py` for the full design rationale.

# %%
import sys, json, time, gc
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np

from cadsel.data import build_manifest, patient_level_split
from ablation_inputs import build_full_bgpatch_tile_box, boxmap_from
from adversarial import train_eval_adversarial
from harness import DEVICE, classes_for, encode_labels

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT / "12_adversarial.json"
SEED = 42
SEEDS = (42, 7, 123)
EPOCHS, BATCH = 15, 32
print("device:", DEVICE)

# %%
store = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {"runs": [], "baseline_noise_floor": {}}
results = store["runs"]
done = {(r["modality"], r["task"], r["condition"], r["seed"]) for r in results}
if done:
    print(f"resuming: {len(done)} run(s) already recorded")


def save():
    RESULTS_PATH.write_text(json.dumps(store, indent=2))


def run_one(D_tr, ytr, D_va, yva, n_classes, modality, task, condition, seed, **kwargs):
    key = (modality, task, condition, seed)
    if key in done:
        return
    t0 = time.time()
    r = train_eval_adversarial(
        D_tr, ytr, D_va, yva, n_classes, seed=seed, epochs=EPOCHS, batch=BATCH,
        tag=f"{modality}/{task}/{condition}/s{seed}", **kwargs,
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
    # "boxmap" is a valid discriminator target (see CONDITIONS below), so the
    # training split needs its own boxmap images too, not just the validation
    # split's (which exists only for the full/bg_patch/boxmap ablation eval).
    D_tr["boxmap"] = boxmap_from(D_tr["box"])
    D_va["boxmap"] = boxmap_from(D_va["box"])
    return D_tr, D_va, train_s, val_s

# %% [markdown]
# ## 12.1 Hyperparameter pre-sweep
#
# Cheap first pass: fix WLE/subtype/seed 42, sweep `lambda_max` and the
# discriminator target set. Pick the setting that most reduces `ret_bg_patch`
# while keeping `Δfull > -0.10` (a mitigation that destroys headline accuracy
# is not useful even if it technically reduces reliance).

# %%
PRESWEEP_LAMBDAS = (0.3, 1.0, 3.0)
PRESWEEP_TARGET_SETS = [("bg_patch",), ("bg_patch", "boxmap")]

D_tr_wle, D_va_wle, _train_wle, _val_wle = build_split("WLE")

n_classes_sub = len(classes_for("subtype"))
ytr_sub = encode_labels(_train_wle, "subtype")
yva_sub = encode_labels(_val_wle, "subtype")

run_one(D_tr_wle, ytr_sub, D_va_wle, yva_sub, n_classes_sub, "WLE", "subtype", "baseline", SEED, targets=())
for lam in PRESWEEP_LAMBDAS:
    for targets in PRESWEEP_TARGET_SETS:
        cond = f"presweep_l{lam}_{'-'.join(targets)}"
        run_one(D_tr_wle, ytr_sub, D_va_wle, yva_sub, n_classes_sub, "WLE", "subtype", cond, SEED,
                targets=targets, lambda_max=lam)

# %%
by_presweep = {(r["modality"], r["task"], r["condition"], r["seed"]): r for r in results}
baseline_wle_sub = by_presweep[("WLE", "subtype", "baseline", SEED)]

candidates = []
for lam in PRESWEEP_LAMBDAS:
    for targets in PRESWEEP_TARGET_SETS:
        cond = f"presweep_l{lam}_{'-'.join(targets)}"
        r = by_presweep.get(("WLE", "subtype", cond, SEED))
        if r is None:
            continue
        d_full = r["full"] - baseline_wle_sub["full"]
        candidates.append((lam, targets, r["ret_bg_patch"], d_full))

print(f"{'lambda':>7}{'targets':>22}{'ret_bg_patch':>14}{'d_full':>9}")
for lam, targets, ret, d_full in candidates:
    ok = "" if d_full > -0.10 else "  (rejected: full acc dropped too much)"
    print(f"{lam:>7}{'+'.join(targets):>22}{ret:>14.1%}{d_full:>+9.3f}{ok}")

feasible = [c for c in candidates if c[3] > -0.10]
pool = feasible or candidates
CHOSEN_LAMBDA_MAX, CHOSEN_TARGETS, _, _ = min(pool, key=lambda c: c[2])
print(f"\nchosen: lambda_max={CHOSEN_LAMBDA_MAX}, targets={CHOSEN_TARGETS}")

# Persisted so Experiment 14 can compose channel attention with these same
# hyperparameters instead of re-running its own pre-sweep.
store["chosen_hyperparameters"] = {"lambda_max": CHOSEN_LAMBDA_MAX, "targets": list(CHOSEN_TARGETS)}
save()

del D_tr_wle
gc.collect()

# %% [markdown]
# ## 12.2 Confirmation grid
#
# Chosen hyperparameters × `{WLE,EUS}` × `{binary,subtype}` ×
# `{baseline, adv_bgpatch, adv_bgpatch_boxmap}` × 3 seeds. `baseline` reuses
# whatever was already recorded by the pre-sweep for WLE/subtype/seed 42.

# %%
CONDITIONS = {
    "baseline": {"targets": ()},
    "adv_bgpatch": {"targets": ("bg_patch",), "lambda_max": CHOSEN_LAMBDA_MAX},
    "adv_bgpatch_boxmap": {"targets": ("bg_patch", "boxmap"), "lambda_max": CHOSEN_LAMBDA_MAX},
}

for modality in ("WLE", "EUS"):
    todo = [(t, c, s) for t in ("binary", "subtype") for c in CONDITIONS for s in SEEDS
            if (modality, t, c, s) not in done]
    if not todo:
        print(f"\n=== {modality}: already complete ===")
        continue

    D_tr, D_va, train_s, val_s = build_split(modality)
    print(f"\n=== {modality}: {len(todo)} run(s) to do ===")

    for task in ("binary", "subtype"):
        classes = classes_for(task)
        n_classes = len(classes)
        ytr = encode_labels(train_s, task)
        yva = encode_labels(val_s, task)
        for cond, kwargs in CONDITIONS.items():
            for seed in SEEDS:
                if (modality, task, cond, seed) in done:
                    continue
                run_one(D_tr, ytr, D_va, yva, n_classes, modality, task, cond, seed, **kwargs)

    del D_tr, D_va
    gc.collect()

# %% [markdown]
# ## 12.3 Verdict
#
# Per (modality, task, condition): mean±std of `full`, `ret_bg_patch`,
# `ret_boxmap` over the 3 seeds. `baseline`'s own 3-seed spread is *this
# experiment's* noise floor (Experiment 11's numeric floor was measured on the
# detector pipeline on different metrics and does not transfer numerically —
# only its decision-rule form does: a change is real only if it exceeds
# roughly 2x the pooled standard deviation). The floor is persisted so
# Experiments 13 and 14 can reuse it instead of recomputing their own baseline.

# %%
def pooled_sd(sd_a: float, sd_b: float) -> float:
    return float(np.sqrt((sd_a ** 2 + sd_b ** 2) / 2))


by = {}
for r in results:
    by.setdefault((r["modality"], r["task"], r["condition"]), []).append(r)

summary_rows = []
print("=" * 100)
print("ADVERSARIAL DISCRIMINATOR — mean±std over 3 seeds")
print("=" * 100)
hdr = f"{'mod':<5}{'task':<9}{'condition':<20}{'full':>16}{'ret_bg_patch':>18}{'ret_boxmap':>16}"
print(hdr); print("-" * len(hdr))

for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        base_rows = by.get((modality, task, "baseline"), [])
        if len(base_rows) < len(SEEDS):
            continue
        base_full = np.array([r["full"] for r in base_rows])
        base_ret = np.array([r["ret_bg_patch"] for r in base_rows])
        base_ret_box = np.array([r["ret_boxmap"] for r in base_rows])
        floor_key = f"{modality}/{task}"
        store["baseline_noise_floor"][floor_key] = {
            "full_sd": round(float(base_full.std()), 4),
            "ret_bg_patch_sd": round(float(base_ret.std()), 4),
            "ret_boxmap_sd": round(float(base_ret_box.std()), 4),
        }

        for cond in CONDITIONS:
            rows = by.get((modality, task, cond), [])
            if len(rows) < len(SEEDS):
                continue
            full = np.array([r["full"] for r in rows])
            ret = np.array([r["ret_bg_patch"] for r in rows])
            ret_box = np.array([r["ret_boxmap"] for r in rows])
            print(f"{modality:<5}{task:<9}{cond:<20}"
                  f"{full.mean():>8.3f}±{full.std():<7.3f}"
                  f"{ret.mean():>10.1%}±{ret.std():<7.1%}"
                  f"{ret_box.mean():>8.1%}±{ret_box.std():<7.1%}")

            if cond == "baseline":
                continue
            d_ret = float(ret.mean() - base_ret.mean())
            d_ret_box = float(ret_box.mean() - base_ret_box.mean())
            sd = pooled_sd(float(base_ret.std()), float(ret.std()))
            real = abs(d_ret) > 2 * sd if sd > 1e-9 else abs(d_ret) > 0
            summary_rows.append({
                "modality": modality, "task": task, "condition": cond,
                "d_full": round(float(full.mean() - base_full.mean()), 4),
                "d_ret_bg_patch": round(d_ret, 3),
                "d_ret_boxmap": round(d_ret_box, 3),
                "pooled_sd": round(sd, 4),
                "real_effect": bool(real),
            })
        print()

store["summary"] = summary_rows
save()

print("Change vs baseline (negative d_ret_bg_patch = less shortcut reliance); "
      "real_effect = |d_ret_bg_patch| > 2x pooled sd")
print(f"{'mod':<5}{'task':<9}{'condition':<20}{'d_full':>9}{'d_ret_bg':>10}{'real?':>7}")
for s in summary_rows:
    print(f"{s['modality']:<5}{s['task']:<9}{s['condition']:<20}"
          f"{s['d_full']:>+9.3f}{s['d_ret_bg_patch']:>+10.1%}{str(s['real_effect']):>7}")

real_wins = [s for s in summary_rows if s["real_effect"] and s["d_ret_bg_patch"] < 0]
if real_wins:
    best = min(real_wins, key=lambda s: s["d_ret_bg_patch"])
    print(f"\nLargest confirmed reduction: {best['condition']} on "
          f"{best['modality']}/{best['task']} ({best['d_ret_bg_patch']:+.1%} retention, "
          f"{best['d_full']:+.3f} balanced accuracy)")
else:
    print("\nNo condition cleared the noise floor.")
