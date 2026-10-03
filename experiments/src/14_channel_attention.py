# %% [markdown]
# # Experiment 14 — Channel attention
#
# Squeeze-and-excite channel gates, inspired by `Channel_Att_Bridge` in
# `/home/rohan/Ultralightunet/models/UltraLight_VM_UNet.py` (lines 115-156) —
# see `experiments/backbones.py`'s docstring for why the cross-scale mixing
# that module uses across a U-Net's 5 simultaneous skip connections doesn't
# carry over to a ResNet's strictly sequential stages, and is dropped here.
#
# **Plain channel attention has no mechanism to suppress shortcut-predictive
# channels specifically.** Cross-entropy optimizes for accuracy, and
# Experiment 3 showed shortcut-only inputs already reach 68-95% of full
# accuracy — an SE gate trained on CE alone has no reason to prefer lesion
# channels over context/geometry channels. So `se_layer4_ce`/`se_all_ce` below
# are run as an **expected-null control**, not a candidate mitigation: they
# should land close to baseline retention. The actually-motivated conditions
# are `se_layer4_adv`/`se_all_adv`, which compose the same SE-augmented
# backbone with Experiment 12's Gradient Reversal signal (reusing its chosen
# `lambda_max`/discriminator targets) — there the reversed gradient gives the
# gate an actual reason to suppress `bg_patch`-predictive channels.

# %%
import sys, json, time, gc
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np

from cadsel.data import build_manifest, patient_level_split
from ablation_inputs import build_full_bgpatch_tile_box, boxmap_from
from adversarial import train_eval_adversarial
from backbones import build_resnet18_with_se
from harness import DEVICE, classes_for, encode_labels

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT / "14_channel_attention.json"
SEED = 42
SEEDS = (42, 7, 123)
EPOCHS, BATCH = 15, 32
print("device:", DEVICE)

# %%
# Reuse Experiment 12's baseline rows (plain ResNet-18, no SE, no
# discriminator, same seeds) and its chosen GRL hyperparameters.
ADV_RESULTS_PATH = OUT / "12_adversarial.json"
if not ADV_RESULTS_PATH.exists():
    raise SystemExit(
        "Experiment 12 must run first -- its baseline rows and chosen "
        "hyperparameters are reused here rather than recomputed."
    )
adv_store = json.loads(ADV_RESULTS_PATH.read_text())
baseline_rows = {
    (r["modality"], r["task"], r["seed"]): r
    for r in adv_store["runs"] if r["condition"] == "baseline"
}
chosen = adv_store.get("chosen_hyperparameters")
if not chosen:
    raise SystemExit("Experiment 12's results are missing 'chosen_hyperparameters' "
                      "-- run its pre-sweep cell to completion first.")
ADV_LAMBDA_MAX = chosen["lambda_max"]
ADV_TARGETS = tuple(chosen["targets"])
print(f"reusing {len(baseline_rows)} baseline row(s); GRL hyperparameters "
      f"lambda_max={ADV_LAMBDA_MAX}, targets={ADV_TARGETS}")

# %%
store = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {"runs": []}
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
    # ADV_TARGETS (reused from Experiment 12) may include "boxmap"; the
    # training split needs its own boxmap images in that case, not just the
    # validation split's (which exists only for the ablation eval).
    D_tr["boxmap"] = boxmap_from(D_tr["box"])
    D_va["boxmap"] = boxmap_from(D_va["box"])
    return D_tr, D_va, train_s, val_s

# %% [markdown]
# ## 14.1 Conditions

# %%
CONDITIONS = {
    "se_layer4_ce": {"targets": (), "backbone_factory": partial(build_resnet18_with_se, stages="layer4")},
    "se_all_ce": {"targets": (), "backbone_factory": partial(build_resnet18_with_se, stages="all")},
    "se_layer4_adv": {"targets": ADV_TARGETS, "lambda_max": ADV_LAMBDA_MAX,
                       "backbone_factory": partial(build_resnet18_with_se, stages="layer4")},
    "se_all_adv": {"targets": ADV_TARGETS, "lambda_max": ADV_LAMBDA_MAX,
                   "backbone_factory": partial(build_resnet18_with_se, stages="all")},
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
        n_classes = len(classes_for(task))
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
# ## 14.2 Verdict
#
# The `*_ce` rows are the expected-null check: they should land close to
# baseline `ret_bg_patch` (a large drop under plain CE would be surprising and
# worth double-checking for a confound before trusting it). The `*_adv` rows
# are compared against Experiment 12's matching condition (same
# `lambda_max`/targets, plain ResNet-18) to isolate what the SE gate adds on
# top of the discriminator signal alone.

# %%
def pooled_sd(sd_a: float, sd_b: float) -> float:
    return float(np.sqrt((sd_a ** 2 + sd_b ** 2) / 2))


by = {}
for r in results:
    by.setdefault((r["modality"], r["task"], r["condition"]), []).append(r)

adv_by = {}
for r in adv_store["runs"]:
    adv_by.setdefault((r["modality"], r["task"], r["condition"]), []).append(r)
adv_condition_for = {"se_layer4_adv": "adv_bgpatch" if ADV_TARGETS == ("bg_patch",) else "adv_bgpatch_boxmap",
                     "se_all_adv": "adv_bgpatch" if ADV_TARGETS == ("bg_patch",) else "adv_bgpatch_boxmap"}

summary_rows = []
print("=" * 100)
print("CHANNEL ATTENTION — mean±std over 3 seeds")
print("=" * 100)
hdr = f"{'mod':<5}{'task':<9}{'condition':<16}{'full':>16}{'ret_bg_patch':>18}{'ret_boxmap':>16}"
print(hdr); print("-" * len(hdr))

for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        base_seed_rows = [baseline_rows.get((modality, task, s)) for s in SEEDS]
        if any(r is None for r in base_seed_rows):
            continue
        base_ret = np.array([r["ret_bg_patch"] for r in base_seed_rows])
        base_full = np.array([r["full"] for r in base_seed_rows])
        print(f"{modality:<5}{task:<9}{'baseline':<16}"
              f"{base_full.mean():>8.3f}±{base_full.std():<7.3f}"
              f"{base_ret.mean():>10.1%}±{np.array([r['ret_bg_patch'] for r in base_seed_rows]).std():<7.1%}"
              f"{np.array([r['ret_boxmap'] for r in base_seed_rows]).mean():>8.1%}±"
              f"{np.array([r['ret_boxmap'] for r in base_seed_rows]).std():<7.1%}")

        for cond in CONDITIONS:
            rows = by.get((modality, task, cond), [])
            if len(rows) < len(SEEDS):
                continue
            full = np.array([r["full"] for r in rows])
            ret = np.array([r["ret_bg_patch"] for r in rows])
            ret_box = np.array([r["ret_boxmap"] for r in rows])
            print(f"{modality:<5}{task:<9}{cond:<16}"
                  f"{full.mean():>8.3f}±{full.std():<7.3f}"
                  f"{ret.mean():>10.1%}±{ret.std():<7.1%}"
                  f"{ret_box.mean():>8.1%}±{ret_box.std():<7.1%}")

            d_ret_vs_baseline = float(ret.mean() - base_ret.mean())
            sd = pooled_sd(float(base_ret.std()), float(ret.std()))
            null_confirmed = abs(d_ret_vs_baseline) <= 2 * sd if sd > 1e-9 else abs(d_ret_vs_baseline) <= 1e-6

            row = {
                "modality": modality, "task": task, "condition": cond,
                "d_full_vs_baseline": round(float(full.mean() - base_full.mean()), 4),
                "d_ret_bg_patch_vs_baseline": round(d_ret_vs_baseline, 3),
            }
            if cond in ("se_layer4_ce", "se_all_ce"):
                row["expected_null_confirmed"] = bool(null_confirmed)
            else:
                adv_rows = adv_by.get((modality, task, adv_condition_for[cond]), [])
                if len(adv_rows) >= len(SEEDS):
                    adv_ret = np.array([r["ret_bg_patch"] for r in adv_rows])
                    row["d_ret_bg_patch_vs_adv_alone"] = round(float(ret.mean() - adv_ret.mean()), 3)
            summary_rows.append(row)
        print()

store["summary"] = summary_rows
save()

print("Summary (vs Experiment 12's plain baseline):")
for s in summary_rows:
    extra = ""
    if "expected_null_confirmed" in s:
        extra = f"  expected_null_confirmed={s['expected_null_confirmed']}"
    elif "d_ret_bg_patch_vs_adv_alone" in s:
        extra = f"  vs_adv_alone={s['d_ret_bg_patch_vs_adv_alone']:+.1%}"
    print(f"{s['modality']:<5}{s['task']:<9}{s['condition']:<16}"
          f"d_full={s['d_full_vs_baseline']:+.3f}  "
          f"d_ret_bg={s['d_ret_bg_patch_vs_baseline']:+.1%}{extra}")
