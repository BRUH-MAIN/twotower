# %% [markdown]
# # Experiment 15 — Does `se_all_adv` actually clear the noise floor?
#
# Experiments 12-14 (3 seeds each) found that channel attention composed with
# the adversarial discriminator (`se_all_adv`: SE blocks on every ResNet stage,
# trained with Experiment 12's chosen Gradient Reversal hyperparameters)
# reduced `ret_bg_patch` versus plain baseline in **all 8 of 8**
# modality/task cells — the only fully consistent direction of any of the
# three methods, and a clean dose-response (`se_all_adv` beat `se_layer4_adv`
# in all 4 modality/task combos). But no individual cell cleared its own
# 3-seed noise floor, which is wide (`ret_bg_patch` sigma 0.15-0.38 at n=3).
#
# This experiment adds 4 more seeds (7 total: the original 3 plus these) to
# exactly two conditions — `baseline` (Experiment 12) and `se_all_adv`
# (Experiment 14) — for all 4 modality/task combos, to test whether the
# consistent direction survives more data or was a lucky 3-seed draw. Only
# these two conditions are extended, not the full Experiment 12/14 grids, to
# keep this targeted rather than re-running everything.
#
# New rows are appended to the *same* results files
# (`experiments/results/12_adversarial.json`'s `baseline` rows and
# `experiments/results/14_channel_attention.json`'s `se_all_adv` rows) so
# every consumer of those files sees the extended seed set. Neither file's
# original 3-seed `summary` is touched or recomputed here (those conditions
# aren't being re-run); this experiment's own pooled verdict is written to a
# new `summary_multiseed_se_all_adv` key in `14_channel_attention.json`.

# %%
import sys, json, time
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
ADV_PATH = OUT / "12_adversarial.json"
SE_PATH = OUT / "14_channel_attention.json"
SEED = 42
NEW_SEEDS = (1, 2, 3, 4)
EPOCHS, BATCH = 15, 32
print("device:", DEVICE)

# %%
if not ADV_PATH.exists() or not SE_PATH.exists():
    raise SystemExit("Experiments 12 and 14 must run to completion first.")

adv_store = json.loads(ADV_PATH.read_text())
se_store = json.loads(SE_PATH.read_text())
chosen = adv_store["chosen_hyperparameters"]
ADV_LAMBDA_MAX, ADV_TARGETS = chosen["lambda_max"], tuple(chosen["targets"])
print(f"extending 'baseline' (12_adversarial.json) and 'se_all_adv' "
      f"(14_channel_attention.json) with seeds {NEW_SEEDS}; "
      f"lambda_max={ADV_LAMBDA_MAX}, targets={ADV_TARGETS}")

adv_runs = adv_store["runs"]
se_runs = se_store["runs"]
adv_done = {(r["modality"], r["task"], r["condition"], r["seed"]) for r in adv_runs}
se_done = {(r["modality"], r["task"], r["condition"], r["seed"]) for r in se_runs}


def save_adv():
    ADV_PATH.write_text(json.dumps(adv_store, indent=2))


def save_se():
    SE_PATH.write_text(json.dumps(se_store, indent=2))


def build_split(modality: str):
    samples = build_manifest(modality)
    train_s, val_s = patient_level_split(samples, 0.2, SEED)
    print(f"  building arrays for {modality}: {len(train_s)} train / {len(val_s)} val ...")
    D_tr = build_full_bgpatch_tile_box(train_s)
    D_va = build_full_bgpatch_tile_box(val_s, progress_every=0)
    D_tr["boxmap"] = boxmap_from(D_tr["box"])
    D_va["boxmap"] = boxmap_from(D_va["box"])
    return D_tr, D_va, train_s, val_s

# %% [markdown]
# ## 15.1 Extra seeds

# %%
for modality in ("WLE", "EUS"):
    todo = [(t, s) for t in ("binary", "subtype") for s in NEW_SEEDS
            if (modality, t, "baseline", s) not in adv_done
            or (modality, t, "se_all_adv", s) not in se_done]
    if not todo:
        print(f"\n=== {modality}: already complete ===")
        continue

    D_tr, D_va, train_s, val_s = build_split(modality)
    print(f"\n=== {modality}: {len(todo)} (task, seed) combo(s) to do ===")

    for task in ("binary", "subtype"):
        n_classes = len(classes_for(task))
        ytr = encode_labels(train_s, task)
        yva = encode_labels(val_s, task)

        for seed in NEW_SEEDS:
            if (modality, task, "baseline", seed) not in adv_done:
                t0 = time.time()
                r = train_eval_adversarial(
                    D_tr, ytr, D_va, yva, n_classes, targets=(),
                    seed=seed, epochs=EPOCHS, batch=BATCH,
                    tag=f"{modality}/{task}/baseline/s{seed}",
                )
                r.update({"modality": modality, "task": task, "condition": "baseline",
                          "seed": seed, "minutes": round((time.time() - t0) / 60, 1)})
                adv_runs.append(r)
                adv_done.add((modality, task, "baseline", seed))
                save_adv()

            if (modality, task, "se_all_adv", seed) not in se_done:
                t0 = time.time()
                r = train_eval_adversarial(
                    D_tr, ytr, D_va, yva, n_classes, targets=ADV_TARGETS, lambda_max=ADV_LAMBDA_MAX,
                    backbone_factory=partial(build_resnet18_with_se, stages="all"),
                    seed=seed, epochs=EPOCHS, batch=BATCH,
                    tag=f"{modality}/{task}/se_all_adv/s{seed}",
                )
                r.update({"modality": modality, "task": task, "condition": "se_all_adv",
                          "seed": seed, "minutes": round((time.time() - t0) / 60, 1)})
                se_runs.append(r)
                se_done.add((modality, task, "se_all_adv", seed))
                save_se()

    del D_tr, D_va

# %% [markdown]
# ## 15.2 Verdict — all recorded seeds pooled (up to 7: the original 3 + these 4)

# %%
def pooled_sd(sd_a: float, sd_b: float) -> float:
    return float(np.sqrt((sd_a ** 2 + sd_b ** 2) / 2))


by_adv = {}
for r in adv_runs:
    if r["condition"] == "baseline":
        by_adv.setdefault((r["modality"], r["task"]), []).append(r)

by_se = {}
for r in se_runs:
    if r["condition"] == "se_all_adv":
        by_se.setdefault((r["modality"], r["task"]), []).append(r)

print("=" * 100)
print("se_all_adv vs baseline -- pooled across all recorded seeds")
print("=" * 100)
hdr = f"{'mod':<5}{'task':<9}{'n(b/s)':>8}{'base_ret':>12}{'se_ret':>12}{'d_ret':>9}{'2xsd':>8}{'real?':>7}"
print(hdr); print("-" * len(hdr))

summary = []
for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        base_rows = by_adv.get((modality, task), [])
        se_rows = by_se.get((modality, task), [])
        if not base_rows or not se_rows:
            continue
        base_ret = np.array([r["ret_bg_patch"] for r in base_rows])
        se_ret = np.array([r["ret_bg_patch"] for r in se_rows])
        d_ret = float(se_ret.mean() - base_ret.mean())
        sd = pooled_sd(float(base_ret.std()), float(se_ret.std()))
        real = abs(d_ret) > 2 * sd if sd > 1e-9 else abs(d_ret) > 0
        print(f"{modality:<5}{task:<9}{len(base_rows)}/{len(se_rows):<6}"
              f"{base_ret.mean():>12.1%}{se_ret.mean():>12.1%}{d_ret:>+9.1%}"
              f"{2*sd:>8.1%}{str(real):>7}")
        summary.append({
            "modality": modality, "task": task,
            "n_baseline_seeds": len(base_rows), "n_se_all_adv_seeds": len(se_rows),
            "baseline_ret_bg_patch": round(float(base_ret.mean()), 4),
            "baseline_ret_bg_patch_sd": round(float(base_ret.std()), 4),
            "se_all_adv_ret_bg_patch": round(float(se_ret.mean()), 4),
            "se_all_adv_ret_bg_patch_sd": round(float(se_ret.std()), 4),
            "d_ret_bg_patch": round(d_ret, 4),
            "pooled_sd": round(sd, 4),
            "real_effect": bool(real),
        })

se_store["summary_multiseed_se_all_adv"] = summary
save_se()

n_real = sum(1 for s in summary if s["real_effect"])
n_negative = sum(1 for s in summary if s["d_ret_bg_patch"] < 0)
print(f"\n{n_negative}/{len(summary)} cells still negative (less shortcut reliance) "
      f"after more seeds; {n_real}/{len(summary)} now clear the noise floor.")
