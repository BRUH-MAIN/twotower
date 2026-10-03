# %% [markdown]
# # Experiment 16 — `se_all_adv` vs baseline, 100 epochs / early-stop 20
#
# Experiment 15 extended `baseline` (Experiment 12) and `se_all_adv`
# (Experiment 14: channel attention on every ResNet stage, composed with the
# adversarial discriminator) from 3 to 7 seeds at a fixed 15 epochs. The
# direction stayed consistently negative (less `bg_patch` reliance) in all 4
# modality/task cells, but none cleared the noise floor — plausibly because 15
# epochs, chosen for fast iteration across the original 3-method sweep, is not
# enough for either model to actually converge, especially on the harder
# 5-class subtype task and the smaller EUS split.
#
# This experiment reruns the same two conditions, same 7 seeds, at up to 100
# epochs with early stopping (patience 20 on validation balanced accuracy) —
# giving both `baseline` and `se_all_adv` a real chance to converge before
# comparing them, rather than comparing two under-trained models. Results are
# written under new condition names (`baseline_100ep` / `se_all_adv_100ep`)
# into the *same* `12_adversarial.json` / `14_channel_attention.json` files
# used throughout, so nothing from Experiments 12-15 is touched or overwritten.

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
SEEDS = (42, 7, 123, 1, 2, 3, 4)  # same 7 seeds as Experiment 15, for a clean comparison
EPOCHS, BATCH = 100, 32
EARLY_STOP_PATIENCE = 20
print("device:", DEVICE)

# %%
if not ADV_PATH.exists() or not SE_PATH.exists():
    raise SystemExit("Experiments 12 and 14 must run to completion first.")

adv_store = json.loads(ADV_PATH.read_text())
se_store = json.loads(SE_PATH.read_text())
chosen = adv_store["chosen_hyperparameters"]
ADV_LAMBDA_MAX, ADV_TARGETS = chosen["lambda_max"], tuple(chosen["targets"])
print(f"lambda_max={ADV_LAMBDA_MAX}, targets={ADV_TARGETS}, "
      f"epochs<= {EPOCHS}, early_stop_patience={EARLY_STOP_PATIENCE}, seeds={SEEDS}")

adv_runs = adv_store["runs"]
se_runs = se_store["runs"]
adv_done = {(r["modality"], r["task"], r["condition"], r["seed"]) for r in adv_runs}
se_done = {(r["modality"], r["task"], r["condition"], r["seed"]) for r in se_runs}
BASELINE_COND, SE_COND = "baseline_100ep", "se_all_adv_100ep"


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
# ## 16.1 Training

# %%
for modality in ("WLE", "EUS"):
    todo = [(t, s) for t in ("binary", "subtype") for s in SEEDS
            if (modality, t, BASELINE_COND, s) not in adv_done
            or (modality, t, SE_COND, s) not in se_done]
    if not todo:
        print(f"\n=== {modality}: already complete ===")
        continue

    D_tr, D_va, train_s, val_s = build_split(modality)
    print(f"\n=== {modality}: {len(todo)} (task, seed) combo(s) to do ===")

    for task in ("binary", "subtype"):
        n_classes = len(classes_for(task))
        ytr = encode_labels(train_s, task)
        yva = encode_labels(val_s, task)

        for seed in SEEDS:
            if (modality, task, BASELINE_COND, seed) not in adv_done:
                t0 = time.time()
                r = train_eval_adversarial(
                    D_tr, ytr, D_va, yva, n_classes, targets=(),
                    seed=seed, epochs=EPOCHS, batch=BATCH, early_stop_patience=EARLY_STOP_PATIENCE,
                    tag=f"{modality}/{task}/{BASELINE_COND}/s{seed}",
                )
                r.update({"modality": modality, "task": task, "condition": BASELINE_COND,
                          "seed": seed, "minutes": round((time.time() - t0) / 60, 1)})
                adv_runs.append(r)
                adv_done.add((modality, task, BASELINE_COND, seed))
                save_adv()

            if (modality, task, SE_COND, seed) not in se_done:
                t0 = time.time()
                r = train_eval_adversarial(
                    D_tr, ytr, D_va, yva, n_classes, targets=ADV_TARGETS, lambda_max=ADV_LAMBDA_MAX,
                    backbone_factory=partial(build_resnet18_with_se, stages="all"),
                    seed=seed, epochs=EPOCHS, batch=BATCH, early_stop_patience=EARLY_STOP_PATIENCE,
                    tag=f"{modality}/{task}/{SE_COND}/s{seed}",
                )
                r.update({"modality": modality, "task": task, "condition": SE_COND,
                          "seed": seed, "minutes": round((time.time() - t0) / 60, 1)})
                se_runs.append(r)
                se_done.add((modality, task, SE_COND, seed))
                save_se()

    del D_tr, D_va

# %% [markdown]
# ## 16.2 Verdict

# %%
def pooled_sd(sd_a: float, sd_b: float) -> float:
    return float(np.sqrt((sd_a ** 2 + sd_b ** 2) / 2))


by_adv = {}
for r in adv_runs:
    if r["condition"] == BASELINE_COND:
        by_adv.setdefault((r["modality"], r["task"]), []).append(r)

by_se = {}
for r in se_runs:
    if r["condition"] == SE_COND:
        by_se.setdefault((r["modality"], r["task"]), []).append(r)

print("=" * 100)
print(f"{SE_COND} vs {BASELINE_COND} -- 100 epochs / early-stop {EARLY_STOP_PATIENCE}, {len(SEEDS)} seeds")
print("=" * 100)
hdr = (f"{'mod':<5}{'task':<9}{'n(b/s)':>8}{'base_ret':>12}{'se_ret':>12}{'d_ret':>9}"
       f"{'2xsd':>8}{'real?':>7}{'base_ep':>9}{'se_ep':>7}")
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
        base_ep = np.mean([r.get("epochs_run", EPOCHS) for r in base_rows])
        se_ep = np.mean([r.get("epochs_run", EPOCHS) for r in se_rows])
        d_ret = float(se_ret.mean() - base_ret.mean())
        sd = pooled_sd(float(base_ret.std()), float(se_ret.std()))
        real = abs(d_ret) > 2 * sd if sd > 1e-9 else abs(d_ret) > 0
        print(f"{modality:<5}{task:<9}{len(base_rows)}/{len(se_rows):<6}"
              f"{base_ret.mean():>12.1%}{se_ret.mean():>12.1%}{d_ret:>+9.1%}"
              f"{2*sd:>8.1%}{str(real):>7}{base_ep:>9.1f}{se_ep:>7.1f}")
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
            "mean_epochs_run_baseline": round(float(base_ep), 1),
            "mean_epochs_run_se_all_adv": round(float(se_ep), 1),
        })

se_store["summary_100epoch_se_all_adv"] = summary
save_se()

n_real = sum(1 for s in summary if s["real_effect"])
n_negative = sum(1 for s in summary if s["d_ret_bg_patch"] < 0)
print(f"\n{n_negative}/{len(summary)} cells still negative; {n_real}/{len(summary)} clear the noise floor "
      f"at 100 epochs / early-stop {EARLY_STOP_PATIENCE}.")
