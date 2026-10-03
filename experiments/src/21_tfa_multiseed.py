# %% [markdown]
# # Experiment 21 — Is TFA rare-class fine-tuning's regression real or noise?
#
# `cadsel.finetune_rare` (TFA, Wang et al. 2020) was run once against
# `runs/mobilenet_320__WLE__multiclass__tier1`: overall AP@0.5 dropped
# 0.4106 -> 0.3992, and Leiomyoma AP@0.5 (the class TFA specifically targets)
# dropped 0.175 -> 0.147. A zero-GPU sanity check first: the *pre-TFA*
# seed-to-seed Leiomyoma AP@0.5 spread across the three tier1-recipe
# checkpoints already on disk (`tier1`, `tier1_all_s123`, `tier1_all_s7`) is
# 0.175 / 0.144 / 0.142 -- the one TFA "regression" (0.147) actually lands
# inside that pre-existing spread, not below it. That single run alone can't
# tell TFA-noise from TFA-regression apart from ordinary seed variance.
#
# Two small sweeps disambiguate this:
# - **Sweep A** -- same base checkpoint, `--split-seed` fixed, only the
#   fine-tuning run's own training seed varies (isolates fine-tuning-step noise).
# - **Sweep B** -- different already-trained base checkpoints of the identical
#   recipe (isolates "is this specific to one weight init").

# %%
import json
from pathlib import Path

import numpy as np

RUNS = Path("runs")
OUT = Path("experiments/results")

# (label, sweep, base_result_json, finetuned_result_json)
DATA_POINTS = [
    ("existing_s42", "A", "mobilenet_320__WLE__multiclass__tier1",
     "mobilenet_320__WLE__multiclass__tier1__finetune_rare"),
    ("sweepA_s123", "A", "mobilenet_320__WLE__multiclass__tier1",
     "mobilenet_320__WLE__multiclass__tier1__finetune_rare_s123"),
    ("sweepA_s7", "A", "mobilenet_320__WLE__multiclass__tier1",
     "mobilenet_320__WLE__multiclass__tier1__finetune_rare_s7"),
    ("sweepB_tier1_all_s123", "B", "tier1_all_s123", "tier1_all_s123__finetune_rare"),
    ("sweepB_tier1_all_s7", "B", "tier1_all_s7", "tier1_all_s7__finetune_rare"),
]

# The pre-existing, pre-TFA seed-to-seed spread for this exact metric (computed
# from the 3 tier1-recipe checkpoints already on disk) -- the correct noise
# floor to judge the Leiomyoma-specific delta against, since Leiomyoma's much
# smaller box count makes it structurally noisier than the project's overall
# ±0.023 AP floor.
_pre_tfa_leiomyoma = [0.1751, 0.1441, 0.1420]  # tier1, tier1_all_s123, tier1_all_s7
LEIOMYOMA_FLOOR = float(np.std(_pre_tfa_leiomyoma))
AP_FLOOR = 0.023  # the project's established overall-AP noise floor (Experiment 11)

# %%
def load_result(run_name: str) -> dict:
    path = RUNS / run_name / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


rows = []
for label, sweep, base_name, ft_name in DATA_POINTS:
    base_path, ft_path = RUNS / base_name / "result.json", RUNS / ft_name / "result.json"
    if not base_path.exists() or not ft_path.exists():
        print(f"  missing result.json for {label} ({base_name} / {ft_name}) -- skipping")
        continue
    base, ft = load_result(base_name), load_result(ft_name)
    base_ap = base["best"]["AP@0.5"]
    ft_ap = ft["best"]["AP@0.5"]
    base_leio = base["best"]["per_class"]["Leiomyoma"]["AP@0.5"]
    ft_leio = ft["best"]["per_class"]["Leiomyoma"]["AP@0.5"]
    rows.append({
        "label": label, "sweep": sweep, "base": base_name, "finetuned": ft_name,
        "base_AP@0.5": round(base_ap, 4), "finetuned_AP@0.5": round(ft_ap, 4),
        "delta_AP@0.5": round(ft_ap - base_ap, 4),
        "base_Leiomyoma_AP@0.5": round(base_leio, 4), "finetuned_Leiomyoma_AP@0.5": round(ft_leio, 4),
        "delta_Leiomyoma_AP@0.5": round(ft_leio - base_leio, 4),
    })
    print(f"{label:<24} sweep {sweep}  AP@0.5 {base_ap:.4f} -> {ft_ap:.4f} "
          f"({ft_ap - base_ap:+.4f})  |  Leiomyoma {base_leio:.4f} -> {ft_leio:.4f} "
          f"({ft_leio - base_leio:+.4f})")

json.dump(rows, open(OUT / "21_tfa_multiseed.json", "w"), indent=2)

# %% [markdown]
# ## 21.1 Verdict -- per-sweep mean +/- std, judged against both noise floors

# %%
print("\n" + "=" * 92)
print("TFA MULTI-SEED VERDICT")
print("=" * 92)
print(f"pre-TFA Leiomyoma AP@0.5 seed spread (std): {LEIOMYOMA_FLOOR:.4f}  "
      f"(from {_pre_tfa_leiomyoma})")
print(f"project overall-AP noise floor: {AP_FLOOR:.4f}\n")

summary = {}
for sweep in ("A", "B"):
    sweep_rows = [r for r in rows if r["sweep"] == sweep]
    if not sweep_rows:
        continue
    d_ap = [r["delta_AP@0.5"] for r in sweep_rows]
    d_leio = [r["delta_Leiomyoma_AP@0.5"] for r in sweep_rows]
    ap_mean, ap_std = float(np.mean(d_ap)), float(np.std(d_ap))
    leio_mean, leio_std = float(np.mean(d_leio)), float(np.std(d_leio))
    ap_verdict = "CONFIRMED regression" if ap_mean < -AP_FLOOR else "within noise floor -- not confirmed"
    leio_verdict = ("CONFIRMED regression" if leio_mean < -LEIOMYOMA_FLOOR
                     else "within noise floor -- not confirmed")
    summary[f"sweep_{sweep}"] = {
        "n": len(sweep_rows), "delta_AP_mean": ap_mean, "delta_AP_std": ap_std,
        "delta_Leiomyoma_mean": leio_mean, "delta_Leiomyoma_std": leio_std,
        "ap_verdict": ap_verdict, "leiomyoma_verdict": leio_verdict,
    }
    print(f"Sweep {sweep} (n={len(sweep_rows)}):")
    print(f"  overall AP@0.5   delta {ap_mean:+.4f} +/- {ap_std:.4f}  -> {ap_verdict}")
    print(f"  Leiomyoma AP@0.5 delta {leio_mean:+.4f} +/- {leio_std:.4f}  -> {leio_verdict}\n")

json.dump(summary, open(OUT / "21_tfa_multiseed_summary.json", "w"), indent=2)

print("Interpretation: Sweep A isolates fine-tuning-step randomness (same base "
      "checkpoint, same held-out patients); Sweep B isolates weight-init "
      "sensitivity (different already-trained base checkpoints, same recipe). "
      "If neither sweep's mean delta clears its noise floor, the original "
      "single-run 'TFA hurts here' finding does not hold up under this "
      "project's own multi-seed discipline (Experiment 11) and should be "
      "downgraded from a negative result to 'no effect detected, more seeds "
      "would be needed to resolve it either way.'")
