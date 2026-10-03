"""Experiment 39 -- summary table for the two-tower backbone ablation.

Collects every completed nested-protocol run (``--inner-val-fraction`` > 0)
of ``34_joint_fusion_train.py`` and tabulates backbone x task: outer-fold
test balanced accuracy for fused / wle_only / eus_only (all val patients,
EUS restricted to patients that have it) and fused on paired patients only,
per seed and averaged across seeds, plus tower size, peak GPU memory and
wall-clock time.

    python -m experiments.src.39_two_tower_backbone_table
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("experiments/results/joint_fusion_runs")
OUT_PATH = Path("experiments/results/39_two_tower_backbone_table.json")
COLUMNS = [("fused", "unpaired"), ("wle_only", "unpaired"), ("eus_only", "unpaired"), ("fused", "paired_only")]


def main() -> None:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(RESULTS_DIR.glob("*__nested*.json")):
        run = json.loads(path.read_text())
        if run.get("inner_val_fraction", 0) <= 0 or run["folds_done"] != run["n_splits"]:
            continue
        config = re.sub(r"__s\d+$", "", run["run_name"])
        groups[(run["task"], run["backbone"], config)].append(run)

    rows = []
    for (task, backbone, config), runs in sorted(groups.items()):
        row = {"task": task, "backbone": backbone, "config": config,
               "seeds": [r["seed"] for r in runs], "metrics": {}}
        for ablation, denom in COLUMNS:
            per_seed = [r["summary"][ablation][denom]["bal"] for r in runs]
            means = [s["mean"] for s in per_seed]
            row["metrics"][f"{ablation}/{denom}"] = {
                "per_seed": per_seed, "mean": round(float(np.mean(means)), 4),
                "seed_std": round(float(np.std(means)), 4) if len(means) > 1 else None,
            }
        n_params = runs[0]["folds"][0].get("n_params", {})
        row["tower_params_M"] = round(n_params.get("total", 0) / 2 / 1e6, 2)   # both towers + head, halved: approx per tower
        row["trainable_M"] = round(n_params.get("trainable", 0) / 1e6, 2)
        row["peak_mem_gb"] = max((f.get("peak_mem_gb") or 0) for r in runs for f in r["folds"])
        row["hours_per_run"] = round(float(np.mean([r["elapsed_sec"] for r in runs])) / 3600, 2)
        rows.append(row)

    for task in ("binary", "subtype"):
        task_rows = sorted((r for r in rows if r["task"] == task),
                           key=lambda r: -r["metrics"]["fused/unpaired"]["mean"])
        if not task_rows:
            continue
        print(f"\n== {task} (outer-fold test balanced accuracy; per-seed fold std in brackets) ==")
        print(f"{'backbone':20s} {'seeds':>5s} " + " ".join(f"{a + ('' if d == 'unpaired' else '(paired)'):>22s}" for a, d in COLUMNS)
              + f" {'params':>7s} {'train':>6s} {'mem':>5s} {'h/run':>5s}")
        for r in task_rows:
            cells = []
            for ablation, denom in COLUMNS:
                m = r["metrics"][f"{ablation}/{denom}"]
                spread = (f"±{m['seed_std']:.3f}s" if m["seed_std"] is not None
                          else f"[{m['per_seed'][0]['std']:.3f}]")
                cells.append(f"{m['mean']:.4f} {spread:>9s}")
            print(f"{r['backbone']:20s} {len(r['seeds']):>5d} " + " ".join(f"{c:>22s}" for c in cells)
                  + f" {r['tower_params_M']:>6.1f}M {r['trainable_M']:>5.1f}M {r['peak_mem_gb']:>5.1f} {r['hours_per_run']:>5.2f}")
    print("\n'±x s' = std across seeds of the 5-fold means; '[x]' = fold std for a single seed.")

    OUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
