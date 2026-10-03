"""Experiment 40 -- split audit for the nested two-tower runs.

The fit / select / test patient sets of a ``34_joint_fusion_train.py
--inner-val-fraction`` run are not stored in its JSON; they are a pure
function of the dataset tree, ``--seed``, ``--n-splits`` and
``--inner-val-fraction``. This rebuilds them exactly as the trainer does and
checks, for every nested run on disk:

  * fit, select and test are pairwise disjoint in every fold
  * the rebuilt sizes equal the recorded ``n_train`` / ``n_val`` / ``n_test``
  * test is the run's outer fold, and (for a finished run) every patient is
    tested exactly once

    python -m experiments.src.40_two_tower_split_audit
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cadsel.data import build_joint_manifest, joint_strata_key, patient_level_folds, patient_level_split

RESULTS_DIR = Path("experiments/results/joint_fusion_runs")


def main() -> None:
    samples = build_joint_manifest()
    bad = 0
    for path in sorted(RESULTS_DIR.glob("*__nested*.json")):
        run = json.loads(path.read_text())
        s = samples
        if run.get("paired_only"):
            paired = {x.patient_key for x in s if x.modality == "EUS"}
            s = [x for x in s if x.patient_key in paired]
        outer = patient_level_folds(s, n_folds=run["n_splits"], seed=run["seed"], strata_key=joint_strata_key)

        tested: Counter = Counter()
        problems = []
        for f in run["folds"]:
            k = f["fold"]
            outer_train, test = outer[k]
            fit, select = patient_level_split(
                outer_train, val_fraction=run["inner_val_fraction"],
                seed=run["seed"] + 1000 + k, strata_key=joint_strata_key,
            )
            fit_k, select_k, test_k = ({x.patient_key for x in g} for g in (fit, select, test))
            if fit_k & select_k or fit_k & test_k or select_k & test_k:
                problems.append(f"fold{k}: patient overlap")
            if (len(fit_k), len(select_k), len(test_k)) != (f["n_train"], f["n_val"], f["n_test"]):
                problems.append(f"fold{k}: sizes recorded=({f['n_train']},{f['n_val']},{f['n_test']}) "
                                f"rebuilt=({len(fit_k)},{len(select_k)},{len(test_k)})")
            tested.update(test_k)
        if run["folds_done"] == run["n_splits"]:
            if len(tested) != len({x.patient_key for x in s}) or set(tested.values()) != {1}:
                problems.append("patients not each tested exactly once")

        bad += bool(problems)
        short = re.sub(r"__fusion-.*?__nested", "__nested", run["run_name"])
        print(f"[{'OK ' if not problems else 'BAD'}] {short}  folds={run['folds_done']}/{run['n_splits']}  "
              f"{'; '.join(problems)}")
    print(f"-> {bad} run(s) with problems")
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
