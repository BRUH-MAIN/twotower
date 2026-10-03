"""Experiment 26 (Track A, Phase 4) — leaderboard over
``experiments/finetune_classifier.py`` runs, mean +/- std per arm across
seeds, modeled on ``19_size_lever_eval.py``'s verdict block.

An "arm" is everything except ``seed`` (backbone/modality/task/freeze/
oversample/loss/bg_swap_aug) -- each run's own JSON already reports a
5-fold-CV mean balanced accuracy for one seed; this averages those means
across whichever seeds have been run for that arm and reports the
``experiments/svm_classifier.py`` frozen-embedding numbers alongside as the
reference floor.

    python experiments/src/26_classifier_leaderboard.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("experiments/results/classifier_runs")
SVM_RESULTS_DIR = Path("experiments/results")

# experiments/svm_classifier.py's published numbers (frozen embeddings + SVM,
# 5-fold patient-grouped balanced accuracy) -- the floor any fine-tuned
# classifier needs to clear to be worth adopting.
SVM_REFERENCE = [
    ("resnet50 (frozen, SVM)", "WLE", "binary", 0.6447),
    ("resnet50 (frozen, SVM)", "WLE", "subtype", 0.2995),
    ("dinov3_vits16 (frozen, SVM)", "WLE", "binary", 0.7353),
    ("dinov3_vits16 (frozen, SVM)", "WLE", "subtype", 0.3813),
    ("dinov3_vits16 (frozen, SVM)", "EUS", "binary", 0.5805),
    ("resnet50 (frozen, SVM)", "EUS", "binary", 0.5817),
]


def arm_key(r: dict) -> tuple:
    return (
        r["backbone"], r["modality"], r["task"], r["freeze"], r["oversample"],
        r.get("size_rfs_threshold"), r["loss"], r.get("seesaw_p"), r["bg_swap_aug"],
    )


def arm_label(key: tuple) -> str:
    backbone, modality, task, freeze, oversample, thr, loss, p, bg_swap = key
    bits = [backbone, f"freeze={freeze}"]
    bits.append(f"os={oversample}" + (f"@{thr}" if oversample == "size" and thr else ""))
    bits.append(f"loss={loss}" + (f"(p={p})" if loss == "seesaw_ce" and p else ""))
    if bg_swap:
        bits.append("bgswap")
    return " ".join(bits)


def main() -> None:
    if not RESULTS_DIR.exists():
        print(f"no runs yet under {RESULTS_DIR}")
        return

    runs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    if not runs:
        print(f"no run JSONs under {RESULTS_DIR}")
        return

    by_arm = defaultdict(list)
    for r in runs:
        by_arm[arm_key(r)].append(r)

    import numpy as np

    print("=" * 100)
    print("CLASSIFIER LEADERBOARD -- 5-fold-CV balanced accuracy, mean +/- std across seeds")
    print("(freeze=head -> FROZEN backbone; freeze=none/partial -> unfrozen/fine-tuned)")
    print("=" * 100)
    print(f"{'modality':<9}{'task':<9}{'arm':<48}{'n_seeds':>8}{'bal_acc':>18}{'roc_auc':>10}{'pr_auc':>9}")
    print("-" * 100)

    rows = []
    for key, group in sorted(by_arm.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        modality, task = key[1], key[2]
        means = [g["balanced_accuracy_mean"] for g in group]
        roc_aucs = [g["roc_auc_mean"] for g in group if g.get("roc_auc_mean") is not None]
        pr_aucs = [g["pr_auc_mean"] for g in group if g.get("pr_auc_mean") is not None]
        seeds = sorted({g["seed"] for g in group})
        entry = {
            "modality": modality, "task": task, "arm": arm_label(key), "freeze": key[3],
            "n_seeds": len(seeds), "seeds": seeds,
            "bal_acc_mean": float(np.mean(means)), "bal_acc_std": float(np.std(means)),
            "roc_auc_mean": float(np.mean(roc_aucs)) if roc_aucs else None,
            "pr_auc_mean": float(np.mean(pr_aucs)) if pr_aucs else None,
            "chance": group[0]["chance"],
        }
        rows.append(entry)
        roc_s = f"{entry['roc_auc_mean']:.3f}" if entry["roc_auc_mean"] is not None else "n/a"
        pr_s = f"{entry['pr_auc_mean']:.3f}" if entry["pr_auc_mean"] is not None else "n/a"
        print(f"{modality:<9}{task:<9}{arm_label(key):<48}{len(seeds):>8}"
              f"{entry['bal_acc_mean']:>10.4f}+/-{entry['bal_acc_std']:.4f}{roc_s:>10}{pr_s:>9}")

    print("\n" + "=" * 100)
    print("REFERENCE -- frozen-embedding SVM (experiments/svm_classifier.py)")
    print("=" * 100)
    for name, modality, task, bal in SVM_REFERENCE:
        print(f"{modality:<9}{task:<9}{name:<48}{'':>8}{bal:>10.4f}")

    out = RESULTS_DIR.parent / "26_classifier_leaderboard.json"
    out.write_text(json.dumps({"arms": rows, "svm_reference": [
        {"name": n, "modality": m, "task": t, "balanced_accuracy": b}
        for n, m, t, b in SVM_REFERENCE
    ]}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
