"""Does fusion help specifically where WLE alone is ambiguous?

`results/fusion.json` shows `fused` barely beating `wle_only` on aggregate
balanced accuracy (0.419 vs 0.426) even though `fused_moddrop` clearly wins on
macro-F1 (0.346 vs 0.316). The aggregate hides two things:

1.  `wle_only`'s reported number is computed over all 641 patients, including
    the ~220 WLE-only patients who have no EUS frames at all. `fused` and
    `fused_moddrop` structurally cannot differ from `wle_only` on those
    patients -- there's no second modality to fuse -- so mixing them into the
    comparison dilutes whatever real effect exists on the 421 *paired*
    patients. This script restricts to those 421 before comparing anything.

2.  Even restricted to paired patients, "fusion helps" could mean two very
    different things: a flat, small improvement everywhere, or a concentrated
    improvement specifically on the patients where WLE alone was uncertain
    (the clinically expected pattern -- a second view should matter most
    exactly when the first one is ambiguous). This script buckets paired
    patients into confidence tertiles by `wle_only`'s own predicted-class
    confidence and reports `fused`/`fused_moddrop` accuracy minus `wle_only`
    accuracy within each tertile.

Reads ``results/fusion_predictions.json`` (written by
``cadsel.train_fusion --predictions-output``, one out-of-fold prediction per
patient per ablation) and writes ``results/fusion_subgroup_analysis.json``.

    python -m cadsel.analyze_fusion_subgroups
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT

# Statistical viability limits (from the 421-paired-patient class counts in
# results/dataset_stats.json's table2.EUS): only groups with enough patients
# support tertile slicing. Thinner groups are reported in aggregate only.
#   subtype: NET_G1=299, NET_G2=26, Leiomyoma=14, Lipoma=49, NonNeoplasm=33
#   binary:  NET=325, non-NET=96
SUBTYPE_TERTILE_OK = {"NET_G1", "Lipoma"}
BINARY_TERTILE_OK = {"NET", "non-NET"}  # non-NET (96) gets tertiles only, no finer split


def load_predictions(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def restrict_to_paired(predictions: list[dict], ablation: str) -> list[dict]:
    """Paired (both-modality) patients only, for one ablation.

    ``paired`` is recorded directly on each record by ``train_fusion.run()``
    (``all(patient.has(m) for m in MODALITY_NAMES)`` at extraction time).
    As a cross-check, also verify this against an independent derivation: the
    set of patient_keys appearing under the ``eus_only`` ablation, which by
    construction (``train_fusion.run()`` filters ``eus_only`` to
    ``p.has("EUS")``) only ever contains paired patients.
    """
    rows = [p for p in predictions if p["ablation"] == ablation]
    paired_flagged = [p for p in rows if p["paired"]]

    eus_only_keys = {p["patient_key"] for p in predictions if p["ablation"] == "eus_only"}
    paired_by_eus_xref = [p for p in rows if p["patient_key"] in eus_only_keys]

    flagged_keys = {p["patient_key"] for p in paired_flagged}
    xref_keys = {p["patient_key"] for p in paired_by_eus_xref}
    if flagged_keys != xref_keys:
        missing = flagged_keys ^ xref_keys
        raise AssertionError(
            f"'paired' flag disagrees with eus_only-derived pairing for "
            f"ablation={ablation!r}: {len(missing)} patient(s) differ, "
            f"e.g. {sorted(missing)[:5]} -- the pairing assumption needs "
            f"re-checking before this comparison means anything."
        )
    return paired_flagged


def confidence(record: dict) -> float:
    return max(record["probs"].values())


def accuracy(records: list[dict]) -> float | None:
    if not records:
        return None
    return float(np.mean([r["correct"] for r in records]))


def balanced_accuracy(records: list[dict]) -> float | None:
    """Mean per-class recall, pooled across all 5 folds' out-of-fold
    predictions (not the mean-of-per-fold value `results/fusion.json` reports
    -- that one averages 5 separate small-sample balanced accuracies; this one
    pools all patients first, which is the more stable estimate once sliced
    down to 421 or fewer patients). Supplementary cross-check only: the
    per-tertile numbers below are plain accuracy, per the task spec, since a
    tertile is not class-balanced and "accuracy" is the natural per-slice
    metric there.
    """
    if not records:
        return None
    recalls = []
    for c in sorted({r["y_true"] for r in records}):
        in_class = [r for r in records if r["y_true"] == c]
        if not in_class:
            continue
        recalls.append(np.mean([r["correct"] for r in in_class]))
    return float(np.mean(recalls)) if recalls else None


def assign_tertiles(records: list[dict]) -> list[int]:
    """0 = lowest WLE confidence third, 2 = highest. Ties broken by patient_key
    so the boundary assignment is deterministic."""
    order = sorted(range(len(records)),
                    key=lambda i: (confidence(records[i]), records[i]["patient_key"]))
    n = len(records)
    tertile_of = [0] * n
    for rank, idx in enumerate(order):
        tertile_of[idx] = min(2, (rank * 3) // n)
    return tertile_of


def binary_label(subtype_label: str) -> str:
    return "NET" if subtype_label.startswith("NET") else "non-NET"


def build_group_table(
    wle_paired: list[dict],
    fused_by_key: dict[str, dict],
    moddrop_by_key: dict[str, dict],
    group_of,          # record -> group label (subtype string, or binary label)
    viable_groups: set[str],
    label: str,
) -> dict:
    """One entry per (group, tertile|'aggregate'): fused/fused_moddrop acc minus
    wle_only acc, restricted to paired patients, using wle_only's own tertile
    (computed within the whole paired set for 'aggregate', or within-group
    when the group is viable for slicing).
    """
    out: dict[str, dict] = {}
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in wle_paired:
        by_group[group_of(r)].append(r)

    for group, rows in sorted(by_group.items()):
        entry: dict = {"n": len(rows)}
        wle_acc = accuracy(rows)
        fused_rows = [fused_by_key[r["patient_key"]] for r in rows]
        moddrop_rows = [moddrop_by_key[r["patient_key"]] for r in rows]
        entry["aggregate"] = {
            "n": len(rows),
            "wle_only_acc": round(wle_acc, 4) if wle_acc is not None else None,
            "fused_acc": round(accuracy(fused_rows), 4) if fused_rows else None,
            "fused_moddrop_acc": round(accuracy(moddrop_rows), 4) if moddrop_rows else None,
            "fused_minus_wle": (
                round(accuracy(fused_rows) - wle_acc, 4) if fused_rows and wle_acc is not None else None
            ),
            "fused_moddrop_minus_wle": (
                round(accuracy(moddrop_rows) - wle_acc, 4) if moddrop_rows and wle_acc is not None else None
            ),
        }

        if group in viable_groups and len(rows) >= 15:
            tertile_of = assign_tertiles(rows)
            entry["tertiles"] = []
            for t in range(3):
                t_rows = [r for r, tt in zip(rows, tertile_of) if tt == t]
                t_fused = [fused_by_key[r["patient_key"]] for r in t_rows]
                t_moddrop = [moddrop_by_key[r["patient_key"]] for r in t_rows]
                t_wle_acc = accuracy(t_rows)
                t_fused_acc = accuracy(t_fused)
                t_moddrop_acc = accuracy(t_moddrop)
                confs = [confidence(r) for r in t_rows]
                entry["tertiles"].append({
                    "tertile": ["low_wle_confidence", "mid", "high_wle_confidence"][t],
                    "n": len(t_rows),
                    "wle_confidence_range": [round(min(confs), 4), round(max(confs), 4)] if confs else None,
                    "wle_only_acc": round(t_wle_acc, 4) if t_wle_acc is not None else None,
                    "fused_acc": round(t_fused_acc, 4) if t_fused_acc is not None else None,
                    "fused_moddrop_acc": round(t_moddrop_acc, 4) if t_moddrop_acc is not None else None,
                    "fused_minus_wle": (
                        round(t_fused_acc - t_wle_acc, 4)
                        if t_fused_acc is not None and t_wle_acc is not None else None
                    ),
                    "fused_moddrop_minus_wle": (
                        round(t_moddrop_acc - t_wle_acc, 4)
                        if t_moddrop_acc is not None and t_wle_acc is not None else None
                    ),
                })
        else:
            entry["tertiles"] = None
            entry["tertile_note"] = (
                f"not sliced: n={len(rows)} below viability threshold, or "
                f"{group!r} not in the viable-groups list for {label}"
            )
        out[group] = entry
    return out


def run(predictions_path: Path, output_path: Path) -> dict:
    blob = load_predictions(predictions_path)
    predictions = blob["predictions"]
    task = blob["task"]

    wle_paired = restrict_to_paired(predictions, "wle_only")
    fused_paired = restrict_to_paired(predictions, "fused")
    moddrop_paired = restrict_to_paired(predictions, "fused_moddrop")

    n_paired_expected = 421
    if len(wle_paired) != n_paired_expected:
        print(f"WARNING: expected {n_paired_expected} paired patients, "
              f"found {len(wle_paired)} under wle_only")

    wle_keys = {r["patient_key"] for r in wle_paired}
    fused_keys = {r["patient_key"] for r in fused_paired}
    moddrop_keys = {r["patient_key"] for r in moddrop_paired}
    assert wle_keys == fused_keys == moddrop_keys, (
        "paired patient sets differ across ablations -- cannot align records "
        f"1:1 (wle-only {len(wle_keys)}, fused {len(fused_keys)}, "
        f"fused_moddrop {len(moddrop_keys)})"
    )
    fused_by_key = {r["patient_key"]: r for r in fused_paired}
    moddrop_by_key = {r["patient_key"]: r for r in moddrop_paired}

    # ---- overall (all 421 paired patients, one tertile split) --------------
    overall_tertile_of = assign_tertiles(wle_paired)
    overall_tertiles = []
    for t in range(3):
        t_rows = [r for r, tt in zip(wle_paired, overall_tertile_of) if tt == t]
        t_fused = [fused_by_key[r["patient_key"]] for r in t_rows]
        t_moddrop = [moddrop_by_key[r["patient_key"]] for r in t_rows]
        confs = [confidence(r) for r in t_rows]
        wle_acc = accuracy(t_rows)
        fused_acc = accuracy(t_fused)
        moddrop_acc = accuracy(t_moddrop)
        overall_tertiles.append({
            "tertile": ["low_wle_confidence", "mid", "high_wle_confidence"][t],
            "n": len(t_rows),
            "wle_confidence_range": [round(min(confs), 4), round(max(confs), 4)],
            "wle_only_acc": round(wle_acc, 4),
            "fused_acc": round(fused_acc, 4),
            "fused_moddrop_acc": round(moddrop_acc, 4),
            "fused_minus_wle": round(fused_acc - wle_acc, 4),
            "fused_moddrop_minus_wle": round(moddrop_acc - wle_acc, 4),
        })

    overall_agg = {
        "n": len(wle_paired),
        "wle_only_acc": round(accuracy(wle_paired), 4),
        "fused_acc": round(accuracy(fused_paired), 4),
        "fused_moddrop_acc": round(accuracy(moddrop_paired), 4),
        "fused_minus_wle": round(accuracy(fused_paired) - accuracy(wle_paired), 4),
        "fused_moddrop_minus_wle": round(accuracy(moddrop_paired) - accuracy(wle_paired), 4),
        # Supplementary: balanced accuracy, restricted to the 421 paired
        # patients, pooled across folds -- the paired-only analogue of the
        # "bal" metric results/fusion.json reports over all 641 patients.
        "wle_only_bal_acc": round(balanced_accuracy(wle_paired), 4),
        "fused_bal_acc": round(balanced_accuracy(fused_paired), 4),
        "fused_moddrop_bal_acc": round(balanced_accuracy(moddrop_paired), 4),
        "fused_minus_wle_bal": round(
            balanced_accuracy(fused_paired) - balanced_accuracy(wle_paired), 4),
        "fused_moddrop_minus_wle_bal": round(
            balanced_accuracy(moddrop_paired) - balanced_accuracy(wle_paired), 4),
    }

    # ---- subtype breakdown (5-way) or binary, sliced only where viable -----
    if task == "binary":
        group_table = build_group_table(
            wle_paired, fused_by_key, moddrop_by_key,
            group_of=lambda r: r["y_true"],
            viable_groups=BINARY_TERTILE_OK, label="binary",
        )
    else:
        group_table = build_group_table(
            wle_paired, fused_by_key, moddrop_by_key,
            group_of=lambda r: r["subtype"],
            viable_groups=SUBTYPE_TERTILE_OK, label="subtype",
        )

    payload = {
        "source": str(predictions_path),
        "task": task,
        "n_paired_patients": len(wle_paired),
        "overall": {
            "aggregate": overall_agg,
            "tertiles": overall_tertiles,
        },
        "by_group": group_table,
        "headline_question": (
            "Does fused's advantage over wle_only concentrate in the "
            "low-WLE-confidence tertile (clinically expected: EUS should "
            "help most exactly where WLE is uncertain), or is it flat/absent "
            "across tertiles?"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_summary(payload)
    print(f"\nwrote {output_path}")
    return payload


def _print_summary(payload: dict) -> None:
    print(f"paired patients: {payload['n_paired_patients']}  (task={payload['task']})")
    print()
    agg = payload["overall"]["aggregate"]
    print("OVERALL (421 paired patients, aggregate)")
    print(f"  {'ablation':<16}{'acc':>8}{'Δ vs wle':>12}{'bal_acc':>10}{'Δ vs wle':>12}")
    print(f"  {'wle_only':<16}{agg['wle_only_acc']:>8.4f}{'':>12}{agg['wle_only_bal_acc']:>10.4f}{'':>12}")
    print(f"  {'fused':<16}{agg['fused_acc']:>8.4f}{agg['fused_minus_wle']:>+12.4f}"
          f"{agg['fused_bal_acc']:>10.4f}{agg['fused_minus_wle_bal']:>+12.4f}")
    print(f"  {'fused_moddrop':<16}{agg['fused_moddrop_acc']:>8.4f}{agg['fused_moddrop_minus_wle']:>+12.4f}"
          f"{agg['fused_moddrop_bal_acc']:>10.4f}{agg['fused_moddrop_minus_wle_bal']:>+12.4f}")
    print()
    print("OVERALL by WLE-confidence tertile (low = WLE most uncertain)")
    print(f"  {'tertile':<22}{'n':>5}{'wle_only':>10}{'fused':>10}{'Δfused':>10}"
          f"{'moddrop':>10}{'Δmoddrop':>10}")
    for t in payload["overall"]["tertiles"]:
        print(f"  {t['tertile']:<22}{t['n']:>5}{t['wle_only_acc']:>10.4f}"
              f"{t['fused_acc']:>10.4f}{t['fused_minus_wle']:>+10.4f}"
              f"{t['fused_moddrop_acc']:>10.4f}{t['fused_moddrop_minus_wle']:>+10.4f}")
    print()
    print("BY GROUP (subtype/class), aggregate; tertiles shown where viable")
    for group, entry in payload["by_group"].items():
        a = entry["aggregate"]
        print(f"  {group:<14} n={a['n']:<4} wle_only={a['wle_only_acc']:.4f}  "
              f"fused={a['fused_acc']:.4f} (Δ{a['fused_minus_wle']:+.4f})  "
              f"moddrop={a['fused_moddrop_acc']:.4f} (Δ{a['fused_moddrop_minus_wle']:+.4f})")
        if entry.get("tertiles"):
            for t in entry["tertiles"]:
                print(f"      {t['tertile']:<20} n={t['n']:<4} "
                      f"wle_only={t['wle_only_acc']:.4f}  "
                      f"fused Δ={t['fused_minus_wle']:+.4f}  "
                      f"moddrop Δ={t['fused_moddrop_minus_wle']:+.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path,
                   default=PROJECT_ROOT / "results" / "fusion_predictions.json")
    p.add_argument("--output", type=Path,
                   default=PROJECT_ROOT / "results" / "fusion_subgroup_analysis.json")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args.predictions, args.output)


if __name__ == "__main__":
    main()
