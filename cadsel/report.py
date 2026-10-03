"""Render the reproduced Table 4 and Table 5 next to the published numbers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import MODEL_NAMES, MODEL_REGISTRY, PROJECT_ROOT

# Published results, for side-by-side comparison.
PAPER_TABLE4 = {
    ("fasterrcnn_resnet", "WLE"): (0.8593, 0.6824, 0.9146, 0.7816),
    ("fasterrcnn_resnet", "EUS"): (0.6112, 0.5938, 0.8206, 0.6890),
    ("fasterrcnn_resnet50_fpn_v2", "WLE"): (0.8792, 0.6772, 0.9290, 0.7834),
    ("fasterrcnn_resnet50_fpn_v2", "EUS"): (0.5769, 0.6071, 0.8129, 0.6951),
    ("fasterrcnn_mobilenet", "WLE"): (0.8817, 0.8014, 0.9067, 0.8508),
    ("fasterrcnn_mobilenet", "EUS"): (0.5768, 0.7764, 0.7561, 0.7661),
    ("fasterrcnn_mobilenet_320", "WLE"): (0.8641, 0.8661, 0.8922, 0.8790),
    ("fasterrcnn_mobilenet_320", "EUS"): (0.5320, 0.7730, 0.6840, 0.7258),
}

PAPER_TABLE5 = {
    ("fasterrcnn_resnet", "WLE"): (0.9119, 0.8068),
    ("fasterrcnn_resnet", "EUS"): (0.8011, 0.4213),
    ("fasterrcnn_resnet50_fpn_v2", "WLE"): (0.9105, 0.8480),
    ("fasterrcnn_resnet50_fpn_v2", "EUS"): (0.7814, 0.3725),
    ("fasterrcnn_mobilenet", "WLE"): (0.8851, 0.8784),
    ("fasterrcnn_mobilenet", "EUS"): (0.8143, 0.3393),
    ("fasterrcnn_mobilenet_320", "WLE"): (0.9023, 0.8259),
    ("fasterrcnn_mobilenet_320", "EUS"): (0.7257, 0.3383),
}


def load_results(runs_dir: Path) -> list[dict]:
    results = []
    for path in sorted(runs_dir.glob("*/result.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def canonical_run_name(model: str, modality: str, task: str) -> str:
    return f"{model}__{modality}__{task}"


def _index(results: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Index the *baseline grid* runs by (model, modality, task).

    Only runs whose ``run_name`` is the canonical ``model__modality__task`` are
    eligible. ``runs/`` also holds ablation and multi-seed runs — ``ms_baseline_s7``,
    ``hn30``, ``*__mitigated`` — which share the same (model, modality, task) key
    and would otherwise overwrite a published-comparison row depending only on
    sort order. That silently replaced the Mobilenet_320 WLE multiclass AP with a
    multi-seed run's 0.4397 in place of the baseline's 0.3809.
    """
    index: dict[tuple[str, str, str], dict] = {}
    for r in results:
        key = (r["model"], r["modality"], r["task"])
        if r.get("run_name") != canonical_run_name(*key):
            continue
        index[key] = r
    return index


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def _delta(ours, paper) -> str:
    if ours is None or paper is None:
        return "—"
    return f"{ours - paper:+.4f}"


def table4_rows(results: list[dict]) -> list[dict]:
    idx = _index(results)
    rows = []
    for model in MODEL_NAMES:
        for modality in ("WLE", "EUS"):
            r = idx.get((model, modality, "multiclass"))
            if r is None:
                continue
            best = r["best"]
            paper = PAPER_TABLE4.get((model, modality), (None, None, None, None))
            rows.append(
                {
                    "model": MODEL_REGISTRY[model]["paper_name"],
                    "modality": modality,
                    "ap50": best["AP@0.5"],
                    "precision": best["precision"],
                    "recall": best["recall"],
                    "f1": best["f1"],
                    "ap50_conf": best.get("AP@0.5_conf"),
                    "paper_ap50": paper[0],
                    "paper_precision": paper[1],
                    "paper_recall": paper[2],
                    "paper_f1": paper[3],
                    "best_epoch": r.get("best_epoch"),
                    "per_class": best.get("per_class", {}),
                }
            )
    return rows


def table5_rows(results: list[dict]) -> list[dict]:
    idx = _index(results)
    rows = []
    for model in MODEL_NAMES:
        for modality in ("WLE", "EUS"):
            r = idx.get((model, modality, "binary"))
            if r is None:
                continue
            per_class = r["best"].get("per_class", {})
            paper = PAPER_TABLE5.get((model, modality), (None, None))
            rows.append(
                {
                    "model": MODEL_REGISTRY[model]["paper_name"],
                    "modality": modality,
                    "ap50_net": per_class.get("NET", {}).get("AP@0.5"),
                    "ap50_nonnet": per_class.get("non-NET", {}).get("AP@0.5"),
                    "paper_ap50_net": paper[0],
                    "paper_ap50_nonnet": paper[1],
                    "best_epoch": r.get("best_epoch"),
                }
            )
    return rows


def render_table4(rows: list[dict]) -> str:
    lines = [
        "### Table 4 — Multi-class detection (reproduction vs. paper)",
        "",
        "| Tested model | Modality | AP@0.5 | Precision | Recall | F1-score | paper AP@0.5 | paper P | paper R | paper F1 | ΔAP | ΔF1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {m} | {mod} | {ap} | {p} | {rc} | {f1} | {pap} | {pp} | {pr} | {pf} | {dap} | {df} |".format(
                m=r["model"],
                mod=r["modality"],
                ap=_fmt(r["ap50"]),
                p=_fmt(r["precision"]),
                rc=_fmt(r["recall"]),
                f1=_fmt(r["f1"]),
                pap=_fmt(r["paper_ap50"]),
                pp=_fmt(r["paper_precision"]),
                pr=_fmt(r["paper_recall"]),
                pf=_fmt(r["paper_f1"]),
                dap=_delta(r["ap50"], r["paper_ap50"]),
                df=_delta(r["f1"], r["paper_f1"]),
            )
        )
    return "\n".join(lines)


def render_table5(rows: list[dict]) -> str:
    lines = [
        "### Table 5 — Binary detection, NET vs non-NET (reproduction vs. paper)",
        "",
        "| Tested model | Modality | AP@0.5 (NET) | AP@0.5 (non-NET) | paper NET | paper non-NET | Δ NET | Δ non-NET |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            "| {m} | {mod} | {a} | {b} | {pa} | {pb} | {da} | {db} |".format(
                m=r["model"],
                mod=r["modality"],
                a=_fmt(r["ap50_net"]),
                b=_fmt(r["ap50_nonnet"]),
                pa=_fmt(r["paper_ap50_net"]),
                pb=_fmt(r["paper_ap50_nonnet"]),
                da=_delta(r["ap50_net"], r["paper_ap50_net"]),
                db=_delta(r["ap50_nonnet"], r["paper_ap50_nonnet"]),
            )
        )
    return "\n".join(lines)


def render_per_class(rows: list[dict]) -> str:
    lines = [
        "### Per-class AP@0.5, multi-class task",
        "",
        "| Tested model | Modality | NET | Leiomyoma | Lipoma | NonNeoplasm |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        pc = r["per_class"]
        cells = [_fmt(pc.get(name, {}).get("AP@0.5")) for name in
                 ("NET", "Leiomyoma", "Lipoma", "NonNeoplasm")]
        lines.append(f"| {r['model']} | {r['modality']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multimodal report (joint detector + fusion head)
# ---------------------------------------------------------------------------
# Kept in a separate runs directory and a separate output file. `_index` above
# keys on (model, modality, task), so a shared-split unimodal baseline would
# collide with the already-completed per-modality-split run of the same name and
# silently overwrite a Table 4 row.
def _split_label(result: dict) -> str:
    return "image" if result.get("split_level") == "image" else "patient"


def joint_rows(results: list[dict]) -> list[dict]:
    """Per-modality AP for every joint run, keyed so protocols do not collide."""
    rows = []
    for r in results:
        if r.get("modality") != "JOINT":
            continue
        best = r["best"]
        per_modality = best.get("per_modality", {})
        rows.append({
            "model": MODEL_REGISTRY[r["model"]]["paper_name"],
            "model_key": r["model"],
            "conditioning": r.get("conditioning", "?"),
            "task": r["task"],
            "protocol": _split_label(r),
            "ap50_wle": per_modality.get("WLE", {}).get("AP@0.5"),
            "ap50_eus": per_modality.get("EUS", {}).get("AP@0.5"),
            "ap50_macro": best.get("AP@0.5_macro"),
            "f1": best.get("f1"),
            "hard_negatives": r.get("hard_negatives", 0.0),
            "best_epoch": r.get("best_epoch"),
            "per_class": best.get("per_class", {}),
        })
    return sorted(rows, key=lambda r: (r["task"], r["protocol"], r["conditioning"]))


def baseline_rows(results: list[dict]) -> dict[tuple[str, str, str, str], float]:
    """AP@0.5 of the shared-split unimodal runs, the denominator for the joint rows.

    The completed baselines in ``runs/`` cannot serve this purpose: they were split
    per modality, which puts 67 of 85 EUS validation patients' WLE frames in the
    training set. Only runs on the shared split are comparable.
    """
    out = {}
    for r in results:
        if r.get("modality") not in ("WLE", "EUS"):
            continue
        key = (r["model"], r["modality"], r["task"], _split_label(r))
        out[key] = r["best"]["AP@0.5"]
    return out


def render_joint(rows: list[dict], baselines: dict) -> str:
    lines = [
        "### Joint WLE+EUS detector — per-modality AP@0.5",
        "",
        "`Δ` compares against the shared-split unimodal baseline for the same "
        "model, task and protocol. Per FINDINGS §8.3 the seed-to-seed spread on "
        "this dataset is ~0.02 AP, so treat anything smaller as noise.",
        "",
        "| Model | Task | Protocol | Cond. | HN | AP WLE | Δ WLE | AP EUS | Δ EUS | macro AP | F1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        base_wle = baselines.get((r["model_key"], "WLE", r["task"], r["protocol"]))
        base_eus = baselines.get((r["model_key"], "EUS", r["task"], r["protocol"]))
        lines.append(
            "| {m} | {t} | {proto} | {c} | {hn} | {aw} | {dw} | {ae} | {de} | {mac} | {f1} |".format(
                m=r["model"], t=r["task"], proto=r["protocol"], c=r["conditioning"],
                hn=f"{r['hard_negatives']:.0%}" if r["hard_negatives"] else "—",
                aw=_fmt(r["ap50_wle"]), dw=_delta(r["ap50_wle"], base_wle),
                ae=_fmt(r["ap50_eus"]), de=_delta(r["ap50_eus"], base_eus),
                mac=_fmt(r["ap50_macro"]), f1=_fmt(r["f1"]),
            )
        )
    return "\n".join(lines)


def render_shared_baselines(results: list[dict]) -> str:
    lines = [
        "### Shared-split unimodal baselines",
        "",
        "Re-run on the joint (leak-free) split, because the completed grid in "
        "`runs/` split each modality independently and is therefore not a valid "
        "comparison for a model that sees both.",
        "",
        "| Model | Modality | Task | Protocol | AP@0.5 | F1 | epoch |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [r for r in results if r.get("modality") in ("WLE", "EUS")]
    for r in sorted(rows, key=lambda r: (r["task"], _split_label(r), r["modality"])):
        lines.append(
            f"| {MODEL_REGISTRY[r['model']]['paper_name']} | {r['modality']} | "
            f"{r['task']} | {_split_label(r)} | {_fmt(r['best']['AP@0.5'])} | "
            f"{_fmt(r['best']['f1'])} | {r.get('best_epoch')} |"
        )
    return "\n".join(lines)


def render_fusion(payload: dict) -> str:
    lines = [
        "### Patient-level fusion head",
        "",
        f"{payload['folds']}-fold patient-level cross-validation, "
        f"classes: {', '.join(payload['classes'])}.",
        "",
        f"**{payload['protocol']}** — this scores a patient, not an image.",
        "",
        "| Ablation | Balanced acc. | Macro F1 | Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for name, stats in payload["ablations"].items():
        lines.append(
            f"| {name} | {stats['bal']['mean']:.4f} ± {stats['bal']['std']:.4f} "
            f"| {stats['macro_f1']['mean']:.4f} ± {stats['macro_f1']['std']:.4f} "
            f"| {stats['acc']['mean']:.4f} ± {stats['acc']['std']:.4f} |"
        )
    return "\n".join(lines)


def write_multimodal_report(
    results: list[dict], results_dir: Path, fusion_path: Path | None = None
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = joint_rows(results)
    baselines = baseline_rows(results)

    parts = ["# CAD-SEL multimodal (WLE + EUS)", ""]
    if any(r.get("modality") in ("WLE", "EUS") for r in results):
        parts += [render_shared_baselines(results), ""]
    if rows:
        parts += [render_joint(rows, baselines), ""]
        parts += [render_per_class(
            [{**r, "modality": f"{r['conditioning']}/{r['protocol']}"} for r in rows]
        ), ""]
    if fusion_path and fusion_path.exists():
        parts += [render_fusion(json.loads(fusion_path.read_text(encoding="utf-8"))), ""]

    (results_dir / "MULTIMODAL.md").write_text("\n".join(parts), encoding="utf-8")

    if rows:
        with open(results_dir / "joint_detection.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["model", "task", "protocol", "conditioning",
                            "hard_negatives", "ap50_wle", "ap50_eus", "ap50_macro",
                            "f1", "best_epoch"],
                extrasaction="ignore",
            )
            w.writeheader()
            w.writerows(rows)


def write_reports(results: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    t4 = table4_rows(results)
    t5 = table5_rows(results)

    parts = ["# CAD-SEL technical validation — reproduction", ""]
    if t4:
        parts += [render_table4(t4), "", render_per_class(t4), ""]
    if t5:
        parts += [render_table5(t5), ""]
    (results_dir / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")

    if t4:
        with open(results_dir / "table4_multiclass.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "model", "modality", "ap50", "precision", "recall", "f1",
                    "ap50_conf", "paper_ap50", "paper_precision", "paper_recall",
                    "paper_f1", "best_epoch",
                ],
                extrasaction="ignore",
            )
            w.writeheader()
            w.writerows(t4)

    if t5:
        with open(results_dir / "table5_binary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "model", "modality", "ap50_net", "ap50_nonnet",
                    "paper_ap50_net", "paper_ap50_nonnet", "best_epoch",
                ],
                extrasaction="ignore",
            )
            w.writeheader()
            w.writerows(t5)

    (results_dir / "all_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Render Table 4 / Table 5 from finished runs")
    p.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument(
        "--joint-runs-dir",
        type=Path,
        default=PROJECT_ROOT / "runs_joint",
        help="multimodal runs; kept apart from --runs-dir so a shared-split "
        "baseline cannot overwrite a Table 4 row",
    )
    p.add_argument("--fusion-results", type=Path,
                   default=PROJECT_ROOT / "results" / "fusion.json")
    p.add_argument("--multimodal-only", action="store_true")
    args = p.parse_args()

    if not args.multimodal_only:
        results = load_results(args.runs_dir)
        if results:
            write_reports(results, args.results_dir)
            print((args.results_dir / "REPORT.md").read_text(encoding="utf-8"))
        else:
            print(f"No result.json found under {args.runs_dir}")

    if args.joint_runs_dir.is_dir():
        joint = load_results(args.joint_runs_dir)
        if joint or args.fusion_results.exists():
            write_multimodal_report(joint, args.results_dir, args.fusion_results)
            print("\n" + (args.results_dir / "MULTIMODAL.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
