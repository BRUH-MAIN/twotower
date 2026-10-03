"""Full write-up of the finished runs: tables, per-class detail, training detail.

python -m cadsel.summarize --filter mobilenet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .report import (
    PAPER_TABLE4,
    PAPER_TABLE5,
    load_results,
    render_table4,
    render_table5,
    table4_rows,
    table5_rows,
)


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def render_training_detail(results: list[dict]) -> str:
    lines = [
        "### Training detail",
        "",
        "| Run | Epochs run | Best epoch | Batch | Img | Train imgs / patients | Val imgs / patients | Wall clock |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda x: x["run_name"]):
        s = r["split"]
        mins = r.get("elapsed_sec", 0) / 60
        lines.append(
            f"| {r['run_name']} | {r.get('epochs_run', '—')} | {r.get('best_epoch', '—')} | "
            f"{r['batch_size']} | {r['img_size']} | "
            f"{s['train']['images']} / {s['train']['patients']} | "
            f"{s['val']['images']} / {s['val']['patients']} | {mins:.0f} min |"
        )
    return "\n".join(lines)


def render_binary_full(results: list[dict]) -> str:
    """Table 5 models also produce precision/recall/F1 worth recording."""
    lines = [
        "### Binary task — overall detection metrics",
        "",
        "| Run | AP@0.5 (mean) | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda x: x["run_name"]):
        if r["task"] != "binary":
            continue
        b = r["best"]
        lines.append(
            f"| {r['run_name']} | {fmt(b['AP@0.5'])} | {fmt(b['precision'])} | "
            f"{fmt(b['recall'])} | {fmt(b['f1'])} | {b['tp']} | {b['fp']} | {b['fn']} |"
        )
    return "\n".join(lines)


def render_per_class_full(results: list[dict]) -> str:
    lines = [
        "### Per-class detail (best epoch of each run)",
        "",
        "| Run | Class | AP@0.5 | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(results, key=lambda x: x["run_name"]):
        for name, pc in r["best"].get("per_class", {}).items():
            lines.append(
                f"| {r['run_name']} | {name} | {fmt(pc['AP@0.5'])} | {fmt(pc['precision'])} | "
                f"{fmt(pc['recall'])} | {fmt(pc['f1'])} | {pc['tp']} | {pc['fp']} | {pc['fn']} |"
            )
    return "\n".join(lines)


def render_modality_comparison(results: list[dict]) -> str:
    """The paper's headline qualitative claim: WLE beats EUS."""
    lines = [
        "### WLE vs EUS (the paper's headline qualitative claim)",
        "",
        "| Model | Task | WLE AP@0.5 | EUS AP@0.5 | WLE better? |",
        "| --- | --- | --- | --- | --- |",
    ]
    by = {(r["model"], r["modality"], r["task"]): r for r in results}
    models = sorted({r["model"] for r in results})
    for model in models:
        for task in ("multiclass", "binary"):
            w = by.get((model, "WLE", task))
            e = by.get((model, "EUS", task))
            if not (w and e):
                continue
            wa, ea = w["best"]["AP@0.5"], e["best"]["AP@0.5"]
            lines.append(
                f"| {model} | {task} | {fmt(wa)} | {fmt(ea)} | "
                f"{'yes' if wa > ea else 'no'} |"
            )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Full results write-up")
    p.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "RESULTS.md")
    p.add_argument("--filter", default=None, help="only runs whose name contains this")
    args = p.parse_args()

    results = load_results(args.runs_dir)
    if args.filter:
        results = [r for r in results if args.filter in r["run_name"]]
    if not results:
        print(f"no results under {args.runs_dir}")
        return

    t4, t5 = table4_rows(results), table5_rows(results)
    parts = [
        "# CAD-SEL reproduction — results",
        "",
        f"{len(results)} completed runs. Held-out validation, patient-level split "
        "(no patient appears in both train and val).",
        "",
    ]
    if t4:
        parts += [render_table4(t4), ""]
    if t5:
        parts += [render_table5(t5), ""]
    parts += [
        render_modality_comparison(results), "",
        render_binary_full(results), "",
        render_per_class_full(results), "",
        render_training_detail(results), "",
    ]

    text = "\n".join(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
