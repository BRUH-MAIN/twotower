"""Run the whole multimodal study in dependency order.

Queues, in this order:

1. **Shared-split unimodal baselines** (WLE/EUS x task). These are the denominator.
   The completed grid in ``runs/`` cannot serve the purpose: it split each modality
   independently, leaving 67 of 85 EUS validation patients with their WLE frames in
   the training set.
2. **The modality-blind joint control** (``--conditioning none``). Without it a
   gain cannot be attributed to conditioning rather than to 1.75x the images.
3. **The conditioned joint runs** (``--conditioning film``).
4. Optionally the same under ``--split-level image``, which is the only column
   comparable to the paper's published figures.

Everything lands in ``runs_joint/`` so a shared-split baseline can never overwrite
a Table 4 row in ``runs/`` — see ``report._index``.

    python -m cadsel.run_multimodal --epochs 100 --early-stop-patience 15
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch

from .config import MODALITIES, MODEL_NAMES, PROJECT_ROOT, TASKS
from .models import CONDITIONING_MODES
from .report import write_multimodal_report
from .train import build_parser as build_train_parser
from .train import run as run_unimodal
from .train_joint import build_parser as build_joint_parser
from .train_joint import run as run_joint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the CAD-SEL multimodal study")
    p.add_argument("--model", choices=MODEL_NAMES, default="fasterrcnn_mobilenet",
                   help="MobileNet keeps a run to 0.3-1.0 h; the ResNet grid is "
                        "deliberately out of scope here")
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    p.add_argument("--conditioning", nargs="+", choices=CONDITIONING_MODES,
                   default=["none", "film"])
    p.add_argument("--image-level", action="store_true",
                   help="also run the image-level protocol column (doubles the grid)")
    p.add_argument("--skip-baselines", action="store_true")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=3,
                   help="3 matches the completed grid; each Windows worker costs "
                        "~600 MB because it re-imports torch")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hard-negatives", type=float, default=0.3,
                   help="the audit's clearest win: false alarms -69%%/-86%%, F1 +0.16")
    p.add_argument("--augment", default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs_joint")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-progress", action="store_true", default=True)
    return p


def _plan(args) -> list[dict]:
    """The queue, baselines first so the denominator exists before the numerator."""
    levels = ["patient"] + (["image"] if args.image_level else [])
    jobs: list[dict] = []

    if not args.skip_baselines:
        for level in levels:
            for task in args.tasks:
                for modality in MODALITIES:
                    suffix = "" if level == "patient" else f"__{level}level"
                    jobs.append({
                        "kind": "unimodal",
                        "run_name": f"shared__{args.model}__{modality}__{task}{suffix}",
                        "modality": modality, "task": task, "split_level": level,
                    })

    for level in levels:
        for task in args.tasks:
            for conditioning in args.conditioning:
                suffix = "" if level == "patient" else f"__{level}level"
                jobs.append({
                    "kind": "joint",
                    "run_name": f"joint_{conditioning}__{args.model}__{task}{suffix}",
                    "conditioning": conditioning, "task": task, "split_level": level,
                })
    return jobs


def _shared(dest, args) -> None:
    dest.model = args.model
    dest.epochs = args.epochs
    dest.early_stop_patience = args.early_stop_patience
    dest.learning_rate = args.learning_rate
    dest.warmup_iters = args.warmup_iters
    dest.batch_size = args.batch_size
    dest.img_size = args.img_size
    dest.num_workers = args.num_workers
    dest.seed = args.seed
    dest.hard_negatives = args.hard_negatives
    if args.augment is not None:
        dest.augment = args.augment
    dest.data_root = args.data_root
    dest.device = args.device
    dest.output_dir = args.output_dir
    dest.quiet = args.quiet
    dest.no_progress = args.no_progress


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    jobs = _plan(args)
    print(f"Multimodal study: {len(jobs)} run(s) into {args.output_dir}\n", flush=True)
    for i, job in enumerate(jobs, start=1):
        print(f"  {i:2d}. {job['run_name']}")
    print(flush=True)

    started = time.time()
    results: list[dict] = []
    failures: list[str] = []

    for i, job in enumerate(jobs, start=1):
        result_path = args.output_dir / job["run_name"] / "result.json"
        if args.skip_existing and result_path.exists():
            print(f"[{i}/{len(jobs)}] {job['run_name']}: reusing existing result",
                  flush=True)
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        print(f"[{i}/{len(jobs)}] {job['run_name']}  "
              f"(elapsed {(time.time() - started) / 60:.0f} min)", flush=True)
        try:
            if job["kind"] == "unimodal":
                run_args = build_train_parser().parse_args([])
                _shared(run_args, args)
                run_args.modality = job["modality"]
                run_args.task = job["task"]
                run_args.split_level = job["split_level"]
                run_args.run_name = job["run_name"]
                # The baselines must sit on the *joint* split to be a valid
                # denominator, so they take the same strata as the joint runs.
                run_args.joint_split = True
                results.append(run_unimodal(run_args))
            else:
                run_args = build_joint_parser().parse_args([])
                _shared(run_args, args)
                run_args.task = job["task"]
                run_args.conditioning = job["conditioning"]
                run_args.split_level = job["split_level"]
                run_args.run_name = job["run_name"]
                results.append(run_joint(run_args))
        except Exception:
            print(f"  FAILED: {job['run_name']}")
            traceback.print_exc()
            failures.append(job["run_name"])
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(flush=True)

    if results:
        write_multimodal_report(results, args.results_dir,
                                args.results_dir / "fusion.json")
        print(f"Report written to {args.results_dir / 'MULTIMODAL.md'}")

    print(f"\ntotal {(time.time() - started) / 3600:.2f} h")
    if failures:
        print(f"{len(failures)} of {len(jobs)} run(s) FAILED: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
