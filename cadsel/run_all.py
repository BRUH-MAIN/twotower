"""Run the full technical-validation grid: 4 detectors x 2 modalities x 2 tasks.

This reproduces the experiments behind Table 4 (multi-class detection) and
Table 5 (binary NET vs non-NET detection).

Example
-------
python -m cadsel.run_all --epochs 40
python -m cadsel.run_all --models fasterrcnn_mobilenet_320 --tasks binary
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
from .experiment_config import RunConfig, apply_to_namespace
from .report import load_results, write_reports
from .train import build_parser as build_train_parser
from .train import run as run_training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the CAD-SEL baseline grid")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=MODEL_NAMES)
    p.add_argument("--modalities", nargs="+", choices=MODALITIES, default=MODALITIES)
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="stop a run after N epochs without a validation AP@0.5 improvement",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=min(8, max(2, (os.cpu_count() or 4) // 2)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--augment", default=None,
                   help="comma-separated augmentations; see cadsel.train --help")
    p.add_argument("--input-mode", choices=["reference", "torchvision"], default="reference")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--device", default=None, help="e.g. cuda:0 / cuda:1")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--skip-existing", action="store_true", help="reuse finished runs")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-progress", action="store_true", help="drop tqdm bars, keep epoch lines")
    p.add_argument(
        "--config-grid",
        type=Path,
        default=None,
        help="JSON list of RunConfig dicts (cadsel.experiment_config.RunConfig), run "
        "instead of the --models/--modalities/--tasks grid; each entry fully "
        "specifies one run (including its run name, via 'name').",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    configs = None
    if args.config_grid is not None:
        configs = [RunConfig.from_dict(d) for d in json.loads(args.config_grid.read_text(encoding="utf-8"))]
        grid = [(c.backbone, c.modality, c.task) for c in configs]
    else:
        grid = [
            (model, modality, task)
            for task in args.tasks
            for modality in args.modalities
            for model in args.models
        ]
    print(f"Running {len(grid)} experiment(s)\n", flush=True)
    grid_started = time.time()

    results = []
    failures: list[str] = []
    for i, (model, modality, task) in enumerate(grid, start=1):
        run_name = configs[i - 1].name if configs is not None else f"{model}__{modality}__{task}"
        result_path = args.output_dir / run_name / "result.json"

        if args.skip_existing and result_path.exists():
            print(f"[{i}/{len(grid)}] {run_name}: reusing existing result", flush=True)
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        elapsed = time.time() - grid_started
        print(f"[{i}/{len(grid)}] {run_name}  (grid elapsed {elapsed / 60:.0f} min)", flush=True)
        train_args = build_train_parser().parse_args([])

        if configs is not None:
            apply_to_namespace(configs[i - 1], train_args, explicit=set())
            train_args.output_dir = args.output_dir
            train_args.quiet = args.quiet
            train_args.no_progress = args.no_progress
        else:
            train_args.model = model
            train_args.modality = modality
            train_args.task = task
            train_args.epochs = args.epochs
            train_args.learning_rate = args.learning_rate
            train_args.warmup_iters = args.warmup_iters
            train_args.early_stop_patience = args.early_stop_patience
            train_args.batch_size = args.batch_size
            train_args.img_size = args.img_size
            train_args.num_workers = args.num_workers
            train_args.seed = args.seed
            train_args.val_fraction = args.val_fraction
            train_args.input_mode = args.input_mode
            if args.augment is not None:
                train_args.augment = args.augment
            train_args.no_class_weights = args.no_class_weights
            train_args.no_amp = args.no_amp
            train_args.output_dir = args.output_dir
            train_args.data_root = args.data_root
            train_args.device = args.device
            train_args.quiet = args.quiet
            train_args.no_progress = args.no_progress

        try:
            results.append(run_training(train_args))
        except Exception:  # keep the grid going if one config fails
            print(f"  FAILED: {run_name}")
            traceback.print_exc()
            failures.append(run_name)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print()

    if results:
        # Re-scan the runs directory rather than trusting this process's own
        # in-memory `results` list -- a partial grid (--config-grid on a
        # handful of runs) would otherwise overwrite the shared aggregate
        # with just those few entries. See FINDINGS.md for the incident this
        # fixes: a `gan_copypaste_grid.json` run wiped `all_results.json`
        # down to its own 8 entries and left REPORT.md's tables empty.
        write_reports(load_results(args.output_dir), args.results_dir)
        print(f"Reports written to {args.results_dir}")

    if failures:
        print(f"\n{len(failures)} of {len(grid)} run(s) FAILED: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
