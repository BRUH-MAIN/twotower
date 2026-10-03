"""Score a detector checkpoint against a held-out acquisition center.

Companion to ``cadsel.train --centers ...`` (Lead 3b: detector-level
leave-one-center-out confirmation). A LOCO training run excludes one or more
centers from `build_manifest`'s pool entirely, so its own validation split
never contains them; this module builds the excluded center's *full* sample
pool (train+val patients both, since none of them were seen in training) and
scores the checkpoint against it, the same post-hoc-rescoring pattern
``experiments/src/09_size_matched_eval.py``'s ``score_subset`` uses for
existing checkpoints.

    python -m cadsel.evaluate_holdout_centers \
        --checkpoint runs/loco_holdout_center1/best_model.pth --held-out-centers Center-1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import TASK_CLASSES
from .data import CadSelDetectionDataset, build_manifest, collate_fn, make_normalizer
from .engine import evaluate
from .models import build_detection_model, configure_transform_for_input_mode


def score_holdout(checkpoint: Path, held_out_centers: list[str], batch: int = 8) -> dict:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_type, task, modality = ck["model_type"], ck["task"], ck["modality"]
    img_size, input_mode = ck["img_size"], ck.get("input_mode", "reference")
    multiscale_sizes = ck.get("multiscale_sizes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_detection_model(model_type, task, pretrained=False)
    configure_transform_for_input_mode(model, input_mode, multiscale_sizes or img_size)
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()

    samples = build_manifest(modality, centers=held_out_centers)
    ds = CadSelDetectionDataset(samples, task, img_size, train=False, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate_fn, num_workers=0)
    metrics = evaluate(model, dl, device, TASK_CLASSES[task], progress=False,
                        normalize=make_normalizer(input_mode))
    return {
        "checkpoint": str(checkpoint),
        "modality": modality,
        "task": task,
        "held_out_centers": held_out_centers,
        "n_images": len(samples),
        "n_patients": len({s.patient_key for s in samples}),
        "AP@0.5": metrics["AP@0.5"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "per_class": metrics["per_class"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--held-out-centers", required=True,
                    help="comma-separated centers this checkpoint was NOT trained on")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", type=Path, default=None,
                    help="default: <checkpoint's run dir>/holdout_result.json")
    args = p.parse_args()

    held_out = [c.strip() for c in args.held_out_centers.split(",") if c.strip()]
    result = score_holdout(args.checkpoint, held_out, batch=args.batch_size)

    out_path = args.output or (args.checkpoint.parent / "holdout_result.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"held out {held_out}: AP@0.5={result['AP@0.5']:.4f} "
          f"(n={result['n_images']} images / {result['n_patients']} patients)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
