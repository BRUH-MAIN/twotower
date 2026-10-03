"""Experiment 28 (Track B, Phase 1) — export a CAD-SEL modality/task slice as
an Ultralytics YOLO dataset, reusing an existing Faster R-CNN run's exact
patient-level split.

    python experiments/src/28_export_yolo_dataset.py --modality WLE \\
        --task multiclass --base-run size_rfs_ms_s42__WLE__multiclass \\
        --out experiments/yolo/wle_multiclass_ms_only_s42

The on-disk labels are already literal YOLO-txt (``class cx cy w h``,
normalised 0-1) -- ``cadsel.data.read_yolo_label`` parses them, and this
script only needs to remap the raw annotation class id (0-3) through
``TASK_ID_MAP[task]`` (still 0-based: Ultralytics has no reserved background
class, unlike Faster R-CNN's ``+1``) before writing it back out.

Numbered as 28, not 26 as a first draft of the plan this follows suggested --
Track A's classifier leaderboard (26) and counterfactual gate (27) claimed
those numbers first when both tracks were built in the same session; see
FINDINGS.md for the final numbering actually used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from cadsel.config import MODALITIES, TASK_CLASSES, TASK_ID_MAP
from cadsel.data import build_manifest, read_yolo_label
from yolo_common import filter_by_split, flat_stem


def export_split(samples, images_dir: Path, labels_dir: Path, task: str) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    id_map = TASK_ID_MAP[task]
    n = 0
    for s in samples:
        stem = flat_stem(s)
        src_image = Path(s.image).resolve()
        dst_image = images_dir / (stem + src_image.suffix)
        if dst_image.exists() or dst_image.is_symlink():
            dst_image.unlink()
        dst_image.symlink_to(src_image)

        lines = []
        for cls, cx, cy, bw, bh in read_yolo_label(s.label):
            remapped = id_map[cls]
            lines.append(f"{remapped} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (labels_dir / (stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modality", choices=MODALITIES, required=True)
    p.add_argument("--task", choices=["binary", "multiclass"], required=True)
    p.add_argument("--base-run", required=True, help="existing runs/<name> to reuse the split.json of")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    samples = build_manifest(args.modality)
    train, val = filter_by_split(samples, args.base_run)
    print(f"{args.modality}/{args.task}: {len(samples)} total, "
          f"{len(train)} train / {len(val)} val (split from runs/{args.base_run}/split.json)")

    out = args.out.resolve()
    n_train = export_split(train, out / "images" / "train", out / "labels" / "train", args.task)
    n_val = export_split(val, out / "images" / "val", out / "labels" / "val", args.task)

    class_names = TASK_CLASSES[args.task]
    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"wrote {out / 'data.yaml'}: nc={len(class_names)} names={class_names}")

    # -- go/no-go checks ------------------------------------------------
    ok = True
    if n_train != len(train) or n_val != len(val):
        print(f"  FAIL: exported {n_train}/{n_val} but expected {len(train)}/{len(val)}")
        ok = False
    else:
        print(f"  OK: exported image count matches filtered sample count "
              f"({n_train} train, {n_val} val)")

    spot = list((out / "labels" / "train").glob("*.txt"))[:3]
    for label_path in spot:
        rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
        ids = {int(r[0]) for r in rows}
        valid_ids = set(TASK_ID_MAP[args.task].values())
        bad = ids - valid_ids
        status = "OK" if not bad else f"FAIL (unexpected ids {bad})"
        print(f"  spot-check {label_path.name}: class ids {sorted(ids)} -> {status}")
        ok = ok and not bad

    yaml_names = yaml.safe_load((out / "data.yaml").read_text())["names"]
    names_ok = yaml_names == class_names
    print(f"  data.yaml names match TASK_CLASSES[{args.task}]: {names_ok}")
    ok = ok and names_ok

    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED -- do not train on this export'}")


if __name__ == "__main__":
    main()
