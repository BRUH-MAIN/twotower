"""Experiment 46 -- download, cache and check external colonoscopy datasets for
the same-protocol comparison with CAD-SEL (``cadsel.external_datasets``).

Everything is written under ``CADSEL_EXT_ROOT`` (default ``<repo>/external_data``,
gitignored). Each step resumes: finished downloads and extracted videos are
skipped on a re-run.

  limuc       Zenodo 10.5281/zenodo.5827695 (CC BY 4.0): patient_based_classified_images.zip
              (1.9 GB), unzipped to limuc/patient_based_classified_images/<patient>/Mayo k/*.bmp.
  real_colon  figshare 22202866 (CC BY 4.0). The CSVs and per-video box annotations are small.
              The frames come as one 6-11 GB tarball per video. Only the 44 videos holding a
              histology-labelled polyp are fetched (~705 GiB), streamed, keeping up to
              --frames-per-lesion frames per polyp and discarding the rest as it goes.
              The kept frames are the polyp's annotated frames among the larger half of its
              boxes, spaced evenly in time, each with a YOLO .txt holding only that polyp's box.

After either step: ``--stats`` writes the per-modality mean/std (full frames,
whole dataset, as ``cadsel/measure_stats.py`` does for CAD-SEL). ``--check``
prints the unit / group / class counts against the published numbers, plus
the split audit (no group in two folds).

    python -m experiments.src.46_external_dataset_prep limuc
    python -m experiments.src.46_external_dataset_prep real_colon --frames-per-lesion 32
    python -m experiments.src.46_external_dataset_prep limuc --stats --check
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.data import patient_level_folds, split_group
from cadsel.external_datasets import EXT_ROOT, REAL_COLON_KEEP, get_spec

LIMUC_URL = "https://zenodo.org/api/records/5827695/files/patient_based_classified_images.zip/content"
REAL_COLON_API = "https://api.figshare.com/v2/articles/22202866/files?page_size=200"
UA = {"User-Agent": "Mozilla/5.0 (CADSEL research download)"}

# published numbers the caches must reproduce
EXPECTED = {
    "limuc": {"units": 11276, "groups": 564,
              "classes": {"Mayo 0": 6105, "Mayo 1": 3052, "Mayo 2": 1254, "Mayo 3": 865}},
    "real_colon": {"units": 112, "groups": 44,
                   "classes": {"AD": 55, "HP": 54, "SSL": 2, "TSA": 1}},   # lesion_info.csv, polyps only
}


def _open(url: str, start: int = 0):
    headers = dict(UA)
    if start:
        headers["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600)


def download(url: str, dest: Path, size: int | None = None) -> Path:
    """Stream ``url`` to ``dest``, resuming a partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if size is not None and have == size:
        return dest
    with _open(url, have) as r, open(dest, "ab" if have else "wb") as fh:
        shutil.copyfileobj(r, fh, length=1 << 22)
    if size is not None and dest.stat().st_size != size:
        raise SystemExit(f"{dest}: got {dest.stat().st_size} bytes, expected {size}")
    return dest


# ---------------------------------------------------------------------------
def prep_limuc() -> None:
    root = EXT_ROOT / "limuc"
    target = root / "patient_based_classified_images"
    if target.is_dir() and any(target.iterdir()):
        print(f"{target} exists, skipping download")
        return
    zpath = download(LIMUC_URL, root / "patient_based_classified_images.zip", 1925339722)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    zpath.unlink()
    print(f"extracted to {target}")


# ---------------------------------------------------------------------------
def _real_colon_boxes(ann_dir: Path) -> dict[str, list[tuple[int, str, tuple, tuple]]]:
    """lesion id -> [(frame index, frame file, (W, H), (xmin, ymin, xmax, ymax))]."""
    out = defaultdict(list)
    for xml in ann_dir.glob("*.xml"):
        root = ET.parse(xml).getroot()
        objs = root.findall("object")
        if not objs:
            continue
        fname = root.findtext("filename")
        W, H = int(root.findtext("size/width")), int(root.findtext("size/height"))
        t = int(Path(fname).stem.rsplit("_", 1)[1])
        for o in objs:
            b = o.find("bndbox")
            box = tuple(int(float(b.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax"))
            out[o.findtext("unique_id")].append((t, fname, (W, H), box))
    return out


def _pick_frames(entries: list, n: int) -> list:
    """The polyp's frames with the larger half of its boxes (a polyp entering
    or leaving view is a sliver), then ``n`` of them spaced evenly in time."""
    area = sorted(entries, key=lambda e: (e[3][2] - e[3][0]) * (e[3][3] - e[3][1]), reverse=True)
    big = sorted(area[:max(len(area) // 2, min(n, len(area)))], key=lambda e: e[0])
    if len(big) <= n:
        return big
    idx = np.linspace(0, len(big) - 1, n).round().astype(int)
    return [big[i] for i in sorted(set(idx.tolist()))]


def prep_real_colon(frames_per_lesion: int) -> None:
    root = EXT_ROOT / "real_colon"
    raw, cache = root / "raw", root / "cache"
    files = json.load(_open(REAL_COLON_API))
    by_name = {f["name"]: f for f in files}
    for name in ("lesion_info.csv", "video_info.csv", "dataset_description.md"):
        download(by_name[name]["download_url"], raw / name, by_name[name]["size"])
    with open(raw / "lesion_info.csv") as fh:
        lesions = [r for r in csv.DictReader(fh) if r["histology_class"] in REAL_COLON_KEEP]
    videos = sorted({r["unique_video_name"] for r in lesions})
    print(f"{len(lesions)} polyps with histology in {len(videos)} videos; frames to stream: "
          f"{sum(by_name[f'{v}_frames.tar.gz']['size'] for v in videos) / 2**30:.0f} GiB")
    (cache / "frames").mkdir(parents=True, exist_ok=True)
    keep_ids = {r["unique_object_id"] for r in lesions}

    for i, video in enumerate(videos):
        done = cache / "done" / video
        if done.exists():
            continue
        ann = by_name[f"{video}_annotations.tar.gz"]
        tgz = download(ann["download_url"], raw / ann["name"], ann["size"])
        with tarfile.open(tgz) as t:
            t.extractall(raw / "annotations", filter="data")
        boxes = _real_colon_boxes(raw / "annotations" / f"{video}_annotations")
        want = defaultdict(list)                     # frame file -> [(lesion, (W, H), box)]
        for lesion, entries in boxes.items():
            if lesion in keep_ids:
                for t_, fname, wh, box in _pick_frames(entries, frames_per_lesion):
                    want[fname].append((lesion, wh, box))
        frames = by_name[f"{video}_frames.tar.gz"]
        print(f"[{i + 1}/{len(videos)}] {video}: streaming {frames['size'] / 2**30:.1f} GiB for "
              f"{len(want)} frames", flush=True)
        got = 0
        with _open(frames["download_url"]) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
            for member in t:
                name = Path(member.name).name
                if name not in want or not member.isfile():
                    continue
                data = t.extractfile(member).read()
                for lesion, (W, H), (x1, y1, x2, y2) in want[name]:
                    out = cache / "frames" / lesion / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(data)
                    cx, cy, w, h = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
                    out.with_suffix(".txt").write_text(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                got += 1
                if got == len(want):
                    break                            # the rest of the tarball isn't needed
        if got != len(want):
            raise SystemExit(f"{video}: found {got} of {len(want)} wanted frames in the tarball")
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(str(got))
    shutil.copy(raw / "lesion_info.csv", cache / "lesions.csv")
    print(f"cache ready: {cache}")


# ---------------------------------------------------------------------------
def write_stats(name: str) -> None:
    from PIL import Image

    spec = get_spec(name)
    samples = spec.manifest()
    stats = {}
    for m in spec.modalities:
        total, total_sq, n_px = np.zeros(3), np.zeros(3), 0
        for s in (x for x in samples if x.modality == m):
            with Image.open(s.image) as im:
                px = np.asarray(im.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
            total += px.sum(0)
            total_sq += (px ** 2).sum(0)
            n_px += len(px)
        mean = total / n_px
        std = np.sqrt(np.maximum(total_sq / n_px - mean ** 2, 0.0))
        stats[m] = [[round(float(v), 6) for v in mean], [round(float(v), 6) for v in std]]
    path = EXT_ROOT / name / "modality_stats.json"
    path.write_text(json.dumps(stats, indent=1))
    print(f"wrote {path}: {stats}")


def check(name: str) -> None:
    spec = get_spec(name)
    samples = spec.manifest()
    units = {s.patient_key: s for s in samples}
    groups = {split_group(s) for s in samples}
    classes = Counter(s.subtype for s in units.values())
    print(f"{name}: {len(samples)} frames, {len(units)} units ({spec.unit}), {len(groups)} groups, "
          f"modalities {dict(Counter(s.modality for s in samples))}")
    print(f"  classes: {dict(sorted(classes.items()))}")
    exp = EXPECTED.get(name)
    if exp:
        ok = len(units) == exp["units"] and len(groups) == exp["groups"] and dict(classes) == exp["classes"]
        print(f"  vs published: units {exp['units']}, groups {exp['groups']}, classes {exp['classes']} -> "
              f"{'MATCH' if ok else 'DIFFERS'}")
    for seed in (42, 123, 7):
        folds = patient_level_folds(samples, n_folds=5, seed=seed, strata_key=spec.strata_key)
        seen = Counter(g for _, test in folds for g in {split_group(s) for s in test})
        assert all(c == 1 for c in seen.values()) and len(seen) == len(groups), "a group crosses folds"
        print(f"  seed {seed}: fold sizes (units) "
              f"{[len({s.patient_key for s in test}) for _, test in folds]}, every group in exactly one fold")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=["limuc", "real_colon"])
    p.add_argument("--frames-per-lesion", type=int, default=32)
    p.add_argument("--stats", action="store_true", help="write modality_stats.json")
    p.add_argument("--check", action="store_true", help="print counts and the split audit")
    p.add_argument("--no-download", action="store_true", help="only --stats / --check on an existing cache")
    args = p.parse_args()
    if not args.no_download:
        prep_limuc() if args.dataset == "limuc" else prep_real_colon(args.frames_per_lesion)
    if args.stats:
        write_stats(args.dataset)
    if args.check:
        check(args.dataset)


if __name__ == "__main__":
    main()
