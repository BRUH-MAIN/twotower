"""Dataset registry for running the two-tower / single-backbone pipeline on
colonoscopy datasets other than CAD-SEL, under CAD-SEL's exact protocol.

A ``DatasetSpec`` says what the pipeline needs to know about a dataset:
- its modalities, in tower order. The first is the stream ``GatedFusion``
  falls back to; one modality gives the single-backbone model.
- its tasks and their class lists, and how a sample's ``subtype`` maps to a
  task's class.
- what a classified *unit* is: a patient (CAD-SEL), a lesion, or an image.
- how to build its ``Sample`` manifest.

Each ``Sample`` carries:
- ``patient_key`` = the unit, the bag of frames the model classifies.
- ``group`` = what must not cross a split (the patient or video).
- ``subtype`` = the finest label available.

External data lives under ``CADSEL_EXT_ROOT`` (default ``<repo>/external_data``,
gitignored). ``experiments/src/46_external_dataset_prep.py`` downloads and
caches each dataset there. Nothing here downloads anything.

Datasets whose layout is not known yet are registered with their modalities and
classes, but their manifest raises until the data is in hand: CPC-Paired,
PIBAdb and PICCOLO all need a request or a confirmed host.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .config import MODALITY_NAMES, PROJECT_ROOT
from .data import IMAGE_EXTENSIONS, Sample, build_joint_manifest, joint_strata_key
from .fusion import BINARY, SUBTYPES, patient_label

EXT_ROOT = Path(os.environ.get("CADSEL_EXT_ROOT", PROJECT_ROOT / "external_data"))


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    modalities: tuple[str, ...]
    tasks: dict[str, tuple[str, ...]]
    label_of: Callable[[str, str], str]          # (subtype, task) -> class name
    manifest: Callable[[], list[Sample]]
    unit: str                                    # "patient" | "lesion" | "image"
    strata_key: Callable[[Sequence[Sample]], object]
    has_boxes: bool                              # lesion-crop possible
    centres: str = ""                            # free-text, for the comparison table

    def classes(self, task: str) -> list[str]:
        if task not in self.tasks:
            raise SystemExit(f"{self.name}: unknown task {task!r}; expected one of {sorted(self.tasks)}")
        return list(self.tasks[task])

    def modality_stats(self) -> dict | None:
        """Per-modality (mean, std) for the "modality"-normalised CNN towers,
        written by 46's ``stats`` step. None = CAD-SEL's MODALITY_STATS."""
        if self.name == "cadsel":
            return None
        path = EXT_ROOT / self.name / "modality_stats.json"
        if not path.exists():
            raise SystemExit(f"{path} missing -- run 46_external_dataset_prep.py {self.name} --stats first")
        return {m: tuple(map(tuple, v)) for m, v in json.loads(path.read_text()).items()}


def majority_strata_key(items: Sequence[Sample]):
    """Stratum of a split group that can hold units of different classes (a
    LIMUC patient's images, a REAL-colon video's polyps): its most frequent
    subtype, ties broken by name."""
    counts = Counter(s.subtype for s in items)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS + (".bmp",))


# ---------------------------------------------------------------------------
# LIMUC: ulcerative colitis, Mayo endoscopic score 0-3, one WL image = one unit.
# Zenodo 10.5281/zenodo.5827695 (CC BY 4.0), patient_based_classified_images.zip:
#   patient_based_classified_images/<patient id>/Mayo <k>/UC_patient_<id>_<n>.bmp
# ---------------------------------------------------------------------------
LIMUC_CLASSES = ("Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3")


def _limuc_label(subtype: str, task: str) -> str:
    if task == "mayo":
        return subtype
    # the usual binary endpoint: endoscopic remission (Mayo 0-1) vs active disease (2-3)
    return "remission" if subtype in ("Mayo 0", "Mayo 1") else "active"


def _limuc_manifest() -> list[Sample]:
    root = EXT_ROOT / "limuc" / "patient_based_classified_images"
    if not root.is_dir():
        raise SystemExit(f"{root} missing -- run 46_external_dataset_prep.py limuc")
    out = []
    for patient_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not patient_dir.is_dir():
            continue
        for grade_dir in sorted(patient_dir.iterdir()):
            if grade_dir.name not in LIMUC_CLASSES:
                continue
            for img in _image_files(grade_dir):
                out.append(Sample(image=str(img), label="", center="LIMUC", modality="WL",
                                  subtype=grade_dir.name, patient=img.stem,
                                  group=f"limuc-{patient_dir.name}"))
    return out


# ---------------------------------------------------------------------------
# REAL-colon: 60 full-procedure WL videos from 4 studies (6 centres); each resected
# polyp is boxed in every frame it appears in (unique_id per box) with histology
# in lesion_info.csv. figshare 22202866 (CC BY 4.0). 46 caches up to N frames per
# polyp with a YOLO box file:
#   real_colon/cache/lesions.csv   (unique_object_id, unique_video_name, histology_class, ...)
#   real_colon/cache/frames/<unique_object_id>/<frame>.jpg + <frame>.txt
# One polyp = one unit, one video (= one patient) = one split group.
# ---------------------------------------------------------------------------
REAL_COLON_KEEP = {"AD": "neoplastic", "SSL": "neoplastic", "TSA": "neoplastic", "HP": "hyperplastic"}


def _real_colon_label(subtype: str, task: str) -> str:
    return REAL_COLON_KEEP[subtype]


def _real_colon_manifest() -> list[Sample]:
    cache = EXT_ROOT / "real_colon" / "cache"
    if not (cache / "lesions.csv").exists():
        raise SystemExit(f"{cache / 'lesions.csv'} missing -- run 46_external_dataset_prep.py real_colon")
    out = []
    with open(cache / "lesions.csv") as fh:
        for row in csv.DictReader(fh):
            if row["histology_class"] not in REAL_COLON_KEEP:
                continue                    # no-polyp / other: not a polyp-histology unit
            lesion, video = row["unique_object_id"], row["unique_video_name"]
            for img in _image_files(cache / "frames" / lesion):
                out.append(Sample(image=str(img), label=str(img.with_suffix(".txt")),
                                  center=f"REAL-{video.split('-')[0]}", modality="WL",
                                  subtype=row["histology_class"], patient=lesion,
                                  group=f"real_colon-{video}"))
    return out


# ---------------------------------------------------------------------------
# Paired WL + NBI datasets whose files are not in hand yet.
# ---------------------------------------------------------------------------
def _pending(name: str, how: str) -> Callable[[], list[Sample]]:
    def manifest() -> list[Sample]:
        raise SystemExit(f"{name}: data not available yet ({how}); its adapter is written once the "
                         f"layout is known")
    return manifest


def _neoplastic_label(subtype: str, task: str) -> str:
    if task == "binary":
        return "neoplastic" if subtype in ("adenoma", "SSL", "TSA") else "hyperplastic"
    return subtype


SPECS: dict[str, DatasetSpec] = {
    "cadsel": DatasetSpec(
        name="cadsel", modalities=tuple(MODALITY_NAMES),
        tasks={"binary": tuple(BINARY), "subtype": tuple(SUBTYPES)},
        label_of=patient_label, manifest=build_joint_manifest, unit="patient",
        strata_key=joint_strata_key, has_boxes=True, centres="6 (Center-1 537 of 641)"),
    "limuc": DatasetSpec(
        name="limuc", modalities=("WL",),
        tasks={"mayo": LIMUC_CLASSES, "remission": ("remission", "active")},
        label_of=_limuc_label, manifest=_limuc_manifest, unit="image",
        strata_key=majority_strata_key, has_boxes=False, centres="1 (Marmara University)"),
    "real_colon": DatasetSpec(
        name="real_colon", modalities=("WL",),
        tasks={"binary": ("neoplastic", "hyperplastic")},
        label_of=_real_colon_label, manifest=_real_colon_manifest, unit="lesion",
        strata_key=majority_strata_key, has_boxes=True, centres="6 in 4 countries (4 studies)"),
    "cpc_paired": DatasetSpec(
        name="cpc_paired", modalities=("WL", "NBI"),
        tasks={"binary": ("neoplastic", "hyperplastic")},
        label_of=_neoplastic_label, manifest=_pending("cpc_paired", "download host unverified"),
        unit="lesion", strata_key=majority_strata_key, has_boxes=True, centres="ISIT-UMR + 1 hospital"),
    "pibadb": DatasetSpec(
        name="pibadb", modalities=("WL", "NBI"),
        tasks={"binary": ("neoplastic", "hyperplastic"), "subtype": ("adenoma", "SSL", "hyperplastic")},
        label_of=_neoplastic_label, manifest=_pending("pibadb", "IIS Galicia Sur biobank request"),
        unit="lesion", strata_key=majority_strata_key, has_boxes=True, centres="1 (Ourense)"),
    "piccolo": DatasetSpec(
        name="piccolo", modalities=("WL", "NBI"),
        tasks={"binary": ("neoplastic", "hyperplastic")},
        label_of=_neoplastic_label, manifest=_pending("piccolo", "Basque Biobank request"),
        unit="lesion", strata_key=majority_strata_key, has_boxes=True, centres="1 (Basque Country)"),
}


def modality_views(modalities: Sequence[str]) -> dict[str, list[str]]:
    """Masked-inference views of a trained model: each modality alone, then
    all of them ("fused"). CAD-SEL gives wle_only / eus_only / fused; a
    single-modality run has only "fused", meaning all of its input."""
    views = {f"{m.lower()}_only": [m] for m in modalities} if len(modalities) > 1 else {}
    views["fused"] = list(modalities)
    return views


def get_spec(name: str) -> DatasetSpec:
    if name not in SPECS:
        raise SystemExit(f"unknown dataset {name!r}; expected one of {sorted(SPECS)}")
    return SPECS[name]
