"""Reproduce the dataset-characteristics section: Table 2 and Fig 3.

python -m cadsel.describe
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import METADATA_XLSX, MODALITIES, PROJECT_ROOT, SUBTYPE_DIRS
from .data import build_manifest, read_yolo_label

# Published Table 2, for verification.
PAPER_TABLE2 = {
    "WLE": {
        "set": (641, 2817), "NET": (455, 1945), "NET_G1": (411, 1749),
        "NET_G2": (44, 196), "Lipoma": (96, 453), "Leiomyoma": (24, 137),
        "NonNeoplasm": (66, 282),
    },
    "EUS": {
        "set": (421, 2095), "NET": (325, 1602), "NET_G1": (299, 1444),
        "NET_G2": (26, 158), "Lipoma": (49, 254), "Leiomyoma": (14, 79),
        "NonNeoplasm": (33, 160),
    },
}


def modality_counts(modality: str, data_root=None) -> dict:
    samples = build_manifest(modality, data_root=data_root)
    patients = defaultdict(set)
    images = Counter()
    for s in samples:
        patients[s.subtype].add(s.patient_key)
        images[s.subtype] += 1

    out = {}
    for subtype in SUBTYPE_DIRS:
        out[subtype] = (len(patients[subtype]), images[subtype])
    out["NET"] = (
        out["NET_G1"][0] + out["NET_G2"][0],
        out["NET_G1"][1] + out["NET_G2"][1],
    )
    out["set"] = (
        sum(len(v) for v in patients.values()),
        sum(images.values()),
    )
    return out


def box_counts(modality: str, data_root=None) -> Counter:
    counts = Counter()
    for s in build_manifest(modality, data_root=data_root):
        for cls, *_ in read_yolo_label(s.label):
            counts[cls] += 1
    return counts


def read_metadata() -> list[dict]:
    try:
        import openpyxl
    except ImportError:
        return []
    if not METADATA_XLSX.exists():
        return []
    wb = openpyxl.load_workbook(METADATA_XLSX, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h) for h in rows[0]]
    return [dict(zip(header, row)) for row in rows[1:] if row[0] is not None]


def clinical_distribution(records: list[dict]) -> dict:
    if not records:
        return {}

    def col(prefix: str):
        for k in records[0]:
            if k.lower().startswith(prefix):
                return k
        return None

    age_col, sex_col, size_col, loc_col = (
        col("age"), col("sex"), col("size"), col("location")
    )

    ages = [r[age_col] for r in records if isinstance(r.get(age_col), (int, float))]
    sexes = Counter(r[sex_col] for r in records if r.get(sex_col) in (0, 1))
    sizes = [r[size_col] for r in records if isinstance(r.get(size_col), (int, float))]
    locations = Counter(
        str(r[loc_col]).strip().lower() for r in records if r.get(loc_col)
    )

    def pct(n, total):
        return round(100.0 * n / total, 1) if total else 0.0

    age_bins = Counter()
    for a in ages:
        if a < 30:
            age_bins["<30"] += 1
        elif a <= 60:
            age_bins["30-60"] += 1
        else:
            age_bins[">60"] += 1

    size_bins = Counter()
    for s in sizes:
        if s < 0.5:
            size_bins["<0.5"] += 1
        elif s <= 1.0:
            size_bins["0.5-1"] += 1
        elif s <= 2.0:
            size_bins["1-2"] += 1
        else:
            size_bins[">2"] += 1

    return {
        "n_patients": len(records),
        "mean_age": round(sum(ages) / len(ages), 1) if ages else None,
        "age_pct": {k: pct(v, len(ages)) for k, v in sorted(age_bins.items())},
        "sex_pct": {
            "female": pct(sexes.get(0, 0), sum(sexes.values())),
            "male": pct(sexes.get(1, 0), sum(sexes.values())),
        },
        "size_pct": {k: pct(v, len(sizes)) for k, v in size_bins.items()},
        "location_pct": {
            k: pct(v, sum(locations.values()))
            for k, v in locations.most_common()
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Summarise the CAD-SEL dataset")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--data-root", type=Path, default=None)
    args = p.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    report = {"table2": {}, "boxes": {}, "clinical": {}}
    lines = ["# CAD-SEL dataset characteristics (reproduced)", ""]

    lines += [
        "### Table 2 — Overview of the data records",
        "",
        "| Category | Patients (ours) | Images (ours) | Patients (paper) | Images (paper) | Match |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    order = ["set", "NET", "NET_G1", "NET_G2", "Lipoma", "Leiomyoma", "NonNeoplasm"]
    total_images = 0
    for modality in MODALITIES:
        counts = modality_counts(modality, args.data_root)
        report["table2"][modality] = counts
        total_images += counts["set"][1]
        for key in order:
            ours = counts[key]
            paper = PAPER_TABLE2[modality][key]
            name = f"{modality} set" if key == "set" else f"{modality} · {key}"
            ok = "yes" if tuple(ours) == tuple(paper) else "**NO**"
            lines.append(f"| {name} | {ours[0]} | {ours[1]} | {paper[0]} | {paper[1]} | {ok} |")

    lines += ["", f"Total images: **{total_images}** (paper: 4,912)", ""]

    lines += [
        "### Bounding boxes per annotation class",
        "",
        "| Modality | NET (0) | Leiomyoma (1) | Lipoma (2) | NonNeoplasm (3) | Total |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for modality in MODALITIES:
        c = box_counts(modality, args.data_root)
        report["boxes"][modality] = dict(c)
        lines.append(
            f"| {modality} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {sum(c.values())} |"
        )
    lines.append("")

    clinical = clinical_distribution(read_metadata())
    if clinical:
        report["clinical"] = clinical
        lines += [
            "### Fig 3 — Clinical data distribution",
            "",
            f"- Patients: {clinical['n_patients']} (paper: 641)",
            f"- Mean age: {clinical['mean_age']} y (paper: 52.0)",
            f"- Age: {clinical['age_pct']} (paper: 68% aged 30-60)",
            f"- Sex: {clinical['sex_pct']} (paper: 51% male / 49% female)",
            f"- Size (cm): {clinical['size_pct']} (paper: <0.5 20%, 0.5-1 54%, 1-2 20%, >2 6%)",
            "- Location: " + ", ".join(f"{k} {v}%" for k, v in clinical["location_pct"].items()),
            "  (paper: rectum 81%, sigmoid 5%, ascending 5%, descending 3%, transverse 3%, ileocecal 3%)",
            "",
        ]

    text = "\n".join(lines)
    (args.results_dir / "DATASET.md").write_text(text, encoding="utf-8")
    (args.results_dir / "dataset_stats.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(text)


if __name__ == "__main__":
    main()
