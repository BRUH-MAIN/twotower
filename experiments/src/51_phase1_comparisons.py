"""Experiment 51 -- the Phase 1 follow-up runs against the two-tower re-baseline.

CPU only, from 41's ``oof/*.npz`` (the cluster pushes them to
``cluster-results``), with 49's clinical models and metadata.

Conditions are run families recognised by their run-name configuration;
every one is backbone x task x 3 seeds. The re-baseline is the ablation
configuration retrained with the modality-dropout fix (``__keep1``).

  * ``conditions.csv``: balanced accuracy [95% CI] and AUROC of every
    condition and inference view, at seed 42 and as 3-seed ensembles, on all
    patients, patients with EUS and rectal lesions.
  * ``versus_baseline.csv``: each condition minus the re-baseline on the
    patients both cover (paired bootstrap, Holm-corrected over conditions),
    and for the lesion-free inputs the retention ``(bal - chance) /
    (baseline - chance)``. Late fusion of the separately trained WLE and EUS
    models is included as a condition.
  * ``external.csv``: held-out centres (Centres 2-6, or Center-3). The
    two-tower, its WLE view, the separately trained models and their late
    fusion, and the clinical model refitted on the training centres only.

    python -m experiments.src.51_phase1_comparisons
    python -m experiments.src.51_phase1_comparisons --baseline ablation   # before set A lands
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadsel.config import PROJECT_ROOT
from cadsel.fusion import patient_classes, patient_label

exp49 = importlib.import_module("experiments.src.49_clinical_baseline")
exp44, exp42 = exp49.exp44, exp49.exp42

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "51_phase1_comparisons"
CENTRES = PROJECT_ROOT / "experiments" / "data" / "patient_centre.csv"
BACKBONES = ("efficientnet_b0", "dinov3_vits16")
VIEWS = ("fused", "wle_only", "eus_only")
# condition -> regex on the configuration part of the run name (between task and seed)
CONDITIONS = {
    "ablation": r"^freeze-\w+__fusion-gated__.*__es20__lesioncrop__nested0\.2$",
    "baseline": r"^freeze-\w+__fusion-gated__.*__es20__lesioncrop__nested0\.2__keep1$",
    "full_frame": r"^freeze-\w+__fusion-gated__.*__es20__nested0\.2__keep1$",
    "bg_patch": r"__es20__nested0\.2__input-bg_patch__keep1$",
    "boxmap": r"__es20__nested0\.2__input-boxmap__keep1$",
    "symmetric": r"__fusion-symmetric__.*__lesioncrop__nested0\.2__keep1$",
    "aux": r"__lesioncrop__nested0\.2__aux1__keep1$",
    "ogm0.1": r"__lesioncrop__nested0\.2__ogm0\.1__keep1$",
    "ogm0.8": r"__lesioncrop__nested0\.2__ogm0\.8__keep1$",
    "warm_start": r"__lesioncrop__nested0\.2__warmstart__keep1$",
    "paired_only": r"__lesioncrop__nested0\.2__paired__keep1$",
    "wle_model": r"__lesioncrop__nested0\.2__mod-WLE$",
    "eus_model": r"__lesioncrop__nested0\.2__mod-EUS$",
}
EXTERNAL = {"C2-6": "Centres 2-6", "C3": "Center-3"}
for tag in EXTERNAL:
    CONDITIONS[f"ext_{tag}"] = rf"__lesioncrop__nested0\.2__ext-{tag}__keep1$"
    CONDITIONS[f"ext_{tag}_wle_model"] = rf"__lesioncrop__nested0\.2__ext-{tag}__mod-WLE$"
    CONDITIONS[f"ext_{tag}_eus_model"] = rf"__lesioncrop__nested0\.2__ext-{tag}__mod-EUS$"
COMPARED = ("ablation", "full_frame", "bg_patch", "boxmap", "symmetric", "aux", "ogm0.1", "ogm0.8",
            "warm_start", "paired_only", "late_models")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_conditions() -> dict:
    """(condition, backbone, task) -> {seed: run}, rows sorted by patient."""
    out = defaultdict(dict)
    for path in sorted((exp42.OOF_DIR).glob("*.npz")):
        m = exp42.RUN_RE.match(path.stem)
        if m is None or m["backbone"] not in BACKBONES:
            continue
        cond = next((c for c, rx in CONDITIONS.items() if re.search(rx, m["config"])), None)
        if cond is None:
            continue
        d = np.load(path)
        order = np.argsort(d["patient_key"])
        out[(cond, m["backbone"], m["task"])][int(m["seed"])] = {k: d[k][order] for k in d.files}
    return out


def pick(runs: dict, tag: str) -> dict | None:
    """Seed 42, or the mean of the three seeds' probabilities per patient."""
    if tag == "s42":
        return runs.get(42)
    if len(runs) < 3:
        return None
    keys = next(iter(runs.values()))["patient_key"]
    assert all(np.array_equal(r["patient_key"], keys) for r in runs.values()), "seeds cover different patients"
    out = {k: v for k, v in next(iter(runs.values())).items() if not k.startswith("probs_")}
    for name in [k for k in next(iter(runs.values())) if k.startswith("probs_")]:
        out[name] = np.mean([r[name] for r in runs.values()], axis=0)
    return out


def align(src: dict, keys: np.ndarray) -> dict:
    """``src`` restricted and reordered to ``keys`` (all must be present)."""
    pos = {k: i for i, k in enumerate(src["patient_key"].tolist())}
    idx = np.array([pos[k] for k in keys])
    return {k: v[idx] for k, v in src.items()}


def late_models(groups: dict, prefix: str, backbone: str, task: str, tag: str) -> dict | None:
    """Mean of the separately trained WLE and EUS models' probabilities on the
    patients with EUS (the EUS model only scores those)."""
    w = groups.get((f"{prefix}wle_model", backbone, task))
    e = groups.get((f"{prefix}eus_model", backbone, task))
    w, e = (pick(w, tag) if w else None), (pick(e, tag) if e else None)
    if w is None or e is None:
        return None
    w = align(w, e["patient_key"])
    return {**{k: e[k] for k in ("patient_key", "y")}, "paired": np.ones(len(e["y"]), bool),
            "probs_fused": (w["probs_fused"] + e["probs_fused"]) / 2,
            "probs_wle_only": w["probs_fused"], "probs_eus_only": e["probs_fused"]}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def conditions_table(groups: dict, meta: dict) -> list[dict]:
    rows = []
    for (cond, backbone, task), runs in sorted(groups.items()):
        n_classes = len(patient_classes(task))
        for tag in exp49.TAGS:
            src = pick(runs, tag)
            if src is None:
                continue
            keys = src["patient_key"]
            pops = [("all", np.ones(len(keys), bool)),
                    ("eus", src["paired"]),
                    ("rectum", exp49.population_mask("rectum", keys, src["paired"], meta))]
            for pop, m in pops:
                y = src["y"][m]
                if m.sum() == 0 or len(np.unique(y)) < 2:
                    continue
                for view in [v for v in VIEWS if f"probs_{v}" in src]:
                    if view == "eus_only" and pop == "all":
                        continue            # an EUS-only read needs EUS: scored under "eus"/"rectum"
                    vm = m & src["paired"] if view == "eus_only" else m
                    yv = src["y"][vm]
                    idx = exp44.boot_indices(yv, np.random.default_rng(0))
                    rows.append({"condition": cond, "backbone": backbone, "task": task, "seeds": tag,
                                 "population": pop, "view": view, "n": int(vm.sum()),
                                 **exp49.score_row(yv, src[f"probs_{view}"][vm], idx, n_classes)})
    return rows


def versus_baseline(groups: dict, meta: dict, baseline: str) -> list[dict]:
    rows = []
    for backbone in BACKBONES:
        for task in exp49.TASKS:
            n_classes = len(patient_classes(task))
            chance = 1 / n_classes
            base_runs = groups.get((baseline, backbone, task))
            for tag in exp49.TAGS:
                base = pick(base_runs, tag) if base_runs else None
                if base is None:
                    continue
                others = {c: pick(groups[(c, backbone, task)], tag)
                          for c in COMPARED if (c, backbone, task) in groups and c != baseline}
                others["late_models"] = late_models(groups, "", backbone, task, tag)
                others = {c: s for c, s in others.items() if s is not None}
                for pop in ("all", "eus", "rectum"):
                    panel = []
                    for cond, src in others.items():
                        keys = np.intersect1d(src["patient_key"], base["patient_key"])
                        b, o = align(base, keys), align(src, keys)
                        m = exp49.population_mask(pop, keys, b["paired"], meta)
                        y = b["y"][m]
                        assert np.array_equal(y, o["y"][m]), f"labels differ for {cond}"
                        idx = exp44.boot_indices(y, np.random.default_rng(0))
                        pb, po = b["probs_fused"][m], o["probs_fused"][m]
                        d, lo, hi, p = exp49.delta(exp49.boot_bal(y, po.argmax(1), idx, n_classes),
                                                   exp49.boot_bal(y, pb.argmax(1), idx, n_classes))
                        bal_b, bal_o = exp44.bal_acc(y, pb.argmax(1), n_classes), exp44.bal_acc(y, po.argmax(1), n_classes)
                        row = {"backbone": backbone, "task": task, "seeds": tag, "population": pop,
                               "condition": cond, "n": int(m.sum()), "bal_baseline": bal_b, "bal": bal_o,
                               "delta_bal": d, "delta_lo": lo, "delta_hi": hi, "p": p,
                               "retention": (bal_o - chance) / (bal_b - chance) if bal_b > chance else None,
                               "auroc_baseline": exp44.auroc(y, pb), "auroc": exp44.auroc(y, po)}
                        if n_classes == 2:
                            da, alo, ahi, ap = exp49.delta(exp49.boot_auroc(y, po[:, 0], idx),
                                                           exp49.boot_auroc(y, pb[:, 0], idx))
                            row.update(delta_auroc=da, delta_auroc_lo=alo, delta_auroc_hi=ahi, p_auroc=ap)
                        panel.append(row)
                    for key in ("p", "p_auroc"):
                        if panel and key in panel[0]:
                            for r, adj in zip(panel, exp44.holm([r[key] for r in panel])):
                                r[f"{key}_holm"] = adj
                    rows += panel
    return rows


def clinical_external(meta: dict, task: str, held_out: set, centres: dict, keys: np.ndarray) -> np.ndarray:
    """Clinical logistic model fitted on the training centres only, mean over
    the three seeds' inner CV; probabilities for ``keys``."""
    classes = patient_classes(task)
    train = np.array(sorted(k for k in meta if centres[k] not in held_out))
    y_train = np.array([classes.index(patient_label(k.split("/")[0], task)) for k in train])
    X_train = exp49.design(meta, train, exp49.FEATURE_SETS["clinical"])
    X_test = exp49.design(meta, keys, exp49.FEATURE_SETS["clinical"])
    return np.mean([exp49.make_model("logreg", s).fit(X_train, y_train).predict_proba(X_test)
                    for s in exp49.SEEDS], axis=0)


def external_table(groups: dict, meta: dict, centres: dict) -> list[dict]:
    rows = []
    for tag_ext, name in EXTERNAL.items():
        held_out = {f"Center-{c}" for c in range(2, 7)} if tag_ext == "C2-6" else {"Center-3"}
        for backbone in BACKBONES:
            for task in exp49.TASKS:
                n_classes = len(patient_classes(task))
                tt_runs = groups.get((f"ext_{tag_ext}", backbone, task))
                for tag in exp49.TAGS:
                    tt = pick(tt_runs, tag) if tt_runs else None
                    if tt is None:
                        continue
                    keys = tt["patient_key"]
                    entries = {"two_tower": tt["probs_fused"], "two_tower_wle_view": tt["probs_wle_only"]}
                    clin = clinical_external(meta, task, held_out, centres, keys)
                    entries["clinical"] = clin
                    entries["two_tower+clinical"] = (tt["probs_fused"] + clin) / 2
                    late = late_models(groups, f"ext_{tag_ext}_", backbone, task, tag)
                    paired_keys = keys[tt["paired"]]
                    for pop in ("all", "eus", "rectum"):
                        m = exp49.population_mask(pop, keys, tt["paired"], meta)
                        y = tt["y"][m]
                        if len(np.unique(y)) < 2:
                            continue
                        idx = exp44.boot_indices(y, np.random.default_rng(0))
                        local = {k: v[m] for k, v in entries.items()}
                        if late is not None and pop != "all":
                            sub = align(late, keys[m]) if set(keys[m]) <= set(late["patient_key"]) else None
                            if sub is not None:
                                local["wle_model"] = sub["probs_wle_only"]
                                local["late_models"] = sub["probs_fused"]
                        for model, probs in local.items():
                            rows.append({"held_out": name, "backbone": backbone, "task": task, "seeds": tag,
                                         "population": pop, "model": model, "n": int(m.sum()),
                                         "n_paired_total": len(paired_keys),
                                         **exp49.score_row(y, probs, idx, n_classes)})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="baseline", choices=sorted(CONDITIONS),
                   help="reference condition (use 'ablation' until the re-baseline runs land)")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = exp49.load_metadata()
    centres = dict(np.loadtxt(CENTRES, delimiter=",", skiprows=1, dtype=str))
    groups = load_conditions()
    found = defaultdict(list)
    for (cond, backbone, task), runs in sorted(groups.items()):
        found[cond].append(f"{backbone}/{task}:{len(runs)}")
    print("runs found (backbone/task:seeds):")
    for cond in CONDITIONS:
        print(f"  {cond:22s}{' '.join(found[cond]) if found[cond] else '-'}")
    for name, rows in (("conditions", conditions_table(groups, meta)),
                       ("versus_baseline", versus_baseline(groups, meta, args.baseline)),
                       ("external", external_table(groups, meta, centres))):
        if rows:
            exp44.write_csv(OUT_DIR / f"{name}.csv", rows)
        else:
            print(f"no rows for {name} yet")


if __name__ == "__main__":
    main()
