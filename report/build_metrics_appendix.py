#!/usr/bin/env python3
"""Extracts real logged metrics from every result JSON in the CADSEL workspace
and emits LaTeX longtable fragments for the report's metrics appendix.

Never fabricates a number: a cell is populated only if the underlying JSON
actually contains it (or it is exactly derivable, e.g. macro precision/recall/F1
from a saved confusion matrix). Anything not logged is rendered as "n/l".
Metrics with no standard definition for a task type (accuracy/ROC-AUC for
object detection) are rendered as "N/A".
"""
from __future__ import annotations
import json
import glob
import os
import re
from pathlib import Path

ROOT = Path("/home/rohan/CADSEL")
OUT = ROOT / "report" / "tables"
OUT.mkdir(exist_ok=True)

NL = r"\nl{}"
NA = r"\napp{}"


def esc(s: str) -> str:
    s = str(s)
    return (s.replace("\\", r"\textbackslash{}")
             .replace("_", r"\_")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("#", r"\#"))


def fmt(x, nd=3):
    if x is None:
        return NL
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return esc(x)


def fmt_pm(mean, std, nd=3):
    if mean is None:
        return NL
    if std is None:
        return fmt(mean, nd)
    return f"{float(mean):.{nd}f}$\\pm${float(std):.{nd}f}"


# ---------------------------------------------------------------------------
# A.1 Track A classifiers (62 runs) -- full 7 metrics
# ---------------------------------------------------------------------------
def macro_prf1(cm, class_order):
    n = len(class_order)
    precisions, recalls, f1s = [], [], []
    col_sums = [sum(cm[r][c] for r in range(n)) for c in range(n)]
    for i in range(n):
        row_sum = sum(cm[i])
        tp = cm[i][i]
        rec = tp / row_sum if row_sum else 0.0
        prec = tp / col_sums[i] if col_sums[i] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n


def build_track_a():
    rows = []
    files = sorted(glob.glob(str(ROOT / "experiments/results/classifier_runs/*.json")))
    for fp in files:
        d = json.load(open(fp))
        cm = d.get("confusion_matrix_sum")
        order = d.get("confusion_matrix_class_order")
        if cm and order:
            p, r, f1 = macro_prf1(cm, order)
        else:
            p = r = f1 = None
        rows.append({
            "run": d["run_name"],
            "backbone": d.get("backbone"),
            "modality": d.get("modality"),
            "task": d.get("task"),
            "freeze": d.get("freeze"),
            "loss": d.get("loss"),
            "oversample": d.get("oversample"),
            "seed": d.get("seed"),
            "acc_mean": d.get("accuracy_mean"), "acc_std": d.get("accuracy_std"),
            "bal_mean": d.get("balanced_accuracy_mean"), "bal_std": d.get("balanced_accuracy_std"),
            "precision": p, "recall": r, "f1": f1,
            "roc_auc": d.get("roc_auc_mean"), "pr_auc": d.get("pr_auc_mean"),
        })
    rows.sort(key=lambda x: (x["modality"], x["task"], x["backbone"], x["freeze"], x["loss"], x["oversample"], x["seed"]))
    return rows


def write_track_a(rows):
    lines = []
    lines.append(r"\begingroup\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{longtable}{L{4.3cm}ccccccc}")
    lines.append(r"\caption{Track~A classifier runs: full 7-metric breakdown, extracted directly from each "
                 r"run's saved result JSON (\texttt{experiments/results/classifier\_runs/*.json}). "
                 r"Precision/recall/F1 are macro-averaged over classes, derived from each run's own saved "
                 r"confusion matrix (5-fold pooled). Accuracy and balanced accuracy are the 5-fold mean $\pm$ std; "
                 r"ROC-AUC and PR-AUC are the 5-fold mean (per-fold std not separately saved).}"
                 r"\label{tab:app-tracka}\\")
    lines.append(r"\toprule")
    hdr = r"\textbf{Run} & \textbf{Acc.} & \textbf{Bal.\ Acc.} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{ROC-AUC} & \textbf{PR-AUC} \\"
    lines.append(hdr)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(hdr)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for r_ in rows:
        run_short = r_["run"].replace("__", "/")
        lines.append(
            f"\\texttt{{\\scriptsize \\seqsplit{{{esc(run_short)}}}}} & "
            f"{fmt_pm(r_['acc_mean'], r_['acc_std'])} & "
            f"{fmt_pm(r_['bal_mean'], r_['bal_std'])} & "
            f"{fmt(r_['precision'])} & {fmt(r_['recall'])} & {fmt(r_['f1'])} & "
            f"{fmt(r_['roc_auc'])} & {fmt(r_['pr_auc'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    (OUT / "track_a.tex").write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


# ---------------------------------------------------------------------------
# A.2 Joint two-tower fusion runs
# ---------------------------------------------------------------------------
def build_joint_fusion():
    rows = []
    files = sorted(glob.glob(str(ROOT / "experiments/results/joint_fusion_runs/*.json")))
    for fp in files:
        d = json.load(open(fp))
        run = d.get("run_name", os.path.basename(fp))
        summary = d.get("summary", {})
        for ablation in ("wle_only", "eus_only", "fused"):
            ab = summary.get(ablation)
            if not ab:
                continue
            denom = ab.get("unpaired") or ab.get("paired_only")
            denom_label = "unpaired" if ab.get("unpaired") else "paired-only"
            if not denom:
                continue
            rows.append({
                "run": run, "ablation": ablation, "denom": denom_label,
                "acc_mean": denom.get("acc", {}).get("mean"), "acc_std": denom.get("acc", {}).get("std"),
                "bal_mean": denom.get("bal", {}).get("mean"), "bal_std": denom.get("bal", {}).get("std"),
                "f1_mean": denom.get("macro_f1", {}).get("mean"), "f1_std": denom.get("macro_f1", {}).get("std"),
            })
    rows.sort(key=lambda x: (x["run"], x["ablation"]))
    return rows


def write_joint_fusion(rows):
    lines = []
    lines.append(r"\begingroup\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{longtable}{L{5.4cm}L{1.3cm}ccc}")
    lines.append(r"\caption{Joint end-to-end two-tower fusion runs (\S\ref{sec:fusion2}): accuracy, balanced "
                 r"accuracy, and macro-F1 as logged in \texttt{experiments/results/joint\_fusion\_runs/*.json} "
                 r"(5-fold mean $\pm$ std). Precision, recall, ROC-AUC, and PR-AUC were not logged for this "
                 r"architecture (no confusion matrix or class-probability curve was saved per fold) --- "
                 r"reported as \nl{} rather than fabricated.}"
                 r"\label{tab:app-jointfusion}\\")
    lines.append(r"\toprule")
    hdr = (r"\textbf{Run} & \textbf{Ablation} & \textbf{Accuracy} & \textbf{Bal.\ Acc.} & \textbf{F1 (macro)} \\"
           r" & & \multicolumn{3}{c}{\scriptsize(Prec./Rec./ROC-AUC/PR-AUC: \nl{})} \\")
    lines.append(hdr)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(hdr)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for r_ in rows:
        run_short = re.sub(r"^joint_efficientnet_b0__", "", r_["run"])
        lines.append(
            f"\\texttt{{\\scriptsize \\seqsplit{{{esc(run_short)}}}}} & {esc(r_['ablation'])} & "
            f"{fmt_pm(r_['acc_mean'], r_['acc_std'])} & {fmt_pm(r_['bal_mean'], r_['bal_std'])} & "
            f"{fmt_pm(r_['f1_mean'], r_['f1_std'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    (OUT / "joint_fusion.tex").write_text("\n".join(lines), encoding="utf-8")
    return len(rows)



# ---------------------------------------------------------------------------
# A.3 Shortcut-audit classifier-style probes (01,02,03,04,05,07,12,13,14,17,18)
# ---------------------------------------------------------------------------
def build_audit_classifier():
    rows = []

    def add(exp, run_list, modality_key="modality", task_key="task", acc_key="acc", bal_key="bal", label_keys=()):
        for it in run_list:
            label = " / ".join(str(it.get(k)) for k in label_keys if it.get(k) is not None)
            rows.append({
                "exp": exp,
                "modality": it.get(modality_key),
                "task": it.get(task_key),
                "label": label,
                "acc": it.get(acc_key),
                "bal": it.get(bal_key),
            })

    # Exp 01: dataset confounds (metadata + geometry probes)
    d = json.load(open(ROOT / "experiments/results/01_dataset_confounds.json"))
    for grp in ("metadata_probes", "geometry_probes"):
        for it in d.get(grp, []):
            rows.append({
                "exp": "01", "modality": it.get("modality", "--"), "task": it.get("target"),
                "label": f"{grp.replace('_probes','')}: {it.get('probe')}",
                "acc": it.get("accuracy"), "bal": it.get("balanced_accuracy"),
            })

    # Exp 02: low-level probes
    d = json.load(open(ROOT / "experiments/results/02_lowlevel_probes.json"))
    add("02", d, task_key="target", acc_key="accuracy", bal_key="balanced_accuracy", label_keys=("probe",))

    # Exp 03: lesion vs background (core counterfactual)
    d = json.load(open(ROOT / "experiments/results/03_lesion_vs_background.json"))
    add("03", d["runs"], label_keys=("variant",))

    # Exp 04: cross-centre
    d = json.load(open(ROOT / "experiments/results/04_cross_centre.json"))
    add("04", d, label_keys=("arm",))

    # Exp 05: split leakage
    d = json.load(open(ROOT / "experiments/results/05_split_leakage.json"))
    add("05", d, label_keys=("split",))

    # Exp 07: mitigation (full/bg_patch/boxmap are balanced-accuracy per variant)
    d = json.load(open(ROOT / "experiments/results/07_mitigation.json"))
    for it in d:
        for variant in ("full", "bg_patch", "boxmap"):
            rows.append({
                "exp": "07", "modality": it.get("modality"), "task": it.get("task"),
                "label": f"{it.get('condition')} / {variant}",
                "acc": None, "bal": it.get(variant),
            })

    # Exp 12/13/14: adversarial / bbox_blur / channel_attention -- same schema as 07
    for exp_id, fname in (("12", "12_adversarial"), ("13", "13_bbox_blur"), ("14", "14_channel_attention")):
        fp = ROOT / f"experiments/results/{fname}.json"
        if not fp.exists():
            continue
        d = json.load(open(fp))
        for it in d.get("runs", []):
            for variant in ("full", "bg_patch", "boxmap"):
                if variant not in it:
                    continue
                rows.append({
                    "exp": exp_id, "modality": it.get("modality"), "task": it.get("task"),
                    "label": f"{it.get('condition')} (s{it.get('seed')}) / {variant}",
                    "acc": None, "bal": it.get(variant),
                })

    # Exp 17: SVM/DINOv3 counterfactual
    d = json.load(open(ROOT / "experiments/results/17_svm_counterfactual.json"))
    add("17", d["runs"], label_keys=("backbone", "variant"))

    # Exp 18: LOCO
    d = json.load(open(ROOT / "experiments/results/18_loco.json"))
    add("18", d["runs"], label_keys=("held_out", "arm"))

    return rows


def write_audit_classifier(rows):
    lines = []
    lines.append(r"\begingroup\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{longtable}{cL{1.1cm}L{1.5cm}L{5.6cm}ccc}")
    lines.append(r"\caption{Shortcut-learning-audit and classifier-level mitigation probes (Experiments "
                 r"1--5, 7, 12--14, 17--18): accuracy and balanced accuracy as logged in each experiment's own "
                 r"\texttt{experiments/results/*.json}. These ResNet-18 probe scripts predate Track~A's "
                 r"infrastructure and never saved a confusion matrix, class-probability scores, or a checkpoint "
                 r"(they resume from JSON only, per \texttt{experiments/README.md}) --- precision, recall, F1, "
                 r"ROC-AUC, and PR-AUC were never computed for these runs, and backfilling them would require "
                 r"fully retraining every one of the $\sim$150 individual model fits below. Reported honestly "
                 r"as \nl{} rather than estimated.}"
                 r"\label{tab:app-auditclf}\\")
    lines.append(r"\toprule")
    hdr = r"\textbf{Exp.} & \textbf{Mod.} & \textbf{Task} & \textbf{Configuration} & \textbf{Acc.} & \textbf{Bal.\ Acc.} & \textbf{P/R/F1/ROC/PR-AUC} \\"
    lines.append(hdr)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(hdr)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for r_ in rows:
        lines.append(
            f"{esc(r_['exp'])} & {esc(r_['modality'])} & {esc(r_['task'])} & {esc(r_['label'])} & "
            f"{fmt(r_['acc'])} & {fmt(r_['bal'])} & {NL} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    (OUT / "audit_classifier.tex").write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    n1 = write_track_a(build_track_a())
    n2 = write_joint_fusion(build_joint_fusion())
    n3 = write_audit_classifier(build_audit_classifier())
    print(f"Track A rows: {n1}")
    print(f"Joint fusion rows: {n2}")
    print(f"Audit classifier rows: {n3}")
    print(f"TOTAL appendix rows: {n1 + n2 + n3}")
