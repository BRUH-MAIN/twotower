# %% [markdown]
# # Experiment 4 — Cross-centre external validation
#
# The review calls this the decisive test (Section 4.5): *"The single most
# reliable detection method reported across this literature remains testing on a
# genuinely external dataset ... A large and reproducible drop in performance under
# such a test, even when internal cross-validation looked excellent, is the
# clearest practical signature of shortcut reliance."*
#
# CAD-SEL is multicentre but heavily skewed — Center-1 contributes 537 of 641
# patients. That makes exactly one meaningful external test possible:
#
# * **Internal**: train on Center-1, validate on held-out Center-1 patients
# * **External**: train on Center-1, validate on Centers 2-6 (never seen)
#
# Both arms use the *same trained model class and the same training set size*, so
# the only thing that changes is whether the test patients come from the training
# site. Experiment 2 established that the site is highly decodable from the
# images; this asks whether that fingerprint actually costs us generalisation.

# %%
import sys, json, gc
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "experiments"))

import numpy as np

from cadsel.data import build_manifest, patient_level_split
from harness import DEVICE, classes_for, decode, encode_labels, label_of, train_eval

OUT = Path("experiments/results")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
print("device:", DEVICE)

# %%
results = []
for modality in ("WLE", "EUS"):
    samples = build_manifest(modality)
    c1 = [s for s in samples if s.center == "Center-1"]
    rest = [s for s in samples if s.center != "Center-1"]

    print(f"\n=== {modality} ===")
    print(f"  Center-1     : {len(c1)} images / "
          f"{len({s.patient_key for s in c1})} patients")
    print(f"  Centers 2-6  : {len(rest)} images / "
          f"{len({s.patient_key for s in rest})} patients")
    print("  external subtype mix:", dict(Counter(s.subtype for s in rest)))

    if len(rest) < 40:
        print("  too few external images to evaluate; skipping")
        continue

    # Internal split of Center-1; the external set is everything else.
    c1_train, c1_val = patient_level_split(c1, 0.2, SEED)

    print("  decoding ...")
    Xtr = decode(c1_train)
    Xin = decode(c1_val, progress_every=0)
    Xex = decode(rest, progress_every=0)

    for task in ("binary", "subtype"):
        classes = classes_for(task)
        ytr = encode_labels(c1_train, task)
        yin = encode_labels(c1_val, task)
        yex = encode_labels(rest, task)

        # Only score classes that actually occur in the external set.
        print(f"  -- {task} --")
        internal = train_eval(Xtr, ytr, Xin, yin, len(classes),
                              f"{modality}/{task}/internal (Center-1 held-out)",
                              seed=SEED)
        external = train_eval(Xtr, ytr, Xex, yex, len(classes),
                              f"{modality}/{task}/EXTERNAL (Centers 2-6)",
                              seed=SEED)
        for name, r in (("internal", internal), ("external", external)):
            r.update({"modality": modality, "task": task, "arm": name})
            results.append(r)
        json.dump(results, open(OUT / "04_cross_centre.json", "w"), indent=2)

    del Xtr, Xin, Xex
    gc.collect()

# %% [markdown]
# ## 4.1 Verdict
#
# The signature of shortcut reliance is a large internal-to-external drop,
# expressed against the chance floor:
#
# ```
# retained = (external - chance) / (internal - chance)
# ```

# %%
by = {(r["modality"], r["task"], r["arm"]): r for r in results}
verdict = []
for modality in ("WLE", "EUS"):
    for task in ("binary", "subtype"):
        if (modality, task, "internal") not in by:
            continue
        i, e = by[(modality, task, "internal")], by[(modality, task, "external")]
        chance = i["chance"]
        denom = i["bal"] - chance
        retained = (e["bal"] - chance) / denom if denom > 1e-6 else None
        verdict.append({
            "modality": modality, "task": task, "chance": chance,
            "internal": i["bal"], "external": e["bal"],
            "drop": round(i["bal"] - e["bal"], 4),
            "retained": None if retained is None else round(retained, 3),
            "n_external": e["n_val"],
        })

json.dump({"runs": results, "verdict": verdict},
          open(OUT / "04_cross_centre_verdict.json", "w"), indent=2)

print("=" * 86)
print("CROSS-CENTRE EXTERNAL VALIDATION — balanced accuracy")
print("=" * 86)
hdr = (f"{'modality':<9}{'task':<9}{'chance':>8}{'internal':>10}{'external':>10}"
       f"{'drop':>8}{'retained':>10}{'n_ext':>7}")
print(hdr); print("-" * len(hdr))
for v in verdict:
    ret = v["retained"]
    print(f"{v['modality']:<9}{v['task']:<9}{v['chance']:>8.3f}{v['internal']:>10.3f}"
          f"{v['external']:>10.3f}{v['drop']:>8.3f}"
          f"{'n/a' if ret is None else format(ret, '>9.1%')}{v['n_external']:>7}")

print()
for v in verdict:
    ret = v["retained"]
    if ret is None:
        continue
    if ret < 0.4:
        msg = "LARGE external drop — consistent with site-specific shortcut reliance"
    elif ret < 0.75:
        msg = "moderate external drop"
    else:
        msg = "transfers well across sites"
    print(f"  {v['modality']:<4} {v['task']:<8} retained {ret:>6.1%}  {msg}")

print("\nCaveat: Centers 2-6 contribute only ~100 patients and a different class")
print("mix, so part of any drop reflects small-sample noise and prevalence shift,")
print("not purely acquisition-site shortcutting.")
