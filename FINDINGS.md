# CAD-SEL — consolidated findings

Working notes for anyone (human or agent) picking this up cold. Covers what was
reproduced, what the shortcut audit found, which mitigations work, and the
implementation traps that cost time.

Last updated: 2026-08-30.

---

## 1. What this workspace is

A PyTorch reproduction of the technical-validation baselines in:

> Xu, L., Chen, S., Li, C. *et al.* **CAD-SEL: A dual-modal colonoscopy dataset of
> subepithelial lesion.** *Scientific Data* (2026).
> doi:10.1038/s41597-026-07331-y

plus a shortcut-learning audit of the dataset, driven by
`Shortcut_Learning_Colonoscopy_Literature_Review.pdf`.

| Path | What |
| --- | --- |
| `cadsel/` | The reproduction pipeline (data, models, metrics, engine, CLIs) |
| `experiments/` | Shortcut audit + mitigation notebooks (`src/*.py` → executed `*.ipynb`) |
| `runs/` | Per-run checkpoints, history, results |
| `runs_partial/` | Interrupted ResNet runs, parked so they don't pollute reports |
| `runs_joint/` | Multimodal study runs, parked separately for the same reason |
| `results/` | Aggregated Table 4 / Table 5, and `MULTIMODAL.md` |
| `kaggle/` | Self-contained notebooks for running the grid on Kaggle |
| `README.md` | Setup and usage |
| `experiments/README.md` | The audit write-up |

Environment: `.venv` with torch 2.11.0+cu128, CUDA working on an RTX 5060 (8 GB).

---

## 2. Status board

| Item | State |
| --- | --- |
| Dataset verified against paper Table 2 | done — **every cell matches exactly** |
| 8 MobileNet runs (Table 4/5) | done |
| 8 ResNet runs | **2/8 done** (WLE, both backbones: AP@0.5 0.28/0.26 — worse than MobileNet's 0.38-0.41); remaining 6 **deliberately dropped**, not paused — see §3 |
| Shortcut audit (Exp 1–6) | done |
| Tier-1 mitigation, classifier (Exp 7) | done |
| Tier-1 mitigation, detector (Exp 8) | done — ambiguous at n=1 |
| Size-matched evaluation (Exp 9) | done |
| Hard negatives (Exp 10) | done — **best mitigation found** |
| Multi-seed A/B (Exp 11) | **done — settles the augmentation question** |
| Exp 12 (adversarial discriminator, gradient reversal) | done |
| Exp 13 (blur non-lesion pixels during training) | done |
| Exp 14 (channel attention, SE blocks) | done |
| Exp 15/16 (`se_all_adv` multi-seed / 100-epoch follow-up) | **done — settles the channel-attention question** |
| Literature review + Tier-1 performance levers (anchors, IoU thresholds, multi-scale, RFS, EMA, WBF TTA) | done, implemented in `cadsel/` |
| Seesaw-p sweep (p=0.2/0.4/0.6/0.8) | **done** — flat 0.334-0.347 AP, all below the plain-augment control (0.370); doesn't help EUS at any p |
| DFR (group=center) | **done** — no effect, see §5m |
| Group DRO (group=center) | **done — confirmed negative**, see §5t |
| JTT, LfF | deferred |
| Multimodal study — code + verification | done, 28/28 checks |
| Multimodal study — training runs (Stage 1 grid + Stage 2 fusion) | **done** — all 8 joint-detector runs + 5-fold fusion complete; see §5b |
| Exp 17 — SVM/DINOv3 through the counterfactual protocol | **done** — see §5e |
| Exp 18 — leave-one-center-out (classifier) | **done** — see §5f |
| Lead 3b — leave-one-center-out (detector, scoped) | **done** — see §5f |
| Size-aware oversampling (`SizeRepeatFactorSampler`) | **done** — see §5g |
| Exp 20 — lesion GAN vs. real copy-paste on EUS (detector, 3 seeds) | **done** — see §5h |
| Exp 21 — TFA rare-class fine-tuning, multi-seed | **done** — see §5h |
| Fusion subgroup analysis + symmetric-attention fusion | **done** — see §5i |
| `report.py`/`run_all.py` aggregate-overwrite bug | **found and fixed** — see §5j |
| Exp 22 — Center-3 black-overlap-fraction diagnostic | **done** — see §5k |
| Center-bucketed oversampling (`CenterRepeatFactorSampler`) | **done — confirmed on the clean-label subset** — see §5l |
| DFR, detector-level, group=center | **done** — no effect, see §5m |
| EndoStyle-style color/style augmentation | **done** — no effect, see §5n |
| DINOv3 as detector backbone (frozen) | **done — promising, not yet confirmed**, see §5o |
| RadImageNet backbone for EUS, isolated | **done — confirmed negative**, see §5p |
| EMA + multiscale combo; center-RFS on full pool | **done — neither helps**, see §5q |
| Dirty>clean AP mystery (item 12) | **resolved — lesion-size confound**, see §5r |
| DINOv3 push (more seeds, trainable backbone, binary, EUS) | **done — settles unconfirmed**, see §5s |
| Group DRO, detector-level, group=center | **done — confirmed negative**, see §5t |
| Binary-task size-RFS, more seeds (5 total) | **done — settles unconfirmed**, see §5g |
| JTT diagnostic (identification-only) | **done — mixed result, doesn't justify full method**, see §5u |
| Cross-model WBF ensembling | **done — confirmed negative**, see §5v |
| Auto-anchors + size-RFS + multiscale combo | **done — confirmed negative**, see §5x |
| Size-RFS threshold sweep (0.1/0.3/0.4 vs adopted 0.2) | **done — 0.2 stays adopted**, see §5y |
| Track A — classification-accuracy campaign (trainable backbones, lever screen, counterfactual gate) | **done — 1 confirmed lever** (`efficientnet_b0` + `weighted_ce`, counterfactual-clean), see §10 |
| Track B — YOLO11 detection track (dataset export, eval bridge, lever screen) | **done — 1 confirmed lever after correction** (class-oversampling), see §11 |
| Track B follow-ups — `close_mosaic` early-stopping correction, `yolo11m` screen, class-oversampling | **done — class-oversampling confirmed positive**, `close_mosaic` corrected to confirmed-negative, see §11.4-11.6 |
| Track A follow-ups — DINOv3 `--freeze partial`, low-LR full fine-tune, binary lever screen, TTA | **done — new best classifier found** (DINOv3 partial-freeze, flagged partial on counterfactual gate), see §12 |
| Round 2, Tiers 1-2 — combined levers, multi-seed ensembling, EUS counterfactual backlog, YOLO TTA/coverage | **done — 3 new campaign bests**: Track A binary 0.8209 / subtype 0.5230 (3-seed ensembles, counterfactual-clean/inherited); Track B EUS/multiclass YOLO beats Faster R-CNN (0.3799±0.0064 vs 0.3163, confirmed) — see §13 |
| Round 3 — WBF retry, EUS-specific lever screen, DINOv3-partial on EUS, classifier-backbone fusion rebuild, `unfreeze_last_n` sweep, 5-crop TTA | **done — new best EUS classifier both tasks** (binary 0.7386, counterfactual-checked); fusion rebuild promising but not confirmed (small paired-pool variance); `unfreeze_last_n=6-8` modestly beats the default 4; 5-crop TTA also null, closing the TTA thread — see §14 |
| Joint end-to-end two-tower WLE+EUS fusion (user-proposed architecture, trainable EfficientNet-B0 towers) | **done for this round — campaign-best confirmed configs**: binary `--freeze none` + `--lesion-crop`, 3-seed fused 0.8110±0.0123 (ensembled 0.8219); subtype `--freeze none` + `--lesion-crop`, 3-seed fused 0.5419±0.0201 (ensembled 0.5706); counterfactual gate PASSED on both the binary `--freeze none` and subtype `--freeze partial` checkpoints (boxmap retention ~0-2%, essentially chance — genuine lesion signal, not shortcut-driven); discriminative backbone LR, symmetric fusion (incl. combined with lesion-crop), paired-only training, and `--bg-remove-prob` (both p=0.5 and a padding-fixed p=0.3 retry) all screened with no further win; domain-specific backbone pretraining researched and not pursued (no accessible dataset found) — see §15 |

---

## 3. Reproduction results (patient-level held-out)

MobileNets are complete. 2 of 8 ResNet runs exist (WLE multiclass, both
backbones) — **AP@0.5 0.2809 (`fasterrcnn_resnet`) / 0.2594
(`fasterrcnn_resnet50_fpn_v2`), both markedly worse than MobileNet's 0.38-0.41
on the same cell.** Given that, and that this project's center has since
moved decisively toward the shortcut-audit findings, the remaining 6 ResNet
runs are **deliberately not being completed** — a full ResNet Table 4/5 row
wasn't going to change any conclusion in this document, only extend a
reproduction table nothing downstream depends on.

### Table 4 — multi-class detection

| Model | Modality | our AP@0.5 | paper | Δ |
| --- | --- | --- | --- | --- |
| Mobilenet | WLE | 0.4098 | 0.8817 | −0.47 |
| Mobilenet | EUS | 0.4514 | 0.5768 | −0.13 |
| Mobilenet_320 | WLE | 0.3809 | 0.8641 | −0.48 |
| Mobilenet_320 | EUS | 0.3476 | 0.5320 | −0.18 |
| Resnet | WLE | 0.2809 | 0.8593 | −0.58 |
| Resnet50_FPN_V2 | WLE | 0.2594 | 0.8792 | −0.62 |

### Table 5 — binary NET vs non-NET

| Model | Modality | AP NET | paper | AP non-NET | paper |
| --- | --- | --- | --- | --- | --- |
| Mobilenet | WLE | 0.8203 | 0.8851 | 0.5015 | 0.8784 |
| Mobilenet | EUS | 0.7456 | 0.8143 | 0.2780 | 0.3393 |
| Mobilenet_320 | WLE | 0.8108 | 0.9023 | 0.4809 | 0.8259 |
| Mobilenet_320 | EUS | 0.6688 | 0.7257 | 0.1876 | 0.3383 |

**NET detection reproduces well** (within 0.06–0.09). The multi-class gap is
driven by rare classes: NET has 1,611 training boxes, Leiomyoma 112. Table 4's AP
is a *mean over classes*, so three weak rare classes sink it however well NET does.

### Why our numbers differ from the paper

Three independent measurements converge:

| Evaluation | Result |
| --- | --- |
| Held-out, patient-level | 0.381 |
| **Training split (seen data)** | **0.853** (paper: 0.864) |
| Image-level split | +0.14 to +0.58 balanced accuracy over patient-level |

Under image-level splitting, **366 of 379 validation patients also appear in
training**. Evaluating on effectively-seen data lands on the published figures.

This is *evidence, not proof* — the paper states patient-level splitting, and its
own Usage Notes warn against image-level. But the pattern is consistent and
reproducible.

---

## 4. Shortcut audit — what's present and what isn't

| Mechanism | Verdict | Evidence |
| --- | --- | --- |
| Centre predicts the label | **not present** | Cramér's V 0.099; probe lift +0.003 |
| Site encoded in pixels | **present (strong)** | centre decodable at 0.81 vs 0.167 chance |
| Label leaks via global appearance | **not present** | max probe lift +0.10 |
| Label predictable from non-lesion pixels | **present (strong)** | 68–95% retention |
| Annotation geometry alone diagnostic | **present (strong)** | 48–189% retention |
| Collapse on unseen site | **partial** | 52–100% retained, small external n |
| Image-level split inflation | **present** | +0.14 to +0.58 |

The headline: **the label is recoverable from context and box geometry, largely
independent of lesion appearance.** But the site fingerprint, though real, is
*not* the route to the label — Center-1 supplies 537/641 patients, so site and
class are near-independent.

**Detection is less shortcut-prone than classification.** A classifier retained
68–95% of accuracy on lesion-free crops; a detector fires there only ~0.3× as
often as on full frames. Box regression forces commitment to a location.

---

## 5. Mitigations — what works

| Method | Effect | Cost |
| --- | --- | --- |
| **Hard negatives (30%)** | **false alarms −69% / −86%** | AP −0.05 / −0.01; F1 **+0.16** on multiclass |
| Background swap (classifier) | reliance −75% on WLE | −0.15 balanced acc |
| Scale jitter + copy-paste (classifier) | reliance −42 to −51% on WLE | ~free, sometimes +acc |
| Scale jitter + copy-paste (detector) | **no effect — within seed noise** | — |
| Size-matched *evaluation* | removes +0.03 avg (up to +0.10) of size prior | free, no training |

### The seed-noise floor (Exp 11) — read before trusting any A/B

Three seeds of `mobilenet_320` / WLE / multiclass. The split stays fixed at seed
42; the seed varies weight init and augmentation RNG only:

| arm | AP@0.5 | F1 | lesion-free ratio |
| --- | --- | --- | --- |
| baseline | 0.4200 ±0.0228 | 0.6136 ±0.0131 | 0.2334 ±0.0643 |
| augment | 0.4296 ±0.0220 | 0.4839 ±**0.1334** | 0.2465 ±0.0917 |
| hardneg | 0.4050 ±0.0297 | **0.6312** ±0.0609 | **0.0491 ±0.0104** |

**The baseline's own spread is ±0.023 AP and ±0.064 ratio.** Several single-run
comparisons made earlier — including Exp 8's "mitigation costs −0.008 AP" — were
smaller than that, and therefore meaningless. Do not report a detector A/B below
roughly 0.05 AP without multiple seeds.

`hardneg` clears the floor decisively (Δratio −0.184, ~3x pooled sd, consistent
across all three seeds: 0.063 / 0.045 / 0.039). `augment` does not (ΔAP +0.010,
Δratio +0.013), and its F1 spread of ±0.133 is ten times the baseline's — an
instability that is itself a reason to avoid it.

**Hard negatives are the clear winner.** They supply supervision the dataset
structurally lacks: every CAD-SEL image contains a lesion, so nothing ever teaches
the model what "no lesion here" looks like.

**The mitigations backfire on EUS.** Every condition made reliance *worse* on EUS
subtype (+23 to +37%). Grayscale, narrow-field ultrasound makes pasted lesions
implausible; the augmentation mostly injects noise. Review §9.6 anticipates this
("naive rebalancing does not fix a genuine visual-cue shortfall").

### Size is a real but secondary confound

Mean size-prior contribution **+0.029 AP**, but up to **+0.10** for
`mobilenet_320` WLE multiclass. AP by lesion-size tertile, WLE binary:
**0.471 → 0.663 → 0.755**. Small lesions are near-unusable — clinically the
important failure, since those are the ones endoscopists miss.

---

## 5a. Adversarial discriminator, bbox blur, channel attention (Exp 12-16)

Three further classifier-level mitigations, following Exp 7's protocol (train
**one** model, ablate at inference) rather than Exp 3's (a mitigation is a
property of a trained model, not of an input). Full design in
`experiments/adversarial.py`, `experiments/bbox_blur.py`,
`experiments/backbones.py`; results in `experiments/results/12_adversarial.json`,
`13_bbox_blur.json`, `14_channel_attention.json`.

| Method | `ret_bg_patch` vs baseline, 4 modality/task cells | Verdict |
| --- | --- | --- |
| **Exp 12** — GRL discriminator on `bg_patch` (+ ablated `boxmap`) | 7/8 conditions negative (less reliance); 1/8 (EUS/subtype) cleared the noise floor | Suggestive, not confirmed at n=3 seeds |
| **Exp 13** — blur non-lesion pixels (feathered, jittered, padded ≥1.15x box extent) | mixed sign on WLE; **backfires on EUS binary** (+34 to +38% retention) | No effect; no boundary-leak warning fired (the feathering/padding safeguards worked) |
| **Exp 14** — SE channel attention × discriminator (`se_all_adv`) | **8/8 cells negative**, clean dose-response (`se_all_adv` beat `se_layer4_adv` in all 4 combos); `se_*_ce` (no-discriminator control) behaved as the expected null | Most consistent direction of the three, not yet confirmed |

**Exp 15/16 — does `se_all_adv` actually hold up?** Extended `baseline` and
`se_all_adv` from 3 to 7 seeds (Exp 15), then reran both at up to 100 epochs
with early-stop patience 20 (Exp 16, `experiments/src/16_se_all_adv_100epoch.py`;
early stopping added to `experiments/adversarial.py::train_eval_adversarial`).
Models converged well before the cap (mean 27-45 epochs). Result: **all 4
cells still negative, none clear 2x the pooled seed-noise floor**, even after
proper convergence — noise floors actually tightened in 3 of 4 cells (e.g.
EUS/subtype: σ 0.379 → 0.221), so this is a settled read, not an
under-powered one.

**Verdict: `se_all_adv` is a real, consistent, but small effect — not strong
enough to confirm as a validated mitigation at the classifier level.** None of
the three methods (12/13/14) cleared their own noise floor with the budget
tested. DFR / Group DRO / LfF remain deferred (queued, unscheduled) as the
next things to try if this line of work continues.

---

## 5b. Multimodal (WLE + EUS) — what the data allows

The paper's baselines are unimodal despite CAD-SEL being a *dual-modal* dataset.
Three measured facts define what a multimodal model can be here:

| Fact | Number | Consequence |
| --- | --- | --- |
| EUS patients are a **subset** of WLE patients | 421 of 641, zero EUS-only | Fusion is possible; 220 patients are WLE-only |
| Frames are **not** paired | median 4 WLE + 4 EUS/patient, no correspondence | Fusion must be patient-level, over two unordered frame sets |
| The domains barely overlap | WLE channel spread 0.264, EUS **0.0004**, ~2.9× darker | Per-modality normalisation is a prerequisite, not a tweak |

**The per-modality baseline splits are cross-modally leaky.** Splitting WLE and EUS
independently at seed 42 puts **67 of the 85 EUS validation patients' WLE frames in
the WLE training set**; only 19 of 85 validation patients agree between the two
splits. So the completed grid in `runs/` *cannot* be the denominator for a joint
model. `cadsel.train --joint-split` exists for exactly this: it partitions patients
over the union and then keeps one modality, so a unimodal baseline is scored on the
joint run's own validation patients.

`CADSEL_MEAN`/`CADSEL_STD` are effectively the WLE statistics (WLE is the brighter,
more numerous half). Applied to EUS they land its channels between −0.9 and −1.8
and halve its dynamic range. `MODALITY_STATS` in `config.py` holds per-modality
values measured over all 4,912 images by `python -m cadsel.measure_stats`; the
unimodal baselines deliberately keep using `CADSEL_*` so their numbers stay
comparable to the completed grid.

Design of the two stages, and why:

- **Stage 1, joint detector.** One Faster R-CNN over the union, batches kept
  modality-homogeneous by `ModalityBatchSampler`, optional per-modality FiLM on the
  FPN outputs. The expected win is rare classes: Leiomyoma goes from 139 WLE boxes
  to 220 pooled, and Table 4's mAP is a mean over classes. Evaluated **per
  modality** so a model that helps WLE by trading EUS away is visible.
  `--conditioning none` is the required control.
- **Stage 2, patient-level fusion.** Frozen Stage 1 → ROI features of the
  top-scoring box per frame (not global pooling: Exp 3 showed 68–95% of the label
  is recoverable from lesion-*free* crops, so a global descriptor fuses two
  shortcuts) → attention-pool within modality → gated fusion → patient subtype.
  Modality dropout p=0.3 so the 220 WLE-only patients train and WLE-only inference
  degrades gracefully. **5-fold CV**, because 421 paired patients with 14 Leiomyoma
  leaves ~3 in a single 20% validation split.
  This metric is patient-level balanced accuracy / macro-F1 and is **not**
  comparable to Table 4/5 AP.

`copy_paste` is rejected outright for joint runs — pasting colour WLE lesions into
grayscale EUS frames is the augmentation class Exp 7 measured making EUS reliance
*worse* (+23 to +37%).

### Two bugs found while building this

**`report._index` silently overwrote Table 4 rows.** It keyed on
`(model, modality, task)`, which `runs/` satisfies more than once: `ms_baseline_s7`,
`hn30`, `*__mitigated` all share a key with the canonical baseline, and whichever
sorted last won. Exp 11's multi-seed runs replaced the Mobilenet_320 WLE multiclass
AP with **0.4397 in place of 0.3809**. `REPORT.md` on disk predated those runs, so
it had never shown the wrong value. `_index` now requires the canonical
`model__modality__task` run name. Any new run family belongs in its own directory.

**Editing a live module killed a running experiment.** Adding an import to
`cadsel/data.py` before the constant existed in `config.py` crashed Exp 11 mid-run
(it re-imports `data.py` per cell). Exp 11 skips runs whose `result.json` exists, so
only the in-flight run was lost — but the lesson is to add the constant before the
import, or not to touch shared modules while a grid is live.

---

## 5c. Detection-performance literature review + Tier-1 levers

A literature review of small-object/imbalanced/limited-data detection
techniques (full findings kept out of this file — see the session record; the
short version and every citation is in `experiments/README.md`), targeting
raw AP rather than shortcut mitigation. Six "Tier 1" (cheap, additive,
default-off) items were implemented directly in `cadsel/` — see the
README's "Performance levers" section for flags and examples:

- **`--auto-anchors`** (`cadsel/anchors.py`, `cadsel.measure_anchors`) — RPN
  anchor sizes/aspect-ratios clustered (IoU k-means) from the training box-size
  distribution, replacing torchvision's COCO-tuned defaults. Verified: WLE @
  640px clusters to sizes ≈(117,181,249,329,446) vs the COCO defaults'
  (32,64,128,256,512) — CAD-SEL's lesions sit at a very different scale.
- **`--rpn-fg-iou-thresh` / `--rpn-bg-iou-thresh` / `--box-fg-iou-thresh` /
  `--box-bg-iou-thresh`** — configurable IoU matching thresholds (torchvision
  defaults 0.7/0.3/0.5/0.5), targeting small-box anchor starvation.
- **`--multiscale-sizes`** (`cadsel/models.py::configure_transform_for_input_mode`)
  — random per-image training scale via torchvision's own `GeneralizedRCNNTransform`
  multi-value `min_size`; content-preserving, so a candidate for the scale
  diversity that `scale_crop`/`copy_paste` measurably hurt EUS with. Only
  meaningful with `--input-mode torchvision`.
- **`--rfs-threshold`** (`cadsel/data.py::RepeatFactorSampler`) — Detectron2-style
  Repeat Factor Sampling; the image-level complement to the existing per-box
  class-weighted loss (loss weighting changes how much a rare-class box's
  gradient counts once seen; RFS changes how often it's seen at all).
- **`--ema`/`--ema-decay`** (`torch.optim.swa_utils.AveragedModel`, wired into
  `cadsel/engine.py::train_one_epoch` and `cadsel/train.py`) — evaluation and
  checkpointing use the averaged weights once enabled.
- **`cadsel.evaluate_tta`** (`cadsel/tta.py`) — multi-scale Weighted Boxes
  Fusion at inference time (hand-rolled in NumPy, no new dependency), reported
  against the checkpoint's own single-scale baseline. Inference-only.

All six were smoke-tested individually and together (a 2-epoch run with every
flag enabled simultaneously trained, evaluated, and checkpointed cleanly).

**First real result** (`fasterrcnn_mobilenet_320`/WLE/multiclass, single seed):
auto-anchors + IoU thresholds + multiscale + RFS + EMA together scored
**AP@0.5 0.4106 vs the existing baseline's 0.3809 (+0.030)** — but P/R/F1 at
the fixed 0.5 confidence threshold *dropped* (F1 0.619→0.546), so ranking
improved while calibration at that cutoff shifted; not yet re-tuned. This is
just outside Exp 11's noise floor for this exact model/modality/task
(±0.023 AP) on a *single* run — encouraging, not confirmed. A 10-run
config-grid (`experiments/configs/tier1_confirmation_grid.json`, via
`cadsel.run_all --config-grid`) extends baseline/combo to 3 seeds each and
ablates each of the 5 levers individually plus the combo without multiscale
(isolating whether the `torchvision` input-mode switch itself, required for
multiscale, explains part of the gain) — not yet run.

Multi-scale WBF TTA (`cadsel.evaluate_tta`) on that same checkpoint made
things *slightly worse* (single-scale 0.394 → multi-scale-fused 0.386,
**-0.009**) — plausibly because a model already trained with
`--multiscale-sizes` has already learned scale-robust features, so
inference-time fusion on top adds noise rather than new information. Worth
retrying against a checkpoint that was *not* trained multi-scale.

**Bug found and fixed while checking this**: checkpoints didn't record
`multiscale_sizes`, so re-evaluating one from a fresh process (`evaluate_tta`,
`visualize`) silently used the wrong resolution and undercounted AP by ~0.03 —
torchvision's transform deterministically uses the *last* (largest) value of
a multi-scale tuple in eval mode, which training's own per-epoch validation
naturally picked up (same live model, transform untouched between epochs) but
a fresh `build_detection_model` + `configure_transform_for_input_mode(...,
img_size)` reconstruction did not. Fixed: checkpoints now save
`multiscale_sizes`, and `evaluate_tta.py` uses it when present.

### Tier 2 — moderate cost, also implemented

- **`--loss-fn seesaw --seesaw-p 0.8`** (`cadsel/models.py::SeesawLossState`) —
  Seesaw Loss (Wang et al., CVPR 2021): tracks *cumulative* per-class counts
  as training progresses (not `compute_class_weights`' one-time snapshot) and
  damps how much a frequent class's logit suppresses a rarer one's, scaled by
  their current count ratio. Verified numerically (mitigation-matrix logic,
  background exclusion, forward+backward) and end-to-end through
  `cadsel.train --loss-fn seesaw`.
- **`speckle_noise` / `ultrasound_gain`** augmentation ops (`cadsel/data.py`)
  — multiplicative speckle noise and a depth-dependent (TGC-style) gain
  gradient, EUS-specific and physically motivated rather than generic, aimed
  at the augmentations that measurably backfire on EUS (§5). Opt-in via
  `--augment ...,speckle_noise,ultrasound_gain`, same pattern as `copy_paste`.
- **`cadsel.finetune_rare`** — TFA-style (Wang et al. 2020) two-stage
  fine-tuning: freeze everything but the box predictor, briefly fine-tune on
  a class-balanced subset (`build_balanced_subset`) that oversamples the
  rarest classes. Verified end-to-end against the tier1 checkpoint (25,625 of
  18.9M params trainable, subset built to 436 images with box counts
  112/115/116/116 across the 4 classes vs the original 1611/112/383/243).
- **`--backbone-weights PATH`** (`cadsel/models.py::load_backbone_weights`) —
  loads a raw state_dict into `model.backbone.body` in place of COCO
  pretraining, for a domain-specific backbone per the literature review
  (RadImageNet for EUS, in-domain endoscopy SSL for WLE). This is the loading
  *mechanism* only, verified against both a correctly-keyed and a
  deliberately-mismatched synthetic state dict (reports matched/missing/
  unexpected key counts either way) — no specific third-party weights were
  acquired or bundled; that's real, separate future work (find/convert a
  compatible file, then point this flag at it).

None of Tier 2 has training results yet — all four are smoke-tested and ready
to use, not yet run to convergence or compared against a baseline.

### Tier 3 — two of four items implemented; two deliberately skipped

The literature review's own recommendation was to hold Tier 3 until Tier 1/2
were measured, since each item there risks the validated pipeline for
uncertain gain. Multiscale training (Tier 1) has since cleared a real
confirmation (§5c above); at the user's request, two Tier 3 items were
implemented and two were scoped out rather than attempted:

- **`fcos_resnet50`** (`cadsel/models.py::build_fcos_model`, registered in a
  separate `FCOS_MODEL_REGISTRY`) — the anchor-free architecture item, using
  torchvision's own tested FCOS (Tian et al. 2019) rather than a from-scratch
  port of a specific paper variant (YOLO-OB's custom BiSPFPN head has no
  public reference implementation to build against; the review's own fallback
  was "prioritize an anchor-free CNN detector (FCOS-style) over full DETR").
  Fully compatible with the existing pipeline unchanged (`train_one_epoch`/
  `evaluate` are generic over any `GeneralizedRCNN`-family model) *except*:
  no RPN/ROI-head IoU matching (FCOS uses center-sampling), so
  `--auto-anchors`/`--rpn-*-iou-thresh`/`--box-*-iou-thresh` have no effect
  and are flagged with a NOTE; no call through `roi_heads.fastrcnn_loss`, so
  `--loss-fn` has no effect either. **Found in testing**: the Faster R-CNN
  family's default `--learning-rate 1e-3` makes FCOS's loss collapse to
  0/NaN within ~1 epoch on this dataset -- `--learning-rate 1e-4
  --warmup-iters 0` trained stably (AP@0.5 0.23→0.29 over 2 epochs). Now
  warned about automatically rather than left as a silent trap.
- **Conditional lesion GAN** (`cadsel/lesion_gan.py`, `cadsel.train_lesion_gan`)
  — the EUS generative-augmentation item. A small class-conditional DCGAN
  (64x64 patches) rather than a diffusion model: the review's own caution was
  that training a *usable* generator on a few hundred crops per class is
  itself a small-data problem (confirmed here -- EUS Leiomyoma has only 64
  crops after `LesionBank`'s extraction), so this is explicitly a tractable
  prototype, not an attempt at diffusion-model quality. `GeneratedLesionBank`
  drops into the existing `copy_paste` augmentation as an alternative patch
  *source* (`--copy-paste-source generated --lesion-gan-checkpoint PATH`),
  reusing `_copy_paste` unchanged. **Real bug found and fixed while wiring
  this in**: generating patches on-the-fly inside `_copy_paste` crashes under
  `--num-workers > 0` (`RuntimeError: Cannot re-initialize CUDA in forked
  subprocess`) since `CadSelDetectionDataset.__getitem__` runs inside forked
  DataLoader workers. Fixed by pre-generating a fixed pool (default 200
  patches/class) once in the main process before workers fork, mirroring
  `LesionBank`'s own real-crop-pool design exactly -- verified working with
  `--num-workers 2`, the config that originally crashed. No training results
  yet (quality of the generated patches themselves is untested).
- **Cross-modal WLE↔EUS domain adaptation (CycleGAN-style) -- skipped.** The
  review itself flagged this as lower priority, and CAD-SEL's WLE/EUS are
  different physical imaging processes of a shared lesion (color endoscopy
  vs. grayscale ultrasound), not the same modality across sites -- translation
  validity is a bigger risk here than for typical cross-modality medical
  imaging work the technique is usually applied to.
- **Paired dual-modality fusion architecture -- skipped, and doesn't apply.**
  The review's suggestion assumed CAD-SEL might pair the same physical lesion
  across WLE and EUS captures. §5b already established the opposite: "Frames
  are not paired... no correspondence." A paired-instance fusion architecture
  isn't just deprioritized here, it's the wrong design for this dataset's
  actual structure. The *patient-level* (unpaired) fusion the data does
  support is already implemented -- `cadsel/fusion.py`, `cadsel.train_joint`,
  `cadsel.train_fusion` (§5b) -- and is the correct next step for multimodal
  work, not a new architecture.

## 5d. Separate per-modality backbones + RadImageNet for EUS

Stage 1's shared backbone + FiLM conditioning (§5b) was a deliberate choice:
"only 421 EUS patients... two towers would share none of it." The
alternative — two independent detectors, one per modality, with a
domain-appropriate pretrained backbone for EUS specifically so it needs less
task data rather than more — is now implemented too, as a smaller and more
surgical change than it might sound, since Stage 2 (`fusion.py`) was already
built to consume frozen features from *a* detector; it just assumed one
shared one.

- **`cadsel.convert_radimagenet_weights`** — `Lab-Rasool/RadImageNet`'s
  `ResNet50.pt` (MIT-licensed, verified in this session to load safely with
  `weights_only=True` -- a plain tensor state_dict, no pickled-object risk)
  wraps a standard torchvision `resnet50()` as `nn.Sequential(...)`, so its
  keys are `backbone.<index>...` instead of the named-submodule convention
  `load_backbone_weights` expects. Converted and verified: 318/318 tensors
  remapped, then 265/265 matched loading into a real `fasterrcnn_resnet`
  backbone via `--backbone-weights` with 0 missing/unexpected keys.
- **`fusion.py::extract_features`** now accepts either the original shared
  model (unchanged) or a `{"WLE": model, "EUS": model}` dict, routing each
  modality's frames to its own detector (`set_active_modality`/FiLM
  conditioning is skipped for the dict path, since two independent models
  have no shared conditioning to select).
- **`train_fusion.py`** gained `--checkpoint-wle`/`--checkpoint-eus` as an
  alternative to `--checkpoint`, via a new `load_stage1_separate` — requires
  both detectors to share `img_size`/`input_mode` (the dataset/transform for
  feature extraction is still built once for every frame regardless of which
  model processes it) and, for the RadImageNet backbone specifically, both
  detectors must be ResNet50-family (`fasterrcnn_resnet` /
  `fasterrcnn_resnet50_fpn_v2`) — RadImageNet's converted weights are
  architecture-specific to that backbone, not MobileNetV3.
- **Verified end-to-end**: trained two real (1-epoch smoke) detectors — WLE
  plain, EUS with the RadImageNet backbone — then ran the full Stage 2 fusion
  pipeline (separate-backbone feature extraction over all 4,912 frames, all 4
  ablations, 2-fold CV) against them without error.

Training results now exist — the separate-backbone detectors
(`wle_resnet_backbone`, `eus_radimagenet_backbone`) are the ones Stage 2
fusion (§5b, §5i) is built on.

---

## 5e. SVM/DINOv3 through the counterfactual protocol — resolves item 10

§8 item 10 asked whether DINOv3's edge over ResNet-50 on frozen embeddings
(WLE binary balanced accuracy 0.735 vs 0.645) reflects genuinely better
lesion-reading or just better shortcut-reading. New `experiments/src/
17_svm_counterfactual.py` runs both backbones' embeddings + SVM through
Experiment 3's 5-variant counterfactual protocol (`full`/`lesion`/`bg_mask`/
`bg_patch`/`boxmap`) — cheap here since both backbones are frozen (20 forward
passes total, no training), unlike Exp 3's literal from-scratch retrain per
variant. `experiments/ablation_inputs.py::build_five_variants` was verified
byte-identical to Exp 3's own in-file `build_variants` before trusting the
comparison.

**The answer splits by modality.** DINOv3-minus-ResNet50 gap at full image,
vs. at `bg_patch` (lesion removed):

| modality/task | gap@full | gap@bg_patch | retained | verdict |
| --- | --- | --- | --- | --- |
| WLE binary | +0.090 | +0.047 | 52% | **partial** — edge shrinks but survives |
| WLE subtype | +0.084 | +0.032 | 38% | **partial** — edge shrinks but survives |
| EUS binary | +0.032 | +0.049 | 152% (grows) | **shortcut-reading** |
| EUS subtype | +0.007 | +0.014 | 200% (grows) | **shortcut-reading** |

On **WLE**, roughly half to three-quarters of DINOv3's advantage evaporates
once the lesion is removed — a real, if partial, lesion-reading edge. On
**EUS**, DINOv3's full-image edge is tiny to begin with (0.7-3.2 points) and
*fully survives or grows* under lesion removal — consistent with EUS already
being the modality where Exp 3 found the strongest geometry-shortcut
retention (93-189%). DINOv3's modest EUS edge over ResNet-50 is not evidence
of better lesion-reading; it's a better reader of the same shortcut.

Full retention table and the join against Exp 3's classifier numbers:
`experiments/results/17_svm_counterfactual.json` and `..._verdict.json`.

---

## 5f. Leave-one-center-out validation — resolves item 5

§8 item 5 flagged that Exp 4's "external" set (~100 patients, same collection
protocol) isn't genuinely external. Two passes, classifier then detector:

**Classifier-level (`experiments/src/18_loco.py`, Exp 18).** Generalizes Exp
4's Center-1-only asymmetric split into true LOCO: train on every center
*except* one, test on the held-out one, for Center-1/2/3 individually plus
Centers-4-6 pooled (each too small — 6-8 patients — to hold out alone). 28 of
32 planned ResNet-18 runs completed (EUS Centers-4-6 auto-skipped, n=38 below
the 40-image guard).

| Modality/task | Center-1 retained | Center-2 retained | Center-3 retained |
| --- | --- | --- | --- |
| WLE binary | **40.6%** (n=2277) | 73.3% | **49.4%** (n=368) |
| WLE subtype | 66.8% | 114.7% | 89.2% |
| EUS binary | **57.3%** (n=1498) | 149.3% | 100.7% |
| EUS subtype | 96.0% | 251.3% | 124.0% |

WLE-binary shows a real, well-supported drop for Center-1 and Center-3 (not
a small-N fluke — Center-1's external set is the largest in the study).
Center-2 shows no drop anywhere, and several EUS cells "retain" >100% (small
external N, balanced-accuracy volatility, not genuine cross-site gain). Note
this is *worse* than Exp 4's original Center-1-trained number (71.5%
retained) — true LOCO's Center-1 fold trains on a small, skewed 5-site pool
(540 WLE images) rather than Exp 4's large, prevalence-representative
Center-1, which may have flattered the original cross-site transfer estimate.

**Detector-level confirmation, scoped (`cadsel.evaluate_holdout_centers`,
new).** Per Exp 18's own recommendation, only the two fold/task combinations
with well-supported classifier-level signal were run at the detector level
(WLE binary, `--centers` now wired through `build_manifest`/`RunConfig`):

| Held-out center | Internal AP@0.5 | External AP@0.5 | |
| --- | --- | --- | --- |
| Center-1 (n=2277 images / 537 patients) | 0.3819 | **0.4022** | no drop at all |
| Center-3 (n=368 images / 68 patients) | 0.6532 | **0.1330** | severe collapse (~80% loss) |

**This inverts the classifier-level ranking.** The classifier flagged
Center-1 as the worse-generalizing site (40.6% retained) and Center-3 as
milder (49.4%); the detector shows the opposite — Center-1 generalizes
cleanly, Center-3 catastrophically fails (non-NET AP@0.5 collapses to 0.048).
**A classifier-level shortcut/generalization diagnostic does not reliably
predict which sites the detector that actually ships will fail on** — worth
treating as a standing caution against using the classifier as a cheap proxy
for detector-level site generalization on this dataset. Center-3's collapse
is a genuine, confirmed external-validation failure and the strongest single
piece of evidence in this session that site fingerprint can matter for
detection specifically, not just classification.

Raw results: `experiments/results/18_loco.json`;
`runs/loco_holdout_center{1,3}_binary/holdout_result.json`.

---

## 5g. Size-aware oversampling — targets the small-lesion gap directly

§5/§8 flagged AP by lesion-size tertile (WLE binary: 0.471 → 0.663 → 0.755)
as the most clinically important failure mode, with nothing targeting it
directly. New `SizeRepeatFactorSampler` (`cadsel/data.py`) reuses
`RepeatFactorSampler`'s stochastic-round mechanism but buckets by
`box_area_frac` tertile instead of class id (`--size-rfs-threshold`, mutually
exclusive with `--rfs-threshold`), combined with the one confirmed-positive
lever, multiscale training. 9 new runs
(`experiments/configs/size_lever_grid.json`) plus the 3 already-on-disk
`tier1_multiscale_only_s{42,123,7}` as the multiscale-alone control, evaluated
by `experiments/src/19_size_lever_eval.py` (which reads each run's own
`split.json` rather than a single fixed seed — multi-seed runs have
different splits, an easy trap the first version of this script fell into).

| Task | ms_only tertile-1 AP | size_rfs_ms tertile-1 AP | delta | vs. floor |
| --- | --- | --- | --- | --- |
| multiclass (3 seeds) | 0.2712 ± 0.0432 | 0.3381 ± 0.0357 | **+0.0669** | **clears** (~0.056) |
| binary (3 seeds) | 0.4748 ± 0.0402 | 0.5027 ± 0.0447 | +0.0279 | within (~0.0601) |
| binary (5 seeds) | 0.4954 ± 0.0404 | 0.5082 ± 0.0353 | +0.0128 | within (~0.0536) |

Overall AP is flat in both tasks (multiclass 0.4294 → 0.4202, binary
0.6711 → 0.6731) — the gain is specific to the small-lesion bucket, not a
general lift, and costs nothing elsewhere. **The multiclass result is a
confirmed win**, the first lever in this project to move the small-lesion
tertile specifically rather than overall AP. **Binary is now settled, not
confirmed**: 2 more seeds shrank the delta from +0.0279 to +0.0128 — the
same direction more data pulls an unconfirmed effect, toward zero, not
toward significance. Don't expect this lever to help the binary task.

---

## 5h. Lesion GAN copy-paste, and TFA re-examined at multiple seeds

**Lesion GAN (resolves the GAN half of items 4/8).** The class-conditional
DCGAN generator (`runs/lesion_gan_EUS_multiclass.pth`) had been trained but
never fed into a detector run. New grid
(`experiments/configs/gan_copypaste_grid.json`, 8 new runs + 1 reused seed)
compares plain augmentation, real-crop `copy_paste`, and
`--copy-paste-source generated` on EUS/multiclass, evaluated by Exp 8's own
hallucination-ratio protocol extended to 3 seeds
(`experiments/src/20_gan_copypaste_reliance.py`; same per-run-split fix as
§5g's script):

| Arm | AP@0.5 (mean ± sd, 3 seeds) | Hallucination ratio |
| --- | --- | --- |
| plain control | 0.3838 ± 0.0480 | 0.109 ± 0.041 |
| real copy-paste | 0.3669 ± 0.0528 | 0.106 ± 0.038 |
| GAN copy-paste | 0.3763 ± 0.0479 | **0.065 ± 0.034** |

Exp 8 was previously flagged "ambiguous at n=1"; at 3 seeds, **real
copy-paste's detector-level hallucination ratio is essentially flat against
plain control** (−0.003), not the backfire pattern §5/Exp 7 found at the
*classifier* level (+23 to +37% reliance on EUS subtype). Read together with
"detection is less shortcut-prone than classification" (§4), this looks like
the backfire is a classifier-task-specific effect (implausible pasted
texture leaking the subtype label) that doesn't materialize the same way in
the detector's box-level objective, rather than a contradiction to resolve.
**GAN copy-paste shows the lowest hallucination ratio of the three arms** —
directionally the answer to "does synthetic copy-paste avoid the backfire
that real copy-paste caused," though the delta (−0.044) is close to the
combined seed noise (~0.053) and shouldn't be called fully confirmed yet.

**TFA rare-class fine-tuning, multi-seed (resolves item 7's TFA half).** The
one prior run showed a Leiomyoma AP@0.5 regression (0.175 → 0.147). A
zero-GPU check first: the *pre-TFA* seed-to-seed Leiomyoma spread across
three `tier1`-recipe checkpoints already on disk is 0.175/0.144/0.142 (sd
0.015) — the reported "regression" landed inside that spread, not below it.
Two 2-run sweeps then disambiguated it (`cadsel/finetune_rare.py` gained
`--split-seed`, decoupled from `--seed`, since it had been silently
conflating held-out-patient selection with fine-tuning randomness — a real
bug, fixed): Sweep A holds the base checkpoint and split fixed, varies only
fine-tuning-step randomness; Sweep B fine-tunes different already-trained
base checkpoints of the same recipe.

| Sweep | n | ΔAP@0.5 (overall) | ΔAP@0.5 (Leiomyoma) | verdict |
| --- | --- | --- | --- | --- |
| A (fine-tune noise only) | 3 | −0.0116 ± 0.0002 | **−0.0277 ± 0.0011** | Leiomyoma: **confirmed regression** |
| B (different base checkpoints) | 2 | +0.0023 ± 0.0025 | +0.0380 ± 0.0258 | not confirmed either way |

TFA reliably hurts Leiomyoma when only fine-tuning randomness varies (a
real, small, consistent effect — Sweep A's std is tiny), but Sweep B swings
from +0.012 to +0.064 depending on which base checkpoint it starts from, so
it isn't a stable property of the recipe on this dataset. **Downgrade from
"TFA hurts" to "TFA has a small, base-checkpoint-dependent effect on
Leiomyoma; not a reliable rare-class fix here as currently configured."**

Raw results: `experiments/results/20_gan_copypaste_reliance*.json`,
`experiments/results/21_tfa_multiseed*.json`.

---

## 5i. Multimodal fusion — subgroup analysis and a symmetric-attention alternative

§5b/§5d's headline (`fused` barely beats `wle_only`, 0.419 vs 0.426 balanced
accuracy) turned out to be measured on the wrong denominator: it's computed
over all 641 patients, including 220 WLE-only patients where fusion
*structurally cannot differ* from `wle_only`. `cadsel/train_fusion.py` now
captures per-patient softmax predictions on every new-best epoch
(`results/fusion_predictions.json`); `cadsel/analyze_fusion_subgroups.py`
restricts to the 421 paired patients before comparing.

**On paired patients only**, accuracy tells a different story than balanced
accuracy: `wle_only` 0.378 → `fused` 0.420 (+0.043) → `fused_moddrop` 0.523
(**+0.145**). Bucketing by `wle_only`'s own confidence tertile, the gain
concentrates in the low/mid tertiles and shrinks or reverses in the
high-confidence tertile — directionally the clinically expected pattern
(EUS helps exactly where WLE is uncertain). But it is **driven almost
entirely by NET_G1** (n=299, the majority class): Lipoma (n=49, the only
other subtype with enough n to slice) shows fusion hurting uniformly across
all three tertiles. And on **balanced accuracy**, which doesn't let the
majority class dominate, `wle_only` actually *beats* both fused variants on
the paired subset. Net: real signal exists, but it is a majority-class
accuracy effect, not a broad, class-general one — a substantive correction
to "does fusion help" that any future multimodal work here should measure on
the paired subset going forward, not the diluted 641-patient number.

**Does the `GatedFusion` module's structural WLE-fallback explain the
shortfall?** `GatedFusion`'s `(1-gate)*a` always falls back to the WLE
stream specifically (`MODALITY_NAMES = ["WLE", "EUS"]`). A new
`SymmetricCrossAttentionFusion` (2-token self-attention over
`[WLE, EUS]` with a learned "modality absent" embedding, no hardcoded
fallback) was compared against it at 3 seeds × 5-fold CV each, reusing the
already-cached patient features (no re-extraction). **No** — 3 of 4
ablation/metric comparisons don't clear seed-to-seed noise (~0.008-0.013),
and the one nominal exception (macro-F1, +0.0105) doesn't replicate across
the other three. The architectural asymmetry isn't what's limiting fusion.

Files: `cadsel/analyze_fusion_subgroups.py` (new),
`results/fusion_subgroup_analysis*.json`, `results/fusion_{gated,symmetric}_s*.json`.

---

## 5j. A second `report.py` data-loss bug, found and fixed

`results/REPORT.md` was found reduced to 48 bytes (just the title) and
`results/all_results.json` overwritten down to 8 entries — a second instance
of the class of bug already documented in §6 (`_index`'s sorted-last-wins
collision), but worse: a near-total wipe, not a row overwrite.

**Root cause**: `cadsel/run_all.py::main()` called
`write_reports(results, args.results_dir)` where `results` was only the
in-memory list of runs *that specific grid invocation* executed or reused —
not a fresh scan of `runs/`. `write_reports` (`cadsel/report.py`)
unconditionally overwrites `all_results.json` from whatever list it's
handed (no read-merge-write). Running `cadsel.run_all --config-grid
gan_copypaste_grid.json` (§5h) therefore wiped the shared aggregate down to
just that grid's 8 runs; since none of those run names are canonical
`model__modality__task` strings, `_index` found zero matches and `REPORT.md`
came out nearly empty.

**Fix**: `run_all.py` now imports `load_results` from `.report` and calls
`write_reports(load_results(args.output_dir), args.results_dir)` — always
"everything currently in this runs directory," matching `report.py::main()`'s
own standalone semantics, for both `--config-grid` and the default grid
path. Regenerated via `python -m cadsel.report`: `all_results.json` back to
74 entries, `REPORT.md`'s Table 4/5 restored (and now includes the 2
completed ResNet rows, §3). `results/MULTIMODAL.md` was never affected by
this bug (a separate code path). **Note**: no current tooling writes
`results/RESULTS.md` by that name — the on-disk file (Aug 4) is a stale,
orphaned artifact, not a target to keep fixing.

---

## 5k. Is Center-3's LOCO collapse a labeling artifact? No.

§5f found Center-3's LOCO detector holdout collapses (internal AP@0.5
0.6532 → external 0.1330) for two compounding reasons: the training pool is
93.7% Center-1 by patient count when Center-3 is held out, and Center-3's
images (100%) are a raw 1920×1080 screen-capture format (UI sidebar, black
letterboxing around a shrunken field of view) essentially unseen in
training. A further wrinkle complicated the obvious fix: decoding Center-3's
YOLO boxes against the raw canvas shows a majority substantially overlap
black background (58.6% of boxes >10% black overlap, 23.3% >50%, several
100% — floating with no visible lesion nearby), and a naive "crop to content
rect, remap boxes" fix tried on one example did not resolve this.

New `experiments/src/22_center3_black_overlap_eval.py` settles the more
tractable question first, with no retraining: bucket each image by its
max-over-boxes black-overlap fraction into "clean" (<10%) vs "dirty" (≥10%),
and re-score the already-trained `runs/loco_holdout_center3_binary`
checkpoint on each bucket separately.

**Result: the opposite of the labeling-artifact hypothesis.** Clean-bucket
AP@0.5 = 0.0953 (n=155), *lower* than dirty-bucket AP@0.5 = 0.1705 (n=213)
and lower than the pooled 0.1330. If mislabeled boxes were dragging the
score down, dirty should score worse, not better. **This pattern replicated
across every one of the 12 checkpoints tested in §5l/§5m/§5n (every arm,
every seed) — always dirty > clean, no exception.** The Center-3 collapse is
not primarily a labeling artifact; it reflects a genuine generalization
failure on Center-3's visual domain, independent of the box-alignment issue
(which remains a real, separate data-quality concern — see item 12). Per the
plan's own gate, this ruled out spending further time this round on
crop-and-retrain, and set the clean-bucket comparison as the more reliable
yardstick for §5l-§5n's levers below (the pooled/dirty numbers are noisier
and, per this finding, measuring something not yet understood).

---

## 5l. Center-bucketed oversampling — confirmed on the clean subset

New `CenterRepeatFactorSampler` (`cadsel/data.py`, `--center-rfs-threshold`):
`RepeatFactorSampler`'s mechanism, bucketed by `Sample.center` instead of
class or size, directly targeting the training-pool imbalance §5k
identifies. Tested against a 3-seed Center-3-LOCO baseline (bringing the
existing single-seed `loco_holdout_center3_binary` to 3 seeds) plus 3 new
`--center-rfs-threshold 0.2` runs, same recipe otherwise:

| | mean external AP@0.5 | std | delta vs. baseline | floor | clears? |
| --- | --- | --- | --- | --- | --- |
| baseline (pooled) | 0.1916 | 0.0567 | — | — | — |
| center-RFS (pooled) | 0.2458 | **0.0049** | +0.0541 | 0.0570 | No (narrowly) |
| baseline (clean bucket) | 0.1113 | 0.0235 | — | — | — |
| **center-RFS (clean bucket)** | **0.1464** | 0.0056 | **+0.0351** | 0.0242 | **Yes** |

The pooled comparison narrowly misses the noise floor — but that
comparison is dominated by the baseline's own instability (one seed spiked
to 0.2684, per §5k's now-established dirty-bucket-inflation pattern).
**Restricted to the clean bucket — the more trustworthy read per §5k — the
lift clears its floor: a real, confirmed win.** Separately notable: center-RFS
collapses the *variance* across seeds by roughly 10× (0.0567 → 0.0049
pooled; 0.0235 → 0.0056 clean) — even setting aside the mean shift, this is
a detector that behaves far more predictably across random seeds when
minority-center oversampling is active, itself a useful property for a
LOCO-style setting where a single outlier seed could otherwise mislead.

---

## 5m. DFR (Deep Feature Reweighting), detector-level, group=center — no effect

DFR (Kirichenko et al. 2023): freeze everything but the box predictor
(the exact TFA/`finetune_rare.py` mechanism), fine-tune briefly on a
group-balanced subset — new `build_balanced_subset_by_center` (sibling of
`build_balanced_subset`, `--balance-by center`), plus a `--centers` flag
finetune_rare.py was missing (needed so fine-tuning against a LOCO
checkpoint doesn't silently leak the held-out center back in). Fine-tuned
`runs/loco_holdout_center3_binary` at 3 seeds (`--split-seed` fixed):

| | mean external AP@0.5 | std |
| --- | --- | --- |
| DFR (pooled) | 0.1399 | 0.0088 |
| DFR (clean bucket) | 0.1016 | 0.0061 |

No lift on either read — pooled is *below* baseline (0.1916), clean-bucket
delta (−0.0097) doesn't clear its floor (0.0243) either. As anticipated
going in: DFR/TFA only unfreezes the box predictor, leaving the backbone
frozen — it structurally cannot fix a *feature-representation* gap (§5k's
mechanism 2, near-zero training exposure to Center-3's visual format), only
a decision-boundary one. Worth noting separately: the balanced subset ended
up tiny (40 images — capped by Centers 4/5/6's own single-digit-patient
sizes), which may also have limited how much the fine-tuning step could
move the decision boundary regardless of mechanism.

---

## 5n. EndoStyle-style color/style augmentation — no effect

New `AUGMENT_OPS` entry `style_jitter` (`cadsel/data.py`): independent
per-channel gain + bias plus an HSV hue rotation, distinct from the existing
`color` op's single global brightness/contrast/saturation triplet — targets
the site-fingerprint signal directly (§4: center decodable at 0.81 vs 0.167
chance; Center-3 is 45-48% darker *per channel* specifically, a signature a
same-magnitude-per-channel jitter can't disrupt). A named-but-never-tried
item from the original shortcut-learning review. Tested the same way, 3
seeds:

| | mean external AP@0.5 | std |
| --- | --- | --- |
| style_jitter (pooled) | 0.1873 | 0.0664 |
| style_jitter (clean bucket) | 0.1421 | 0.0347 |

Indistinguishable from baseline on both reads (pooled: −0.0044, well within
noise; clean-bucket: +0.0308, directionally positive but doesn't clear its
floor of 0.0419 — one seed's spike to 0.2780 drives most of the apparent
lift). No confirmed effect this round.

---

## 5o. DINOv3 as the detector backbone — initial 3-seed read (see §5s for the settled verdict)

§5e found DINOv3's frozen-embedding classifier advantage over ResNet-50
partly survives lesion removal on WLE (genuine lesion-reading, not just a
better shortcut-reader) — that was only ever tested as a classifier probe.
This wires the same frozen DINOv3 representation in as an actual detector
backbone, to see whether the signal transfers to detection and helps close
the reproduction gap (§3).

**New model type `fasterrcnn_dinov3`** (`cadsel/models.py::Dinov3Backbone`,
`build_dinov3_detection_model`; registry in `cadsel/config.py::
DINOV3_MODEL_REGISTRY`). DINOv3 is a ViT (patch tokens, not a conv
hierarchy), so it can't use the existing `MODEL_REGISTRY` pattern (a
torchvision-prebuilt-constructor call) — this hand-builds a `FasterRCNN`
from a single-scale custom backbone (torchvision's own documented
non-FPN-backbone pattern: `AnchorGenerator` + `MultiScaleRoIAlign` over one
feature map), following the same separate-registry precedent FCOS already
established. The backbone is **frozen by default**
(`--dinov3-trainable-backbone` to override) — only the RPN and ROI heads
train, deliberately isolating "does the pretrained representation itself
help" from "can a full fine-tune find something better." Only
`--input-mode torchvision` is supported: DINOv3 was pretrained with
ImageNet normalisation (torchvision's own `GeneralizedRCNNTransform`
default), and `reference` mode would feed it CADSEL_MEAN/STD-normalised
pixels instead, silently degrading the representation — `cadsel/train.py`
now raises rather than allowing that combination silently.

Tested on WLE multiclass, 3 seeds, against the standard
`fasterrcnn_mobilenet_320` baseline (fully fine-tuned):

| | mean AP@0.5 | std | delta | floor | clears? |
| --- | --- | --- | --- | --- | --- |
| MobileNet_320 baseline (fully fine-tuned) | 0.4200 | 0.0228 | — | — | — |
| **DINOv3 (frozen backbone)** | **0.4326** | 0.0369 | +0.0126 | 0.0433 | No |

Doesn't clear the noise floor — but **all three individual seeds landed at
or above their corresponding baseline seed** (0.3809 vs 0.3881, 0.4530 vs
0.4323, 0.4640 vs 0.4397), and this is a *frozen* backbone: only the RPN and
ROI heads are trainable, a small fraction of the parameters the fully
fine-tuned MobileNet gets to use. A frozen ViT representation being
competitive with — and directionally ahead of — a fully fine-tuned CNN is
notable even without statistical confirmation yet. Worth a follow-up with
more seeds, or with `--dinov3-trainable-backbone` to see whether letting the
representation adapt pushes this from "competitive" to "confirmed win" (at
the cost of losing the clean "is the pretrained representation itself
useful" isolation). Note this is still far from the paper's 0.8641 on this
cell — nothing tested this session meaningfully closes that gap, which §3
already attributes to suspected evaluation leakage rather than model
capability.

---

## 5p. RadImageNet backbone for EUS, isolated — confirmed negative

§5d/§5i built and used a RadImageNet-pretrained ResNet50 backbone for EUS
inside the multimodal fusion study, but never tested it as a standalone
single-modality EUS detector against the vanilla COCO-pretrained backbone —
so it was unclear whether RadImageNet's ultrasound-domain pretraining
actually helps EUS detection on its own, separate from the fusion question.

Tested on EUS multiclass, 3 seeds, `fasterrcnn_resnet` (the ResNet50-FPN
family RadImageNet's converted weights are architecture-specific to):

| | mean AP@0.5 | std |
| --- | --- | --- |
| vanilla COCO-pretrained (control) | 0.3559 | 0.0484 |
| RadImageNet backbone | **0.2844** | **0.0056** |

Delta −0.0716 clears the noise floor (0.0487) — a genuine, confirmed
**negative** result. Notably, the RadImageNet arm's seed-to-seed variance is
tiny (0.0056 vs the control's 0.0484) — it's not just worse on average, it's
worse consistently, every seed landing in a narrow band below every control
seed. RadImageNet's chest/abdominal ultrasound pretraining domain
apparently doesn't transfer usefully to endoscopic ultrasound's very
different imaging characteristics (narrow-field, different tissue/probe
geometry) — worth remembering before reaching for a "domain-appropriate"
pretrained backbone again without checking it's actually the same domain.

---

## 5q. EMA stacked with multiscale, and center-bucketed oversampling on the full pool — neither helps

Two more candidate AP-boosting levers, tested in isolation from each other
this time (unlike the diluted 5-lever `tier1_all` combo in §5c/item 6).

**EMA + multiscale, isolated.** EMA alone is real but weak (+0.013 over
baseline, doesn't clear its floor); multiscale alone is the one confirmed
overall-AP win in this project (+0.028, clears). Stacking them (3 seeds,
WLE multiclass):

| | mean AP@0.5 | std |
| --- | --- | --- |
| plain baseline | 0.4200 | 0.0228 |
| multiscale-alone | 0.4447 | 0.0258 |
| EMA-alone | 0.4297 | 0.0176 |
| **EMA + multiscale combo** | **0.4363** | 0.0287 |

The combo's mean is actually *below* multiscale-alone (delta −0.0084,
within noise either way) — consistent with §5c's finding that stacking
levers here tends to dilute rather than add. **No reason to adopt this
combo over multiscale alone.**

**Center-bucketed oversampling, on the standard full-pool split.** §5l
confirmed a real win from `CenterRepeatFactorSampler` in the LOCO-Center-3
setting (clean-bucket external AP +0.035, clears its floor) — untested
until now on the normal all-6-centers training run, where Center-1's
dominance is milder (≈81% vs. 94% in the Center-3-LOCO pool) and there's no
unseen-format problem to correct for. Tested on WLE multiclass, 3 seeds:

| | mean AP@0.5 | std |
| --- | --- | --- |
| plain baseline (`ms_baseline_s*`) | 0.4200 | 0.0228 |
| center-RFS (full pool) | **0.3848** | 0.0199 |

Delta −0.0352 clears the noise floor (0.0302) — a confirmed **negative**
result here, the opposite direction from §5l's LOCO finding. **The lever's
value is specific to the severe-imbalance LOCO setting, not a general
improvement** — oversampling minority centers that are 0.3–4% of the full
pool (Centers 4/5/6 especially) means repeatedly re-showing the model a
very small, non-diverse image set each epoch, which apparently costs more
in overfitting/reduced majority-center exposure than it gains here. Don't
apply this lever outside the LOCO context without re-testing.

---

## 5r. Resolved: why "dirty" scores higher than "clean" — a lesion-size confound

Item 12 flagged this as unexplained: across all 12 checkpoints tested this
session, the "dirty" (≥10% black-overlap) bucket consistently scored
*higher* AP than "clean," the opposite of a labeling-artifact explanation.
The leading hypothesis was that black-overlap fraction might be an
accidental proxy for capture format (the "wide" layout has more letterbox
by construction). New `experiments/src/23_dirty_clean_mystery.py` tested
this directly against Center-3's 368 images and the `loco_holdout_center3_
binary` checkpoint.

**The capture-format hypothesis is refuted.** "Wide" layout images are
actually *under*-represented in the dirty bucket (3/18, 16.7%) versus clean
(15/18, 83.3%) — the opposite of what would explain dirty scoring higher.
Layout mode's own AP difference is negligible (standard 0.1397, wide
0.1481, n=345/18).

**Lesion size is the real driver.** Mean `box_area_frac` (largest box's
area fraction) is 0.1115 in the clean bucket vs. **0.1865 in the dirty
bucket — 67% larger on average.** A larger box is both more likely to touch
the black margin by simple geometry (more edge pixels at a fixed
frame position) *and* easier to detect — this project has already
independently confirmed lesion size is a strong AP driver everywhere else
(Exp 9's tertile evaluation, §5g's size-RFS lever). The "dirty > clean"
pattern is a lesion-size confound riding on the black-overlap-fraction
bucketing, not evidence about label quality or capture format. (Subtype
distribution shifts the same direction — dirty over-represents Leiomyoma
and Lipoma, both larger/more prominent submucosal presentations than
NET_G1 — consistent with, not independent of, the size explanation.)

This doesn't change §5k's headline (the Center-3 collapse isn't primarily a
labeling artifact — the *clean* bucket's own AP, 0.0953, is still far below
Center-1's no-drop external benchmark), but it does resolve why the
clean/dirty split itself doesn't look like a clean labeling-quality signal:
it's mostly re-deriving the already-known size effect.

---

## 5s. DINOv3 as detector backbone, pushed further — settles as unconfirmed, and one clean falsification

§5o found DINOv3 (frozen backbone, only RPN/ROI heads trainable) competitive
with the fully fine-tuned MobileNet_320 baseline on WLE multiclass at 3
seeds (+0.0126, didn't clear its floor). Four follow-ups: more seeds, an
unfrozen backbone, the binary task, and a falsification check on EUS (where
§5e found DINOv3's classifier-probe edge over ResNet-50 is pure
shortcut-reading, predicting the backbone should *not* help there).

**More seeds settle it as unconfirmed, trending toward zero.** WLE
multiclass, 5 seeds: mean 0.4266 ± 0.0330 vs. baseline 0.4200 ± 0.0228 —
delta shrinks from +0.0126 (3 seeds) to **+0.0066** (5 seeds), further from
clearing its floor (0.0401), not closer. Same pattern as §5g's binary
size-RFS lever: more data pulling an unconfirmed positive toward "no
effect," which is what more seeds are supposed to do.

**Unfreezing the backbone — confirmed worse, not better.**
`--dinov3-trainable-backbone`, 3 seeds, same recipe: mean 0.3862 ± 0.0233.
Clears the floor as *worse* than both the frozen version (Δ −0.0464, floor
0.0436) and the baseline (Δ −0.0338, floor 0.0326). On this small dataset
(~2249 training images), full fine-tuning destabilizes the ViT rather than
improving on the frozen representation — keep it frozen.

**WLE binary, frozen — roughly comparable, not a win.** 3 seeds: mean
0.6527 ± 0.0293, close to the single-seed plain baseline (0.6458) and not
distinguishable from the multiscale-alone lever's own 3-seed mean
(0.6711 ± 0.0282; Δ −0.0184, floor 0.0407, doesn't clear).

**EUS multiclass, frozen — a clean, confirmed falsification.** 3 seeds:
mean **0.2217 ± 0.0166** vs. the EUS plain-augment baseline's 0.3838 ± 0.0480
— Δ **−0.1621**, clears the floor by more than 3×. This is the sharpest
result in this section: forced to detect lesions using only the frozen
EUS-side representation, DINOv3 fails badly, exactly as predicted by §5e's
finding that its embedding edge on EUS was never genuine lesion-reading.
The classifier-probe theory and the detector-level test agree.

**Net verdict on DINOv3 as a backbone**: not adopted. The one surviving
open thread (frozen, WLE, multiclass) is weaker with more data, not
stronger, and every other configuration tested (trainable, WLE binary, EUS)
is either confirmed-worse or merely comparable.

---

## 5t. Group DRO, detector-level, group=center — the worst of the four group-aware methods tried

New `GroupDROLossState` (`cadsel/models.py`, Sagawa et al. 2020): an online
exponentiated-gradient adversarial reweighting over groups (here,
acquisition center), applied to the *whole* per-batch loss rather than
reweighting classes within one loss computation (contrast Seesaw) or
resampling which images the optimiser sees (contrast
`CenterRepeatFactorSampler`/DFR). Requires group-homogeneous batches — new
`CenterBatchSampler` (`cadsel/data.py`, `ModalityBatchSampler`'s pattern,
grouped by `Sample.center`) — and a per-image `center` field threaded
through the target dict so `cadsel/engine.py::train_one_epoch` can declare
the active group before each forward pass (`--loss-fn group_dro
--groupdro-step-size 0.01`).

Tested on the LOCO-Center-3 setting (3 seeds), the setting where the actual
problem (§5f/§5k) lives:

| | mean external AP@0.5 | std |
| --- | --- | --- |
| plain LOCO baseline | 0.1916 | 0.0567 |
| **Group DRO** | **0.0848** | 0.0373 |

Δ −0.1068 clears the floor (0.0679) decisively — **the worst-performing of
all four group-aware interventions tried this session** (center-RFS
confirmed a real clean-bucket win in §5l; DFR was neutral in §5m; Group DRO
is the only one that actively hurts). A plausible mechanism: Centers 4/5/6
have single-digit patient counts, so a batch from one of them is an
extremely noisy estimate of "that group's loss" — the online adversarial
weight update can swing wildly on one bad batch from a tiny group, actively
destabilizing training rather than correcting for imbalance. The
theoretically most sophisticated of the four methods tried turned out to be
the least suited to this dataset's specific imbalance profile (a few
centers with only a handful of patients each, not a moderate multi-way
split Group DRO's own literature is usually validated on).

---

## 5u. JTT diagnostic — a mixed result, doesn't clearly justify the full method

JTT (Liu et al. 2021) needs no group labels: identify which *training
images* a vanilla-ERM model gets wrong, then upweight them in a second
training run. Its appeal over Group DRO/DFR/`CenterRepeatFactorSampler`
(all group=center, mixed-to-negative results above) is that it could
surface an unnamed interaction nobody hypothesized. Per a scoping pass done
before writing any code, only the identification half was built as a
diagnostic first — new `experiments/src/24_jtt_diagnostic.py`: score every
training image's loss from a plain-ERM checkpoint (`ms_baseline_s42__WLE__
multiclass`) at batch-size-1 (so the batch total *is* that image's loss, no
per-ROI ambiguity), flag the top 20% as "hard," and cross-tabulate against
center/subtype/lesion-size — the three axes already directly probed this
session.

| | overall pool | hard set (top 20%, n=450) |
| --- | --- | --- |
| Center-3 | 13.2% | **19.8%** (1.5x enriched) |
| NonNeoplasm subtype | 10.1% | **15.1%** (1.5x enriched) |
| Lesion size (mean `box_area_frac`) | 0.161 (easy) | **0.122** (hard, smaller) |

The hard set is **partially** explained by axes already tried (Center-3
enrichment, small-lesion enrichment — both consistent with §5f/§5l/§5g) —
not a clean rediscovery, but not clearly orthogonal either. The one
modestly novel signal is NonNeoplasm subtype enrichment, never directly
probed by a group-aware method this session, though the effect size is
modest (1.5x, not a dramatic multi-fold enrichment). Given every group-aware
method actually tried this session (DFR, Group DRO, center-RFS on the full
pool) has been neutral-to-negative, and this diagnostic doesn't show a
strongly novel hard group, **the full two-stage JTT retrain isn't clearly
justified by this result** — logged as a diagnostic finding, not escalated
to a second training run.

---

## 5v. Cross-model WBF ensembling — confirmed negative, and consistently so

Weighted Boxes Fusion (already implemented in `cadsel/tta.py` for
multi-scale TTA within one model) is a standard way to fuse predictions
from *different* trained checkpoints at inference time — free signal from
models already trained, no retraining needed. New
`cadsel/tta.py::predict_at_native_scale`/`predict_ensemble_wbf` bring every
checkpoint's boxes into a common native-pixel coordinate space first (since
different checkpoints use different `img_size`/`input_mode` conventions —
`"reference"` mode's square resize is inverted with independent x/y scale
factors; `"torchvision"` mode's boxes come back already in native space, a
guarantee `predict_multiscale_wbf` already relied on). Verified first that
every candidate checkpoint shares the byte-identical seed-42 validation
split, so the comparison is uncontaminated.

New `experiments/src/25_cross_model_ensemble.py`, WLE multiclass, tested 3
subsets of the available checkpoint zoo:

| Ensemble | Models | AP@0.5 | vs. best single model |
| --- | --- | --- | --- |
| All 5 | MobileNet, ResNet50-FPN, ResNet50-FPN-v2, size-RFS+multiscale MobileNet, DINOv3 (0.5x weight) | 0.3207 | **−0.1144** |
| Top 3 by solo AP | size-RFS+multiscale, MobileNet, DINOv3 | 0.4066 | **−0.0285** |
| Top 2 by solo AP | size-RFS+multiscale, DINOv3 | 0.4071 | **−0.0280** |

**Confirmed negative, and the pattern is consistent, not noisy**: every
subset tested underperforms simply using the single best checkpoint
(`size_rfs_ms_s42__WLE__multiclass`, solo AP@0.5 0.4351) alone, and adding
more (weaker) models makes it worse, not better. WBF ensembling needs
genuinely decorrelated errors across models to help; these checkpoints are
all trained on the identical small dataset and split, and evidently share
enough correlated failure modes (the same shortcut-learning tendencies and
annotation quirks this project has spent most of its effort characterizing)
that averaging in a weaker model's boxes mostly adds noise around the
strongest model's already-good predictions rather than correcting them.
**Don't use cross-model WBF ensembling on this checkpoint zoo** — the
single best model beats every combination tried.

---

## 5w. Research notes: medical-pretrained backbones, ensembling, GAN augmentation

A literature/Hugging Face search for further avenues, given everything
above:

- **GastroNet-5M** (`tgwboers/GastroNet-5M_Pretrained_Weights`) — an
  endoscopy-specific SSL-pretrained ResNet-50/ViT-B, trained on 4.82M
  images from ~500K GI procedures. Unlike RadImageNet (general/abdominal
  ultrasound, confirmed −0.072 AP on EUS, §5p), this is genuinely
  in-domain for WLE and architecture-matched to the existing ResNet50-FPN
  backbone via `--backbone-weights`. **Not yet tried**: the weights are
  gated behind a commercial platform ("Theta Vision Cortex"), not a
  standard HF license-click gate like DINOv3's — needs external account
  setup outside what this session can do autonomously. Worth pursuing if
  access is arranged; the closest EUS-side candidate (URFM, general
  ultrasound) is a repeat of RadImageNet's already-failed bet and isn't
  recommended.
- **Cross-model WBF ensembling** — tested directly, see §5v: confirmed
  negative on this project's checkpoint zoo. (Same-architecture multi-seed
  ensembling and SVM probability-averaging across the ResNet-50/DINOv3
  embedding classifiers were flagged as lower-cost variants worth a quick
  look, but weren't tested this round.)
- **GAN/diffusion augmentation beyond the existing lesion-GAN** — a
  diffusion+ControlNet upgrade path exists in the literature (patch
  synthesis + composite-with-automatic-box, the same recipe the existing
  64×64 DCGAN already uses, just with a stronger pretrained generator).
  **Not recommended right now**: it's the same fundamental strategy the
  project's own lesion-GAN already returned a marginal/noise-adjacent
  result from (§5h), with a heavier engineering lift (Stable Diffusion
  fine-tuning + LoRA + ControlNet) and thin medical-detection-specific
  evidence in the literature (most positive GAN-augmentation evidence is
  classification-accuracy gains in other domains, not object-detection AP
  gains in a similarly small-N medical setting).

---

## 5x. Auto-anchors stacked on size-RFS + multiscale — confirmed negative

Item 6/11's last untested piece: does IoU-clustered RPN anchor sizing add
anything on top of the two confirmed detector levers (multiscale +
`SizeRepeatFactorSampler`), tested cleanly this time as a 3-way stack with no
other confounds (unlike the diluted 5-lever `tier1_all` combo that first
raised the question). 3 seeds, WLE multiclass,
`--input-mode torchvision --multiscale-sizes 256,320,384
--size-rfs-threshold 0.2 --auto-anchors`:

| seed | overall AP@0.5 | tertile-1 (small) AP@0.5 |
|---|---|---|
| 42  | 0.3879 | 0.2538 |
| 123 | 0.4320 | 0.2218 |
| 7   | 0.4310 | 0.4098 |
| **mean +/- std** | **0.4170 +/- 0.0206** | **0.2951 +/- 0.0821** |

Against the already-confirmed `size_rfs_ms` arm (tertile-1 0.3381 +/- 0.0357):
delta -0.0430, noise floor ~0.0895 (driven by auto-anchors' own high
seed-to-seed spread -- seed 7 alone nearly doubles seeds 42/123's tertile-1
score). **Within noise floor -- not confirmed**, and directionally negative
if anything. Auto-anchors adds nothing on top of size-RFS + multiscale, and
if anything makes the small-lesion metric noisier across seeds. Consistent
with the original single-run Tier-1 ablation (§5c) where anchors/IoU/RFS
individually looked flat-to-negative -- this closes out that open question
(item 11) rather than reopening it. **Verdict: do not adopt `--auto-anchors`
on top of the existing size_rfs_ms recipe.**

## 5y. Size-RFS threshold sweep -- 0.2 (the adopted value) is at the peak

Single-seed (42) screen of `--size-rfs-threshold` at 0.1/0.3/0.4, otherwise
identical to the confirmed `size_rfs_ms` recipe (multiscale 256/320/384, WLE
multiclass), following this project's established coarse-screen-then-confirm
pattern (same as the Seesaw p-sweep, §8 item 7):

| threshold | overall AP@0.5 | tertile-1 (small) AP@0.5 |
|---|---|---|
| 0.1 | 0.4215 | 0.2561 |
| **0.2 (adopted)** | 0.4131 | **0.3806** |
| 0.3 | 0.3998 | 0.3733 |
| 0.4 | 0.4017 | 0.2724 |

Single seed, so no formal noise-floor claim -- but the shape is clean and
non-monotonic: both extremes (0.1, 0.4) score ~0.26-0.27 on the small-lesion
tertile, while 0.2 and 0.3 sit together around ~0.37-0.38, a real gap not
plausibly explained by seed noise alone (it mirrors a sensible bias/variance
story: too low a threshold barely oversamples small lesions, too high one
oversamples so aggressively it distorts the class/size mix). **0.2 (the
value already adopted from §5g) is at or right at the peak** -- 0.3 is close
enough (delta 0.007) that it isn't a clearly better alternative, and neither
0.1 nor 0.4 is worth pursuing further. No change to the adopted recipe.

---

## 6. Implementation traps (these cost real time)

**Windows DataLoader IPC.** Ship **uint8** through workers, never float32. A
640×640 float32 image is 4.9 MB and exhausts Windows shared memory
(`OSError 1455`) and Kaggle's small `/dev/shm`. Normalisation happens on GPU via
`make_normalizer`.

**`make_normalizer` expects uint8** and does the `/255` itself. Passing pre-scaled
floats divides by 255 twice and silently produces near-zero images — Exp 8 first
reported all-zero detections from AP-0.38 models because of this.

**Validation loader workers.** Give the val loader ≤2 non-persistent workers. On
Windows each worker re-imports torch (~580 MB); train+val at 6 each is ~7 GB per
run and will OOM.

**Orphaned workers.** Windows doesn't reap DataLoader workers when the parent
dies. Clean up with:
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*cadsel*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

**Empty targets are meaningful.** They mark hard negatives; torchvision's RPN/ROI
heads treat every anchor as background. `engine._target_is_usable` keeps them and
drops only degenerate boxes. Do **not** filter on `len(boxes) > 0`.

**`patient_key` excludes the centre** deliberately — pseudonyms are globally
unique, and including the centre made the figshare and Hugging Face layouts split
differently, breaking comparability.

**Long runs must survive teardown.** Use detached `Start-Process`, not the tool's
background shell — that gets killed on session teardown. `_py2nb.py` flushes the
notebook after every cell, and Exps 3, 7, 10, 11 resume from their results JSON.

**Network.** The sandbox blocks `download-r2.pytorch.org` (WinError 10013); pip
also restarts its 2.7 GB torch download from zero each retry. The Hugging Face
mirror must be fetched with **`git clone` + `git lfs pull`**, not
`snapshot_download` — 9,824 individual file requests trip HF rate limits (429).
And verify by **bytes, not filenames**: LFS pointer files are still named `.tiff`.

---

## 7. Known deviations from the paper / reference repo

- **4 detection classes, not 5.** Released labels collapse NET_G1/NET_G2 into
  class id 0. NET grade is kept as directory metadata and used to stratify splits.
- **Training resolution is not stated in the paper.** The reference repo defaults
  to 640 for all models; here the detector's internal transform is pinned to
  `--img-size` (640, or 320 for the `_320` variant).
- **Normalisation happens once.** The reference normalises with CAD-SEL stats and
  then lets torchvision normalise again; treated here as unintended.
- **AP@0.5** is VOC all-point interpolation over *all* detections. The reference
  scores only detections above confidence 0.5; that variant is reported alongside
  as `AP@0.5_conf`.
- Batch sizes 4/4/8/16 to fit 8 GB (reference used 16 throughout).

---

## 8. Open items

0. **Multimodal grid — done.** Stage 1 (8 joint-detector runs) and Stage 2
   (5-fold patient-level fusion) have both run to completion; see §5b/§5i for
   results and the correction to how "does fusion help" should be measured
   (restrict to the 421 paired patients, not all 641).
1. **The 8 ResNet runs — deliberately not being finished** (see §3). 2/8 exist
   and are already worse than MobileNet on the same cell; the remaining 6
   were a Table 4/5-completeness task with no bearing on this project's
   actual findings, and were dropped in favor of the Center-3 LOCO
   follow-ups (§5j-§5n) instead.
2. **Exp 12–16 done** (§5a): none of the three classifier-level mitigations
   (adversarial discriminator, bbox blur, channel attention) cleared their own
   noise floor, even `se_all_adv` at 7 seeds / full convergence. **DFR, Group
   DRO, LfF remain deferred** (Exp 17+, unscheduled) as the next things to try
   if this line of work continues.
3. **Seed variance is already visible**: the same config scored 0.3809 and 0.4000
   on two runs. Any single-run comparison below ~0.02 AP is noise. This is why
   Exp 11 exists.
4. **EUS generative augmentation — tried, see §5h.** GAN copy-paste gives the
   lowest hallucination ratio of the three EUS arms tested (plain/real-cp/
   gan-cp), though the margin over plain control is close to the seed noise.
   Also found: real copy-paste's detector-level backfire on EUS, at 3 seeds,
   doesn't replicate Exp 8's original n=1 "ambiguous" reading — it's flat
   against control. Bbox-blur backfiring on EUS binary (Exp 13, classifier
   level) stands unchanged.
5. **External validation — done, see §5f.** Classifier-level LOCO across all
   6 centers, plus a scoped detector-level confirmation on the two
   fold/task combinations with real signal (WLE binary, Center-1 and
   Center-3 held out). Headline: the detector-level result *inverts* the
   classifier-level ranking (Center-1 generalizes cleanly at the detector
   level despite a classifier-level red flag; Center-3 collapses at the
   detector level despite a milder classifier-level drop) — a classifier
   proxy for detector generalization is not reliable on this dataset.
6. **A second confirmed lever: size-aware oversampling, see §5g.**
   `SizeRepeatFactorSampler` + multiscale lifts small-lesion (tertile-1) AP
   by +0.067 on multiclass (clears the noise floor), overall AP unchanged —
   the first lever in this project to move the small-lesion bucket
   specifically, which is also the failure mode endoscopists actually care
   about. Binary is directionally positive but doesn't yet clear its floor.
   **Tier-1/Tier-3 architecture levers: multiscale training is the other
   confirmed win.** 3-seed grids done (§5c): `multiscale only` (paired mean
   +0.028 AP, sd 0.009, 3/3 seeds positive, clears a classical significance
   threshold). `EMA only` is real but weaker (+0.013, doesn't clear). The full
   5-lever combo undershoots either lever alone — anchors/IoU/RFS
   individually looked flat-to-negative in the (single-run) ablation, so they
   may be diluting EMA's contribution when stacked; not re-tested at more
   seeds. **FCOS looked promising at one seed (+0.031) but resolved to noise
   at three** (paired mean +0.005, sd 0.024, only 2/3 positive) — not a win,
   despite looking like one initially; a reminder of how wide this dataset's
   noise floor is. **Adopt `--input-mode torchvision --multiscale-sizes ...`
   on its own** — the one thing out of everything tested this session that's
   actually earned trust.
7. **Tier 2: Seesaw p-sweep — done, doesn't help.** All four values
   (p=0.2/0.4/0.6/0.8) score 0.334-0.347 AP on EUS, essentially flat and
   below the plain-augment control (0.370) — confirms §5c's hypothesis with
   data rather than leaving it as a plan. TFA-style rare-class fine-tuning —
   done at multiple seeds, see §5h: a real but small, base-checkpoint-
   dependent effect on Leiomyoma, not a reliable fix as configured. Custom
   backbone-weight loading (`--backbone-weights`) has a real, verified
   weight source for EUS — see item 9.
8. **Tier 3: FCOS confirmed away (see item 6); the lesion GAN now has real
   training results, see §5h.** Both needed a real bug fix during
   integration (FCOS's stable LR, the lesion GAN's DataLoader-fork/CUDA
   crash). GAN copy-paste gives the lowest EUS hallucination ratio of the
   arms tested, though the margin is close to seed noise. Cross-modal domain
   adaptation and paired-instance fusion were scoped out, not attempted (see
   §5c for why the latter is actually the wrong design for this dataset, not
   just lower priority).
9. **Separate per-modality backbones + RadImageNet for EUS — now has real
   training results** (§5b/§5d/§5i). `wle_resnet_backbone` and
   `eus_radimagenet_backbone` are the detectors Stage 2 fusion is built on;
   §5i's subgroup analysis and symmetric-fusion comparison both run against
   them for real.
10. **SVM on frozen embeddings — checked against the counterfactual
   protocol, see §5e.** DINOv3's edge over ResNet-50 is partly genuine on
   WLE (survives at 38-52% of its full-image size once the lesion is
   removed) but is essentially pure shortcut-reading on EUS (the edge
   survives or grows under the same ablation) — resolves the question this
   item originally flagged.
11. **Still genuinely open**: GAN copy-paste's hallucination-ratio
    improvement (§5h, real direction, margin close to noise); LfF (still
    deferred, unscheduled — DFR, Group DRO, and JTT's identification half
    were all tried, see §5m/§5t/§5u; DFR neutral, Group DRO confirmed
    negative, JTT mixed); root-causing the Center-3/Center-2 raw-canvas
    label-misalignment (§5k found it, ruled it out as the AP-collapse
    driver, but never explained *why* it exists); and the 6 still-missing
    ResNet runs (item 1, deliberately not pursued). **Settled, not open**:
    binary-task size-RFS (§5g, 5 seeds, delta shrank to +0.0128 — not
    confirmed); DINOv3 as a detector backbone (§5o/§5s, every
    configuration tested is either confirmed-worse or merely comparable to
    baseline — not a lever worth adopting); and anchors/IoU/RFS re-tested as
    a clean 3-seed combo stack on top of size-RFS+multiscale (§5x, confirmed
    negative — auto-anchors adds nothing and, if anything, adds noise).
12. **Resolved, see §5r**: the "dirty" (>10% black-overlap) bucket scoring
    *higher* AP than "clean" is a lesion-size confound (dirty-bucket boxes
    are 67% larger on average) — the capture-format hypothesis (§5k's "wide"
    layout) was tested directly and refuted (wide is under-, not
    over-represented in the dirty bucket).

---

## 9. Commands

```powershell
# dataset check against paper Table 2
.\.venv\Scripts\python.exe -m cadsel.describe

# one run
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 `
    --modality WLE --task multiclass --epochs 100 --early-stop-patience 15

# with mitigations
... --augment flip,color,rotate,scale_crop,copy_paste --hard-negatives 0.3

# the grid, then the tables
.\.venv\Scripts\python.exe -m cadsel.run_all --epochs 100 --early-stop-patience 15
.\.venv\Scripts\python.exe -m cadsel.report
.\.venv\Scripts\python.exe -m cadsel.summarize --filter mobilenet

# audit notebooks (convert + execute; resumable)
.\.venv\Scripts\python.exe experiments\_py2nb.py       # all
.\.venv\Scripts\python.exe experiments\_py2nb.py 10    # one

# ---- multimodal study ----
# always run the pre-flight first: split integrity, normalisation, conditioning
.\.venv\Scripts\python.exe -m cadsel.verify_joint

# the whole grid in dependency order (baselines -> control -> conditioned)
.\.venv\Scripts\python.exe -m cadsel.run_multimodal --epochs 100 --early-stop-patience 15
.\.venv\Scripts\python.exe -m cadsel.run_multimodal --image-level   # + paper-protocol column

# one joint run
.\.venv\Scripts\python.exe -m cadsel.train_joint --conditioning film --task multiclass `
    --hard-negatives 0.3 --output-dir runs_joint

# stage 2: patient-level fusion (5-fold CV, all four ablations)
.\.venv\Scripts\python.exe -m cadsel.train_fusion `
    --checkpoint runs_joint\joint_film__fasterrcnn_mobilenet__multiclass\best_model.pth

# re-measure the per-modality normalisation statistics
.\.venv\Scripts\python.exe -m cadsel.measure_stats

# ---- Tier-1 performance levers (§5c) ----
.\.venv\Scripts\python.exe -m cadsel.measure_anchors     # inspect recommended anchors first

.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality WLE --task multiclass `
    --auto-anchors --rpn-fg-iou-thresh 0.6 --box-fg-iou-thresh 0.5 --box-bg-iou-thresh 0.4 `
    --input-mode torchvision --multiscale-sizes 256,320,384 --rfs-threshold 0.2 --ema

# multi-scale WBF at inference, against an already-trained checkpoint (needs --input-mode torchvision)
.\.venv\Scripts\python.exe -m cadsel.evaluate_tta --checkpoint runs\...\best_model.pth --scales 256,320,384

# extra seeds for Exp 12/14's se_all_adv (Exp 15), then the 100-epoch/early-stop settle (Exp 16)
.\.venv\Scripts\python.exe experiments\src\15_se_all_adv_multiseed.py
.\.venv\Scripts\python.exe experiments\src\16_se_all_adv_100epoch.py

# multi-seed confirmation + per-lever ablation of the Tier-1 combo (10 runs)
.\.venv\Scripts\python.exe -m cadsel.run_all --config-grid experiments\configs\tier1_confirmation_grid.json

# Tier 2: Seesaw Loss, EUS-specific augmentation, two-stage rare-class fine-tuning
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality WLE --task multiclass `
    --loss-fn seesaw --seesaw-p 0.8

.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality EUS --task multiclass `
    --augment flip,color,rotate,speckle_noise,ultrasound_gain

.\.venv\Scripts\python.exe -m cadsel.finetune_rare --checkpoint runs\...\best_model.pth --epochs 10

# follow-up: 3-seed confirmation of EMA/multiscale alone + Seesaw split apart on EUS
.\.venv\Scripts\python.exe -m cadsel.run_all --config-grid experiments\configs\tier1_lever_seeds_and_tier2_split.json
# Seesaw's p hyperparameter sweep (0.2/0.4/0.6 vs the tested 0.8) on EUS
.\.venv\Scripts\python.exe -m cadsel.run_all --config-grid experiments\configs\seesaw_p_sweep.json

# Tier 3: anchor-free FCOS (needs a much lower LR than the Faster R-CNN family)
.\.venv\Scripts\python.exe -m cadsel.train --model fcos_resnet50 --modality WLE --task multiclass `
    --learning-rate 1e-4 --warmup-iters 0

# Tier 3: conditional lesion GAN for EUS, then generated-lesion copy-paste
.\.venv\Scripts\python.exe -m cadsel.train_lesion_gan --modality EUS --task multiclass --epochs 300
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality EUS --task multiclass `
    --augment flip,color,rotate,copy_paste --copy-paste-source generated `
    --lesion-gan-checkpoint runs\lesion_gan_EUS_multiclass.pth

# separate per-modality backbones + RadImageNet for EUS (§5d)
.\.venv\Scripts\python.exe -m cadsel.convert_radimagenet_weights
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_resnet --modality WLE --task multiclass `
    --run-name wle_resnet_backbone
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_resnet --modality EUS --task multiclass `
    --backbone-weights runs\radimagenet_resnet50.pth --run-name eus_radimagenet_backbone
.\.venv\Scripts\python.exe -m cadsel.train_fusion `
    --checkpoint-wle runs\wle_resnet_backbone\best_model.pth `
    --checkpoint-eus runs\eus_radimagenet_backbone\best_model.pth

# SVM on frozen embeddings (ResNet-50 / DINOv3 -- the latter needs HF gate access, see experiments/README.md)
.\.venv\Scripts\python.exe -m experiments.svm_classifier --backbone resnet --modality WLE --task binary
.\.venv\Scripts\python.exe -m experiments.svm_classifier --backbone dinov3 --modality WLE --task binary

# ---- this session's follow-ups (§5e-§5i) ----
# SVM/DINOv3 through the counterfactual protocol (Exp 17)
.\.venv\Scripts\python.exe experiments\src\17_svm_counterfactual.py

# leave-one-center-out: classifier (Exp 18), then scoped detector confirmation
.\.venv\Scripts\python.exe experiments\src\18_loco.py
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality WLE --task binary `
    --centers Center-2,Center-3,Center-4,Center-5,Center-6 --run-name loco_holdout_center1_binary
.\.venv\Scripts\python.exe -m cadsel.evaluate_holdout_centers `
    --checkpoint runs\loco_holdout_center1_binary\best_model.pth --held-out-centers Center-1

# size-aware oversampling (mutually exclusive with --rfs-threshold)
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality WLE --task multiclass `
    --input-mode torchvision --multiscale-sizes 256,320,384 --size-rfs-threshold 0.2
.\.venv\Scripts\python.exe -m cadsel.run_all --config-grid experiments\configs\size_lever_grid.json
.\.venv\Scripts\python.exe experiments\src\19_size_lever_eval.py

# lesion GAN copy-paste vs. real copy-paste, multi-seed reliance check (Exp 20)
.\.venv\Scripts\python.exe -m cadsel.run_all --config-grid experiments\configs\gan_copypaste_grid.json
.\.venv\Scripts\python.exe experiments\src\20_gan_copypaste_reliance.py

# TFA multi-seed (needs --split-seed fixed to the base checkpoint's own split when sweeping --seed)
.\.venv\Scripts\python.exe -m cadsel.finetune_rare --checkpoint runs\...\best_model.pth `
    --split-seed 42 --seed 123
.\.venv\Scripts\python.exe experiments\src\21_tfa_multiseed.py

# fusion subgroup analysis (restricts to the 421 paired patients) + symmetric-attention fusion
.\.venv\Scripts\python.exe -m cadsel.analyze_fusion_subgroups
.\.venv\Scripts\python.exe -m cadsel.train_fusion `
    --checkpoint-wle runs\wle_resnet_backbone\best_model.pth `
    --checkpoint-eus runs\eus_radimagenet_backbone\best_model.pth --fusion symmetric --seed 123

# ---- Center-3 LOCO follow-ups (§5j-§5n) ----
# regenerate the aggregate report (also the fix verification for §5j)
.\.venv\Scripts\python.exe -m cadsel.report

# Center-3 black-overlap-fraction diagnostic against any checkpoint (Exp 22)
.\.venv\Scripts\python.exe experiments\src\22_center3_black_overlap_eval.py `
    --checkpoint runs\loco_holdout_center3_binary\best_model.pth --held-out-centers Center-3

# center-bucketed oversampling (mutually exclusive with --rfs-threshold / --size-rfs-threshold)
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality WLE --task binary `
    --centers Center-1,Center-2,Center-4,Center-5,Center-6 --center-rfs-threshold 0.2

# DFR, detector-level, group=center (needs --centers matching the base checkpoint's own exclusion)
.\.venv\Scripts\python.exe -m cadsel.finetune_rare --checkpoint runs\...\best_model.pth `
    --centers Center-1,Center-2,Center-4,Center-5,Center-6 --balance-by center --epochs 10

# EndoStyle-style color/style augmentation
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality WLE --task binary `
    --centers Center-1,Center-2,Center-4,Center-5,Center-6 --augment flip,color,rotate,style_jitter

# ---- "closer to the paper" experiments (§5o-§5t) ----
# DINOv3 as detector backbone -- frozen by default, needs --input-mode torchvision
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_dinov3 --modality WLE --task multiclass `
    --input-mode torchvision
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_dinov3 --modality WLE --task multiclass `
    --input-mode torchvision --dinov3-trainable-backbone   # confirmed worse than frozen, see §5s

# RadImageNet backbone for EUS, isolated (confirmed negative, see §5p)
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_resnet --modality EUS --task multiclass `
    --backbone-weights runs\radimagenet_resnet50.pth

# Group DRO, detector-level, group=center (confirmed negative on LOCO-Center-3, see §5t)
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality WLE --task binary `
    --centers Center-1,Center-2,Center-4,Center-5,Center-6 --loss-fn group_dro --groupdro-step-size 0.01

# dirty>clean AP mystery diagnostic, re-runnable against any checkpoint (Exp 23)
.\.venv\Scripts\python.exe experiments\src\23_dirty_clean_mystery.py
```

**---- Track A / Track B (§10-11) ----** this session ran on Linux against
`.venv-linux`, not the Windows `.venv` above — bash syntax:
```bash
# Track A: fair-baseline gate, then backbone screen (frozen + unfrozen), best backbone -> lever screen
python -m experiments.finetune_classifier --backbone resnet50 --modality WLE --task binary \
  --freeze none --oversample none --loss ce --seed 42
python -m experiments.finetune_classifier --backbone efficientnet_b0 --modality WLE --task subtype \
  --freeze none --oversample none --loss weighted_ce --seed 42
python experiments/src/26_classifier_leaderboard.py

# Track A: mandatory counterfactual gate on a confirmed win (needs --baseline-run-name for the WLE gate)
python experiments/src/27_classifier_counterfactual.py \
  --run-name efficientnet_b0__WLE__subtype__freeze-none__os-none__loss-weighted_ce__s42 \
  --baseline-run-name efficientnet_b0__WLE__subtype__freeze-none__os-none__loss-ce__s42

# Track B: export a modality/task slice (reuses an existing Faster R-CNN run's split.json)
python experiments/src/28_export_yolo_dataset.py --modality WLE --task multiclass \
  --base-run size_rfs_ms_s42__WLE__multiclass --out experiments/yolo/wle_multiclass_ms_only_s42

# Track B: baseline + a lever (project= MUST be an absolute path -- see §11's env note)
yolo detect train model=runs_yolo/yolo11n.pt data=experiments/yolo/wle_multiclass_ms_only_s42/data.yaml \
  imgsz=640 epochs=100 seed=42 batch=16 patience=20 device=0 \
  project=/home/rohan/CADSEL/runs_yolo name=yolo11n_wle_multiclass_baseline_s42

# Track B: validate the eval bridge against a known-good Faster R-CNN checkpoint BEFORE trusting it on YOLO
python experiments/src/29_yolo_eval_bridge.py --model-family fasterrcnn \
  --weights runs/size_rfs_ms_s42__WLE__multiclass/best_model.pth \
  --modality WLE --task multiclass --base-run size_rfs_ms_s42__WLE__multiclass
# then score a YOLO checkpoint the same way
python experiments/src/29_yolo_eval_bridge.py --model-family yolo \
  --weights runs_yolo/yolo11n_wle_multiclass_baseline_s42/weights/best.pt \
  --modality WLE --task multiclass --base-run size_rfs_ms_s42__WLE__multiclass --conf 0.001

# §12 follow-ups: DINOv3 partial-freeze (new best classifier) and low-LR full fine-tune
python -m experiments.finetune_classifier --backbone dinov3_vits16 --modality WLE --task binary \
  --freeze partial --oversample none --loss ce --seed 42
python -m experiments.finetune_classifier --backbone dinov3_vits16 --modality WLE --task binary \
  --freeze none --oversample none --loss ce --lr 1e-5 --seed 42

# §12.4: TTA (horizontal-flip averaging) eval against a fine-tuned checkpoint
python experiments/src/30_classifier_tta_eval.py --run-name \
  dinov3_vits16__WLE__binary__freeze-partial__os-none__loss-ce__s42

# §11.5: class-oversampled YOLO export (confirmed positive lever) + train
python experiments/src/31_export_yolo_dataset_oversampled.py --modality WLE --task multiclass \
  --base-run size_rfs_ms_s42__WLE__multiclass --threshold 0.2 \
  --out experiments/yolo/wle_multiclass_ms_only_s42_oversample
yolo detect train model=runs_yolo/yolo11n.pt \
  data=experiments/yolo/wle_multiclass_ms_only_s42_oversample/data.yaml \
  imgsz=640 epochs=100 seed=42 batch=16 patience=20 device=0 \
  project=/home/rohan/CADSEL/runs_yolo name=yolo11n_wle_multiclass_oversample_s42
```

**Always launch long runs detached**, otherwise they die with the session:
```powershell
Start-Process -FilePath 'd:\CADSEL\.venv\Scripts\python.exe' `
  -ArgumentList '-u','-m','cadsel.run_all','--models','fasterrcnn_resnet','--epochs','100' `
  -WorkingDirectory 'd:\CADSEL' -RedirectStandardOutput 'runs\x.log' `
  -RedirectStandardError 'runs\x.err' -WindowStyle Hidden
```
## 10. Track A — pure classification-accuracy campaign

Everything above used classification only as a shortcut-learning diagnostic
(§5e's frozen-embedding SVM) or as a component of the multimodal fusion
classifier (§5b/§5i) — never pushed for accuracy as an end in itself, and
never with a *trainable* backbone (only frozen embeddings had been tried).
This section builds that infrastructure and pushes single-modality lesion
classification (binary NET/non-NET, and 5-way subtype) as far as a single
8 GB GPU reasonably allows, holding it to the same discipline as the rest of
this project: a fair-baseline gate before trusting the new pipeline, single-
lever screens before combining, ≥3 seeds + a noise-floor check before calling
anything confirmed, and — because this project's own §5e found DINOv3's
apparent classifier edge on EUS is ~100% shortcut-reading — a mandatory
counterfactual check before any win is recorded as genuine.

**New infrastructure**: `cadsel/classification_data.py`
(`CadSelClassificationDataset`, `ClassificationSizeRepeatFactorSampler`,
`make_classification_normalizer`) and `experiments/finetune_classifier.py`
(4 backbones × `{none, partial, head}` freeze × `{ce, weighted_ce,
seesaw_ce}` loss × size-aware oversampling × background-swap augmentation,
5-fold patient-grouped CV, mean±std across seeds). Every run also writes
per-fold loss/LR curves, a confusion matrix, and ROC/PR curves
(`experiments/results/classifier_runs/<run>/fold*_{loss_curve,lr_curve,
confusion_matrix,roc_pr_curve}.png`) plus ROC-AUC/PR-AUC in its JSON —
`sklearn.metrics` is used here (unusually for this project, which avoids
scikit-learn for its decodability probes, see `experiments/probes.py`) since
these are standard report metrics, not a probe, and scikit-learn is already
a dependency (`experiments/svm_classifier.py`). Leaderboard:
`experiments/src/26_classifier_leaderboard.py`. Counterfactual gate:
`experiments/src/27_classifier_counterfactual.py`.

### 10.1 Fair-baseline gate — PASS

A fine-tuned ResNet-50 (`--freeze none --loss ce`, no oversampling) must
clearly beat `experiments/svm_classifier.py`'s frozen-embedding SVM numbers,
or the new pipeline has a bug (LR, undertraining, a data leak). It does,
decisively:

| modality/task | frozen-SVM (existing) | fine-tuned ResNet-50 (new) |
|---|---|---|
| WLE binary | 0.6447 | **0.7616 ± 0.0234** |
| WLE subtype | 0.2995 | **0.4398 ± 0.0457** |

### 10.2 Backbone screen

32 runs (4 backbones × {WLE, EUS} × {binary, subtype} × {frozen,
unfrozen}), single seed (42), 20 epochs / early-stop patience 8:

| modality | task | backbone | frozen (head) | unfrozen (none) |
|---|---|---|---|---|
| WLE | binary | resnet50 | 0.6669 | 0.7668 |
| WLE | binary | convnext_tiny | 0.6916 | 0.7607 |
| WLE | binary | efficientnet_b0 | 0.6724 | **0.7711** |
| WLE | binary | dinov3_vits16 | **0.7072** | 0.5711 |
| WLE | subtype | resnet50 | 0.2928 | 0.4164 |
| WLE | subtype | convnext_tiny | 0.3220 | 0.4327 |
| WLE | subtype | efficientnet_b0 | 0.3069 | **0.4348** |
| WLE | subtype | dinov3_vits16 | 0.3115 | 0.3007 |
| EUS | binary | resnet50 | 0.5279 | **0.6925** |
| EUS | binary | convnext_tiny | 0.5828 | 0.6918 |
| EUS | binary | efficientnet_b0 | 0.5761 | 0.6836 |
| EUS | binary | dinov3_vits16 | 0.5308 | 0.5000 |
| EUS | subtype | resnet50 | 0.2000 | 0.3595 |
| EUS | subtype | convnext_tiny | 0.2415 | 0.3295 |
| EUS | subtype | efficientnet_b0 | 0.2327 | **0.3602** |
| EUS | subtype | dinov3_vits16 | 0.2055 | 0.2330 |

(chance: 0.500 binary, 0.200 subtype; every unfrozen CNN result above clears
its SVM reference where one exists — resnet50 WLE 0.6447/0.2995, dinov3
WLE 0.7353/0.3813, dinov3 EUS-binary 0.5805, resnet50 EUS-binary 0.5817.)

**efficientnet_b0 is the best unfrozen backbone on WLE** (0.7711 binary,
0.4348 subtype) — carried forward into the lever screen, alongside
resnet50 kept as the fixed SVM-comparison reference per this project's own
convention.

**DINOv3 is the one backbone where full fine-tuning backfires.** Frozen
DINOv3 beats *unfrozen* DINOv3 by a wide margin on WLE binary (0.7072 vs
0.5711) — the only backbone in the grid where that happens — and unfrozen
DINOv3 collapses to nearly exact chance on EUS (binary 0.5000, subtype
0.2330, both flat across all 5 folds). This is consistent with this
project's own §5e finding that DINOv3 was already the *strongest
frozen*-embedding backbone (WLE binary SVM 0.7353, ahead of ResNet-50's
0.6447): full fine-tuning at `--lr 3e-4` for 20 epochs actively degrades
that pretrained representation rather than improving on it, a known ViT
failure mode on small datasets. Not pursued further (would need a much
lower LR / layer-wise decay to test properly, and the CNN backbones already
outperform it unfrozen) — recorded as a confirmed-negative for
`--freeze none` specifically, not a verdict on DINOv3 as a backbone.

### 10.3 Lever screen (`efficientnet_b0`, WLE, single seed 42) — one confirmed lever

| lever | binary Δ vs. plain-CE (0.7711) | subtype Δ vs. plain-CE (0.4348) |
|---|---|---|
| oversample=size@0.2 | 0.7642 (−0.0069) | 0.4161 (−0.0187) |
| loss=weighted_ce | — | **0.4569 (+0.0221)** |
| loss=seesaw_ce | — | 0.4128 (−0.0220) |
| bg-swap-aug | 0.7404 (−0.0307) | 0.4317 (−0.0031) |

(bg-swap-aug also tested on EUS: binary 0.6836→0.6823, subtype
0.3602→0.3711 — no meaningful effect either direction; unlike the
frozen-feature bg-swap mitigation in §7's `07_mitigation.py`
(−75% reliance / −0.15 balanced acc on WLE), swapping backgrounds during
fine-tuning of a trainable backbone does not reproduce that effect at the
scale tested here.)

Only `weighted_ce` on the subtype task showed a directional single-seed
win, so it alone was carried to multi-seed confirmation.

### 10.4 Multi-seed confirmation — CONFIRMED

`efficientnet_b0`/WLE/subtype, `weighted_ce` vs. plain `ce`, seeds
{42, 123, 7}:

| seed | ce | weighted_ce |
|---|---|---|
| 42 | 0.4348 | 0.4569 |
| 123 | 0.4394 | 0.4603 |
| 7 | 0.4105 | 0.4655 |
| **mean ± std** | **0.4282 ± 0.0127** | **0.4609 ± 0.0035** |

Delta +0.0327, noise floor `sqrt(0.0127² + 0.0035²)` ≈ 0.0132 — **delta
clears the floor by ~2.5×, all 3 seeds agree in direction.** Confirmed.

### 10.5 Counterfactual gate (mandatory) — PASSES, confirmed as genuine

Scored the actual fine-tuned checkpoints (not a frozen-embedding SVM) on
`ablation_inputs.build_five_variants`'s 5 variants, same protocol as
§5e/Exp 3/Exp 17:

| run | full | lesion | bg_mask | bg_patch | boxmap |
|---|---|---|---|---|---|
| weighted_ce (seed 42) | 0.457 | 0.342 (55%) | 0.290 (35%) | 0.285 (33%) | 0.200 (−0%) |
| plain-CE baseline (seed 42) | 0.435 | 0.363 (70%) | 0.293 (40%) | 0.298 (42%) | 0.197 (−1%) |

Gate: the win's lesion-removal retention must be comparable to or *lower*
than the baseline's own retention (the lever must not be adding shortcut
reliance on top of what the baseline already has). `ret_bg_patch`: 33% vs.
baseline's 42% — **lower, not higher**. `weighted_ce`'s accuracy gain is not
explained by increased background/shortcut reliance; if anything it leans
slightly more on genuine lesion signal than the plain-CE baseline does.
**Recorded as a validated classification-accuracy improvement.**

One EUS result was also spot-checked against this gate (`resnet50`/EUS/
binary, unfrozen, the largest EUS win over its SVM reference: 0.6925 vs.
0.5817): `ret_bg_patch` = 42% — clearly below the DINOv3-EUS shortcut
precedent's ~100% (§5e), so *not* purely shortcut-driven, but 42% is a
non-trivial chunk of the full-image performance recoverable from background
alone. **Mixed / partial** — genuine signal is present, but the win over the
SVM baseline should not be read as clean. The other 15 EUS backbone-screen
cells were not individually re-run through the gate (out of scope given
time budget); treat all EUS numbers in §10.2 as shortcut-suspect until
checked, per this project's standing caution about this modality.

### 10.6 Verdict

**Confirmed**: `efficientnet_b0` (unfrozen) is the best WLE classification
backbone found so far, beating the frozen-SVM baseline by a wide,
counterfactual-gate-clean margin (binary 0.6447→0.7711, subtype
0.2995→0.4348); `--loss weighted_ce` adds a further confirmed +0.0327
balanced accuracy on the subtype task specifically, also counterfactual-gate
clean. **Confirmed-negative**: `--freeze none` (full fine-tuning) for
DINOv3 — use it frozen instead, where it remains the strongest single
embedding this project has measured. **Not confirmed / inconclusive at
single-seed**: size-aware oversampling and `seesaw_ce` (no effect),
bg-swap-aug (no effect on the fine-tuned model, unlike the frozen-feature
version). **Not independently validated**: the EUS backbone screen beyond
the one spot-checked cell — treat as shortcut-suspect.

---

## 11. Track B — YOLO11 detection track

A second detection architecture family, scored through the exact same
`cadsel/metrics.py` code (`DetectionAccumulator`/`summarize`) as the
existing Faster R-CNN Table 4/5 numbers, for a fair AP@0.5 comparison.
**No medical/endoscopy-pretrained YOLO checkpoint exists with verified
public weights** (checked HuggingFace, GitHub, Roboflow Universe, papers
2022-2026, prior to this track starting) — this starts from COCO-pretrained
`yolo11n.pt`/`yolo11s.pt`, a real gap stated plainly rather than assumed
away. **License**: `ultralytics` is AGPL-3.0; its own stated position is
that even internal R&D use technically requires either open-sourcing under
AGPL or an Enterprise License — a genuine ambiguity worth the user's
awareness, not a blocker to the results below.

**New infrastructure**: `experiments/yolo_common.py` (`filter_by_split`
reuses an existing Faster R-CNN run's own `split.json` so both stacks are
scored on the literal same held-out patients; `flat_stem` avoids filename
collisions across patient folders), `experiments/src/28_export_yolo_dataset.py`
(exports a modality/task slice to Ultralytics' expected `images/labels/
{train,val}` + `data.yaml` layout, remapping class ids via `TASK_ID_MAP`),
`experiments/src/29_yolo_eval_bridge.py` (scores either a YOLO or an
existing Faster R-CNN checkpoint through identical ground-truth-parsing +
`DetectionAccumulator` code, at `--conf 0.001` to match Faster R-CNN's own
permissive `box_score_thresh=0.05` scoring — not YOLO's `0.25` CLI default,
which would truncate the PR curve before AP@0.5 is computed).

**Environment note**: Ultralytics' global `settings.json` defaults
`runs_dir: runs` and silently prefixes any *relative* `project=` path with
`runs/<task>/` (confirmed by reading `ultralytics/cfg/__init__.py::
get_save_dir`) — `project=runs_yolo` lands at `runs/detect/runs_yolo/...`,
not the intended top-level `runs_yolo/`. Pass an **absolute** `project=`
path (bypasses the prefix entirely) to get the intended layout; all runs
below use `project=/home/rohan/CADSEL/runs_yolo`.

### 11.1 Eval bridge validated before trusting it on YOLO

Ran the bridge's own ground-truth-parsing + `DetectionAccumulator` code
against an existing Faster R-CNN checkpoint (`size_rfs_ms_s42__WLE__
multiclass`, `--input-mode torchvision` so its native-resolution
predictions match the bridge's native-resolution ground truth) and compared
to that checkpoint's already-published AP@0.5:

published 0.4280 vs. bridge-recomputed 0.4358 — delta +0.0078, **PASS**
(single-image inference vs. the original batched DataLoader eval accounts
for the small residual delta). The class-id remap and ground-truth
extraction are trustworthy.

### 11.2 Baseline vs. Faster R-CNN reference

`yolo11n`, default hyperparameters, WLE/multiclass, seed 42, same val split
as the Faster R-CNN reference above:

| model | AP@0.5 |
|---|---|
| Faster R-CNN (`size_rfs_ms_s42`, multiscale+size-RFS, tuned) | 0.4280 published / 0.4358 bridge |
| **YOLO11n (untuned baseline)** | **0.3963** |

An untuned nano model modestly underperforms the already lever-tuned Faster
R-CNN reference — expected, and the reason for the lever screen below.

### 11.3 Single-lever screen (single seed 42, vs. 0.3963 baseline)

| lever | AP@0.5 | P | R | verdict |
|---|---|---|---|---|
| imgsz=1280 | 0.3745 | 0.8184 | 0.4781 | worse — much more conservative (R crashed) at this batch/epoch budget |
| yolo11s (bigger model) | 0.3861 | 0.7617 | 0.6296 | worse |
| close_mosaic=30 | 0.4021 | 0.7653 | 0.6094 | marginal positive, see §11.4 |
| copy_paste=0.3 | 0.3963 | 0.7477 | 0.6835 | **exact no-op** — see below |

`copy_paste` reproduced the baseline's numbers to 4 decimal places
(P/R/F1 identical). Verified in Ultralytics' source
(`ultralytics/data/augment.py::CopyPaste.__call__`): `if len(labels
["instances"].segments) == 0 or self.p == 0: return labels` — this
augmentation requires segmentation polygons, which a pure bounding-box
detection dataset (ours) never has. **Confirmed no-op on this dataset, not
a bug** — not a meaningful lever here regardless of what value is passed.

### 11.4 Multi-seed confirmation — early-stopping trap found, then corrected — CONFIRMED NEGATIVE

First attempt: retrained `close_mosaic=30` and the baseline at seeds
{123, 7}, `patience=20` (matching §11.3). Seeds 123 and 7 produced
**literally identical model weights** between the baseline and lever arms
(different `.pt` file hashes, but every parameter tensor equal). Root
cause, confirmed via `results.csv`: both seeds found their best-mAP50 epoch
(55 and 50 respectively) *before* either config's mosaic-off window opened
(epoch 70 for `close_mosaic=30`, epoch 90 for the `close_mosaic=10`
default), and `patience=20` stopped training soon after — so with a fixed
seed and `deterministic=True`, training was bit-for-bit identical through
the only point that mattered, and the lever never got a chance to act.

Corrected test: reran **all three seeds** with `patience=100` (no early
stop, full 100-epoch budget guaranteed for every seed, not just 42):

| seed | baseline (clean) | close_mosaic=30 (clean) |
|---|---|---|
| 42 | 0.3963 | 0.4021 |
| 123 | 0.4205 | 0.3654 |
| 7 | 0.4319 | 0.4065 |
| **mean ± std** | **0.4162 ± 0.0148** | **0.3913 ± 0.0184** |

(the clean baseline numbers at seeds 123/7 are themselves higher than the
`patience=20` numbers above — 0.4205 vs 0.3822, 0.4319 vs 0.3758 — meaning
`patience=20` was cutting training short *in general*, not just masking
this one lever.)

Delta −0.0249, noise floor `sqrt(0.0148² + 0.0184²)` ≈ 0.0237 — **clears
the floor**, and 2 of 3 seeds show a clear decrease. **Verdict: CONFIRMED
NEGATIVE** — `close_mosaic=30` modestly *hurts* AP@0.5 once every seed gets
a genuine, uncontaminated test, reversing the earlier "not confirmed"
read. Generalisable methodological note this leaves behind: for any lever
whose mechanism is scoped to "the final N epochs of a fixed budget," early
stopping can silently prevent it from ever being exercised, and low
cross-seed variance under that condition is an artifact of the lever never
firing, not evidence it doesn't matter — always sanity-check *when* the
best epoch was found relative to the lever's own trigger point before
trusting a multi-seed result.

### 11.5 Follow-up leads (user-requested, see §12 for the classification-side leads run alongside these)

**`yolo11m` single-seed screen** — 0.4135 AP@0.5, beating both `yolo11n`
(0.3963) and `yolo11s` (0.3861). The plan's original caution about
overfitting risk from a bigger model on this ~2800-image dataset didn't
materialise at single-seed screening; not yet multi-seed confirmed.

**Class-oversampled training (targeting Leiomyoma, the rarest class) —
CONFIRMED POSITIVE, Track B's first confirmed lever.**
`experiments/src/31_export_yolo_dataset_oversampled.py` physically
duplicates training images on disk following
`cadsel.data.RepeatFactorSampler`'s own Detectron2-style formula
(`r(c) = max(1, sqrt(threshold / f(c)))`, threshold 0.2, reused directly —
702/2249 images oversampled, mean factor 1.11, train set grows to 2358
images; val set is untouched and identical to the plain export):

| seed | plain export | oversampled export |
|---|---|---|
| 42 | 0.3963 | 0.4295 |
| 123 | 0.3822 | 0.3963 |
| 7 | 0.3758 | 0.4070 |
| **mean ± std** | **0.3848 ± 0.0086** | **0.4109 ± 0.0138** |

Delta +0.0262, noise floor `sqrt(0.0086² + 0.0138²)` ≈ 0.0163 — **clears
the floor by ~1.6×, all 3 seeds positive.** Confirmed. Per-class AP shows
the gain isn't purely a Leiomyoma fix (still low, ~0.17 AP@0.5 at seed 42)
— Lipoma and overall balance improved too, so this reads as "oversampling
correctly rebalances the whole training signal," the same mechanism that
made size-aware oversampling work for Faster R-CNN (FINDINGS.md §5g), not
a targeted fix for one class specifically.

### 11.6 Verdict (updated)

**Confirmed**: class-oversampled training (§11.5) — the first confirmed
positive Track B lever, +0.0262 AP@0.5. **Confirmed-negative**:
`imgsz=1280`, `yolo11s` (worse, single-seed), and now `close_mosaic=30`
(§11.4, corrected from "not confirmed" to confirmed-negative once tested
without early-stopping contamination). **Confirmed no-op**: `copy_paste`
(dataset has no segmentation masks). **Promising, not yet multi-seed
confirmed**: `yolo11m` (+0.0172 over `yolo11n` baseline at single seed).
Best untuned/single-lever YOLO11n number found (oversampled, mean across 3
seeds: 0.4109) is now within noise of the tuned Faster R-CNN reference
(0.4280 published / 0.4358 bridge-validated) rather than clearly behind it
— combining the oversample lever with `yolo11m` is the natural next step,
not pursued in this round.

**Critical files**: `cadsel/classification_data.py`,
`experiments/finetune_classifier.py`,
`experiments/src/26_classifier_leaderboard.py`,
`experiments/src/27_classifier_counterfactual.py`,
`experiments/src/30_classifier_tta_eval.py` (Track A);
`experiments/yolo_common.py`, `experiments/src/28_export_yolo_dataset.py`,
`experiments/src/29_yolo_eval_bridge.py`,
`experiments/src/31_export_yolo_dataset_oversampled.py` (Track B).

---
## 12. Track A follow-up leads (user-requested) — a new best classifier

Four leads pursued alongside §11.5's Track B follow-ups, same discipline
(single-lever screen → ≥3-seed confirmation → mandatory counterfactual
gate for any classification win).

### 12.1 Binary-task lever screen — no confirmed lever

§10.3's lever screen only covered the subtype task ("where imbalance is
real" per the original campaign plan). Filling the gap on binary
(`efficientnet_b0`/WLE, seed 42, vs. the 0.7711 plain-CE baseline):

| lever | balanced accuracy | delta |
|---|---|---|
| loss=weighted_ce | 0.7640 ± 0.0215 | −0.0071 |
| loss=seesaw_ce | 0.7603 ± 0.0377 | −0.0108 |

(oversample=size and bg-swap-aug were already tested for binary in §10.3:
0.7642 and 0.7404 respectively, both also non-wins.) Neither loss variant
helps — expected, since binary class imbalance (NET vs. non-NET) is much
milder than the 5-way subtype split. No further action.

### 12.2 DINOv3 `--freeze partial` — CONFIRMED, new best classifier on both WLE tasks

§10.2 found DINOv3 `--freeze none` (full fine-tune) catastrophically
degrades its own strong frozen representation. The natural next lever,
unfreezing only the last few transformer blocks
(`Dinov3Classifier`'s existing `unfreeze_last_n=4` default — no new code
needed), was screened and then confirmed at 3 seeds:

| task | seed 42 | seed 123 | seed 7 | mean ± std |
|---|---|---|---|---|
| binary | 0.7963 | 0.8136 | 0.8105 | **0.8068 ± 0.0075** |
| subtype | 0.4741 | 0.4899 | 0.4697 | **0.4779 ± 0.0087** |

Binary: vs. the previous best confirmed WLE-binary result (`efficientnet_b0`
unfrozen, 0.7711 single-seed) — a decisive +0.0357 jump, far outside any
plausible per-arm noise band given DINOv3-partial's own std is 0.0075.
Subtype: vs. the already-confirmed `weighted_ce` winner (§10.4, 0.4609 ±
0.0035) — delta +0.0170, noise floor `sqrt(0.0087² + 0.0035²)` ≈ 0.0094,
**clears it**. Confirmed as the new best on both tasks.

**Counterfactual gate — PARTIAL, not a clean pass.** Scored against the
strongest prior baseline for each task (binary: `efficientnet_b0`
unfrozen; subtype: the confirmed `weighted_ce` winner):

| run | full | lesion (isolated) | bg_patch |
|---|---|---|---|
| DINOv3-partial, binary | 0.796 | 0.759 (**87%**) | 0.659 (54%) |
| efficientnet_b0 baseline, binary | 0.771 | 0.648 (55%) | 0.624 (46%) |
| DINOv3-partial, subtype | 0.474 | 0.427 (**83%**) | 0.311 (40%) |
| weighted_ce baseline, subtype | 0.457 | 0.342 (55%) | 0.285 (33%) |

The strict WLE gate (target's `bg_patch` retention should be ≤ baseline's)
technically fails on both tasks (54%>46%, 40%>33%) — by the letter of this
project's own rule, that's a WARN, not a clean pass. But the fuller picture
is genuinely mixed, not simply "more shortcut-driven": DINOv3-partial's
retention on the **isolated lesion crop** (context removed, only the
lesion + padding visible) is dramatically higher than either baseline's
(87%/83% vs. 55%/55%) — the strongest evidence of genuine lesion-reading
seen anywhere in this campaign. Reading both numbers together: DINOv3's
richer pretrained representation appears to extract more signal from
*both* the lesion and the surrounding context than the CNN backbones do,
not to be substituting shortcut-reading for lesion-reading. **Recorded as
the new best result, flagged PARTIAL on the counterfactual gate** (not a
clean pass like §10.5's `weighted_ce` result) — a real caveat for anyone
building on this, not a rejection.

### 12.3 DINOv3 `--freeze none` at a much lower LR — confirms the LR hypothesis, recovers most of the gap

§10.2's read on DINOv3 full fine-tuning ("a known ViT failure mode... would
need a much lower LR to test properly") was tested directly: `--lr 1e-5`
instead of the project default `3e-4`, single seed 42, `--freeze none`:

| task | freeze=none, lr=3e-4 (§10.2) | freeze=none, lr=1e-5 | freeze=partial, lr=3e-4 (§12.2) |
|---|---|---|---|
| binary | 0.5711 (catastrophic) | **0.7764** | 0.7963 |
| subtype | 0.3007 (catastrophic) | **0.4388** | 0.4741 |

Confirms the hypothesis cleanly: the earlier "DINOv3 full fine-tuning
fails" result was an LR problem, not a fundamental incompatibility between
DINOv3 and full unfreezing. At the right LR, full fine-tuning recovers
almost all the way to partial-freeze's level, though partial-freeze still
edges it out on both tasks. Single-seed screen only (not pursued to
confirmation) — partial-freeze is already confirmed and slightly ahead, so
this doesn't change the leaderboard, but it does correct the earlier
framing: don't write off `--freeze none` for a ViT backbone generally, only
at this project's CNN-tuned default LR.

**Bug found and fixed while running this**: `--lr` was not part of
`finetune_classifier.py`'s output `run_name`, so this lead's `--freeze none
--lr 1e-5` runs collided with and **silently overwrote** §10.2's original
catastrophic-failure JSON/checkpoints for
`dinov3_vits16__WLE__{binary,subtype}__freeze-none__os-none__loss-ce__s42`.
The original 0.5711/0.3007 numbers above are preserved from this
conversation's own transcript (recorded independently at the time, before
the collision) and were not re-derived after the fact. Fixed: `run_name`
now includes `__lr<value>` whenever `--lr` differs from the 3e-4 default
(default runs are unaffected, so every existing filename elsewhere in this
project stays valid). The collided files were renamed to their correct
`__lr1e-05__s42` names post hoc. **Lesson for future runs**: any CLI flag
that can distinguish two otherwise-identical configs needs to be in the
output filename, or a later run can silently destroy an earlier one's
results with no error or warning.

### 12.4 Test-time augmentation (horizontal-flip averaging) — no effect

`experiments/src/30_classifier_tta_eval.py` (new): averages softmax
probabilities over the image and its horizontal flip, evaluated against a
plain single-view pass computed through identical code (not the number
recorded during training) so the comparison isn't confounded by any
code-path difference:

| checkpoint | plain | TTA | delta |
|---|---|---|---|
| `efficientnet_b0`/subtype/weighted_ce (confirmed winner) | 0.4569 ± 0.0465 | 0.4560 ± 0.0479 | −0.0009 |
| `efficientnet_b0`/subtype/plain-CE | 0.4348 ± 0.0495 | 0.4256 ± 0.0529 | −0.0092 |
| `efficientnet_b0`/binary/plain-CE | 0.7711 ± 0.0419 | 0.7721 ± 0.0391 | +0.0010 |

All three deltas are noise-floor territory. Unsurprising in hindsight:
`RandomHorizontalFlip` is already in the training augmentation pipeline
(`cadsel/classification_data.py`), so the model has already learned
flip-invariance — flip-TTA has little left to add. Confirmed no-op; not
pursued further (a multi-crop or multi-scale TTA variant might behave
differently, but wasn't tested).

### 12.5 Verdict

**New best classifier, confirmed**: DINOv3 `--freeze partial` — WLE binary
0.8068 ± 0.0075 (was 0.7711), WLE subtype 0.4779 ± 0.0087 (was 0.4609) —
**but flagged PARTIAL, not clean, on the mandatory counterfactual gate**
(§12.2): report both the confirmation and the caveat together, don't cite
the headline number without it. `efficientnet_b0` + `weighted_ce` (§10)
remains the cleanest fully-passing result if a strict counterfactual pass
is required. **Confirmed-negative**: DINOv3 `--freeze none` at the
project's default LR (3e-4) — but only at that LR; §12.3 shows the same
config at `lr=1e-5` recovers to near partial-freeze levels, so this is a
hyperparameter-sensitivity finding, not a backbone-level rejection. **No
effect**: binary-task loss reweighting (§12.1), horizontal-flip TTA
(§12.4).

**Critical files (additions)**: `experiments/src/30_classifier_tta_eval.py`
(new); `experiments/finetune_classifier.py` (`run_name` bug fix, §12.3).

---
## 13. Round-2 follow-up leads (user-requested: "anything else to get more impressive results?")

Executed per the approved plan (`do-you-think-theres-jazzy-valiant.md`), same
discipline as §10-§12: single-seed screen → ≥3-seed confirmation → mandatory
counterfactual gate for any Track A win.

### 13.1 Tier 1a — Track B: `yolo11m` + class-oversampling combo — confirmed-negative interaction

Combining the two individually-positive Track B levers backfired:

| config | AP@0.5 |
|---|---|
| `yolo11n` baseline | 0.3963 |
| `yolo11n` + oversample (§11.5) | 0.4295 |
| `yolo11m` alone (§11.5) | 0.4135 |
| **`yolo11m` + oversample (combo)** | **0.3724** |

Worse than either lever alone, and worse than the plain baseline. Single-seed
screen only (not pursued further given the negative direction) — a real
lesson that two independently-positive levers don't automatically compose;
plausibly `yolo11m`'s larger capacity combined with the oversampled set's
~5% larger, more duplicate-heavy training data at the same `batch=8` made
optimization harder within the 100-epoch budget, not explored further.

### 13.2 Tier 1b — Track A: DINOv3 `--freeze partial` + `--loss weighted_ce` — CONFIRMED, counterfactual-clean

Combining Track A's two independently-confirmed levers **did** compose, on
the subtype task specifically:

| config | seed 42 | seed 123 | seed 7 | mean ± std |
|---|---|---|---|---|
| DINOv3-partial + weighted_ce (subtype) | 0.5003 | 0.4928 | 0.4941 | **0.4957 ± 0.0033** |
| DINOv3-partial + weighted_ce (binary) | 0.8015 | — | — | (screen only, see below) |

Subtype: vs. DINOv3-partial alone (0.4779 ± 0.0087, §12.2), delta +0.0178,
floor `sqrt(0.0033²+0.0087²)` ≈ 0.0093 — confirmed. Vs. `weighted_ce` alone
(0.4609 ± 0.0035, §10.4), delta +0.0348, floor ≈ 0.0048 — confirmed
decisively. Binary: single-seed screen (0.8015) landed statistically flat
against DINOv3-partial alone (0.8068 ± 0.0075) — no compounding benefit
there, consistent with `weighted_ce` never helping binary regardless of
backbone (§12.1) — not pursued to multi-seed confirmation.

**Counterfactual gate (subtype, vs. its own plain-CE sibling — the correct
isolate-one-variable comparison) — clean pass:**

| run | full | lesion | bg_patch | boxmap |
|---|---|---|---|---|
| DINOv3-partial + weighted_ce | 0.500 | 0.445 (82%) | 0.282 (**27%**) | 0.215 (5%) |
| DINOv3-partial + plain-ce (baseline) | 0.474 | 0.427 (83%) | 0.311 (**40%**) | 0.199 (−0%) |

`bg_patch` retention *dropped* (40%→27%) while isolated-lesion retention
held steady (83%→82%) — the accuracy gain is not shortcut-driven. One minor
WARN on `boxmap` (5% vs. −0%), but both are noise-level on an
already-near-zero metric, not a meaningful signal. **Recorded as a fully
validated classification-accuracy improvement — Track A's new best subtype
result.**

### 13.3 Tier 1c — Track B: Ultralytics' built-in TTA (`--augment`) — confirmed negative

Added `--augment` to `experiments/src/29_yolo_eval_bridge.py`'s
`predict_yolo()` (Ultralytics' own multi-scale+flip TTA, merged
internally). Tested against the best available YOLO checkpoint at the time
(`yolo11n` + oversample, 0.4295):

| | AP@0.5 | Leiomyoma AP@0.5 |
|---|---|---|
| plain | 0.4295 | ~0.17 |
| `--augment` | **0.3972** | **0.0258** |

Worse (−0.0323), and Leiomyoma detection specifically collapsed. Confirmed
negative for this checkpoint; not pursued further.

### 13.4 Tier 2a — Track A: multi-seed probability ensembling — two new campaign bests, zero additional GPU training

New script `experiments/src/32_classifier_multiseed_ensemble.py` averages
softmax probabilities across the 3 already-trained seed checkpoints of the
same config, aligned per-image by `Sample.image` (not by fold index —
each seed's `patient_level_folds` partitions patients differently; see the
script's own docstring for the alignment logic this had to get right).
Built-in sanity check: each seed's reconstructed out-of-fold balanced
accuracy is printed alongside its own recorded 5-fold mean and matches
closely (e.g. 0.7955 reconstructed vs. 0.7963 recorded) — confirms the
reconstruction is correct before trusting the ensemble number.

| config | individual seeds (OOF) | mean ± std | **ensemble** | delta |
|---|---|---|---|---|
| DINOv3-partial, binary | 0.7955, 0.8134, 0.8101 | 0.8063 ± 0.0078 | **0.8209** | +0.0146 |
| DINOv3-partial + weighted_ce, subtype | 0.5026, 0.5018, 0.4919 | 0.4988 ± 0.0049 | **0.5230** | +0.0242 |

Both ensembles beat every individual seed and the seed mean, for zero
additional GPU time (pure inference on checkpoints that already existed).
**These are now Track A's best numbers overall**: binary 0.8209, subtype
0.5230. Inherits §13.2's counterfactual-gate status (the ensemble aggregates
predictions from the same underlying models, not a new mechanism — not
re-gated separately, no script currently exists to counterfactually-gate an
ensemble directly).

### 13.5 Tier 2b — Track A: EUS counterfactual-gate backlog cleared

All 17 remaining EUS backbone-screen checkpoints (§10.2) scored against the
gate (2 more collapsed to exact chance like the previously-known cases and
produce no retention signal — division-by-zero guard, not a bug):

| run | bg_patch retention | boxmap retention |
|---|---|---|
| convnext_tiny binary frozen | 29% | 66% |
| convnext_tiny binary unfrozen | 41% | −8% |
| convnext_tiny subtype frozen | 15% | 16% |
| convnext_tiny subtype unfrozen | 24% | −23% |
| dinov3 binary frozen | 49% | 62% |
| dinov3 binary unfrozen | n/a (collapsed to exact chance) | n/a |
| dinov3 subtype frozen | 33% | 12% |
| dinov3 subtype unfrozen | 22% | 4% |
| efficientnet_b0 binary frozen | 57% | −43% |
| efficientnet_b0 binary unfrozen+bgswap | 21% | 16% |
| efficientnet_b0 binary unfrozen | 50% | 0% |
| **efficientnet_b0 subtype frozen** | **89%** | −15% |
| efficientnet_b0 subtype unfrozen+bgswap | −0% | 0% |
| efficientnet_b0 subtype unfrozen | 35% | 0% |
| resnet50 binary frozen | 25% | −43% |
| resnet50 subtype frozen | n/a (collapsed to exact chance) | n/a |
| resnet50 subtype unfrozen | 21% | −5% |
| resnet50 binary unfrozen (already checked, §10.5) | 42% | −4% |

**Standout**: `efficientnet_b0`/EUS/subtype/**frozen** retains **89%** of its
full-image balanced accuracy with the lesion completely removed — the
closest any result in this campaign has come to the ~100% "pure
shortcut-reading" DINOv3-EUS precedent from §5e, on a *different* backbone
than the one that precedent was originally about. Do not trust this
specific number (0.2327 balanced accuracy, §10.2) as a genuine EUS subtype
result. Most other EUS cells land in the 15-50% range (mixed/partial, same
character as the one cell already checked in §10.5) — no other clean
result emerged. **Verdict: EUS classification results in this campaign
should continue to be treated as shortcut-suspect by default; only
`efficientnet_b0`/EUS/subtype/unfrozen+bgswap (−0% retention) looks clean,
and it's also one of the weaker accuracy numbers (0.3711, §10.3) — no EUS
result in this campaign is both strong and counterfactually clean.**

Raw results: `experiments/results/27_classifier_counterfactual__*.json` (one
file per checkpoint checked).

### 13.6 Tier 2c — Track B: binary and EUS coverage for YOLO — EUS is the first case YOLO beats Faster R-CNN

Track B had only ever tested WLE/multiclass. Filled the two coverage gaps,
single-seed screen, same `yolo11n` defaults as §11.2's original baseline:

| modality/task | YOLO11n AP@0.5 | Faster R-CNN reference AP@0.5 | delta |
|---|---|---|---|
| WLE/binary | 0.6059 | 0.6731 (`size_rfs_ms_s42__WLE__binary`, published) | −0.0672 |
| **EUS/multiclass** | **0.3776** | 0.3163 (`fasterrcnn_mobilenet_320__EUS__multiclass`, published) | **+0.0613** |

WLE/binary reproduces the same modest-underperformance pattern already seen
on WLE/multiclass (§11.2). **EUS/multiclass is the first result anywhere in
this campaign where an untuned YOLO11n beats the Faster R-CNN reference** —
plausibly because EUS's smaller, more homogeneous 4-class distribution and
grayscale imagery suit YOLO11's single-stage architecture better than the
harder, more class-imbalanced WLE task.

**3-seed confirmation — CONFIRMED**:

| seed | AP@0.5 |
|---|---|
| 42 | 0.3776 |
| 123 | 0.3735 |
| 7 | 0.3886 |
| **mean ± std** | **0.3799 ± 0.0064** |

Delta vs. the Faster R-CNN published reference (0.3163): **+0.0636** — about
10× YOLO's own seed-to-seed std, decisively clear of any plausible noise
floor even without a matching multi-seed spread for the Faster R-CNN side.
**Confirmed: untuned YOLO11n beats the tuned Faster R-CNN reference on
EUS/multiclass** — the first track-level win for YOLO11 anywhere in this
campaign (Track B's other confirmed win, class-oversampling, still trails
the Faster R-CNN WLE reference; this is the first case of outright beating
it). Worth revisiting with EUS-specific levers (the WLE lever screen's
findings don't necessarily transfer) as a follow-up, not pursued further
in this round.

**Critical files (round-2 additions)**:
`experiments/src/32_classifier_multiseed_ensemble.py` (new, §13.4);
`experiments/src/29_yolo_eval_bridge.py` (`--augment` flag, §13.3);
`experiments/yolo/wle_binary_s42/`, `experiments/yolo/eus_multiclass_s42/`
(new exports, §13.6).
## 14. Round-3 follow-up leads (further "anything else to get more impressive results?")

### 14.1 WBF multi-scale TTA vs. a non-multiscale-trained checkpoint — closes the open thread, still negative

The open thread from §5c ("worth retrying [multi-scale WBF TTA] against a
checkpoint that was not trained multi-scale") was closed. `cadsel/
evaluate_tta.py` didn't support the DINOv3 detector-backbone model family
(only the standard `MODEL_REGISTRY` torchvision families) — fixed with a
small dispatch on `model_type` to `build_dinov3_detection_model` when
appropriate.

`runs/dinov3_wle_multiclass_s42` (input_mode=torchvision, `multiscale_sizes:
None` — confirmed not multiscale-trained): single-scale AP@0.5=0.3767 →
multi-scale WBF (`[256,320,400]`)=0.3679, **delta −0.0088**. Negative even
without the "already scale-robust from multiscale training" confound that
explained the earlier negative result on a multiscale-trained checkpoint
(§5c). Caveat: DINOv3-as-detector-backbone is itself a confirmed-not-useful
backbone (§5o/§5s), so this doesn't cleanly settle whether WBF TTA would
help a *strong* non-multiscale checkpoint — but the thread as originally
posed (retest against any non-multiscale checkpoint) is closed, negative.

### 14.2 EUS-specific lever screen for YOLO — WLE's oversampling win does not transfer

The confirmed WLE oversampling lever (§13.6, §11.5) was tested on EUS via
the same `31_export_yolo_dataset_oversampled.py` mechanism
(threshold 0.2, 420/1660 EUS images oversampled, mean factor 1.12):

| | AP@0.5 |
|---|---|
| EUS plain (3-seed mean, §13.6) | 0.3799 ± 0.0064 |
| EUS class-oversampled (single seed 42) | 0.3680 |

Oversampling *hurts* on EUS (single-seed screen; not pursued to
confirmation given the negative direction). A real, useful negative: the
lever that helped WLE doesn't automatically transfer to a different
modality with a different class distribution — EUS's own imbalance profile
and lesion-size distribution differ enough that the same oversampling
recipe isn't a free win there.

### 14.3 DINOv3 `--freeze partial` on EUS — new best EUS classifier on both tasks, counterfactual-checked

Never tried before (Track A's DINOv3-partial work was WLE-only). Single-seed
screen (seed 42):

| task | DINOv3-partial | previous best (unfrozen CNN) |
|---|---|---|
| binary | **0.7386 ± 0.0313** | 0.6925 (resnet50) |
| subtype | **0.3764 ± 0.0507** | 0.3711 (efficientnet_b0+bgswap) |

Binary is a decisive jump (+0.046 over the previous best); subtype is a
modest new best, within its own wide fold-to-fold std. Single-seed screens
only, not yet multi-seed confirmed.

**Counterfactual gate (mandatory for any EUS claim, given this modality's
shortcut history):**

| task | bg_patch retention | boxmap retention |
|---|---|---|
| binary | 35% | 4% |
| subtype | 15% | −13% |

Both comfortably below the ~100% "pure shortcut" DINOv3-EUS precedent
(§5e), and both are among the *lowest* (cleanest) retentions of any EUS
result checked in this campaign (§13.5's table ranged 15-89% across 16
other cells). The large accuracy jump looks genuinely substantive, not
shortcut-driven — the most encouraging EUS classification result found so
far, though still only single-seed screened.

### 14.4 Multimodal fusion rebuilt on classifier backbones — promising, but noise-limited on the methodologically-correct comparison

New module `cadsel/classifier_fusion_features.py` (DINOv3-only scope — see
its docstring for why the CNN backbones were deferred) and new script
`experiments/src/33_classifier_fusion.py`, both built to slot into
`cadsel.fusion`'s existing `FusionHead`/`make_batch`/`PatientFeatures` and
`cadsel.train_fusion`'s `train_fold`/`ABLATIONS` **completely unchanged** —
only the feature-extraction front-end differs (a DINOv3 CLS-token pooled
feature + softmax posterior per frame, in place of the detector's ROI-head
feature + posterior). Ran using the confirmed-best WLE (`weighted_ce`
combo, §13.2) and EUS (§14.3) subtype checkpoints (fold 0 of each as the
frozen feature extractor).

**On all 641 patients** (includes the 220 WLE-only patients where fusion
*structurally* cannot differ from `wle_only` — §5i's own documented caveat
about this exact denominator):

| ablation | balanced accuracy |
|---|---|
| wle_only | 0.8836 ± 0.0280 |
| eus_only | 0.8934 ± 0.0367 |
| fused | 0.9271 ± 0.0317 |
| **fused_moddrop** | **0.9324 ± 0.0290** |

**On the 421 paired patients only** (the methodologically correct
comparison per §5i's own established correction):

| ablation | balanced accuracy |
|---|---|
| wle_only | 0.8847 ± 0.0225 |
| eus_only | 0.8960 ± 0.2014 |
| fused | 0.9406 ± 0.1122 |
| fused_moddrop | 0.9304 ± 0.1343 |

Fusion beats `wle_only` by a real margin on both denominators (paired-only:
+0.0559 for `fused`). **But** the paired-only fold-to-fold variance is very
wide (std up to 0.20) — noise floor `sqrt(0.0225² + 0.1122²)` ≈ 0.115
exceeds the +0.0559 delta, so this does **not** clear this project's own
confirmation bar. This is a pre-existing, already-documented limitation of
the whole patient-level fusion protocol, not something new introduced
here: `cadsel/train_fusion.py`'s own module docstring already flags that
"only 421 patients have both modalities and Leiomyoma has 14 of them — a
single 20% split leaves about three Leiomyoma patients in validation, which
cannot support a claim." The classifier-backbone version's numbers are
high enough (~0.88-0.94 vs. the original detector-based fusion's ~0.38-0.52
*accuracy* on the same paired subset, §5i) that the same small-sample
instability shows up starkly in relative terms.

**Contrast with the original detector-based fusion (§5i)**: that version's
apparent gain was found to be a majority-class accuracy effect — on
*balanced* accuracy specifically, `wle_only` actually *beat* both fused
variants there. Here, fusion beats `wle_only` on balanced accuracy on
*both* denominators, directionally the opposite conclusion. **Verdict: a
genuinely promising, structurally different result from the original
fusion work, but not confirmed** — the paired-patient pool is too small
for this single-5-fold-split protocol to produce a stable read. A repeated
CV (multiple different 5-fold partitions, not just multiple seeds of the
CV network's own init) or bootstrapped confidence interval would be needed
to confirm properly; not pursued further in this round given the scope
already covered.

### 14.5 `unfreeze_last_n` sweep — the default (4) is close but not quite optimal

`Dinov3Classifier`'s `unfreeze_last_n` (how many of the 12 transformer
blocks `--freeze partial` unfreezes, from the end) was hardcoded, never
exposed or varied. Added `--unfreeze-last-n` to
`experiments/finetune_classifier.py` (included in `run_name` when
non-default, following §12.3's established naming-collision fix). Swept
`{2, 6, 8, 12}` against the default 4, WLE binary, single seed 42:

| unfreeze_last_n | balanced accuracy |
|---|---|
| 2 | 0.7968 ± 0.0120 |
| 4 (default, §12.2) | 0.7963 ± 0.0189 |
| 6 | 0.8068 ± 0.0125 |
| **8** | **0.8085 ± 0.0142** |
| 12 | 0.7670 ± 0.0178 |

`n=8` (and `n=6`) modestly beat the default `n=4` (+0.0122 for `n=8`,
single-seed, not yet multi-seed confirmed). `n=12` is clearly worse than
2-8, but **not catastrophic** the way `--freeze none` was (0.5711, §10.2) —
a real architectural difference explains why: even at `unfreeze_last_n=12`
(all 12 transformer blocks), the patch/position embedding layers stay
frozen (they live outside `self.backbone.model.layer`, which is all
`--freeze partial` ever touches), while `--freeze none` leaves literally
everything trainable including those embeddings. Keeping the embedding
layer frozen alone appears to be most of what prevents the catastrophic
collapse — worth remembering as a cheaper alternative to the `--lr 1e-5`
fix (§12.3) if a use case specifically wants the embeddings frozen.
Sweet spot is 6-8 blocks, not the default 4 or the full 12; not pursued to
multi-seed confirmation in this round.

### 14.6 5-crop TTA — also negative, closing the TTA thread

`experiments/src/30_classifier_tta_eval.py` gained `--mode {flip,5crop}`
(default `flip`, unchanged behaviour). `5crop` decodes each image ~15%
oversized and averages predictions over 4 corner crops + 1 center crop at
the model's own `img_size` — perturbs scale/framing rather than left-right
orientation, the natural complement to §12.4's flip-TTA:

| checkpoint | plain | 5-crop TTA | delta |
|---|---|---|---|
| DINOv3-partial+weighted_ce, WLE subtype | 0.5003 ± 0.0248 | 0.4989 ± 0.0247 | −0.0014 |
| DINOv3-partial, EUS binary (§14.3) | 0.7386 ± 0.0313 | 0.7223 ± 0.0277 | −0.0162 |

Both negative or null, same conclusion as flip-TTA (§12.4) and for the same
reason: `RandomResizedCrop(scale=(0.7,1.0))` already gives the model
scale/crop augmentation during training, leaving TTA nothing new to
contribute. Closes the TTA thread for this project — neither variant
tested (flip, 5-crop) helps any checkpoint tried.

**Critical files**: `cadsel/evaluate_tta.py` (DINOv3 dispatch fix, §14.1);
`experiments/src/31_export_yolo_dataset_oversampled.py` (reused unchanged
for EUS, §14.2); `cadsel/classifier_fusion_features.py`,
`experiments/src/33_classifier_fusion.py` (new, §14.4);
`experiments/finetune_classifier.py` (`--unfreeze-last-n` addition, §14.5);
`experiments/src/30_classifier_tta_eval.py` (`--mode 5crop` addition, §14.6).

---

## 15. Joint end-to-end two-tower WLE+EUS fusion (user-proposed architecture) — CONFIRMED on both binary and subtype tasks

### 15.1 Motivation and architecture

Every fusion result before this section — the original detector-ROI version
(`cadsel/fusion.py`, §5i) and this session's DINOv3 classifier-feature
rebuild (`cadsel/classifier_fusion_features.py`, §14.4) — trains
`FusionHead` on **frozen, pre-extracted, single-pass features**. The
per-modality backbones are trained once beforehand and never see a
fusion-loss gradient, and no image augmentation happens anywhere in the
fusion-training stage, since the "dataset" by that point is a static table
of feature vectors, not images.

The user proposed a structurally different architecture: two independently
trainable backbones (one per modality), jointly fine-tuned end-to-end, with
independent per-modality data augmentation applied every training step.
Built as three new files, using `efficientnet_b0` for both towers per the
user's request:

- **`cadsel/joint_fusion_data.py`** — `JointPatientDataset`: one item per
  patient, up to `max_frames` (default 4) raw frames per modality, each
  independently run through its own modality's `CadSelClassificationDataset`-style
  transform (`RandomResizedCrop`/`HFlip`/`ColorJitter`/`RandomRotation`).
  Frame *selection* is deterministic per patient index (mirrors
  `CadSelClassificationDataset`'s own seeding pattern); each frame's actual
  augmentation is freshly redrawn every epoch via torchvision's transform
  internals, exactly as the single-modality classifiers already do. A
  present modality is sampled *with replacement* to always fill
  `max_frames` — zero-padding is reserved for a genuinely absent modality
  (a WLE-only patient's EUS side), not ordinary frame scarcity.
- **`cadsel/joint_fusion_model.py`** — `JointTwoTowerModel`: two independent
  `efficientnet_b0` towers (ImageNet-pretrained, `--freeze partial`
  unfreezes the last two `features` stages, mirroring
  `finetune_classifier.py::build_cnn_backbone`'s CNN partial-freeze slice)
  feeding their live per-frame pooled features directly into an
  **unmodified** `cadsel.fusion.FusionHead` (`AttentionPool`,
  `GatedFusion`/`SymmetricCrossAttentionFusion`, `modality_dropout`,
  missing-modality masking — all reused as-is).
- **`experiments/src/34_joint_fusion_train.py`** — training/CV driver,
  modeled on `finetune_classifier.py::run_fold` (AdamW, mixed precision,
  `ReduceLROnPlateau`, per-fold checkpointing) since `train_fusion.py::train_fold`
  assumes features already sit in RAM with no image/augmentation/backbone
  step. CV via `build_joint_manifest()` + `patient_level_folds(...,
  strata_key=joint_strata_key)`. One deliberate compute-driven
  simplification vs. `train_fusion.py`'s 4-way `ABLATIONS` protocol: rather
  than training 4 separate models per fold, this trains ONE joint model per
  fold with `modality_dropout=0.3` active throughout training, then reports
  `wle_only`/`eus_only`/`fused` as three *masked-inference* views of that
  single trained model (`eus_only` always restricted to patients who
  actually have EUS frames, matching `train_fusion.py`'s own precedent).

Verified the backbones genuinely receive gradients, not just architecturally
claim to: after one fold, `features.8` (trainable under `--freeze partial`)
had moved a mean absolute diff of 0.004072 from its ImageNet initialization,
while `features.0` (frozen) showed exactly 0.0 diff.

### 15.2 Single-seed screens — task and training-budget sweep

Three single-seed (42) screens, `--loss weighted_ce`, `--fusion gated`,
`--freeze partial`, `max_frames=4`, one flat `lr=1e-4` for everything:

| config | ablation | denominator | balanced accuracy |
|---|---|---|---|
| subtype, 30 epochs (no early stop) | wle_only | unpaired | 0.4638 ± 0.0398 |
| | wle_only | paired-only | 0.4178 ± 0.0688 |
| | eus_only | (paired) | 0.2512 ± 0.0569 |
| | **fused** | unpaired | **0.5009 ± 0.0152** |
| | fused | paired-only | 0.4692 ± 0.0552 |
| subtype, 100 epochs / patience 20 | wle_only | unpaired | 0.4589 ± 0.0550 |
| | wle_only | paired-only | 0.4140 ± 0.0950 |
| | eus_only | (paired) | 0.3182 ± 0.0767 |
| | **fused** | unpaired | **0.4963 ± 0.0371** |
| | fused | paired-only | 0.4636 ± 0.0955 |
| binary, 100 epochs / patience 20 | wle_only | unpaired | 0.6871 ± 0.0880 |
| | wle_only | paired-only | 0.6409 ± 0.0787 |
| | eus_only | (paired) | 0.6411 ± 0.1132 |
| | **fused** | unpaired | **0.7490 ± 0.0728** |
| | fused | paired-only | 0.7207 ± 0.0783 |

`fused` beat `wle_only` in every screen (+0.037 to +0.037 subtype, +0.062
binary), but none individually cleared its own fold-to-fold noise floor.
Giving subtype 100 epochs + patience instead of a fixed 30 barely moved
`wle_only`/`fused` at all (both changes are a fraction of either run's own
std) — ruling out "undertrained" as the explanation for this architecture's
much lower absolute numbers vs. §14.4's frozen-feature approach (~0.50 vs.
~0.93 fused balanced accuracy on subtype). The gap is better explained by
DINOv3 being a stronger backbone than EfficientNet-B0 (already established
earlier in this campaign) and by §14.4's frozen extractors being
individually-converged checkpoints with a softmax-posterior feature this
from-scratch joint model doesn't have, not by an optimization bug here.
`eus_only`'s jump on the 100-epoch run (0.3182 vs. 0.2512) did not
replicate under discriminative LR (§15.3) — noise, not a real epoch-budget
effect.

Binary showed the largest raw gap of the three, motivating the confirmation
run in §15.4.

### 15.3 Discriminative backbone LR — screened, doesn't help

Hypothesis: a flat `lr=1e-4` from step 1 risks the randomly-initialized
`FusionHead`'s early, noisy gradients corrupting the pretrained backbone
tail's features (fold-level evidence: val_loss climbed well past the best
epoch in most folds, e.g. fold 0 subtype: 1.98 at epoch 30 vs. a much
better epoch 12). Added `--backbone-lr` to `34_joint_fusion_train.py`: two
AdamW param groups, backbone tower params at `--backbone-lr` and
`FusionHead` params at `--lr`, `None` (default) keeping the original
single-group behaviour. Verified via a smoke test that both groups
genuinely update independently before running it for real.

Single-seed (42) screens, `backbone_lr=1e-5` / `lr=1e-4`, 100 epochs /
patience 20, vs. the flat-LR runs from §15.2:

| task | ablation | denom | flat LR | discriminative LR |
|---|---|---|---|---|
| subtype | wle_only | unpaired | 0.4589 ± 0.0550 | 0.4643 ± 0.0365 |
| subtype | fused | unpaired | 0.4963 ± 0.0371 | 0.5096 ± 0.0479 |
| binary | wle_only | unpaired | 0.6871 ± 0.0880 | 0.7193 ± 0.0609 |
| binary | fused | unpaired | 0.7490 ± 0.0728 | 0.7533 ± 0.0613 |

Subtype barely moved. On binary, `wle_only` moved up more than `fused` did
— **the discriminative-LR change helped the single-modality branch, not
fusion specifically**, narrowing the `fused`-`wle_only` gap (+0.062 → +0.034
unpaired) rather than widening it. A genuine, useful negative result:
protecting the pretrained backbone from early disruption helped the model
overall, but didn't make fusion more valuable relative to WLE alone. Not
carried into the confirmation run below.

### 15.4 3-seed confirmation (binary, flat LR, 100 epochs / patience 20) — CONFIRMED

Seeds 123 and 7 added to seed 42's binary/flat-LR run (§15.2, the config
with the largest single-seed gap):

| ablation | denom | seed 42 | seed 123 | seed 7 | **3-seed mean ± std** |
|---|---|---|---|---|---|
| wle_only | unpaired | 0.6871 | 0.7223 | 0.7365 | **0.7153 ± 0.0208** |
| wle_only | paired-only | 0.6409 | 0.7031 | 0.7176 | **0.6872 ± 0.0333** |
| **fused** | unpaired | 0.7490 | 0.7609 | 0.7690 | **0.7596 ± 0.0082** |
| fused | paired-only | 0.7207 | 0.7573 | 0.7626 | **0.7469 ± 0.0186** |

Noise-floor check (`sqrt(std_a² + std_b²)`):
- Unpaired: delta +0.0443, noise floor 0.0223 (`sqrt(0.0208²+0.0082²)`) — delta is **~2x** the floor. **CONFIRMED.**
- Paired-only: delta +0.0597, noise floor 0.0381 (`sqrt(0.0333²+0.0186²)`) — delta is **~1.6x** the floor. **CONFIRMED.**
- Cross-checked on macro-F1 (not just balanced accuracy): delta +0.0635 vs. noise floor ~0.0373 — also confirmed, so this isn't a single-metric artifact.

This is a materially different outcome from the original detector-based
fusion (§5i), where the apparent `fused` gain was a majority-class accuracy
effect that vanished on balanced accuracy specifically (`wle_only` actually
*beat* `fused` there on that metric). Here, balanced accuracy is the
primary metric throughout and `fused` still wins, confirmed at 3 seeds — a
genuine, class-general fusion benefit on the binary task, from a jointly
end-to-end-trained (not frozen-feature) two-tower rig.

**Caveats, stated plainly rather than rounded off:**
- **Subtype (5-way) was never confirmed** across any of the 3 configs tried
  (§15.2, §15.3) — this result is binary-task-specific.
- **The counterfactual gate does not yet cover this architecture.**
  `experiments/src/27_classifier_counterfactual.py` is built for a
  single-image, single-modality classifier; a patient-level, multi-frame,
  multi-modality model needs its own adaptation (e.g. the 5 image-variant
  substitutions applied per frame before pooling) before this result counts
  as genuine signal rather than merely accuracy-confirmed — not built yet,
  required follow-up, not skipped quietly.
- Absolute numbers (~0.72-0.76 balanced accuracy) remain well below §14.4's
  frozen-feature fusion on the equivalent comparison, consistent with
  §15.2's explanation (weaker backbone, no lesion localization, head
  trained from scratch in one run rather than reusing individually-converged
  checkpoints).

### 15.5 Multi-seed probability ensemble — new campaign best, zero additional training

Zero-cost follow-up on §15.4's 3 confirmed `--freeze partial` checkpoints
(15 fold checkpoints already on disk): built
`experiments/src/35_joint_fusion_multiseed_ensemble.py`, mirroring
`32_classifier_multiseed_ensemble.py`'s pattern at the patient level —
reconstruct each seed's own `patient_level_folds` split (a joint-fusion seed
reshuffles the fold assignment too, not just model init), collect one
out-of-fold `fused` prediction per seed keyed by `patient_key`, then average
softmax probabilities across the 3 seeds before scoring.

Reconstruction sanity check passed for all 3 seeds (matched each run's own
recorded 5-fold mean to within 0.0003-0.0004):

| | seed 42 | seed 123 | seed 7 | mean ± std |
|---|---|---|---|---|
| individual OOF `fused`/unpaired bal | 0.7486 | 0.7606 | 0.7691 | 0.7594 ± 0.0084 |

| | unpaired (641 patients) | paired-only (421 patients) |
|---|---|---|
| ensemble acc | 0.8237 | 0.8409 |
| **ensemble balanced accuracy** | **0.7741** | 0.7575 |
| ensemble macro-F1 | 0.7807 | 0.7661 |

Unpaired balanced accuracy: **0.7741**, up **+0.0147** from the 3-seed mean
(0.7594) for zero additional GPU training — a new campaign best for this
architecture at the time it was run (superseded a few hours later by
§15.6's `--freeze none` result, 0.8007). Paired-only came in lower (0.7575)
than unpaired this round, the reverse of most other configs in this
section — noted but not investigated further given `--freeze none` was
already the clearer thread to pursue by that point.

**Re-run on the `--freeze none` checkpoints (§15.6, the config that
superseded this one) — new overall campaign best**:

| | unpaired (641 patients) | paired-only (421 patients) |
|---|---|---|
| ensemble acc | 0.8471 | 0.8622 |
| **ensemble balanced accuracy** | **0.8160** | 0.8043 |
| ensemble macro-F1 | 0.8150 | 0.8043 |

Individual-seed OOF reconstruction (sanity check, matched each run's own
recorded mean to within 0.0001-0.0002): 0.8029, 0.8076, 0.7915 (mean
0.8007 ± 0.0068). Ensembling lifts this to **0.8160**, +0.0153 over the mean
individual seed — the highest number this architecture has produced,
free. (Naming note: `35_joint_fusion_multiseed_ensemble.py`'s output
filename originally keyed only on `task`, not on which specific run_names
were ensembled — this second run silently overwrote the first result's JSON
on disk before the fix landed, the same run_name-collision bug class hit
repeatedly elsewhere this campaign. No data was actually lost since the
`--freeze partial` numbers above were already transcribed into this section
before the overwrite; the script now tags its output by the ensembled
config, not just the task, so this can't recur.)

### 15.6 `--freeze none` (full backbone unfreeze) — CONFIRMED, new best config

Follow-up lever screen after §15.4's confirmation: does letting gradients
reach the *entire* backbone (not just the last two `features` stages) help
further, at the cost of more overfitting risk on this small a dataset?
Single-seed screen (42) came back the strongest single fold of the whole
campaign (0.8591), prompting immediate 3-seed confirmation rather than
waiting, per the plan's own single-seed-then-confirm discipline applied to
whichever config looks most promising in the moment:

| ablation | denom | seed 42 | seed 123 | seed 7 | **3-seed mean ± std** |
|---|---|---|---|---|---|
| wle_only | unpaired | 0.7656 | 0.7734 | 0.7543 | **0.7644 ± 0.0093** |
| wle_only | paired-only | 0.7536 | 0.7545 | 0.7149 | **0.7410 ± 0.0185** |
| **fused** | unpaired | 0.8028 | 0.8077 | 0.7916 | **0.8007 ± 0.0067** |
| fused | paired-only | 0.8034 | 0.8030 | 0.7666 | **0.7910 ± 0.0173** |

**Noise-floor checks:**
- `fused` vs. `wle_only` (within `--freeze none`), unpaired: delta +0.0363,
  noise floor `sqrt(0.0093²+0.0067²)` ≈ 0.0115 — **CONFIRMED**, ~3.2x the
  floor (the cleanest margin of any comparison in this section).
- `fused` vs. `wle_only`, paired-only: delta +0.0500, noise floor
  `sqrt(0.0185²+0.0173²)` ≈ 0.0253 — **CONFIRMED**, ~2x the floor.
- **`--freeze none`'s `fused` vs. `--freeze partial`'s `fused` (§15.4),
  direct between-config comparison**: delta +0.0411, noise floor
  `sqrt(0.0067²+0.0082²)` ≈ 0.0106 — **CONFIRMED**. Full unfreeze is a
  genuinely better base config for this architecture, not just a noisier
  version of the same result.

`--freeze none` **supersedes §15.4 as the best confirmed configuration** for
this architecture. The overfitting risk flagged when this lever was first
proposed didn't materialize at 3 seeds — worth remembering that "more
trainable parameters on a small dataset" isn't automatically worse when the
dataset is patient-frame-multiplied (up to `max_frames` independently-
augmented views per patient per epoch) rather than a single static image per
example. Every subsequent lever in this section (counterfactual gate,
`--lesion-crop`, `--bg-remove-prob`) should be evaluated against `--freeze
none` checkpoints going forward, not the now-superseded `--freeze partial`
ones.

**Counterfactual gate, adapted and run (closes §15.4's open caveat)**: built
`experiments/src/36_joint_fusion_counterfactual.py`, the patient-level,
multi-frame, two-modality adaptation of `27_classifier_counterfactual.py`'s
protocol flagged as required follow-up in §15.4 — swaps both towers to the
same `ablation_inputs.build_five_variants` variant simultaneously per
patient, while keeping each patient's frame *selection* identical to a
normal eval pass (only pixel content changes). Run against the confirmed
`--freeze none` seed-42 checkpoint:

| variant | balanced accuracy | retention vs. full |
|---|---|---|
| full (real images, both towers) | 0.803 | — |
| lesion (crop to box + padding) | 0.632 | 44% |
| bg_mask (lesion erased, background kept) | 0.644 | 48% |
| bg_patch (pure background, no lesion anywhere) | 0.627 | 42% |
| **boxmap (black canvas, box outline only — zero real tissue)** | **0.494** | **-2%** |

`full` (0.803) closely reproduces the training run's own recorded `fused`
balanced accuracy (0.8028), the expected sanity check. With every trace of
real tissue removed (`boxmap`), balanced accuracy falls to essentially exact
chance (0.494 vs. chance 0.500) — no exploitable signal survives from box
position/geometry alone. `bg_patch` (real background, zero lesion) drops to
42% retention, comfortably below the "survives or grows" shortcut threshold.
**Gate: PASSED — the confirmed win is genuine lesion-dependent signal, not
shortcut-reading.** (`lesion` scoring below `bg_mask` — 44% vs. 48% — is
plausibly a scale/framing artifact: this checkpoint was trained on full,
un-cropped frames, so evaluating it on a cropped-and-resized `lesion` variant
is mildly out-of-distribution for scale in a way `bg_mask`, which keeps the
original framing, is not.)

### 15.7 Other single-seed lever screens this round — no further wins

Screened alongside `--freeze none`, all single-seed (42), binary task,
against the §15.4 `--freeze partial` baseline (`fused` unpaired 0.7490):

| lever | fused (unpaired) | fused (paired-only) | verdict |
|---|---|---|---|
| `--fusion symmetric` | 0.7593 | 0.7461 | no clear win — within noise of gated, narrows the fused-wle_only gap rather than widening it (0.045 vs. gated's 0.062), same pattern discriminative LR showed (§15.3) |
| `--paired-only` (train on the 421 paired patients only) | 0.7634 | 0.7634 (identical by construction — no unpaired patients exist in this arm) | comparable delta to full-pool training; fused edges up slightly but single-seed, not pursued to confirmation given `--freeze none` was the clearer win |

`SymmetricCrossAttentionFusion` also surfaced a real, separate finding along
the way: an intermittent `val_loss=nan` during training, present in *every*
joint-fusion run since §15.2 (including the ones behind the §15.4
confirmation), not something introduced by the symmetric-fusion change.
Root-caused via direct diagnosis: `train_loss` never went NaN (0/206 epochs
checked); loading the actual saved `fold0_best.pt` from the confirmed
seed-42 run and running 180 repeated eval passes on real validation data
produced zero NaN events; neither best-checkpoint selection nor the LR
scheduler ever reads `val_loss` (both use `bal`, which stayed smooth through
every NaN epoch in the logs). Read as a transient BatchNorm running-statistic
artifact (plausibly from the zero-padded EUS blocks on WLE-only patients)
that occasionally pushes one epoch's eval-mode forward pass into fp16
overflow inside `F.cross_entropy`'s internal log-softmax under autocast,
self-correcting by the next epoch as running stats update further —
**cosmetic, does not affect saved checkpoints or any reported number in this
section.** `cadsel/fusion.py::SymmetricCrossAttentionFusion.forward` was
still given an fp32-forced attention/layernorm block as a general hardening
fix (a no-op for `cadsel.train_fusion`, which never uses autocast) since it's
good practice regardless, even though it turned out not to be the actual
source.

### 15.8 `--lesion-crop` (crop each frame to its lesion box) — EUS confirmed better, fused not confirmed better than the current best

User-proposed augmentation lead: crop each frame to its lesion box + 10%
context padding before the standard augmentation pipeline, instead of
feeding the whole frame — directly targets the "global average pooling
mostly encodes context" weakness `cadsel/fusion.py`'s own docstring cites as
the reason the original detector-based fusion used ROI features instead.
Added to `cadsel/joint_fusion_data.py` (`_boxes_for`/`_lesion_crop`,
duplicated from `experiments/ablation_inputs.py`'s box logic rather than
imported, for the same layering reason `joint_fusion_model.py` duplicates
`_freeze_all_but`) and wired through `--lesion-crop` in
`34_joint_fusion_train.py`. Single-seed screen (42, on top of the confirmed
`--freeze none` base) came back the strongest single fold of the whole
campaign (0.8631), prompting immediate 3-seed confirmation:

| ablation | denom | seed 42 | seed 123 | seed 7 | **3-seed mean ± std** |
|---|---|---|---|---|---|
| wle_only | unpaired | 0.7685 | 0.7680 | 0.7833 | **0.7733 ± 0.0071** |
| eus_only | (paired) | 0.6820 | 0.7248 | 0.7102 | **0.7057 ± 0.0178** |
| **fused** | unpaired | 0.8169 | 0.7938 | 0.8222 | **0.8110 ± 0.0123** |
| fused | paired-only | 0.8003 | 0.7740 | 0.8197 | 0.7980 ± 0.0188 |

**Noise-floor checks:**
- `fused` vs. `wle_only` (within `--lesion-crop`): delta +0.0377, noise
  floor `sqrt(0.0123²+0.0071²)` ≈ 0.0142 — **CONFIRMED**, ~2.65x the floor.
- **`eus_only` with `--lesion-crop` vs. `eus_only` without it** (§15.6's
  `--freeze none` baseline, 0.6197 ± 0.0216): delta **+0.0860**, noise floor
  `sqrt(0.0178²+0.0216²)` ≈ 0.0280 — **CONFIRMED**, ~3.1x the floor. The
  cleanest, strongest result of this whole lead — lesion-cropping genuinely
  and substantially helps the EUS tower specifically, consistent with EUS
  frames plausibly carrying more irrelevant background/probe artifact per
  frame than WLE, so restricting to the lesion region removes more noise.
- **`fused` with `--lesion-crop` vs. `fused` without it** (§15.6's `--freeze
  none` baseline, 0.8007 ± 0.0067) — the comparison that would make this a
  new best config: delta +0.0103, noise floor `sqrt(0.0123²+0.0067²)` ≈
  0.0140. **Delta is smaller than the noise floor — NOT confirmed.**

**Honest read**: `--lesion-crop` gives the EUS tower a real, substantial,
cleanly-confirmed improvement, but that doesn't propagate into a
statistically confirmed improvement of the *fused* number over the current
best (`--freeze none` alone). Plausible explanation: `GatedFusion`'s own
docstring notes its gate structurally favours WLE (`(1-gate)*a` always falls
back to WLE specifically, never EUS) — a stronger EUS branch doesn't
translate one-for-one into a better fusion under a fusion mechanism that
isn't modality-symmetric to begin with. `--freeze none` (§15.6, without
lesion-crop) remains the best *confirmed* fused configuration; `--lesion-crop`
is recorded as a real, confirmed win for the EUS single-modality signal
specifically, not (yet) for the headline fused metric.

**Multi-seed ensemble of the 3 lesion-crop checkpoints — new overall campaign
best.** (One real bug caught and fixed in the process: `35_joint_fusion_
multiseed_ensemble.py`'s `collect_oof_probs` never read `lesion_crop`/
`bg_remove_prob` from the run's own metadata, so its first attempt built an
eval dataset with plain full frames — completely out-of-distribution for a
model trained exclusively on lesion crops. Sanity check caught it
immediately: reconstructed OOF balanced accuracy came back at 0.52-0.54
against a recorded 0.79-0.82, an unmissable mismatch, not a subtle one.
Fixed by threading both flags through from the run JSON; re-run then
matched each seed's recorded mean to within 0.0001-0.0004, as expected.)

| | unpaired (641 patients) | paired-only (421 patients) |
|---|---|---|
| ensemble acc | 0.8487 | 0.8646 |
| **ensemble balanced accuracy** | **0.8219** | 0.8169 |
| ensemble macro-F1 | 0.8183 | 0.8111 |

Individual-seed OOF: 0.8170, 0.7942, 0.8220 (mean 0.8111 ± 0.0121). Ensemble
reaches **0.8219**, +0.0108 over the mean individual seed — higher than the
plain `--freeze none` ensemble (§15.5, 0.8160), making lesion-crop the best
overall config once ensembling is accounted for, even though its fused
number didn't clear 3-seed confirmation against `--freeze none` on its own
(above). Worth the distinction: "best ensemble result" and "confirmed best
single-model config" are two different claims in this section, and only the
latter carries the noise-floor discipline this campaign uses throughout.

**Follow-up: does `--fusion symmetric` let lesion-crop's EUS gain reach the
fused output?** Targeted hypothesis test, single seed 42: if `GatedFusion`'s
WLE-favoring gate really is what caps lesion-crop's fused improvement, a
modality-symmetric fusion mechanism combined with lesion-crop should do
better than gated+lesion-crop. Result: **it doesn't.**

| | wle_only | eus_only | fused |
|---|---|---|---|
| gated + lesion-crop (seed 42) | 0.7685 | 0.6820 | 0.8169 |
| symmetric + lesion-crop (seed 42) | 0.7786 | **0.7317** | **0.8118** |

`eus_only` itself is genuinely higher under symmetric fusion (+0.0497) —
consistent with the WLE-bias story at the pooled-descriptor level — but
`fused` still comes out *lower* than the plain gated version, not higher.
So the specific hypothesis is wrong: `GatedFusion`'s WLE bias isn't the
(sole) reason lesion-crop's EUS gain fails to propagate into a confirmed
fused improvement — something else about how the two streams combine caps
it regardless of fusion mechanism. Not pursued to 3-seed confirmation given
the negative direction; `--fusion gated` (the default throughout this
section) remains the better choice.

### 15.9 `--bg-remove-prob` (stochastic background erasure) — single-seed negative

Second user-proposed augmentation lead, complementary to §15.8: rather than
always cropping to the lesion (a permanent framing change), stochastically
erase everything *outside* the lesion box on a fraction of training frames
only, forcing those specific frames to be classified without background
signal — directly motivated by Experiment 3's finding that labels are
68-95% recoverable from lesion-free crops alone. Filled with the frame's own
mean colour (not black) for the same reason `ablation_inputs.py`'s `bg_mask`
variant makes that choice: a stark black rectangle is itself an artificial
shortcut cue. Added `_bg_remove` to `cadsel/joint_fusion_data.py` and
`--bg-remove-prob` to `34_joint_fusion_train.py` (train-time only, mutually
exclusive with `--lesion-crop`).

Single-seed screen (42, `--freeze none` base, p=0.5):

| ablation | fused-config baseline (`--freeze none`, no aug) | `--bg-remove-prob 0.5` |
|---|---|---|
| wle_only (unpaired) | 0.7656 | 0.7363 |
| eus_only | 0.6081 | 0.6404 |
| **fused** (unpaired) | 0.8028 | **0.7631** |

Clearly negative direction for `fused` (−0.0397) and `wle_only` (−0.0293),
even below the 3-seed confirmed `--freeze none` mean (0.8007) — not a close
call, so not escalated to 3-seed confirmation. Plausible mechanism: unlike
`--lesion-crop` (crops *before* the standard augmentation, so
`RandomResizedCrop` always sees real lesion content), `--bg-remove-prob`
erases the background and then still runs `RandomResizedCrop` on top — at
p=0.5 a meaningful fraction of crops can land entirely inside the
mean-filled region, especially for small lesion boxes, producing a
near-featureless training example with a real label still attached. That's
noisy supervision, not useful regularization.

**Retry with a fix, not abandoned outright**: `_bg_remove` in
`cadsel/joint_fusion_data.py` was given a wider kept island (`pad_frac=0.3`
around each lesion box, vs. `_lesion_crop`'s 0.1) to make it far less likely
`RandomResizedCrop` lands entirely inside the erased region, and the
probability was lowered to p=0.3. Single-seed screen (42, `--freeze none`
base):

| config | wle_only (unpaired) | fused (unpaired) |
|---|---|---|
| plain `--freeze none` (no bg-removal) | 0.7656 | 0.8028 |
| `bg-remove-prob 0.5` (original) | 0.7363 | 0.7631 |
| `bg-remove-prob 0.3` + wider padding (retry) | 0.7573 | **0.7792** |

The fix genuinely helped — fused recovered from 0.7631 to 0.7792
(+0.0161), confirming the diagnosed mechanism was real — but it's still
below the plain baseline (0.8028) by a real margin (−0.0236), not a close
call. **Verdict: still negative even fixed — closing this thread.** The
lever doesn't help this architecture regardless of the crop-landing-on-
erased-region issue; not worth further probability sweeps.

(One resumability note along the way: this retry died mid-fold-1 when the
underlying session was interrupted, sitting idle for several days before
being noticed. `34_joint_fusion_train.py`'s per-epoch resume state
[built specifically after this campaign's earlier session-interruption
losses] picked it up cleanly — skipped fold 0 entirely since it was already
complete, and resumed fold 1 from its last saved epoch rather than
retraining either from scratch.)

### 15.10 Subtype 3-seed confirmation (discriminative backbone LR) — CONFIRMED, closing the open subtype thread

Subtype (5-way) was never confirmed across the 3 single-seed configs
screened in §15.2/§15.3. Given the discriminative-LR config (§15.3) had the
highest single-seed `fused` number of the three, seeds 123 and 7 were added
to close out whether subtype would confirm the way binary eventually did —
the same escalation pattern that worked for §15.6's `--freeze none`.

(One relaunch note: seed 123's first attempt died mid-fold-3, 3/5 folds
completed, when the underlying session/environment restarted — the same
class of interruption noted earlier in this campaign for background jobs.
Relaunched from scratch rather than resumed, since no checkpoint-resume path
exists; folds 0-2 reproduced byte-identical balanced-accuracy values to the
interrupted attempt, confirming the relaunch was a clean re-run, not a
different draw.)

| ablation | denom | seed 42 | seed 123 | seed 7 | **3-seed mean ± std** |
|---|---|---|---|---|---|
| wle_only | unpaired | 0.4643 | 0.4453 | 0.5017 | **0.4704 ± 0.0234** |
| wle_only | paired-only | 0.4259 | 0.4122 | 0.5002 | **0.4461 ± 0.0387** |
| **fused** | unpaired | 0.5096 | 0.4812 | 0.5282 | **0.5063 ± 0.0193** |
| fused | paired-only | 0.4873 | 0.4631 | 0.5345 | **0.4950 ± 0.0296** |

**Noise-floor checks:**
- Unpaired, balanced accuracy: delta +0.0359, noise floor
  `sqrt(0.0234²+0.0193²)` ≈ 0.0304 — **CONFIRMED**, ~1.18x the floor (a
  real margin, but much tighter than the binary confirmations in this
  section).
- Unpaired, macro-F1 (cross-check): wle_only 0.3386 ± 0.0202, fused
  0.4055 ± 0.0110 — delta +0.0669, noise floor `sqrt(0.0202²+0.0110²)` ≈
  0.0230 — **CONFIRMED**, ~2.9x the floor. Macro-F1 gives a substantially
  cleaner margin than balanced accuracy here, so the overall confirmation
  doesn't rest on one borderline metric.
- Paired-only, balanced accuracy: delta +0.0489, noise floor
  `sqrt(0.0387²+0.0296²)` ≈ 0.0487 — clears by **0.0002**, essentially the
  noise floor itself. Reported honestly as a razor-thin, not-clean
  confirmation on this specific denominator, rather than rounded up.

**Verdict: subtype is CONFIRMED on the unpaired denominator** (both metrics,
comfortably on macro-F1), closing the thread that was open since §15.2. The
paired-only read is too close to call cleanly and shouldn't be leaned on.

**Counterfactual gate, run against this checkpoint (seed 42) — PASSED**:

| variant | balanced accuracy | retention vs. full |
|---|---|---|
| full | 0.510 | — |
| lesion (crop to box + padding) | 0.377 | 57% |
| bg_mask (lesion erased, background kept) | 0.306 | 34% |
| bg_patch (pure background, no lesion) | 0.315 | 37% |
| **boxmap (black canvas, box outline only)** | **0.206** | **2%** |

`full` (0.510) closely reproduces the recorded fused/unpaired mean (0.5096,
§15.3's single-seed screen this checkpoint came from). With all real tissue
removed (`boxmap`), balanced accuracy falls to essentially exact chance
(0.206 vs. 0.200) — no exploitable signal from box geometry alone.
`bg_patch` retention (37%) is comfortably below the "survives/grows"
shortcut threshold. **Genuine lesion-dependent signal, mirroring the binary
result (§15.6) exactly** — closes the counterfactual-gate caveat for
subtype the same way it was closed for binary.

(Run against the discriminative-LR/`--freeze partial` checkpoint
specifically, not the newer `--lesion-crop`/`--freeze none` config from
§15.8.1 below — a lesion-crop-trained model would need its own protocol
variant here, since it never sees uncropped content during training, so a
low `bg_patch` retention on it could reflect distribution shift as much as
genuine lesion-dependence. Not built this round; noted as an open
methodological question rather than assumed away.)

**Superseded almost immediately by `--lesion-crop` on `--freeze none`** —
see §15.8.1 below, run once the binary-side `--lesion-crop` lead (§15.8)
was extended to subtype. `--freeze none` alone was never separately
screened on subtype at 3 seeds before lesion-crop was tried on top of it
(the binary-side escalation order — screen `--freeze none` alone, confirm
it, *then* layer `--lesion-crop` on top — wasn't repeated step-for-step on
subtype; lesion-crop went straight on top of `--freeze none` given how well
that combination had just worked on binary). Recorded as a limitation: the
subtype comparison below is against §15.3's confirmed `--freeze partial`
config, not an isolated `--freeze none`-alone baseline the way §15.6's
binary comparison was.

### 15.8.1 `--lesion-crop` on subtype — CONFIRMED, new subtype best

Single-seed screen (42, `--freeze none` base) came back the best subtype
result of the whole campaign (fused 0.5548 unpaired), prompting the same
immediate 3-seed escalation used throughout this section:

| ablation | denom | seed 42 | seed 123 | seed 7 | **3-seed mean ± std** |
|---|---|---|---|---|---|
| wle_only | unpaired | 0.5273 | 0.4694 | 0.5179 | **0.5049 ± 0.0254** |
| eus_only | (paired) | 0.3713 | 0.3709 | 0.3503 | 0.3642 ± 0.0097 |
| **fused** | unpaired | 0.5548 | 0.5135 | 0.5574 | **0.5419 ± 0.0201** |
| fused | paired-only | 0.5713 | 0.5046 | 0.5432 | 0.5397 ± 0.0273 |

**Noise-floor checks:**
- `fused` vs. `wle_only` (within this config): delta +0.0370, noise floor
  `sqrt(0.0254²+0.0201²)` ≈ 0.0324 — **CONFIRMED**, ~1.14x the floor.
- Macro-F1 cross-check: wle_only 0.3854 ± 0.0273, fused 0.4379 ± 0.0266 —
  delta +0.0525, noise floor ≈ 0.0381 — **CONFIRMED**, ~1.38x the floor.
- **Vs. the previously confirmed subtype best** (§15.10, discriminative-LR/
  `--freeze partial`: fused 0.5063 ± 0.0193): delta +0.0356, noise floor
  `sqrt(0.0201²+0.0193²)` ≈ 0.0279 — **CONFIRMED**, ~1.28x the floor. This
  is a genuine new subtype best, not just a noisier read of the same
  ceiling — mirroring exactly what happened on the binary task (§15.6/
  §15.8): `--freeze none` (+ `--lesion-crop`) beats whatever `--freeze
  partial` config held the record before it.

**Verdict: `--lesion-crop` on `--freeze none` is the new confirmed subtype
best**, at 0.5419 ± 0.0201 fused/unpaired — up from 0.5063 (§15.10) and
well above the original 30-epoch screen's 0.5009 (§15.2) that opened this
whole thread.

**Multi-seed ensemble (free, zero additional training)**: sanity checks
matched each seed's recorded mean closely (0.5574/0.5548, 0.5209/0.5135,
0.5572/0.5574).

| | unpaired (641 patients) | paired-only (421 patients) |
|---|---|---|
| ensemble acc | 0.6037 | 0.6485 |
| **ensemble balanced accuracy** | **0.5706** | 0.5995 |
| ensemble macro-F1 | 0.4753 | 0.4903 |

Individual-seed OOF mean 0.5452 ± 0.0172 → ensemble **0.5706**, +0.0254 —
the new overall subtype campaign best. Not yet run through the
counterfactual gate — as noted in §15.10, testing a `--lesion-crop`-trained
model against the gate's raw-frame variants needs its own protocol
adaptation first (the model never sees uncropped content during training),
so this is left as an open methodological question rather than run naively.

### 15.11.1 Patient-level multi-frame TTA — null/negative, closing the TTA question for this architecture

One more genuinely new idea, distinct from the flip/5-crop TTA already
found null on the plain classifiers (§12.4/§14.6): this architecture's eval
mode fixes ONE deterministic frame-subset draw per patient (seeded by
patient index) whenever a patient has more raw frames than `max_frames`. A
single-image classifier never had this axis — averaging predictions over
K *different* frame-subset draws per patient is specific to the multi-frame
design. Built `experiments/src/37_joint_fusion_frame_tta.py`: for each fold,
runs K=5 independent `JointPatientDataset` instances (draw 0 reproduces the
plain recorded eval exactly; draws 1-4 use different frame-selection seeds),
averages softmax probabilities across draws, and compares to the plain
single-draw balanced accuracy.

| config (campaign-best checkpoint, seed 42) | plain | TTA (k=5) | delta |
|---|---|---|---|
| binary (`--freeze none` + `--lesion-crop`) | 0.8169 ± 0.0333 | 0.8108 ± 0.0434 | −0.0061 (mixed sign, 2/5 folds up) |
| subtype (`--freeze none` + `--lesion-crop`) | 0.5553 ± 0.0343 | 0.5357 ± 0.0476 | −0.0196 (4/5 folds down) |

Both null-to-negative. Plausible reason: `--lesion-crop` already constrains
*which region* gets shown (always the lesion box + padding) — the only
thing that varies between draws is *which raw frame* gets picked, and
different frames of the same lesion are similarly informative rather than
meaningfully complementary, so averaging mostly adds sampling noise from
alternate draws rather than new signal. Closes the TTA question for this
architecture the same way §12.4/§14.6 closed it for the single-image
classifiers — no TTA variant tried anywhere in this campaign (flip, 5-crop,
multi-frame) has helped any checkpoint.

### 15.11 Domain-specific backbone pretraining (WLE/EUS) — researched, not pursued

Follow-up idea: pretrain the two towers separately on other single-modal
datasets in this domain before assembling the joint rig, rather than
starting both from plain ImageNet weights. This project already ran this
exact experiment once, for the EUS **detector** backbone:

- **RadImageNet** (general/abdominal ultrasound, `cadsel/convert_radimagenet_weights.py`)
  — isolated and tested against a COCO-pretrained control on EUS detection,
  3 seeds: 0.2844 vs. 0.3559 AP@0.5, a **confirmed negative** delta
  (−0.0716, clears the 0.0487 noise floor), consistently across every seed
  (§5p). General/abdominal ultrasound pretraining apparently doesn't
  transfer to endoscopic ultrasound's different imaging characteristics.
  URFM (general ultrasound) was scouted as the next-closest EUS candidate
  and assessed as "a repeat of RadImageNet's already-failed bet," not
  pursued (§5w).
- **GastroNet-5M** (in-domain GI-endoscopy SSL, ResNet-50/ViT-B, 4.82M
  images / ~500K procedures) — genuinely in-domain for WLE, but the weights
  sit behind a commercial platform gate ("Theta Vision Cortex") requiring
  external account setup outside what this session can do autonomously
  (§5w) — an open lead, not a tested one.

The user then supplied the CAD-SEL paper's own Table 1 (comparison against
prior SEL-related datasets), which adds a practical blocker independent of
the domain-transfer question above. Every comparison dataset's "publicly
available" column reads **"No"** except He et al. 2023 ("Planned") and
CAD-SEL itself ("Yes"):

Full transcription of the paper's Table 1 (Cls = classification labels, Det =
detection/bounding-box annotations; "—" = not explicitly reported in the
source):

| dataset | year | public? | location | modality | centers | patients | images | classes | labels |
|---|---|---|---|---|---|---|---|---|---|---|
| Oh et al. | 2021 | No | Gastric | EUS | 1 | 168 | 546 | 2 | Cls+Det |
| Yang et al. | 2022 | No | Upper GI | EUS | 4 | 752 | 10,439 | 2 | Cls |
| Hirai et al. | 2022 | No | Upper GI | EUS | 12 | 631 | 16,110 | 5 | Cls |
| Kim et al. | 2020 | No | Gastric | EUS | 1 | — | 1,117 | 3 | Cls |
| Seven et al. | 2022 | No | Upper GI | EUS | 1 | 145 | 1,362 | 2 | Cls |
| Minoda et al. | 2020 | No | Gastric | EUS | 1 | 60 | — | 2 | Cls |
| Zhu et al. | 2022 | No | Gastric | WLE+EUS | 2 | 1,366 | — | 3 | Cls |
| Dong et al. | 2024 | No | Upper GI | EUS | 5 | 1,342 | 5,419 | 2 | Cls+Det |
| He et al. | 2023 | Planned | Upper GI | EUS | 1 | 514 | — | 2 | Cls+Det |
| **CAD-SEL (this project)** | 2025 | **Yes** | Colorectal | WLE+EUS | 6 | 641 | 4,912 | 5 | Cls+Det |

None of these are actually downloadable, so none can be used for
pretraining as things stand — a hard blocker distinct from (and in addition
to) RadImageNet's confirmed domain-transfer failure and GastroNet-5M's
access gate. Also worth noting for future reference: none of the
comparison datasets are WLE-only either (they're all EUS or WLE+EUS), so
even a released one wouldn't cleanly serve a "single-modal WLE pretraining"
half of this idea — only Zhu et al. 2022 is multi-modal like CAD-SEL, and
it's also private. **Verdict: not pursued** — no accessible, domain-matched
pretraining dataset currently identified for either tower. Worth revisiting
if He et al.'s "Planned" release materializes.

**Critical files**: `cadsel/joint_fusion_data.py`, `cadsel/joint_fusion_model.py`,
`experiments/src/34_joint_fusion_train.py` (all new, §15.1); the
`--backbone-lr` addition to `34_joint_fusion_train.py` (§15.3);
`experiments/src/35_joint_fusion_multiseed_ensemble.py` (new, §15.5);
`experiments/src/36_joint_fusion_counterfactual.py` (new, §15.6); the
`--lesion-crop`/`--bg-remove-prob` additions to `34_joint_fusion_train.py`
and `cadsel/joint_fusion_data.py` (§15.7); the fp32-forced attention block
in `cadsel/fusion.py::SymmetricCrossAttentionFusion` (§15.7, hardening fix,
unrelated to its trigger).

---

## 16. EfficientNet-B0 as a per-modality Track B detector backbone

User-proposed follow-up to §15's joint classifier fusion work: adapt the
same EfficientNet-B0 backbone for detection, as a genuinely new Track B
backbone option ("dual" meaning one independently-trained EfficientNet-B0
detector per modality, matching how every other backbone in this project is
trained — not the §15 fusion architecture itself, which doesn't apply to
per-frame bounding-box prediction since WLE and EUS frames have no
correspondence to fuse over).

### 16.1 Architecture

New `EFFICIENTNET_MODEL_REGISTRY` in `cadsel/config.py`
(`fasterrcnn_efficientnet_b0`) and `build_efficientnet_detection_model` /
`_efficientnet_b0_fpn_backbone` in `cadsel/models.py`, wired into
`cadsel/train.py` and `cadsel/evaluate_tta.py`'s dispatch the same way
DINOv3 already is. Unlike DINOv3 (a ViT, single patch-token resolution,
§5o's single-scale fallback), EfficientNet-B0 **is** a genuine conv
hierarchy — verified empirically via a forward-shape probe rather than
assumed (`efficientnet_b0().features`, stages 0/2/3/4/6 halve spatial
resolution: 32/16/24/40/80/112/192/320/1280 channels at strides
2/2/4/8/16/16/32/32/32) — so it gets a real multi-level FPN, a fairer
comparison to the existing ResNet50-FPN/MobileNetV3-FPN backbones than a
single feature map would be.

torchvision ships no `fasterrcnn_efficientnet_b0_fpn` constructor (unlike
the stock MobileNetV3/ResNet50 options), so the backbone+FPN is hand-
assembled the way torchvision's own `_mobilenet_extractor` does it for
MobileNet — except EfficientNet's blocks don't carry MobileNet's `_is_cn`
stage-boundary marker (verified: `False` on every stage), so stage
indices/channels came from the probe above, not that helper. Four FPN
levels are picked, one per resolution (the deepest/most-refined stage at
each, mirroring `resnet_fpn_backbone`'s C2-C5 convention): stage 2 (/4,
24ch), stage 3 (/8, 40ch), stage 5 (/16, 112ch — deeper than stage 4's
80ch at the same resolution), stage 8 (/32, 1280ch — deeper than stages
6/7 at the same resolution). `LastLevelMaxPool` adds a 5th (P6) level for
the RPN. `num_classes` set directly at construction (same as
`build_dinov3_detection_model`/`build_fcos_model` — no COCO-pretrained head
to swap out of for a from-scratch backbone+FPN). No `FrozenBatchNorm2d`
conversion (unlike torchvision's own COCO-pretrained backbones) — a
deliberate simplification given this project's already-small detection
batch sizes, not an oversight.

**One real bug caught mid-implementation**: the first draft of the channel-
count logic used `isinstance(backbone[i], nn.Sequential)` to distinguish
MBConv stages from the final head conv — but `Conv2dNormActivation`
(stage 8) is *also* an `nn.Sequential`, and its last child is a bare `SiLU`
activation with no `out_channels` attribute, so that check would have
silently returned the wrong channel count for stage 8 specifically. Caught
before it ever ran by checking `hasattr(block, "out_channels")` empirically
rather than trusting the `isinstance` branch; fixed by reading the real
channel count off a live forward-shape probe (robust to any block's
internal structure) instead of introspecting submodule attributes.

Verified at two levels before any real training: a synthetic forward/
backward/eval pass (21.3M trainable params, all 4 standard FasterRCNN
losses present, backward+optimizer step clean, valid eval-mode output
structure), then a real 1-epoch training run on actual WLE data end-to-end
through `cadsel/engine.py`'s standard AP@0.5 evaluation — both clean.

### 16.2 `batch_size` OOM — a real gap in the initial verification, found and fixed

The registry's first `batch_size` value (6, picked by analogy to the other
entries without being empirically load-tested at full scale — the smoke
tests above only ever exercised batch_size 2 at a smaller image size) OOM'd
partway through a real 100-epoch run: sporadic *recoverable* CUDA
allocation-failure warnings appeared as early as epoch 1, then a *fatal*
OOM crashed the run at epoch 10, inside the FPN's lateral-connection add
(`torchvision/ops/feature_pyramid_network.py`, `last_inner = inner_lateral
+ inner_top_down`). That pattern — early recoverable warnings, later fatal
crash, not a one-shot "this batch is simply too big" failure on the very
first step — is the signature of GPU memory fragmentation accumulating
over time on a tight 8GB card, consistent with `cadsel/config.py`'s own
comment that every registry `batch_size` is meant to be "tuned for an 8GB
GPU" — this one hadn't actually been.

Lowered to `batch_size=4` (matching `fasterrcnn_resnet`'s own conservative
value) and re-validated with a real 20-epoch run specifically chosen to run
well past the epoch-10 failure point: **zero OOM warnings**, full 20 epochs
clean, AP@0.5 climbing to ~0.27 by the end (a normal, unremarkable
trajectory for that few epochs — not evidence about final quality, just
confirmation the fix holds). `batch_size=4` is now the registry default.

### 16.3 First full result — competitive, not a clear win, single seed only

100 epochs, WLE/multiclass, seed 42 (registry defaults: `batch_size=4`,
`img_size=640`), ~3.5 hours, best epoch 46:

| model | AP@0.5 |
|---|---|
| Mobilenet (Table 4 baseline) | 0.4098 |
| Mobilenet_320 | 0.3809 |
| **EfficientNet-B0 + FPN (new)** | **0.3902** |

Lands *between* the two existing MobileNet variants — competitive with,
but not better than, the current best WLE/multiclass backbone. Per-class:
NET 0.732 (the majority class, driving most of the aggregate), Lipoma
0.497, Leiomyoma 0.218, NonNeoplasm 0.114 — the same "rare classes struggle
most" pattern already documented for every other backbone in this project,
not something new to EfficientNet-B0 specifically.

**Verdict: a genuinely working new backbone option, not (yet) a confirmed
lever.** Single seed only — per this campaign's own discipline throughout
(§10-§15), this would need ≥3 seeds and a noise-floor check against
Mobilenet's own seed-to-seed spread before either "competitive" or "worse"
could be called confirmed rather than a single-run read. Not pursued to
multi-seed confirmation in this round; a reasonable next step if this
backbone is revisited (WLE and EUS both, matching how every other Track B
backbone was eventually covered).

**Critical files**: `cadsel/config.py` (`EFFICIENTNET_MODEL_REGISTRY`, new);
`cadsel/models.py` (`_efficientnet_b0_fpn_backbone`,
`build_efficientnet_detection_model`, new); `cadsel/train.py`,
`cadsel/evaluate_tta.py` (dispatch wiring, same pattern as DINOv3).
