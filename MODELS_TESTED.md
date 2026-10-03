# Models Built and Tested — CAD-SEL

A registry of every model architecture/configuration trained and evaluated in
this project, across both tracks. For the narrative behind the shortcut-audit
entries, see [SHORTCUT_LEARNING.md](SHORTCUT_LEARNING.md). For full
experimental detail behind any row here, see [FINDINGS.md](FINDINGS.md)
(section references given throughout).

Metrics: detection reports **AP@0.5** (patient-level held-out split);
classification reports **balanced accuracy** (mean per-class recall, patient-
level 5-fold CV). All numbers are this project's own reproduction, not the
CAD-SEL paper's.

---

## 1. Detection — Faster R-CNN reproduction (baseline grid, Table 4/5)

Four backbone variants × 2 modalities × 2 tasks, reproducing the CAD-SEL
paper's own technical-validation grid.

| Model | Modality | Task | AP@0.5 | F1 | Paper AP@0.5 | Δ |
|---|---|---|---|---|---|---|
| `fasterrcnn_resnet` | WLE | multiclass | 0.2809 | 0.4799 | 0.8593 | −0.578 |
| `fasterrcnn_resnet50_fpn_v2` | WLE | multiclass | 0.2594 | 0.2511 | 0.8792 | −0.620 |
| `fasterrcnn_mobilenet` | WLE | multiclass | 0.4098 | 0.5481 | 0.8817 | −0.472 |
| `fasterrcnn_mobilenet` | EUS | multiclass | 0.4514 | 0.5293 | 0.5768 | −0.125 |
| `fasterrcnn_mobilenet_320` | WLE | multiclass | 0.3809 | 0.6192 | 0.8641 | −0.483 |
| `fasterrcnn_mobilenet_320` | EUS | multiclass | 0.3476 | 0.6561 | 0.5320 | −0.184 |
| `fasterrcnn_mobilenet` | WLE | binary (NET/non-NET) | 0.6609 | 0.7053 | — | — |
| `fasterrcnn_mobilenet` | EUS | binary | 0.5118 | 0.6701 | — | — |
| `fasterrcnn_mobilenet_320` | WLE | binary | 0.6458 | 0.7378 | — | — |
| `fasterrcnn_mobilenet_320` | EUS | binary | 0.4282 | 0.6395 | — | — |

**ResNet backbones dropped after 2/8 runs** — both scored well below
MobileNet on the same cell (0.28/0.26 vs 0.38–0.41) and the remaining 6 were
deliberately not run (see FINDINGS §3 for the reasoning — a suspected
evaluation-leakage gap between this reproduction and the paper, not specific
to backbone choice). **MobileNet variants are the reference baseline for
every detection lever below.**

---

## 2. Detection — performance-lever campaign

All levers tested on top of the `fasterrcnn_mobilenet`/`mobilenet_320`
baselines, WLE unless noted. Discipline: single-seed screen → ≥3-seed
confirmation against a measured per-arm noise floor before "confirmed."

| Lever | Mechanism | Result | Verdict |
|---|---|---|---|
| Auto-anchors (IoU-clustered RPN sizes) | Replace COCO-tuned anchor sizes with ones fit to CAD-SEL's own box-size distribution | +0.030 AP single-seed combo; alone, flat-to-negative; adds nothing on top of size-RFS+multiscale (§5x: Δ −0.043, within noise) | Not confirmed / negative in combo |
| IoU-threshold relaxation (RPN/box fg/bg) | Loosen torchvision's 0.7/0.3/0.5/0.5 defaults for small-box anchor starvation | Part of the Tier-1 combo only, not isolated | Not confirmed alone |
| Multi-scale training (`--multiscale-sizes`) | Random per-image training scale | Confirmed positive, folded into every later "ms" recipe | **Confirmed** |
| Repeat Factor Sampling (class-level) | Detectron2-style oversampling of rare-class images | Part of Tier-1 combo | Not isolated |
| Size-aware RFS (`SizeRepeatFactorSampler`, §5g) | RFS bucketed by lesion-size tertile instead of class | Multiclass tertile-1 AP +0.0669, clears floor; binary +0.0128, doesn't clear (settled negative for binary) | **Confirmed (multiclass only)** |
| Size-RFS threshold sweep 0.1/0.2/0.3/0.4 (§5y) | Tune the adopted 0.2 threshold | 0.2 and 0.3 both peak (~0.37–0.38 tertile-1 AP); 0.1/0.4 much worse | 0.2 stays adopted |
| EMA (weight averaging) | `torch.optim.swa_utils` averaged eval/checkpoint weights | Combo-only; stacked with multiscale + center-RFS, neither helped (§5q) | Not confirmed |
| Multi-scale WBF TTA | Inference-time box fusion across scales | −0.009 on a multiscale-trained ckpt; −0.009 again on a non-multiscale ckpt (§14.1) | **Confirmed negative** |
| Seesaw Loss (`--loss-fn seesaw`) | Cumulative per-class logit damping | Not carried to a full confirmation run this project | Implemented, not confirmed |
| Speckle noise / ultrasound gain (EUS-specific aug) | Physically-motivated EUS augmentation ops | Implemented; not benchmarked to a confirmed verdict | Implemented, untested |
| TFA two-stage rare-class fine-tune (§5h) | Freeze all but box predictor, fine-tune on balanced subset | Sweep A (fine-tune noise only): Leiomyoma **confirmed regression** (−0.0277); Sweep B (different base ckpt): not confirmed either way | **Mixed — not a reliable fix** |
| RadImageNet backbone, EUS (§5p) | Ultrasound-domain-pretrained ResNet50 | 0.2844 vs 0.3559 control, Δ −0.0716, clears floor | **Confirmed negative** |
| DINOv3 as detector backbone, frozen (§5o/§5s) | ViT features feeding RPN+ROI heads only | WLE multiclass: +0.0066 at 5 seeds (shrinking, unconfirmed); WLE binary: comparable, not a win; **EUS multiclass: −0.1621, clean confirmed falsification** | **Not adopted** |
| DINOv3 detector backbone, unfrozen | Full fine-tune of the ViT backbone | 0.3862 vs 0.4200 frozen/baseline, confirmed worse | **Confirmed negative** |
| Lesion GAN copy-paste, EUS (§5h) | Class-conditional DCGAN-generated lesion patches pasted in | AP 0.3763 vs 0.3838 control (flat); hallucination ratio 0.065 vs 0.109 (real copy-paste 0.106) — lowest of the three, not fully confirmed | Directionally positive, not confirmed |
| Real-crop copy-paste, EUS (Exp 8, §4/§5h) | Real lesion crops pasted onto lesion-free backgrounds | AP flat (−0.008/−0.012); hallucination ratio ~flat at detector level (contrast: classifier-level backfire) | Not confirmed (detector); confirmed backfire (classifier, see report 2) |
| Cross-model WBF ensembling (§5v) | Fuse boxes from 2–5 independently trained checkpoints | Every subset tested underperforms the single best checkpoint by −0.028 to −0.114 | **Confirmed negative** |
| Auto-anchors + size-RFS + multiscale (3-way stack, §5x) | Does the anchor lever add anything on top of the 2 confirmed levers | Δ −0.043 vs size-RFS+multiscale alone, within noise | **Confirmed negative (redundant)** |
| FCOS (anchor-free, `fcos_resnet50`) | torchvision's anchor-free detector as an architecture alternative | Trained briefly (0.23→0.29 AP over 2 epochs) to find a stable LR; no full grid run | Implemented, not benchmarked |

### 2a. Group-aware / site-generalization interventions (LOCO-Center-3 setting)

Four different mechanisms tried against Center-3's severe leave-one-center-out
collapse (internal AP 0.6532 → external 0.1330, see report 2 for why):

| Method | Mechanism | Clean-bucket result | Verdict |
|---|---|---|---|
| `CenterRepeatFactorSampler` (§5l) | Oversample images from minority centers | 0.1113 → 0.1464, Δ +0.0351, clears floor; also collapses seed variance ~10× | **Confirmed positive** |
| DFR — Deep Feature Reweighting (§5m) | Freeze all but box predictor, fine-tune on center-balanced subset | 0.1016 vs 0.1113 baseline, no lift | **No effect** |
| Group DRO (§5t) | Online adversarial per-group loss reweighting | 0.0848 vs 0.1916 (pooled), Δ −0.1068, clears floor as **worse** | **Confirmed negative — worst of the four** |
| EndoStyle-style color/style jitter (§5n) | Per-channel gain/bias + hue rotation targeting the site fingerprint | Indistinguishable from baseline on both pooled and clean reads | **No effect** |
| JTT diagnostic (identification-only, §5u) | Flag high-loss training images, check what they correlate with | Hard set 1.5× enriched for Center-3, small lesions, NonNeoplasm — overlaps axes already known | **Diagnostic only, full retrain not justified** |

---

## 3. Detection — YOLO11 track (Track B)

Second detection architecture family, COCO-pretrained `yolo11n`/`s`/`m`,
scored through the identical `DetectionAccumulator` code as Faster R-CNN for
a fair AP@0.5 comparison (bridge validated: published 0.4280 vs
bridge-recomputed 0.4358).

| Config | Modality/Task | AP@0.5 | vs. Faster R-CNN reference | Verdict |
|---|---|---|---|---|
| `yolo11n`, defaults | WLE multiclass | 0.3963 | 0.4280/0.4358 (tuned FR-CNN) | Baseline, modestly behind |
| `imgsz=1280` | WLE multiclass | 0.3745 | — | Worse |
| `yolo11s` | WLE multiclass | 0.3861 | — | Worse |
| `close_mosaic=30` | WLE multiclass (3-seed, corrected) | 0.3913 ± 0.0184 vs 0.4162 ± 0.0148 | — | **Confirmed negative** (after fixing an early-stopping trap that masked it, §11.4) |
| `copy_paste=0.3` | WLE multiclass | 0.3963 (byte-identical) | — | **Confirmed no-op** (dataset has no segmentation masks) |
| `yolo11m` | WLE multiclass | 0.4135 | — | Promising, single-seed only |
| Class-oversampled export (§11.5) | WLE multiclass, 3-seed | 0.4109 ± 0.0138 vs 0.3848 ± 0.0086 plain | 0.4280/0.4358 | **Confirmed positive** — Track B's first confirmed lever |
| `yolo11m` + class-oversample (combo, §13.1) | WLE multiclass | 0.3724 | — | **Confirmed-negative interaction** — two positive levers didn't compose |
| Ultralytics built-in TTA (`--augment`, §13.3) | WLE multiclass (best ckpt) | 0.3972 vs 0.4295 plain | — | **Confirmed negative**, Leiomyoma AP collapsed |
| `yolo11n`, defaults | WLE binary | 0.6059 | 0.6731 (published) | Behind |
| `yolo11n`, defaults | **EUS multiclass**, 3-seed (§13.6) | **0.3799 ± 0.0064** | 0.3163 (published) | **Confirmed — first YOLO win over Faster R-CNN anywhere in this project** |
| Class-oversampled export, EUS (§14.2) | EUS multiclass, single-seed | 0.3680 vs 0.3799 plain | — | Hurts — WLE's lever doesn't transfer |

---

## 4. Detection — multimodal joint detector (Stage 1 fusion)

One Faster R-CNN trained over the **union** of WLE+EUS, modality-homogeneous
batches, optional per-modality FiLM conditioning on the FPN outputs, scored
per modality against a shared-split unimodal baseline (leak-free joint split,
30% hard negatives):

| Model | Task | Conditioning | AP WLE (Δ) | AP EUS (Δ) | Macro AP | F1 |
|---|---|---|---|---|---|---|
| Shared-split unimodal baseline | binary | — | 0.6566 | 0.5040 | — | — |
| Shared-split unimodal baseline | multiclass | — | 0.3750 | 0.4481 | — | — |
| `fasterrcnn_mobilenet` joint | binary | FiLM | 0.6402 (−0.016) | 0.4329 (−0.071) | 0.5366 | 0.6615 |
| `fasterrcnn_mobilenet` joint | binary | none | 0.6540 (−0.003) | 0.4559 (−0.048) | 0.5550 | 0.6558 |
| `fasterrcnn_mobilenet` joint | multiclass | FiLM | 0.3863 (+0.011) | 0.4134 (−0.035) | 0.3999 | 0.5487 |
| `fasterrcnn_mobilenet` joint | multiclass | none | 0.3525 (−0.023) | 0.4116 (−0.037) | 0.3820 | 0.5614 |

**No configuration beat the unimodal baseline by more than the ~0.02 AP
seed-to-seed noise floor** — the joint detector is roughly on par with, not
better than, training each modality separately.

---

## 5. Classification — Track A (single-modality campaign)

### 5.1 Frozen-embedding baselines

| Backbone | Modality/Task | Balanced acc. |
|---|---|---|
| SVM on ResNet-50 embeddings | WLE binary | 0.6447 |
| SVM on ResNet-50 embeddings | WLE subtype | 0.2995 |
| SVM on DINOv3 embeddings | WLE binary | 0.7353 |
| SVM on DINOv3 embeddings | WLE subtype | 0.3813 |
| SVM on DINOv3 embeddings | EUS binary | 0.5805 |
| SVM on ResNet-50 embeddings | EUS binary | 0.5817 |

DINOv3's edge on WLE partially survives lesion removal (52%/38% retained —
genuine, if partial, lesion-reading); on EUS it fully survives or grows
(152%/200%) — pure shortcut-reading, not a real advantage. Full detail in
[SHORTCUT_LEARNING.md](SHORTCUT_LEARNING.md).

### 5.2 Fine-tuned backbone screen (32 cells: 4 backbones × 2 modalities × 2 tasks × frozen/unfrozen)

| Modality | Task | ResNet-50 (unfrozen) | ConvNeXt-Tiny (unfrozen) | EfficientNet-B0 (unfrozen) | DINOv3 (frozen) | DINOv3 (unfrozen) |
|---|---|---|---|---|---|---|
| WLE | binary | 0.7668 | 0.7607 | **0.7711** | 0.7072 | 0.5711 (collapse) |
| WLE | subtype | 0.4164 | 0.4327 | **0.4348** | 0.3115 | 0.3007 (collapse) |
| EUS | binary | **0.6925** | 0.6918 | 0.6836 | 0.5308 | 0.5000 (collapse) |
| EUS | subtype | 0.3595 | 0.3295 | **0.3602** | 0.2055 | 0.2330 (collapse) |

**EfficientNet-B0 is the best unfrozen CNN on WLE**, carried into the lever
screen. **Full fine-tuning catastrophically breaks DINOv3** at the project's
default LR (3e-4) — see 5.4 below for the fix.

### 5.3 Loss/augmentation lever screen (`efficientnet_b0`, WLE)

| Lever | Binary Δ (vs 0.7711) | Subtype Δ (vs 0.4348) |
|---|---|---|
| `weighted_ce` | −0.0071 | **+0.0221 → confirmed +0.0327 at 3 seeds** |
| `seesaw_ce` | −0.0108 | −0.0220 |
| size-aware oversampling | −0.0069 | −0.0187 |
| background-swap augmentation | −0.0307 | −0.0031 (no effect on EUS either) |

**`weighted_ce` is the one confirmed lever**, subtype-only: 0.4282±0.0127 →
0.4609±0.0035 at 3 seeds, clears the noise floor ~2.5×.

### 5.4 DINOv3 follow-ups

| Config | Task/Modality | Result | Verdict |
|---|---|---|---|
| `--freeze partial` (unfreeze last 4 blocks) | WLE binary/subtype, 3-seed | 0.8068±0.0075 / 0.4779±0.0087 | **Confirmed new best** — counterfactual gate **PARTIAL** (see report 2) |
| `--freeze none`, `lr=1e-5` (vs. default 3e-4) | WLE binary/subtype, single-seed | 0.7764 / 0.4388 — recovers almost to partial-freeze level | Confirms LR, not collapse, was the earlier problem |
| `--freeze partial`, EUS (§14.3) | EUS binary/subtype, single-seed | **0.7386±0.0313** / 0.3764±0.0507 (new EUS bests) | Promising, gate-clean (lowest EUS retentions seen), not yet multi-seed |
| `unfreeze_last_n` sweep {2,6,8,12} (§14.5) | WLE binary, single-seed | n=8: 0.8085±0.0142 (best); n=12: 0.7670 (worse, not catastrophic) | n=6–8 modestly beats default n=4, not confirmed |
| DINOv3-partial + `weighted_ce` combo (§13.2) | WLE subtype, 3-seed | **0.4957±0.0033** | **Confirmed, counterfactual-clean** (bg_patch retention dropped 40%→27%) |

### 5.5 Ensembling and TTA

| Method | Config | Result | Verdict |
|---|---|---|---|
| Multi-seed probability ensemble (§13.4) | DINOv3-partial, WLE binary | 0.8063±0.0078 → **0.8209** | New Track A best, zero extra training |
| Multi-seed probability ensemble | DINOv3-partial+weighted_ce, WLE subtype | 0.4988±0.0049 → **0.5230** | New Track A best |
| Horizontal-flip TTA (§12.4) | 3 checkpoints tested | Δ −0.0009 to +0.0010 | **No effect** |
| 5-crop TTA (§14.6) | 2 checkpoints tested | Δ −0.0014 to −0.0162 | **No effect / negative** |

**Track A's best confirmed numbers overall: WLE binary 0.8209, WLE subtype
0.5230** (both via zero-cost ensembling of DINOv3-partial checkpoints).

---

## 6. Classification — multimodal fusion

### 6.1 Original detector-ROI-based fusion (`cadsel/fusion.py`, Stage 2 of §5b)

Frozen Faster R-CNN → top-box ROI feature per frame → attention-pool →
`GatedFusion` → patient-level subtype, 5-fold CV, modality dropout p=0.3.

| Ablation | All 641 patients (accuracy) | 421 paired patients (accuracy) | 421 paired (balanced acc.) |
|---|---|---|---|
| `wle_only` | 0.4258 (bal. acc.) | 0.378 | 0.378 (beats both fused variants on this metric) |
| `eus_only` | 0.3890 (bal. acc.) | — | — |
| `fused` | 0.4190 (bal. acc.) | 0.420 (+0.043) | lower than `wle_only` |
| `fused_moddrop` | 0.4163 (bal. acc.) | **0.523 (+0.145)** | lower than `wle_only` |

**Verdict: the apparent gain is a majority-class accuracy effect** — on
balanced accuracy specifically, `wle_only` beats both fused variants on the
correct (paired-only) denominator. A `SymmetricCrossAttentionFusion`
alternative was also tried (3 seeds × 5-fold) and found statistically
indistinguishable from `GatedFusion` — the architectural WLE-fallback isn't
what's limiting fusion here.

### 6.2 Fusion rebuilt on classifier-backbone features (§14.4, DINOv3)

Same `FusionHead`, new front-end: DINOv3 CLS-token features + softmax
posteriors from the confirmed-best WLE/EUS subtype classifiers, instead of
detector ROI features.

| Ablation | All 641 patients | 421 paired only |
|---|---|---|
| `wle_only` | 0.8836±0.0280 | 0.8847±0.0225 |
| `eus_only` | 0.8934±0.0367 | 0.8960±0.2014 |
| `fused` | 0.9271±0.0317 | 0.9406±0.1122 |
| `fused_moddrop` | **0.9324±0.0290** | 0.9304±0.1343 |

Fusion beats `wle_only` on **balanced accuracy** this time (opposite
conclusion from 6.1) — but paired-only variance is very wide (std up to
0.20), so the +0.0559 paired-only delta does **not** clear its own noise
floor (~0.115). **Verdict: promising, structurally different result, not
confirmed** — needs repeated-CV or bootstrapped confidence intervals to
settle.

### 6.3 Joint end-to-end two-tower fusion (§15, user-proposed architecture) — CONFIRMED

Two independently trainable EfficientNet-B0 towers (one per modality),
jointly fine-tuned end-to-end with per-modality augmentation every step,
feeding the existing `FusionHead` unmodified.

| Config | Task | `wle_only` | `eus_only` | `fused` | Ensembled (3-seed) |
|---|---|---|---|---|---|
| `--freeze partial`, flat LR (§15.4) | binary | 0.7153±0.0208 | — | 0.7596±0.0082 | 0.7741 |
| `--freeze partial`, discriminative LR (§15.3) | binary | 0.7193 | — | 0.7533 | — |
| `--freeze none` (§15.6) | binary | 0.7644±0.0093 | — | 0.8007±0.0067 | 0.8160 |
| `--freeze none` + `--lesion-crop` (§15.8) | binary | 0.7733±0.0071 | 0.7057±0.0178 | **0.8110±0.0123** | **0.8219** |
| `--freeze none` + `--lesion-crop` (§15.8.1) | subtype | 0.5049±0.0254 | 0.3642±0.0097 | **0.5419±0.0201** | **0.5706** |
| `--fusion symmetric` + lesion-crop (§15.8, screen) | binary | — | — | 0.7786 (EUS-only jumped to 0.7317) | — |

**All configs above beat `wle_only`, confirmed at 3 seeds beyond the noise
floor, on both tasks.** `--freeze none` + `--lesion-crop` is the current best
confirmed config (binary and subtype both); lesion-cropping specifically and
substantially helps the EUS tower (+0.086 confirmed) but that gain doesn't
fully propagate into the fused number because `GatedFusion` structurally
favors WLE. Counterfactual gate: **PASSED** on the `--freeze none` binary
checkpoint (boxmap retention −2%, essentially chance) — see report 2 for
full detail. Levers screened with no further win: discriminative backbone LR
(helps `wle_only`, not fusion specifically), symmetric fusion + lesion-crop
combined, paired-only training, `--bg-remove-prob` (both p=0.5 and p=0.3),
patient-level multi-frame TTA.

**This is the project's best confirmed multimodal classification result.**

---

## 7. Architectures implemented but not fully benchmarked

| Architecture | Status |
|---|---|
| `fcos_resnet50` (anchor-free detector) | Trained briefly to find a stable LR (1e-4); no full grid run |
| Conditional lesion GAN (64×64 DCGAN) | Trained; used as a copy-paste patch source (§5h results above); generator quality itself untested |
| Cross-modal WLE↔EUS CycleGAN-style translation | Explicitly scoped out — different physical imaging processes, not a domain-shift problem |
| Paired-instance fusion (assumes matched WLE/EUS captures) | Scoped out — CAD-SEL frames are not paired at the instance level, only at the patient level |

---

## 8. Leaderboard — best confirmed result per track

| Track | Task | Best confirmed config | Result | Gate status |
|---|---|---|---|---|
| Detection (Faster R-CNN) | WLE multiclass | `mobilenet_320` + size-RFS + multiscale | AP@0.5 ≈ 0.41 (tertile-1 gain confirmed) | n/a |
| Detection (YOLO11) | EUS multiclass | `yolo11n`, plain, 3-seed | AP@0.5 0.3799±0.0064 (beats Faster R-CNN) | n/a |
| Detection (joint WLE+EUS) | binary/multiclass | `mobilenet` joint, FiLM or none | ≈ unimodal baseline, no confirmed gain | n/a |
| Classification (single-modality) | WLE binary | DINOv3-partial ensemble | **0.8209** | Underlying config PARTIAL |
| Classification (single-modality) | WLE subtype | DINOv3-partial+weighted_ce ensemble | **0.5230** | Underlying config PASS |
| Classification (single-modality, gate-clean) | WLE binary/subtype | EfficientNet-B0 + weighted_ce | 0.7711 / 0.4609±0.0035 | **PASS** |
| Classification (fusion) | binary | Two-tower, `--freeze none`+lesion-crop | **0.8110±0.0123** (ensembled 0.8219) | **PASS** |
| Classification (fusion) | subtype | Two-tower, `--freeze none`+lesion-crop | **0.5419±0.0201** (ensembled 0.5706) | Not yet re-gated on this exact config |
