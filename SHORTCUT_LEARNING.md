# Shortcut Learning in CAD-SEL — Detection, Mitigation, and Whether It Worked

This is the project's account of one specific risk: that a model reaches a
good validation score by reading something other than the lesion — image
background, annotation-box geometry, acquisition-site fingerprints, or a
leaky evaluation split — rather than genuine lesion morphology. This is the
failure mode Geirhos et al. (2020, *Nature Machine Intelligence*) define as
shortcut learning, and it is a known risk in endoscopic imaging specifically
(the literature review behind this audit, `Shortcut_Learning_Colonoscopy_
Literature_Review.pdf`, catalogues prior cases). Full experimental detail
lives in [FINDINGS.md](FINDINGS.md) §4–§5 and `experiments/README.md`; this
document is the narrative summary: what was tested, what was found, what was
tried against it, and whether any of it actually worked.

For the model inventory this audit informed, see
[MODELS_TESTED.md](MODELS_TESTED.md).

---

## 1. Why this was worth checking

A shortcut learner looks fine on an ordinary held-out split drawn from the
same dataset — it's only exposed by a test where a lesion-reading model and
a shortcut-reading model would have to behave *differently*. Two properties
of CAD-SEL make this a live risk rather than a formality:

- Every image in the dataset **contains a lesion** — nothing ever teaches a
  model what "no lesion here" looks like, so there is no structural pressure
  against learning to key off whatever else correlates with the label.
- The dataset is pooled across **6 acquisition centers** with different
  capture equipment/format, and the label distribution is not independent of
  center in practice (Center-1 alone supplies 537/641 patients) — a
  classic setup for a model to pick up a site fingerprint that happens to
  correlate with class prevalence.

---

## 2. How it was detected — the audit methodology

Six stress tests (`experiments/src/01`–`06`, ResNet-18 classifiers, patient-
level splits throughout, balanced accuracy as the metric since the class
mix is ~15:1 and plain accuracy is flattered by always predicting the
majority class):

| Test | Question asked |
|---|---|
| `01_dataset_confounds` | Is the label predictable from centre, demographics, size, or location alone (no image)? |
| `02_lowlevel_probes` | Is centre — or the label — decodable from global colour/exposure statistics? |
| `03_lesion_vs_background` | **The core test.** Train separate models on 5 pixel-content variants of the same images, see what survives when the lesion is removed. |
| `04_cross_centre` | Train on Center-1, test on Centers 2–6 — does performance transfer? |
| `05_split_leakage` | Does patient-level vs. image-level splitting change the score, and by how much? |
| `06_synthesis` | Collate 01–05 into one verdict per mechanism. |

Experiment 3 is the load-bearing test and set the protocol every later
counterfactual check in this project reuses. Five variants of the same
image, each used to train a *separate* model from scratch (not one model
ablated at inference — that distinction matters, see §6):

- **`full`** — the ordinary image.
- **`lesion`** — cropped to the annotation box + padding, background removed.
- **`bg_mask`** — the annotation box filled with mean colour (background
  visible, lesion erased). Later found to be a flawed control — see below.
- **`bg_patch`** — a crop from the largest rectangle the annotation *never*
  touches. No lesion, no visible marker.
- **`boxmap`** — a black canvas with only the annotation box drawn in white.
  Position and size only, zero real tissue.

A methodological correction made along the way, worth stating explicitly:
the first version of this test used only `bg_mask`. That is **not** a clean
ablation — a mean-colour-filled rectangle still encodes the box's size and
position, which turned out to itself be class-associated. `bg_patch` and
`boxmap` were added specifically to separate "background reliance" from
"geometry reliance," and `bg_mask` is kept in later tables only to show that
contamination, not as trustworthy evidence on its own.

---

## 3. What was found

| Mechanism | Verdict | Evidence |
|---|---|---|
| Centre predicts the label | **Not present** | Cramér's V 0.099; probe lift +0.003 |
| Acquisition site encoded in the pixels | **Present (strong)** | Centre decodable at 0.81 vs. 0.167 chance |
| Label leaks via global appearance (colour/exposure) | **Not present** | Max probe lift +0.10 |
| Label predictable from non-lesion pixels (`bg_patch`) | **Present (strong)** | 68–95% of full-image accuracy retained |
| Annotation geometry alone is diagnostic (`boxmap`) | **Present (strong)** | 48–189% retained (sometimes *exceeds* the full image) |
| Collapse on an unseen site | **Partial** | 52–100% retained, small external n |
| Image-level split inflates the score | **Present** | +0.14 to +0.58 balanced accuracy |

The core numbers (Experiment 3, balanced accuracy, retention vs. full image
in brackets):

| | chance | full | lesion | bg_patch | boxmap |
|---|---|---|---|---|---|
| WLE binary | 0.500 | 0.706 | 0.754 (123%) | 0.695 (**95%**) | 0.598 (48%) |
| WLE subtype | 0.200 | 0.326 | 0.448 (197%) | 0.299 (**79%**) | 0.300 (80%) |
| EUS binary | 0.500 | 0.679 | 0.749 (139%) | 0.621 (**68%**) | 0.666 (93%) |
| EUS subtype | 0.200 | 0.322 | 0.599 (326%) | 0.292 (**75%**) | 0.431 (**189%**) |

**Headline: the label is recoverable from context and from box geometry,
largely independent of lesion appearance.** A model can score respectably
having never learned what a NET or a lipoma actually looks like.

Two distinct problems came out of this, and they are not the same thing:

1. **A genuine shortcut** — the pattern above: context and geometry carry
   real, exploitable signal.
2. **A leakage trap** — image-level splitting inflates balanced accuracy by
   up to +0.58 because most validation patients also appear in training.
   Not shortcut learning in the Geirhos sense, but the single largest source
   of over-optimistic numbers found in this project, and the most plausible
   explanation for published figures on this data sitting far above this
   project's own reproduction. **Every result in this project uses
   patient-level splitting for exactly this reason.**

The site-fingerprint finding is a partial exception worth flagging on its
own: acquisition centre is strongly decodable from the pixels (0.81 vs 0.167
chance), yet cross-centre transfer only degrades moderately in most cases.
The fingerprint is real but, on this dataset, mostly *not* the dominant route
to the label — because Center-1 supplies the large majority of patients, so
site and class are close to independent overall. (Center-3 is the sharp
exception to this — see §4.)

**Detection is markedly less shortcut-prone than classification.** The same
test at the detector level (Experiment 8): a classifier retains 68–95%
accuracy on lesion-free crops, but a Faster R-CNN fires on those same crops
only ~0.3× as often as on full frames. Box regression forces the model to
commit to a location, which structurally resists the pure-context shortcut
classification is vulnerable to.

---

## 4. Extended findings: where the shortcut concentrates

### DINOv3's embedding edge is genuine on WLE, fake on EUS (§5e)

DINOv3 frozen embeddings beat ResNet-50 embeddings by 0.7–9.0 points at full
image. Run through the same 5-variant protocol:

| Modality/task | Gap @ full | Gap @ `bg_patch` | Retained | Verdict |
|---|---|---|---|---|
| WLE binary | +0.090 | +0.047 | 52% | Partial — a real edge, shrinks but survives |
| WLE subtype | +0.084 | +0.032 | 38% | Partial — real edge, shrinks but survives |
| EUS binary | +0.032 | +0.049 | 152% (grows) | **Pure shortcut-reading** |
| EUS subtype | +0.007 | +0.014 | 200% (grows) | **Pure shortcut-reading** |

On EUS, DINOv3's apparent advantage isn't lesion-reading at all — it's a
*better reader of the same shortcut* that Experiment 3 already found EUS
carries more strongly than WLE.

### The detector-level test independently confirms the same thing (§5s)

Wiring the frozen DINOv3 representation into an actual Faster R-CNN backbone
(only RPN/ROI heads trainable) gave a clean falsification, not just an
echo: EUS multiclass AP@0.5 fell to 0.2217±0.0166 against a 0.3838±0.0480
plain baseline — Δ −0.1621, clearing the noise floor by >3×. **The
classifier-probe theory and an independent detector-level test agree**: on
EUS specifically, DINOv3's apparent strength is not lesion signal.

### Center-3's catastrophic LOCO collapse — is it a labeling problem? No (§5f, §5k, §5r)

Holding out Center-3 entirely, a WLE-binary detector's AP@0.5 collapses from
0.6532 (internal) to 0.1330 (external) — the single sharpest generalization
failure found anywhere in this project. Two follow-up investigations ruled
out the tempting explanations:

- **Is it mislabeled boxes overlapping black letterboxing?** No — bucketing
  Center-3 images by black-overlap fraction, the "dirty" bucket (more
  overlap) scores *higher* AP (0.1705) than "clean" (0.0953), the opposite
  of what a labeling-artifact story predicts. This pattern replicated across
  all 12 checkpoints later tested with this bucketing.
- **Then why does dirty score higher?** A lesion-size confound, not
  capture format: the dirty bucket's lesions are 67% larger on average
  (larger boxes both touch the frame edge more often *and* are easier to
  detect — already an established effect elsewhere in this project). Not a
  label-quality signal.
- **Net verdict:** Center-3's collapse is a genuine generalization failure
  on its visual domain (near-zero training exposure to its specific capture
  format), not a data-quality artifact — the strongest confirmed evidence in
  this project that site fingerprint can matter for detection specifically,
  not just classification.

---

## 5. Mitigations tried, and whether they worked

Two different mitigation *protocols* were used, and the distinction matters
for reading the table below:

- **Experiment-3 style**: train a *separate* model per input variant — asks
  "is the label learnable from this input at all."
- **Experiment-7 style** (used for every mitigation below): train **one**
  model on ordinary images, then ablate it at **inference** — asks "does
  this specific trained model rely on the lesion." A mitigation is a
  property of a trained model, not of an input, so this is the protocol used
  once the question shifted from "is there a shortcut" to "can we stop a
  model from taking it."

| Mitigation | Mechanism | Result | Worked? |
|---|---|---|---|
| **Hard negatives (30% lesion-free frames)** | Supplies the negative supervision the dataset structurally lacks | False alarms **−69%/−86%**; AP cost only −0.05/−0.01; F1 **+0.16** on multiclass; clears the seed-noise floor at 3 seeds (Δratio −0.184, ~3× pooled sd) | **Yes — the clear winner, best mitigation found in this project** |
| Background-swap augmentation (classifier) | Paste the lesion onto a different patient's lesion-free background | WLE: reliance **−75%**, cost −0.153 acc. (both tasks respond); EUS: reliance **+23 to +37%** (backfires) | **Works on WLE only** |
| Scale-jitter + copy-paste (classifier) | Randomly rescale the lesion crop before compositing | WLE: reliance −42 to −51%, sometimes *improves* accuracy; EUS: same backfire pattern | **Works on WLE only** |
| Scale-jitter + copy-paste (detector, Exp 8) | Same augmentation, at the detector level | AP moved by only −0.008/−0.012 (noise); reliance improved on binary (−0.09), worsened on multiclass (+0.05) | **No clear transfer from the classifier-level win** |
| Size-matched evaluation | Remove the lesion-size prior at eval time only, no retraining | Removes up to +0.10 AP of size-driven inflation, for free | **Yes, and it's free** — use this whenever comparing across models |
| Adversarial discriminator (GRL on `bg_patch`, Exp 12) | Push the backbone away from representations a discriminator can read the label from without the lesion | 7/8 conditions moved in the right direction; only 1/8 cleared the noise floor at 3 seeds | **Suggestive, not confirmed** |
| BBox blur (feathered, jittered, Exp 13) | Blur everything outside a padded region around the lesion during training | Mixed sign on WLE; **backfires on EUS binary** (+34 to +38% retention) | **No effect / backfires on EUS** |
| SE channel attention + discriminator (`se_all_adv`, Exp 14) | Squeeze-excite gates composed with the adversarial signal | 8/8 cells negative direction, cleanest dose-response of the three — but at 7 seeds / 100 epochs (Exp 15/16), **none cleared 2× the noise floor** | **Real but too small to confirm** |
| DFR — Deep Feature Reweighting, group=center (§5m) | Freeze all but box predictor, fine-tune on a center-balanced subset | No lift on either pooled or clean-bucket read | **No effect** — structurally can't fix a representation gap, only a decision boundary |
| Group DRO, group=center (§5t) | Online adversarial per-group loss reweighting | Δ −0.1068, clears floor as **worse** — worst of four group-aware methods tried | **Confirmed negative** (tiny per-center batches destabilize the online weight update) |
| Center-bucketed oversampling (§5l) | Oversample images from minority-patient centers | Clean-bucket Δ +0.0351, clears floor; also collapses cross-seed variance ~10× | **Yes, confirmed** — the one group-aware method that worked |
| EndoStyle-style color/style jitter (§5n) | Per-channel gain/bias + hue rotation, targeting the site fingerprint directly | Indistinguishable from baseline on both reads | **No effect** |
| JTT diagnostic (§5u) | Identify high-loss training images, see what they correlate with | Hard set 1.5× enriched for Center-3/small lesions/NonNeoplasm — overlaps axes already known, nothing clearly novel | **Diagnostic only** — full two-stage retrain not justified |
| Lesion GAN copy-paste, EUS (§5h) | Synthetic (DCGAN) lesion patches instead of real crops | Hallucination ratio 0.065 vs. 0.109 real-copy-paste control — lowest of the three arms tested, but within combined seed noise | **Directionally positive, not confirmed** |
| TFA rare-class fine-tuning (§5h) | Freeze all but box predictor, briefly fine-tune on a class-balanced subset | Fine-tune-noise-only sweep: Leiomyoma **confirmed regression** (−0.0277); different-base-checkpoint sweep: not confirmed either way | **Small, inconsistent, not a reliable fix** |

**The one clear, cheap, working mitigation in this project is hard negative
mining.** Everything more architecturally sophisticated that was tried
against it (adversarial training, channel attention, DFR, Group DRO) either
produced an effect too small to confirm or made things actively worse.

---

## 6. The protocol that survived: the mandatory counterfactual gate

The audit's most durable output wasn't a single mitigation — it was
adopting Experiment 3's variant-swap protocol as a **standing gate** that
every subsequent classification/fusion "win" in this project must clear
before being trusted, not just a one-off diagnostic. Concretely: score the
same trained checkpoint on `full`, `lesion`, `bg_mask`, `bg_patch`, and
`boxmap`; a genuine win should not show a *higher* background/geometry
retention than the baseline it's being compared against, and a fusion
architecture's headline number should collapse toward chance under
`boxmap` (all real tissue removed) if the gain is truly lesion-driven.

This gate was applied to every confirmed result in the Track A classification
campaign and the joint WLE+EUS fusion work:

| Result | `bg_patch` retention | `boxmap` retention | Gate verdict |
|---|---|---|---|
| EfficientNet-B0 + `weighted_ce`, WLE (subtype win) | 33% (vs. 42% baseline — **lower**) | −0% | **PASS** |
| DINOv3 `--freeze partial`, WLE (binary/subtype new best) | 54%/40% (vs. baseline's 46%/33% — **higher**, technically a WARN by the letter of the rule) | 5%/4% | **PARTIAL** — isolated-lesion retention (87%/83%) is the highest seen anywhere, read together as genuine-but-caveated, not a clean pass |
| DINOv3-partial + `weighted_ce` combo, WLE subtype | 27% (vs. 40% baseline — **lower**) | 5% | **PASS** |
| Joint two-tower fusion, `--freeze none`, binary | `lesion` 44%, `bg_mask` 48%, `bg_patch` 42% | **−2% (essentially exact chance)** | **PASS** |
| EUS backbone-screen backlog, 17 checkpoints (§13.5) | 15–89% across cells; one cell (`efficientnet_b0`/subtype/frozen) at **89%** — the closest any result has come to the ~100% "pure shortcut" DINOv3-EUS precedent | mostly noisy/near-zero | **Mostly shortcut-suspect** — only one clean cell, and it's also one of the weaker-accuracy configs |
| DINOv3-partial, EUS (new EUS best, §14.3) | 35%/15% | 4%/−13% | **PASS** — among the cleanest EUS retentions found, and the strongest EUS accuracy |

**Did the gate actually catch things?** Yes, on both sides:

- It **flagged** the highest-scoring single-modality result in the whole
  Track A campaign (DINOv3 partial-freeze) as not a clean pass, preventing
  it from being reported as the unqualified best result.
- It **passed** the project's best fusion result cleanly, with the sharpest
  possible evidence: removing every trace of real tissue drops balanced
  accuracy to within noise of exact chance (0.494 vs. 0.500).
- It **confirmed EUS classification should still be treated as
  shortcut-suspect by default** — most of the 17-checkpoint backlog landed
  in a mixed 15–50% retention band, and the one very clean-looking cell
  (89% retention) is flagged explicitly as *not* trustworthy despite scoring
  respectably on the raw metric.

A related but distinct pitfall the audit also caught, worth noting alongside
shortcut learning proper: the original detector-ROI-based fusion model's
apparent gain (§5i) turned out to be a **majority-class accuracy effect** —
on balanced accuracy specifically, `wle_only` actually beat the fused model.
Not a shortcut in the Geirhos sense (no non-lesion signal was being
exploited), but the same underlying discipline — don't trust a headline
number without checking what metric and denominator produced it — is what
caught it.

---

## 7. Bottom line: did it work?

**Detection of the problem: yes, decisively.** The audit found a large,
reproducible shortcut (label recoverable from context and geometry alone,
48–189% retention) and a separate, even larger leakage trap (+0.58 balanced
accuracy from image-level splitting) — both real, both would have silently
inflated every result in this project if unaddressed.

**Fixing it architecturally: mostly no.** Of eight mitigation mechanisms
tried at the classifier/detector-architecture level (adversarial
discriminator, bbox blur, channel attention, DFR, Group DRO, EndoStyle
augmentation, TFA, lesion-GAN copy-paste), only **one** — hard negative
mining — produced a clean, confirmed, meaningfully-sized win. One other
(center-bucketed oversampling) worked for the narrower site-generalization
problem specifically. Everything else was either too small to clear its own
noise floor, backfired on EUS, or actively made things worse (Group DRO).
This is itself a useful negative result: naive rebalancing and standard
group-robustness methods do not reliably fix a genuine visual-cue shortfall
on a dataset this small and this imbalanced.

**Preventing it from contaminating future results: yes.** The mandatory
counterfactual gate, built directly from the audit's own Experiment 3
protocol, is now load-bearing infrastructure for every classification and
fusion result in this project. It has concretely changed what gets reported
as "best" (demoting the raw-highest-scoring DINOv3 result to a caveated
PARTIAL) and concretely validated the project's headline result (the joint
fusion classifier's boxmap collapse to chance). **The audit's most durable
contribution wasn't fixing the shortcut — it was making it impossible to
report an unvalidated win by accident going forward.**
