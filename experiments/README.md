# Shortcut-learning audit of CAD-SEL

Diagnostics testing whether the CAD-SEL dataset lets detectors reach the label
without reading lesion morphology — the failure mode defined in Geirhos et al.
(2020) and catalogued for colonoscopy imaging in the accompanying literature
review.

Sources live in `src/` as percent-format `.py` files; `_py2nb.py` converts and
executes them into the `.ipynb` notebooks alongside, flushing after every cell so
a long run is never lost. Saved results land in `results/`.

```powershell
.\.venv\Scripts\python.exe experiments\_py2nb.py          # all
.\.venv\Scripts\python.exe experiments\_py2nb.py 03       # one
.\.venv\Scripts\python.exe experiments\_py2nb.py 03 --no-exec
```

Experiments 03-05 resume from `results/*.json`; delete those to force a rerun.

## Why stress tests rather than the validation score

A shortcut learner looks fine on a held-out split drawn from the same dataset
(review §2). Every experiment here therefore constructs a situation where a
lesion-reading model and a shortcut-reading model must behave *differently*.

All tests report **balanced accuracy** (mean per-class recall) against a chance
floor, because the class mix is ~15:1 and plain accuracy is flattered by always
predicting NET. All splits are patient-level unless the experiment is explicitly
about splitting.

| Notebook | Test | Review § |
| --- | --- | --- |
| `01_dataset_confounds` | Is the label predictable from centre, demographics, size, location, image geometry? | 3, 7.1 |
| `02_lowlevel_probes` | Is centre — or the label — decodable from global colour/exposure statistics? | 4.4, 7.1 |
| `03_lesion_vs_background` | Counterfactual ablation: remove the lesion and see what survives | 4.2 |
| `04_cross_centre` | Train on Center-1, test on Centers 2-6 | 4.5 |
| `05_split_leakage` | Patient-level vs image-level splitting | CAD-SEL Usage Notes |
| `06_synthesis` | Collates 01-05 into a verdict per mechanism | — |

## Findings

| Mechanism | Verdict |
| --- | --- |
| Centre predicts the label | **not present** — Cramér's V 0.099 |
| Acquisition site encoded in pixels | **present (strong)** — centre decodable at 0.81 vs 0.167 chance |
| Label leaks through global appearance | **not present** — max lift +0.10 |
| Label predictable from non-lesion pixels | **present (strong)** — 68-95% retained |
| Annotation geometry alone is diagnostic | **present (strong)** — 48-189% retained |
| Collapse on an unseen site | **partial** — 52-100% retained, small external n |
| Image-level splitting inflates results | **present** — +0.14 to +0.58 balanced accuracy |

### The core result (Experiment 3)

Balanced accuracy, retention against the full-image reference in brackets:

| | chance | full | lesion | bg_patch | boxmap |
| --- | --- | --- | --- | --- | --- |
| WLE binary | 0.500 | 0.706 | 0.754 (123%) | **0.695 (95%)** | 0.598 (48%) |
| WLE subtype | 0.200 | 0.326 | 0.448 (197%) | **0.299 (79%)** | 0.300 (80%) |
| EUS binary | 0.500 | 0.679 | 0.749 (139%) | **0.621 (68%)** | 0.666 (93%) |
| EUS subtype | 0.200 | 0.322 | 0.599 (326%) | **0.292 (75%)** | 0.431 (189%) |

`bg_patch` is a crop from the largest rectangle the annotation never touches — no
lesion, no marker. It retains most of the signal.

`boxmap` is a black canvas with the bounding boxes drawn white: position and size
only, no tissue at all. That alone rivals the full image.

A methodological note worth carrying forward: the first version of this
experiment used only `bg_mask` (boxes filled with mean colour). That is **not** a
clean ablation — the filled rectangle still encodes the annotation's size and
position, which Experiment 1 showed is class-associated. `bg_patch` and the
`boxmap` control were added to separate the two effects, and `bg_mask` is
retained only to show the contamination.

### Interpretation

Two distinct problems, and they are not the same thing:

1. **A genuine shortcut.** The label is recoverable from context and from
   annotation geometry, largely independent of lesion appearance. This is the
   review's §2 definition, and it means a strong internal number is not evidence
   of lesion-reading behaviour.

2. **A leakage trap.** Image-level splitting inflates balanced accuracy by up to
   +0.58 because 366 of 379 validation patients would also appear in training.
   Not shortcut learning, but the largest single source of over-optimistic
   numbers here — and the most likely explanation for published figures far above
   ours on the same data.

The site-fingerprint result is a partial exception: the acquisition centre is
strongly decodable from the pixels, yet cross-centre transfer degrades only
moderately. The fingerprint exists but is not, on this dataset, the dominant
route to the label — largely because Center-1 supplies 537 of 641 patients, so
site and class are nearly independent.

## Mitigation results (Experiment 7)

Two targeted augmentations, tested against the measurements above:

- **`scale`** — random rescale of the lesion (0.5-1.6x), aimed at the geometry shortcut
- **`bgswap`** — lesion pasted onto a *different patient's* lesion-free background, aimed
  at the context shortcut
- **`both`**

Protocol differs from Experiment 3 on purpose: one model is trained on ordinary
frames and ablated at **inference**, which asks "does this trained model rely on
the lesion?" rather than "is the label learnable from this input?".

Change in background reliance versus baseline (negative = less shortcut):

| | `scale` | `bgswap` | `both` |
| --- | --- | --- | --- |
| WLE binary | −0.6% | **−75.2%** | −50.9% |
| WLE subtype | −46.5% | −41.2% | −42.4% |
| EUS binary | −28.7% | −30.9% | +1.0% |
| EUS subtype | +37.3% | +26.8% | +23.2% |

**WLE responds; EUS does not.** On WLE the mitigations work and the cost is
small — `both` cuts background reliance from 69% to 18% on the binary task for
−0.069 balanced accuracy, and on subtype it cuts 45% to 3% while *improving*
accuracy (+0.042). `bgswap` alone is the strongest single intervention
(−75.2%) but costs the most headline accuracy (−0.153).

On EUS the same augmentations make reliance *worse* on the subtype task
(+23 to +37%). A plausible reading: EUS frames are grayscale with a narrow
field, so a lesion pasted onto another patient's ultrasound background is far
less plausible than in white light, and the augmentation mostly injects noise.
This is the "naive rebalancing does not fix a genuine visual-cue shortfall"
caution of review §9.6, and points to conditional generative augmentation
rather than cut-and-paste for EUS.

### Does it transfer to the detector? (Experiment 8)

Two Faster R-CNN runs (`mobilenet_320`, WLE), baseline vs
`scale_crop + copy_paste`, identical settings otherwise. AP cannot be computed on
a lesion-free crop, so reliance is measured as the **hallucination ratio** —
detections above 0.5 confidence on lesion-free crops, relative to full frames.

| task | arm | AP@0.5 | det/full | det/lesion-free | ratio |
| --- | --- | --- | --- | --- | --- |
| multiclass | baseline | 0.3809 | 1.18 | 0.33 | 0.28 |
| multiclass | mitigated | 0.3730 | 1.18 | 0.39 | 0.33 |
| binary | baseline | 0.6458 | 1.08 | 0.37 | 0.34 |
| binary | mitigated | 0.6340 | 1.11 | 0.27 | 0.24 |

**The mitigation is close to free, but its benefit did not clearly transfer.**
AP moved by −0.008 and −0.012 — negligible. Reliance improved on binary (−0.09)
and slightly worsened on multiclass (+0.05). With one seed per arm that is within
noise; the large reduction seen for the classifier in Experiment 7 is not
reproduced here.

The more interesting number is the **baseline** ratio: ~0.3. A detector fires on
lesion-free tissue about a third as often as on full frames, whereas the
classifier retained 69-95% of its accuracy on the same crops. Detection appears
inherently less shortcut-prone than classification on this dataset, presumably
because box regression forces the model to commit to a location. That is worth
knowing on its own: the shortcut measured in Experiments 1-5 is real, but its
severity depends on the task formulation.

### An important limitation of the `boxmap` column here

Under this inference-time protocol, `boxmap` retention at baseline is ~0 for
three of four settings — a model trained on natural frames simply cannot parse a
white rectangle on black. That is distribution shift, **not** evidence that the
geometry shortcut is absent. The evidence for the geometry shortcut is
Experiment 3's train-on-variant result (48-189% retention); the `boxmap` column
in Experiment 7 should not be read as contradicting it.

## Adversarial discriminator, bbox blur, channel attention (Experiments 12-14)

Three further mitigations, building directly on Experiment 7's protocol (train
**one** model on ordinary frames, ablate it at inference) rather than
Experiment 3's (train a separate model per input variant) — a mitigation is a
property of a trained model, not of an input. All three reuse Experiment 3's
`full`/`bg_patch`/`boxmap` variant definitions and retention formula, and
Experiment 11's noise-floor discipline (a change counts only if it clears
~2x the *arm's own* multi-seed standard deviation — the numeric floor from
Exp 11 was measured on the detector pipeline on different metrics and does not
transfer, only the decision rule's form does).

- **`12_adversarial_discriminator`** — a shared ResNet-18 backbone feeds both
  a main classification head (full image, ordinary loss) and a discriminator
  head (behind a Gradient Reversal Layer) that tries to predict the true label
  from that sample's `bg_patch` crop alone. Reversed gradients push the
  backbone toward representations from which the discriminator cannot predict
  the label from shortcut-only pixels — directly targeting `ret_bg_patch`.
  `boxmap` is a secondary, ablatable discriminator target (see
  `experiments/adversarial.py` for why it's riskier: a synthetic near-black
  canvas is out-of-distribution for a natural-image backbone).
- **`13_bbox_blur`** — Gaussian-blurs everything outside a heavily feathered,
  generously padded, per-sample-jittered region around the lesion box during
  training only. Guards explicitly against the trap Experiment 3 already
  documented for `bg_mask`: a hard blur boundary would itself encode box
  geometry. The verdict re-checks `ret_boxmap` for a *rise*, which would mean
  the model learned to read the blur boundary instead — see
  `experiments/bbox_blur.py`.
- **`14_channel_attention`** — squeeze-and-excite gates on ResNet-18's stages
  (inspired by, but simplified from, `Channel_Att_Bridge` in the
  UltraLight-VM-UNet repo — its cross-scale mixing across simultaneous U-Net
  skip connections has no analogue in a strictly sequential ResNet). Plain
  cross-entropy training is run as an **expected-null control** (channel
  attention alone has no mechanism to prefer lesion channels over shortcut
  channels); the motivated condition composes the same SE-augmented backbone
  with Experiment 12's discriminator signal.

Results land in `experiments/results/{12_adversarial,13_bbox_blur,14_channel_attention}.json`
as each experiment completes; see those files (and `cadsel/experiment_config.py`
for the shared, plug-and-play `RunConfig` used across the audit and the
production detector pipeline) for the latest numbers.

## SVM on frozen embeddings (ResNet-50, DINOv3)

`experiments/svm_classifier.py` -- a different kind of classifier entirely:
no fine-tuning, just an SVM (`sklearn.svm.SVC`) on top of frozen, pretrained
image embeddings, evaluated with the same patient-grouped CV
(`probes.grouped_folds`) as the decodability probes above.

```
python -m experiments.svm_classifier --backbone resnet --modality WLE --task binary
python -m experiments.svm_classifier --backbone dinov3 --modality WLE --task binary
```

Two embedding backbones (`experiments/embeddings.py`), cached to disk per
`(backbone, modality)` since extraction is the expensive part:

- **ResNet-50** (torchvision, ImageNet-pretrained) -- works out of the box.
- **DINOv3** (Meta, via `transformers`) -- the `facebook/dinov3-*` checkpoints
  on Hugging Face are **gated**. Request access on the model page (e.g.
  https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), then
  authenticate locally (`huggingface-cli login` or `HF_TOKEN` env var) before
  this will download weights -- verified in this session that it fails
  cleanly (a clear 401 from `transformers`, not a confusing crash) without
  that setup, but the successful path itself is untested here since it needs
  your own HF account and approval.

`scikit-learn` and `transformers` are new dependencies, used only by these two
files -- the rest of the project deliberately avoids scikit-learn (see
`probes.py`'s docstring); an actual SVM needs a real solver that isn't
practical to hand-roll the way the gradient-based decodability probes are.

## Caveats

- Experiments 3-5 use ResNet-18 classification, not the Faster R-CNN detectors of
  the main reproduction. Classification isolates "is the label predictable" from
  "is the lesion localisable", which is the question the shortcut literature
  asks; the detection results may differ in magnitude.
- The external set in Experiment 4 is ~100 patients with a different class mix,
  so prevalence shift contributes alongside any site effect. Its internal
  reference is also weak (subtype balanced accuracy 0.24-0.40), which makes the
  retention ratio noisy.
- Fixed seed, single split, 15 epochs per arm. These are diagnostics sized to
  separate large effects, not tuned benchmarks.
- Saliency/attribution methods (review §4.1) are deliberately not used as
  evidence: DeGrave et al. showed a plausible-looking saliency map is compatible
  with a provably shortcutting model.

## Mitigations, mapped to the review

1. Report worst-group (per-centre) performance alongside the mean (§9.8)
2. Keep patient-level splitting and state it explicitly (Usage Notes)
3. Mine hard negatives — lesion-free frames from the same patients (§7.1, §9.4)
4. Control for box geometry priors, or evaluate with size-matched controls (§5.1)
5. Style/colour randomisation across centres, EndoStyle-like (§5.1, §9.5)
6. Treat every number as provisional until validated on an external centre (§9.1)
