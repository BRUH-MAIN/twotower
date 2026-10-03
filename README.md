# CAD-SEL — reproduction of the technical-validation baselines

A PyTorch reimplementation of the baseline detection experiments in:

> Xu, L., Chen, S., Li, C. *et al.* **CAD-SEL: A dual-modal colonoscopy dataset of
> subepithelial lesion.** *Scientific Data* (2026).
> <https://doi.org/10.1038/s41597-026-07331-y>

It reproduces the dataset characterisation (Table 2, Fig 3) and the technical
validation grid behind **Table 4** (multi-class detection) and **Table 5**
(binary NET vs non-NET detection): four Faster R-CNN variants × two colonoscopy
modalities × two labelling schemes = 16 training runs.

> **Start with [FINDINGS.md](FINDINGS.md)** — consolidated results, the
> shortcut-learning audit, which mitigations work, and the Windows/PyTorch traps
> worth knowing before running anything. [experiments/README.md](experiments/README.md)
> has the audit in detail.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The `cu128` index is required for Blackwell (RTX 50-series, `sm_120`) GPUs. Drop
the `--index-url` for CPU or an older CUDA runtime.

`CAD-SEL-Dataset/` is expected next to this README, with the layout shipped on
figshare:

```
CAD-SEL-Dataset/
├── Images/Center-{1..6}/{WLE,EUS}-Set/<Subtype>/<Patient>/<n>.tiff
├── Labels/Center-{1..6}/{WLE,EUS}-Set/<Subtype>/<Patient>/<n>.txt
└── metadata.xlsx
```

## Usage

```powershell
# Verify the dataset against Table 2 / Fig 3
.\.venv\Scripts\python.exe -m cadsel.describe

# One configuration
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet --modality WLE --task multiclass --epochs 100

# The full 16-run grid, then render Tables 4 and 5
.\.venv\Scripts\python.exe -m cadsel.run_all --epochs 100
.\.venv\Scripts\python.exe -m cadsel.report

# Fig 4-style qualitative panel
.\.venv\Scripts\python.exe -m cadsel.visualize --checkpoint runs/fasterrcnn_mobilenet__WLE__binary/best_model.pth
```

Per-run artefacts land in `runs/<model>__<modality>__<task>/`
(`best_model.pth`, `history.json`, `result.json`, `split.json`); aggregate
tables land in `results/`.

### Performance levers

Additive, opt-in flags from a literature review of small-object/imbalanced
detection techniques (see `experiments/README.md`), targeting the confirmed
small-lesion AP gap (0.47 / 0.66 / 0.76 by size tertile) and the 112-vs-1611
class imbalance. All default off; none change existing runs unless passed.

```powershell
.\.venv\Scripts\python.exe -m cadsel.measure_anchors   # inspect recommended anchor sizes first

.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_mobilenet_320 --modality WLE --task multiclass `
    --auto-anchors --rpn-fg-iou-thresh 0.6 --box-fg-iou-thresh 0.5 --box-bg-iou-thresh 0.4 `
    --input-mode torchvision --multiscale-sizes 256,320,384 `
    --rfs-threshold 0.2 --ema --ema-decay 0.999
```

- `--auto-anchors` — replaces torchvision's COCO-tuned RPN anchor sizes with
  ones clustered (IoU k-means) from this run's own training box-size
  distribution (`cadsel/anchors.py`). `cadsel.measure_anchors` reports the
  per-modality/per-resolution recommendation without training anything.
- `--rpn-fg-iou-thresh` / `--rpn-bg-iou-thresh` / `--box-fg-iou-thresh` /
  `--box-bg-iou-thresh` — lower torchvision's 0.7/0.3/0.5/0.5 defaults; small
  boxes rarely reach high IoU with any anchor from discretisation alone,
  independent of anything visual.
- `--multiscale-sizes 256,320,384` — random per-image scale during training
  (only meaningful with `--input-mode torchvision`, since `reference` mode
  has already resized every image to one fixed size before the transform
  runs). Content-preserving, unlike `scale_crop`/`copy_paste` — a candidate
  for the augmentations that measurably backfire on EUS.
- `--rfs-threshold 0.1-0.3` — Repeat Factor Sampling (`cadsel/data.py`'s
  `RepeatFactorSampler`): oversamples *images* containing an
  under-threshold-frequency class, the image-level complement to the
  existing per-box class-weighted loss.
- `--ema` / `--ema-decay` — tracks an EMA of the weights (`torch.optim.swa_utils`);
  evaluation and checkpointing use the averaged weights instead of the raw
  ones once enabled.
- `cadsel.evaluate_tta --checkpoint ... --scales 480,640,800` — multi-scale
  Weighted Boxes Fusion at inference time (`cadsel/tta.py`), reported against
  the checkpoint's own single-scale baseline. Inference-only; needs a
  checkpoint trained with `--input-mode torchvision`.

A second, moderate-cost batch (Tier 2 of the literature review):

- `--loss-fn seesaw --seesaw-p 0.8` — Seesaw Loss (Wang et al. 2021) in place
  of the paper's static inverse-frequency class weighting: tracks *cumulative*
  per-class counts as training progresses and damps how much a frequent
  class's logit suppresses a rarer one's, adapting instead of using a fixed
  pre-training snapshot. Mutually exclusive with `--no-class-weights`, which
  only applies to the default `weighted_ce`.
- `--augment flip,color,rotate,speckle_noise,ultrasound_gain` — two
  physically-motivated EUS augmentations (`cadsel/data.py`): multiplicative
  speckle noise and a depth-dependent gain gradient (simulating
  time-gain-compensation), replacing the content-altering augmentations that
  measurably backfire on EUS. Opt these in only for `--modality EUS` runs.
- `cadsel.finetune_rare --checkpoint ... --epochs 10` — TFA-style two-stage
  fine-tuning: freezes everything except the box predictor and briefly
  fine-tunes on a class-balanced subset that oversamples the rarest classes
  (`cadsel/finetune_rare.py`), run against an already-trained checkpoint.
- `--backbone-weights PATH` — loads a raw `state_dict` into the backbone
  (`cadsel/models.py::load_backbone_weights`) in place of COCO pretraining,
  for a domain-specific backbone (e.g. RadImageNet for EUS, an in-domain
  endoscopy self-supervised checkpoint for WLE). This provides the loading
  mechanism only — no specific third-party weights are bundled or fetched;
  find and convert them yourself, then point this flag at the resulting file.
  Reports matched/missing/unexpected key counts so an incompatible file is
  obvious immediately rather than silently loading nothing.

A third batch (Tier 3 — architecture-level; see `FINDINGS.md` §5c for why the
other two Tier-3 items, cross-modal domain adaptation and paired dual-modality
fusion, were skipped rather than implemented):

- `--model fcos_resnet50` — an anchor-free detector (FCOS, via torchvision's
  own implementation) as an alternative to the Faster R-CNN family.
  `--auto-anchors`/IoU-threshold flags/`--loss-fn` don't apply to it (flagged
  with a NOTE if passed). **Needs a much lower learning rate than the other
  models' default** — `--learning-rate 1e-4 --warmup-iters 0` trained stably
  in testing; the default 1e-3 makes it diverge within ~1 epoch (a WARNING
  fires if you forget).
- `cadsel.train_lesion_gan --modality EUS --epochs 300` trains a small
  class-conditional GAN on real lesion crops (`cadsel/lesion_gan.py`); then
  `cadsel.train --augment ...,copy_paste --copy-paste-source generated
  --lesion-gan-checkpoint runs/lesion_gan_EUS_multiclass.pth` pastes
  synthetic lesions instead of real ones cropped from other patients. This is
  a small, tractable prototype (not a diffusion model) — EUS's rarest class
  has only ~64 real crops to train the generator on, so treat generated
  patches with real skepticism until you've looked at some.

### Multimodal (WLE + EUS)

The paper's baselines are unimodal. This is a two-stage multimodal system: a joint
modality-conditioned detector, then a patient-level fusion head. See FINDINGS.md
§5b for what the data does and does not allow — in particular, the per-modality
baseline splits leak across modalities and cannot be used as the comparison.

```powershell
# Pre-flight: split integrity, per-modality normalisation, conditioning. Run first.
.\.venv\Scripts\python.exe -m cadsel.verify_joint

# The grid in dependency order: shared-split baselines -> blind control -> conditioned
.\.venv\Scripts\python.exe -m cadsel.run_multimodal --epochs 100 --early-stop-patience 15

# Stage 2: patient-level fusion, 5-fold CV, four ablations
.\.venv\Scripts\python.exe -m cadsel.train_fusion `
    --checkpoint runs_joint\joint_film__fasterrcnn_mobilenet__multiclass\best_model.pth

.\.venv\Scripts\python.exe -m cadsel.report        # writes results/MULTIMODAL.md too
```

### Separate per-modality backbones (alternative to FiLM conditioning)

Stage 1 above shares one backbone across both modalities (with per-modality
FiLM conditioning) — a deliberate choice given only 421 EUS patients. The
alternative: two independent detectors, one per modality, each with its own
backbone — e.g. an ultrasound-domain-pretrained backbone for EUS specifically,
so it needs less task data to be useful instead of more.

```powershell
# one-time: convert Lab-Rasool/RadImageNet's ResNet50 to torchvision's key convention
.\.venv\Scripts\python.exe -m cadsel.convert_radimagenet_weights

# train two independent detectors -- both must use a ResNet50-family model
# (fasterrcnn_resnet / fasterrcnn_resnet50_fpn_v2) for the RadImageNet backbone
# to be architecture-compatible, and must share --img-size/--input-mode
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_resnet --modality WLE --task multiclass `
    --run-name wle_resnet_backbone
.\.venv\Scripts\python.exe -m cadsel.train --model fasterrcnn_resnet --modality EUS --task multiclass `
    --backbone-weights runs\radimagenet_resnet50.pth --run-name eus_radimagenet_backbone

# Stage 2 fusion over the two separate checkpoints instead of one joint model
.\.venv\Scripts\python.exe -m cadsel.train_fusion `
    --checkpoint-wle runs\wle_resnet_backbone\best_model.pth `
    --checkpoint-eus runs\eus_radimagenet_backbone\best_model.pth
```

`cadsel/fusion.py::extract_features` accepts either one shared model (the
original Stage 1 path, unchanged) or a `{"WLE": model, "EUS": model}` dict,
routing each modality's frames to its own detector. Verified end-to-end in
this session (RadImageNet backbone loaded with 0 missing/unexpected keys;
full extraction + all 4 fusion ablations ran cleanly) — no training results
yet, and there's no shared-backbone multimodal baseline run yet either to
compare against (see `FINDINGS.md` §8 item 0).

Multimodal runs land in `runs_joint/`, kept apart from `runs/` so a shared-split
baseline can never overwrite a Table 4 row (`report._index` requires the canonical
`model__modality__task` name for exactly this reason).

Joint runs print per-modality AP every epoch, since a pooled number would hide a
model that helps WLE by trading EUS away:

```
  epoch  12/100  train_loss=0.4102  val_loss=0.5013  macroAP=0.4551  [EUS=0.4580  WLE=0.4522]  F1=0.6104  lr=1.00e-03  (128s) *
```

Every epoch prints one flushed line, so progress is visible live:

```
  epoch  12/100  train_loss=0.3418  val_loss=0.4127  AP@0.5=0.7264  P=0.7810  R=0.8033  F1=0.7920  lr=1.00e-03  (85s) *
```

A trailing `*` marks a new best AP@0.5 (that epoch's weights are checkpointed).
Add `--no-progress` to drop the tqdm bars but keep these lines — useful when
redirecting to a log file:

```powershell
.\.venv\Scripts\python.exe -u -m cadsel.run_all --epochs 100 --no-progress 2>&1 | Tee-Object runs\grid.log
```

## Runtime

Measured on an RTX 5060 Laptop (8 GB) with 16 CPU threads, AMP on, steady-state
seconds per epoch (the first epoch of each run adds ~3-4 min of worker start-up,
cuDNN autotuning and one-off weight downloads):

| Model | img | batch | WLE s/epoch | EUS s/epoch |
| --- | --- | --- | --- | --- |
| `fasterrcnn_resnet` | 640 | 4 | 181 | ~135 |
| `fasterrcnn_resnet50_fpn_v2` | 640 | 4 | ~215 (est.) | ~160 (est.) |
| `fasterrcnn_mobilenet` | 640 | 8 | 85 | ~63 |
| `fasterrcnn_mobilenet_320` | 320 | 16 | ~55 | 40 |

One epoch across all 16 runs is roughly **31 min** of serial GPU time.

### Running two at once

A single run does not saturate the GPU, so two concurrent runs are nearly free.
Measured, pairing `fasterrcnn_resnet`/WLE with `fasterrcnn_mobilenet_320`/EUS at
`--num-workers 6` each:

| Run | Serial s/epoch | Paired s/epoch |
| --- | --- | --- |
| `fasterrcnn_resnet` WLE | 181 | 182-189 |
| `fasterrcnn_mobilenet_320` EUS | 40 | 40-67 |

The heavy run lost ~4%. Peak VRAM leaves room for it — 2.58 GiB (ResNet),
3.34 (ResNet-v2), 1.43 (MobileNet), 1.72 (MobileNet-320) against 8 GiB total.

Two is the practical limit: three runs need 3 x 5 workers on 16 threads and two
640 px ResNets already occupy ~5.9 GiB. Pair one 640 px ResNet with one
MobileNet rather than stacking both ResNets, and drop `--num-workers` to 6 so
the two processes do not oversubscribe the CPU.

Split the grid into two balanced streams by cost (~940 s/epoch each):

```powershell
# stream 1
.\.venv\Scripts\python.exe -u -m cadsel.run_all --models fasterrcnn_resnet50_fpn_v2 fasterrcnn_mobilenet_320 `
    --epochs 60 --early-stop-patience 12 --num-workers 6 --no-progress 2>&1 | Tee-Object runs\stream1.log

# stream 2, in a second terminal
.\.venv\Scripts\python.exe -u -m cadsel.run_all --models fasterrcnn_mobilenet fasterrcnn_resnet `
    --epochs 60 --early-stop-patience 12 --num-workers 6 --no-progress 2>&1 | Tee-Object runs\stream2.log

# then merge both into the tables
.\.venv\Scripts\python.exe -m cadsel.report
```

Both streams write into the same `runs/` tree under distinct run names, and
`cadsel.report` picks up whatever is finished.

### Grid cost

| Epochs | Serial | Two streams |
| --- | --- | --- |
| 100 (reference default) | ~53 h | ~31 h |
| 60 | ~32 h | ~19 h |
| 30 | ~17 h | ~10 h |
| 20 | ~11 h | ~7 h |

Add roughly 4 min of start-up per run (worker spin-up and cuDNN autotuning);
`--early-stop-patience` typically cuts the totals well below these ceilings.

TIFF decoding, not the GPU, is the bottleneck for the lighter models — a decode
plus resize costs ~26 ms/image against ~28 s of GPU time per MobileNet-320
epoch. `--num-workers` defaults to `min(8, cores/2)`.

## Layout

| File | Purpose |
| --- | --- |
| `cadsel/config.py` | Paths, class schemes, detector registry, normalisation stats |
| `cadsel/data.py` | Manifest building, patient-level splits, dataset, augmentation |
| `cadsel/models.py` | The four detectors + class-weighted ROI classification loss |
| `cadsel/metrics.py` | AP@0.5 (VOC all-point), precision, recall, F1 |
| `cadsel/engine.py` | Train / evaluate loops |
| `cadsel/train.py` | Single-run CLI (`--joint-split` for a shared-split baseline) |
| `cadsel/run_all.py` | The 16-run grid |
| `cadsel/report.py` | Table 4 / Table 5 + multimodal rendering |
| `cadsel/describe.py` | Table 2 / Fig 3 reproduction |
| `cadsel/visualize.py` | Fig 4 reproduction |
| `cadsel/train_joint.py` | Joint WLE+EUS detector (Stage 1) |
| `cadsel/fusion.py` | Patient-level two-stream fusion head (Stage 2) |
| `cadsel/train_fusion.py` | Stage 2 training, 5-fold CV + ablations |
| `cadsel/run_multimodal.py` | The multimodal grid, in dependency order |
| `cadsel/verify_joint.py` | Pre-flight checks for the multimodal pipeline |
| `cadsel/measure_stats.py` | Per-modality normalisation statistics |

## Reproduction choices

**Label space.** The paper describes five subtypes (`NET_G1`, `NET_G2`,
`Leiomyoma`, `Lipoma`, `NonNeoplasm`), but the released bounding-box annotations
use **four** class ids — `0 = NET` (G1 and G2 collapsed), `1 = Leiomyoma`,
`2 = Lipoma`, `3 = NonNeoplasm`. The multi-class detection task therefore has
four foreground classes; NET grade is preserved as directory metadata and is
used to stratify the split. The binary task maps `{1,2,3} → non-NET`.

**Patient-level splits.** Assignment is by patient, never by image, as the
paper's Usage Notes require. Splits are stratified by subtype so every class is
represented in validation; `--folds K` switches to stratified patient-level
K-fold. The realised split is written to `split.json` for every run.

**Class weighting.** Weights are inversely proportional to per-class box
frequency in the training split and normalised to sum to the number of
foreground classes, matching `calculate_class_weights` in the authors'
repository. They are applied to the ROI-head classification loss by installing a
weight-aware `fastrcnn_loss`; the box-regression term is left identical to
torchvision's.

**Optimisation.** AdamW at lr 1e-3, `ReduceLROnPlateau(patience=5, factor=0.5)`
on validation loss, gradient-norm clipping at 1.0, COCO-pretrained backbones,
model selection on best validation AP@0.5 — all following the reference
repository. Batch sizes default to 4/4/8/16 across the four detectors to fit an
8 GB GPU (the reference used 16 throughout); override with `--batch-size`.

**Epoch budget.** The reference README trains for 100 epochs, and that remains
the default. Because every run checkpoints on best validation AP@0.5, training
longer only costs time, never accuracy — so the cheaper route is
`--epochs 60 --early-stop-patience 12`, which halts a run once validation AP has
not improved for 12 epochs. With `ReduceLROnPlateau(patience=5)` the first LR cut
lands around epoch 6-15, and fine-tuning COCO-pretrained detectors on ~2,200
images normally settles well before 60. Treat that as a planning heuristic, not
a measured convergence point for this dataset.

One addition: a linear LR warmup over the first epoch (`--warmup-iters`,
default 500, capped at one epoch). Without it, AdamW at 1e-3 on a randomly
initialised box predictor spikes the first-epoch loss by an order of magnitude
(4.97 vs 0.61 measured on `fasterrcnn_mobilenet_320`/EUS/binary) and converges
to a worse epoch-3 AP. Pass `--warmup-iters 0` for the unmodified reference
schedule.

**Metrics.** AP@0.5 is VOC all-point interpolation over *all* detections, which
is the standard definition. The reference implementation instead scores only
detections above the 0.5 confidence threshold, which truncates the PR curve; that
variant is reported alongside as `AP@0.5_conf`. Precision, recall and F1 use the
paper's formulas at IoU 0.5 and confidence 0.5, micro-averaged over classes.

**Training resolution.** The paper does not state one. It reports only the
native image sizes — 362 x 290 to 1920 x 1080, median 1160 x 1008, with "no
resizing, resampling, color normalization, or contrast enhancement" applied to
the released files. The training resolution therefore comes from the reference
repository, whose `--img_size` defaults to **640** for all four models. That
value is not the whole story: the reference passes its 640 x 640 tensors to the
torchvision detector, whose internal `GeneralizedRCNNTransform` resizes a second
time — to min side 800 for the ResNet variants, and min 320 / max 640 for the
`_320` model. Here the internal transform is pinned to `--img-size` so the
network sees exactly that: 640 px for three models and 320 px for
`fasterrcnn_mobilenet_320`, which matches both its effective reference input and
the architecture's intent. Pass `--img-size 640` to put every model on 640.

**Input pipeline.** `--input-mode reference` (default) resizes to a square
`--img-size` and normalises with the CAD-SEL channel statistics from the authors'
dataloader, with the detector's internal transform neutralised so normalisation
happens exactly once. The reference code leaves torchvision's transform active
on top of its own normalisation, i.e. normalises twice; that is treated here as
unintended. `--input-mode torchvision` hands raw `[0, 1]` images to the detector
and lets its built-in transform resize and normalise.

**Augmentation.** Horizontal flip, brightness/contrast/saturation jitter and
±15° rotation with box recomputation, each at p=0.5 on the training split,
mirroring the reference dataloader.

## Verification

`python -m cadsel.describe` reproduces Table 2 **exactly** — all 14 patient and
image counts across both modalities, and 4,912 images in total. Fig 3 matches on
mean age (52.0 y), sex (51.3% male / 48.7% female) and location (rectum 81.4%).
The only divergence is the lesion-size histogram: this code bins `0.5-1 cm` as
`0.5 < s <= 1.0`, giving 59.8% against the paper's 54%, so the paper evidently
treats exactly-1.0 cm lesions as `1-2 cm`. Nothing downstream depends on it.

## Caveats

The paper does not state the validation-split ratio, fold count, or random seed,
and the authors' published splits were not available, so absolute numbers will
not match published values exactly. `results/REPORT.md` prints the reproduction
and the published figures side by side with deltas, so the comparison is
explicit.

Table 4 reports a single AP@0.5 per model/modality, i.e. the mean over the
foreground classes. With four classes here rather than the five subtypes named
in the text, and with the rare classes (Leiomyoma has only 137 WLE / 79 EUS
images) dominating that mean, the multi-class AP is sensitive to which patients
land in validation. The binary task is far more stable.
