# Paper pack: two-tower WLE + EUS classification of colorectal subepithelial lesions (CAD-SEL)

Written for: the CAD-SEL team drafting the research paper. Compiled 2026-10-22 from the experiment outputs in this repository (branch `external-datasets`, commit 8cf29b0 or later).

Every number here comes from a script output, with its source path given. Numbers that will change when pending runs land are marked **[pending]**. This file has two companions:
- `LITERATURE_REVIEW.md`: related work, verified citations and the research-gap matrix.
- `EXPLAINABILITY_INFERENCE.md`: a plot-by-plot interpretation of the explainability figures.

---

## 1. One-paragraph summary (draft abstract material)

We study patient-level classification of colorectal subepithelial lesions from white-light endoscopy (WLE) and endoscopic ultrasound (EUS) on the public CAD-SEL dataset (641 patients, 6 centres). The model is a two-tower network: one image encoder per modality, attention pooling over each patient's frames, and a gated fusion head. Our protocol avoids selection leakage: nested, patient-level 5-fold cross-validation in which the checkpoint, LR schedule and early stop are chosen on an inner split, and the outer fold is only scored.

Across seven tower backbones and three seeds, DINOv3 ViT-S/16 towers perform best:
- **binary NET vs non-NET:** balanced accuracy 0.774 (3-seed ensemble 0.806, 95% CI 0.769–0.838)
- **5-way subtype:** 0.455 (ensemble 0.479; the MobileViT-S ensemble reaches 0.509, not significantly different)

Learned end-to-end fusion adds little over WLE alone, and the EUS tower is under-used. On the same folds, late fusion of separately trained WLE and EUS models is never worse than the end-to-end two-tower, and for DINOv3 subtype it is better by +0.105 (95% CI +0.037 to +0.177).

Counterfactual inputs show the models rely on lesion pixels: removing the lesion drops every model to chance. But they also depend on a ground-truth lesion crop: on full frames they fall to near chance. On an external ulcerative-colitis dataset (LIMUC), the same pipeline reaches quadratic weighted κ 0.847, which suggests it is competitive outside CAD-SEL.

---

## 2. Dataset: CAD-SEL

Source: Xu L, Chen S, Li C, et al. *Sci Data* 2026;13:1007, doi:10.1038/s41597-026-07331-y, CC BY 4.0.

| Item | Value |
|---|---|
| Patients | 641; 421 have EUS; there are no EUS-only patients, so 220 are WLE-only |
| Images | 4,912 (WLE 2,817, EUS 2,095); median about 4 frames per modality per patient |
| Centres | 6. Center-1 has 537 patients, Center-3 68, Center-2 25, Centers 4/5/6 have 4/4/3 |
| Subtype counts (all 641) | NET G1 411, NET G2 44, lipoma 96, leiomyoma 24, non-neoplasm 66 |
| Binary | NET (G1 + G2) 455 vs non-NET 186 |
| Pairing | WLE and EUS frames are linked by patient only, with no frame correspondence |
| Annotations | YOLO lesion boxes on every frame (used for the lesion crop) |
| Splits | patient-level only. Patient keys are globally unique pseudonyms (`cadsel/data.py`) |

Sources: `README.md`, `FINDINGS.md` §5b, `experiments/data/patient_centre.csv`.

Correction for the old report: `report/main.tex` line 101 says "NET_G1 alone is 299 of 641". 299 is the paired-subset count; the full count is 411.

---

## 3. Methods

### 3.1 Model (`cadsel/joint_fusion_model.py`, `cadsel/fusion.py`)

- **Towers:** one encoder per modality, with the same architecture for both and no weight sharing.
  - Only real frames are encoded; absent or padded frames get a zero descriptor.
  - Each tower outputs a pooled descriptor per frame: the CLS token for the ViTs, global pooling for the CNNs.
- **Per-modality attention pooling** over a patient's frames: Linear(D→128) → tanh → Linear(128→1), then a masked softmax.
- **Per-modality encoder:** Linear(D→256) → ReLU → Dropout(0.3); multiplied by the modality's presence flag.
- **Gated fusion** (`GatedFusion`): `g = σ(MLP([w, e, avail]))` and `h = g·proj([w, e]) + (1−g)·w`.
  - It falls back to WLE when EUS is absent.
  - `avail` is the two presence flags.
- **Classifier:** Linear(256→256) → ReLU → Dropout(0.3) → Linear(256→C).
- **Modality dropout 0.3** in training: a stream is zeroed at random, but never the last one present.
- **Single-backbone variant** (§6.6): the same code with one modality. The pooled, encoded stream goes straight to the classifier.

### 3.2 Backbones (towers)

| Backbone | Source / pretraining | Fine-tuning | Normalisation |
|---|---|---|---|
| EfficientNet-B0 | torchvision ImageNet | full | per-modality dataset stats |
| ResNet-50 | torchvision ImageNet | full | per-modality dataset stats |
| ConvNeXt-Tiny | torchvision ImageNet | full | per-modality dataset stats |
| MobileNetV3-Large | torchvision ImageNet | full | per-modality dataset stats |
| MobileViT-S | HF `apple/mobilevit-small` | full | raw [0, 1], BGR |
| DeiT-S | HF `facebook/deit-small-patch16-224` | last 4 blocks + final norm | ImageNet |
| DINOv3 ViT-S/16 | HF `facebook/dinov3-vits16-pretrain-lvd1689m` (LVD-1689M, self-supervised) | last 4 blocks + final norm | ImageNet |

Per-modality stats (whole dataset, `cadsel/config.py` `MODALITY_STATS`): WLE mean (0.489, 0.290, 0.224), std (0.295, 0.216, 0.176); EUS mean 0.116, std 0.136 (grey).

### 3.3 Input

- **Lesion crop:** the largest ground-truth box plus 10% context padding on each side, resized to 224×224 without keeping the aspect ratio (`cadsel/joint_fusion_data.py`).
- **Frames:** 4 per modality per patient, sampled with replacement. The draw is deterministic per patient at evaluation.
- **Augmentation** (training only): RandomResizedCrop(224, scale 0.7–1.0), horizontal flip, ColorJitter(0.2, 0.2, 0.2), RandomRotation(10°). Each frame and modality is augmented independently.

### 3.4 Training (`experiments/src/34_joint_fusion_train.py`)

- **Optimisation:** AdamW, lr 1e-4, weight decay 1e-4, batch 8 patients, fp16 autocast.
- **Loss:** class-weighted cross-entropy. Weights are inverse class frequency on the fit split, normalised to sum to C.
- **Schedule:** ReduceLROnPlateau on select-split balanced accuracy (mode max, patience 5, factor 0.5); at most 100 epochs; early stop with patience 20.
- **Checkpoint:** the epoch with the best select-split balanced accuracy.

### 3.5 Evaluation protocol: nested, patient-level, leak-free

1. **Outer folds:** 5-fold patient-level CV, stratified on (subtype, has-EUS) (`patient_level_folds`). Fold sizes are 132 / 131 / 129 / 126 / 123 patients.
2. **Inner split:** within each outer-train fold, 20% of patients form a select split (stratified the same way, seed + 1000 + k). The checkpoint, LR plateau and early stop use the select split only. **The outer fold is only scored.**
3. **Seeds:** 42, 123 and 7. The split depends on the seed, and is identical across backbones and tasks for a given seed.
4. **Masked-inference views** of the same trained model, with the other modality's mask zeroed at test time:
   - WLE-only, on all 641 patients
   - EUS-only, on the 421 with EUS
   - fused, on all 641 patients and on the 421
5. **Metrics:**
   - balanced accuracy (primary; macro recall)
   - macro-F1
   - ROC AUC: binary uses the NET class; subtype uses the macro one-vs-rest
   - accuracy (reported but misleading: always predicting NET G1 gives 0.64)
6. **Reported as:** the mean over seeds of the 5-fold mean, plus fold SD. Pooled out-of-fold scores are used for the confusion matrices, bootstrap CIs and clinical analyses.
7. **Ensembles:** the mean of the class probabilities of the 3 seeds per patient.
8. **Leakage audit:** `experiments/src/40_two_tower_split_audit.py`. No patient appears in two of fit / select / test, and there are no cross-patient duplicate frames.

### 3.6 Statistics (`experiments/src/44_two_tower_stats.py`)

- **CIs:** patient-level stratified bootstrap, 2000 resamples, 95% percentile intervals.
- **Comparisons:** paired bootstrap of the balanced-accuracy difference plus exact McNemar on per-patient correctness, Holm-corrected within each panel.
- These CIs cover patient sampling, not training randomness; the seed spread is reported separately.

### 3.7 Explainability (`experiments/src/43_two_tower_explain.py`)

- **Grad-CAM** for all seven towers, from the fused logit of the predicted class. CNNs use their last conv stage; ViTs use the token sequence entering the last block, since the last block's output is read only at CLS.
- **CLS attention** (last layer, mean over heads) for the ViTs.
- **Attention-pool weights**, the **mean fusion gate**, and **fused-embedding kNN decodability** of class and centre.
- **Box energy:** the share of CAM mass inside the ground-truth box.
- **Model-parameter randomisation test** (Adebayo et al. 2018).
- **Counterfactual input gate** (`experiments/src/36_joint_fusion_counterfactual.py`): lesion crop vs full frame, lesion filled with its mean colour, a lesion-free background patch, and a box outline only.

Every map comes from a fold model that never saw that patient. Recomputed probabilities match the cached out-of-fold predictions; `experiments/results/43_two_tower_explain/verification.json` holds the per-run drift.

---

## 4. Main results: backbone ablation (3 seeds, 42 runs)

Source: `experiments/results/41_two_tower_ablation/metrics_summary.csv` (fold means) and `experiments/results/42_two_tower_confusion/per_class.csv` (pooled ensembles).

### 4.1 Binary NET vs non-NET (fused balanced accuracy)

| Backbone | Fused bal (3-seed mean) | AUROC | Macro-F1 | WLE-only view | EUS-only view (421) | 3-seed ensemble bal |
|---|---|---|---|---|---|---|
| **DINOv3 ViT-S/16** | **0.774** | **0.862** | **0.760** | 0.765 | 0.687 | **0.806** |
| DeiT-S | 0.753 | 0.831 | 0.753 | 0.737 | 0.675 | 0.761 |
| MobileViT-S | 0.752 | 0.839 | 0.739 | 0.723 | 0.698 | 0.771 |
| MobileNetV3-L | 0.751 | 0.838 | 0.748 | 0.735 | 0.664 | 0.787 |
| EfficientNet-B0 | 0.741 | 0.821 | 0.727 | 0.731 | 0.628 | 0.765 |
| ConvNeXt-T | 0.728 | 0.811 | 0.727 | 0.713 | 0.595 | 0.754 |
| ResNet-50 | 0.724 | 0.790 | 0.707 | 0.699 | 0.613 | 0.751 |

### 4.2 Five-way subtype (fused balanced accuracy; chance 0.20)

| Backbone | Fused bal (3-seed mean) | AUROC | Macro-F1 | WLE-only view | EUS-only view (421) | 3-seed ensemble bal |
|---|---|---|---|---|---|---|
| **DINOv3 ViT-S/16** | **0.455** | **0.777** | **0.406** | 0.447 | 0.361 | 0.479 |
| MobileViT-S | 0.447 | 0.746 | 0.357 | 0.447 | 0.318 | **0.509** |
| MobileNetV3-L | 0.424 | 0.732 | 0.363 | 0.406 | 0.337 | 0.472 |
| EfficientNet-B0 | 0.421 | 0.733 | 0.361 | 0.418 | 0.339 | 0.454 |
| ConvNeXt-T | 0.405 | 0.709 | 0.357 | 0.396 | 0.292 | 0.406 |
| DeiT-S | 0.395 | 0.722 | 0.344 | 0.377 | 0.351 | 0.425 |
| ResNet-50 | 0.384 | 0.690 | 0.329 | 0.377 | 0.308 | 0.411 |

Per-seed values are in `metrics_summary.csv` (`per_seed` column). Seed spread is largest for MobileViT-S subtype (0.413–0.486) and smallest for DINOv3 binary (0.773–0.776).

### 4.3 Uncertainty and significance (`experiments/results/44_two_tower_stats/`)

- **DINOv3 pooled out-of-fold balanced accuracy, with 95% bootstrap CI:**
  - binary: 0.774 [0.736, 0.808] at seed 42; ensemble 0.806 [0.769, 0.838]
  - subtype: 0.449 [0.398, 0.502] at seed 42; ensemble 0.479 [0.424, 0.533]
- **Seed 42, single model:** after Holm correction, no backbone difference from DINOv3 is significant (smallest p = 0.11).
- **3-seed ensembles, binary:** DINOv3 beats ResNet-50 (Δ +0.054, p = 0.012) and ConvNeXt-T (+0.051, p = 0.035). Against the others, p ranges from 0.09 to 0.24.
- **3-seed ensembles, subtype:** DINOv3 beats ConvNeXt-T (+0.072, p = 0.042). The MobileViT-S ensemble is higher than DINOv3 (Δ −0.030 [−0.091, +0.030], p = 0.98); not significant.
- **Claim to make:** "DINOv3 towers are the most consistent choice and significantly better than the ResNet-50 / ConvNeXt baselines as ensembles; differences among the top four backbones are within noise."

### 4.4 Per-class behaviour (DINOv3 3-seed ensemble, pooled)

| Class | Support | Recall | Precision |
|---|---|---|---|
| NET (binary) | 455 | 0.864 | 0.893 |
| non-NET (binary) | 186 | 0.747 | 0.692 |
| NET G1 | 411 | 0.754 | 0.805 |
| NET G2 | 44 | **0.091** | 0.071 |
| Leiomyoma | 24 | 0.500 | 0.279 |
| Lipoma | 96 | 0.823 | 0.670 |
| Non-neoplasm | 66 | 0.227 | 0.385 |

NET G2 is almost never recognised: 32 of 44 patients are called NET G1 (`plots/two_tower_confusion/cm_subtype_fused_ens_grid.png`). Lipoma is the easiest non-NET class. The grade split (G1 vs G2) and non-neoplasm are the open problems.

### 4.5 Clinical framing (binary, `clinical.csv`)

| | Bal acc | Sens at 90% spec | Spec at 95% sens | ECE | Bal at 80% coverage | Bal at 50% coverage |
|---|---|---|---|---|---|---|
| DINOv3 seed 42 | 0.774 | 0.40 | 0.54 | 0.143 | 0.825 | 0.873 |
| DINOv3 3-seed ensemble | 0.806 | 0.60 | 0.58 | 0.051 | 0.854 | 0.918 |
| EfficientNet-B0 ensemble | 0.765 | 0.49 | 0.51 | 0.044 | 0.831 | 0.891 |

- Operating points are read off the same patients they are scored on, so they are optimistic.
- Ensembling cuts calibration error about threefold.
- Referring the 20% least-confident patients raises balanced accuracy by about 5 points.

---

## 5. What fusion does

### 5.1 Learned fusion adds little; EUS is under-used

Source: `fusion_value.csv` (seed 42, the 421 patients with EUS).

| Backbone | WLE-only view | EUS-only view | Late fusion of the views | Learned fusion | EUS fixes / breaks (McNemar p) |
|---|---|---|---|---|---|
| DINOv3, binary | 0.750 | 0.670 | 0.761 | 0.742 | +22 / −13 (0.18) |
| DINOv3, subtype | 0.469 | 0.381 | 0.465 | **0.441** | +58 / −18 (<0.001) |
| EfficientNet-B0, subtype | 0.461 | 0.321 | 0.453 | 0.441 | +49 / −18 (<0.001) |

- On subtype, adding EUS fixes more patients than it breaks, but almost all the fixes are majority-class NET G1. Minority-class recall drops (non-neoplasm and NET G2 are pulled into NET G1). Accuracy rises while balanced accuracy falls (`plots/two_tower_confusion/cm_subtype_fusion_delta_s42_grid.png`).
- On binary, the training-free late fusion of the model's own two views is as good as or better than learned fusion for 5 of 7 backbones.
- The mean fusion gate is about 0.49–0.57, and the gated path keeps a direct WLE branch, so the architecture is biased towards WLE by design.
- Adding EUS raises confidence in the right answer for NET G1 and lipoma (lipomas are distinctive on EUS) and lowers it for NET G2, leiomyoma and non-neoplasm (`plots/two_tower_explain/gate_and_eus_contribution_subtype.png`).

### 5.2 Separately trained single-modality models vs the two-tower

Source: `experiments/results/47_cross_dataset/cadsel_fusion_strategies.csv`, figure `plots/cross_dataset/cadsel_fusion_strategies.png`. Seed 42, the 421 patients with EUS, identical outer folds for every model.

| Backbone / task | WLE-only model | EUS-only model | Two-tower fused | Late fusion of the two models | Late − two-tower [95% CI] |
|---|---|---|---|---|---|
| EfficientNet-B0, binary | 0.740 | 0.713 | 0.742 | 0.739 | −0.003 [−0.052, +0.047] |
| DINOv3, binary | 0.750 | 0.676 | 0.742 | 0.747 | +0.005 [−0.040, +0.050] |
| EfficientNet-B0, subtype | 0.386 | 0.435 | 0.441 | 0.440 | +0.001 [−0.078, +0.083] |
| **DINOv3, subtype** | 0.487 | 0.429 | 0.441 | **0.545** | **+0.105 [+0.037, +0.177]** |

On all 641 patients (late fusion falls back to the WLE model where there is no EUS):
- EfficientNet binary: 0.765 (late) vs 0.744 (two-tower)
- DINOv3 binary: 0.765 vs 0.774
- EfficientNet subtype: 0.471 vs 0.415
- DINOv3 subtype: 0.490 vs 0.449

What this establishes:
- A **dedicated EUS model is far better than the two-tower's EUS view**: binary 0.68–0.71 vs 0.64–0.67; subtype 0.43 vs 0.32–0.38. So the end-to-end model under-trains its EUS tower.
- **EUS carries subtype information on its own.** With EfficientNet, the EUS-only model beats the WLE-only model on subtype (0.435 vs 0.386).
- These comparisons use seed 42 only. Confirming them with seeds 123 and 7 is a cheap follow-up (8 more single-modality runs). **[pending, optional]**

### 5.3 Single-modality CAD-SEL baselines under the nested protocol

Seed 42; `comparison.csv`:
- WLE-only model: binary 0.764 (EfficientNet) / 0.762 (DINOv3); subtype 0.441 / 0.461
- EUS-only model (421 patients): binary 0.713 / 0.676; subtype 0.435 / 0.429

The earlier per-modality classifier numbers in `MODELS_TESTED.md` are image-level and non-nested, so they are not comparable.

---

## 6. Does the model look at the lesion? (explainability and causal checks)

Full interpretation: `EXPLAINABILITY_INFERENCE.md`.

### 6.1 Counterfactual inputs (exp. 36, seed 42; balanced accuracy)

| Run | Lesion crop (as trained) | Full frame | Lesion filled with mean colour | Background patch | Box outline only |
|---|---|---|---|---|---|
| DINOv3 binary | 0.773 | 0.567 | 0.506 | 0.513 | 0.494 |
| EfficientNet-B0 binary | 0.743 | 0.516 | 0.456 | 0.499 | 0.460 |
| DINOv3 subtype | 0.450 | 0.317 | 0.213 | 0.215 | 0.214 |
| ResNet-50 subtype | 0.429 | 0.234 | 0.191 | 0.209 | 0.255 |

- Removing the lesion, or keeping only its box geometry, drops every model to chance (0.5 / 0.2). The predictions come from lesion pixels, not context or box shape.
  - Caveat: box geometry alone scores 0.66 binary / 0.35 subtype with a simple classifier (`FINDINGS.md`). The trained towers don't exploit it; the exception is ResNet-50 subtype, which retains 24% on the box outline.
- **Full frames also fail.** Every model needs the ground-truth lesion crop. That makes this an oracle-box result, not a deployable one (§9).

### 6.2 Saliency

Source: `experiments/results/43_two_tower_explain/summary.csv`, figure `plots/two_tower_explain/box_energy.png`.

- **Grad-CAM mass inside the box** vs a uniform map's 0.71: WLE 0.71–0.85 depending on backbone; EUS 0.66–0.84. The lesion crop leaves only a 10% margin, so this is a weak test.
- **CAM peak inside the box:**
  - DINOv3: 0.81 (WLE) and 0.76–0.82 (EUS)
  - EfficientNet: 0.72–0.80 (WLE) and 0.43–0.46 (EUS)
  - ConvNeXt: 0.35 on EUS, mostly in the crop margin
- **Sanity check:** median Spearman ρ between trained and tower-randomised CAMs is −0.05 to +0.05 (DINOv3 and EfficientNet). The maps depend on the learned weights (`plots/two_tower_explain/sanity_randomisation.png`).
- **Embedding decodability** (5-NN within each fold model's test patients, `knn_decodability.csv`): centre (Center-1 vs the rest) 0.52–0.64, class 0.68–0.75 (binary).
  - For the binary models, DINOv3 has the lowest centre decodability (0.52) and the highest class decodability (0.75), so there's no sign of a site shortcut in its embedding. For subtype, centre decodability is 0.52–0.62 across backbones (DINOv3 0.58).
  - ConvNeXt has the highest centre decodability (0.64).
- **A non-tissue cue to report:** some EUS frames carry on-screen calipers and measurement text. Example: NonNeoplasm-62 is predicted NET G1 at 1.00. `EXPLAINABILITY_INFERENCE.md` §6.5 lists recurring cues.

---

## 7. Reproducibility and robustness

- **Training curves** (`plots/two_tower_loss/`, `experiments/results/48_two_tower_curves/training_summary.csv`): over 210 folds, the kept checkpoint is at a median epoch of 17 (IQR 10–24), and folds trained a median of 37 epochs. Training loss falls to about 0.2 while select-split loss stays flat or rises after epochs 10–20: classic small-data overfitting, so early checkpoint selection matters.
  - Training loss is class-weighted CE and select loss is plain CE; compare trends, not the gap between them.
- **Compute:** a laptop RTX 5060 (8 GB; 34 runs) and NVIDIA A100 80 GB (cluster; 8 seed runs and the variant runs). Peak memory is ≤5.2 GB per run. Total training wall time is about 47 GPU-hours.
- **Cross-machine check:** the DINOv3 binary seed-42 run was re-trained on the cluster (A100, torch 2.3.1 / transformers 4.57.6) and compared with the laptop (RTX 5060, torch 2.11 / transformers 5.15).
  - Fold assignment, labels and patient order are identical.
  - Fused balanced accuracy is 0.793 vs 0.773 (within one fold SD of 0.04). Per-patient probability correlation is 0.81, and predicted classes agree 87%.
  - Pretrained DINOv3 embeddings match across the two installs to within 4e-6 (`experiments/src/45_two_tower_dinov3_parity.py`).
- **Determinism:** re-scoring a checkpoint reproduces the recorded fold scores exactly for ViT towers. CNN towers can differ by one near-tie patient (fp16 autocast in training vs fp32 re-scoring).
- **Software:**
  - laptop: PyTorch 2.11.0+cu128, torchvision 0.26, transformers 5.15.1, scikit-learn 1.9
  - cluster: PyTorch 2.3.1+cu121, transformers 4.57.6
- **Code map:**

| Script | Purpose |
|---|---|
| 34 | training |
| 35 | seed ensembles |
| 36 | counterfactual gate |
| 40 | split audit |
| 41 | re-scoring, tables, bar graphs |
| 42 | confusion matrices |
| 43 | explainability |
| 44 | statistics |
| 45 | DINOv3 parity |
| 46 | external dataset prep |
| 47 | cross-dataset comparison |
| 48 | training curves |

  The queue script is `experiments/run_two_tower_ablation.sh`.
- **Checkpoints:** laptop runs are on the local results volume. Cluster checkpoints for 9 runs are in a private Hugging Face repo (`RohanRamesh/cadsel-twotower-cluster-checkpoints`).

---

## 8. External datasets (same protocol)

Figure `plots/cross_dataset/normalised_balanced_accuracy.png`; table `experiments/results/47_cross_dataset/comparison.csv`.

Research finding: no public dataset pairs WLE with EUS, so CAD-SEL is unique. The closest multimodal colonoscopy data pairs white light with NBI of polyps.

| Dataset | Modalities / model | Units, groups | Task | Seed-42 result | Status |
|---|---|---|---|---|---|
| CAD-SEL | WLE+EUS two-tower | 641 patients | binary / subtype | 0.774 / 0.449 (DINOv3) | done |
| LIMUC (Zenodo 5827695, CC BY) | WL single-backbone | 11,276 images, 564 patients | Mayo 0–3 | DINOv3 bal 0.713 [0.702, 0.723], **QWK 0.847**, AUROC 0.922; EfficientNet bal 0.699, QWK 0.810 | done |
| LIMUC | WL single-backbone | same | remission (Mayo 0–1 vs 2–3) | DINOv3 bal 0.923 [0.917, 0.929], AUROC 0.978; EfficientNet 0.917 | done |
| REAL-colon (figshare 22202866, CC BY) | WL single-backbone, lesion crop | 112 polyps, 44 videos, 4 studies | neoplastic vs hyperplastic | — | **[pending]**: 22/44 videos cached; 2 runs after the user's go-ahead |
| CPC-Paired | WL+NBI two-tower | about 420 pairs | adenoma vs hyperplastic | — | **[pending]**: download host unverified |
| PIBAdb (IIS Galicia Sur biobank) | WL+NBI two-tower | 1,176 polyps | neoplastic vs hyperplastic, subtype | — | **[pending]**: access request (drafts prepared) |
| PICCOLO (Basque Biobank) | WL+NBI two-tower | 76 lesions, 40 patients | neoplastic vs hyperplastic | — | **[pending]**: access request |

Caveats:
- The tasks differ, so compare normalised balanced accuracy and behaviour, not raw scores. Normalised scores: CAD-SEL binary 0.55 and subtype 0.31; LIMUC Mayo 0.62 and remission 0.85.
- LIMUC's QWK of 0.847 under patient-level nested CV is in the range LIMUC papers report. Verify against the specific baselines before claiming parity; see `LITERATURE_REVIEW.md` or add LIMUC papers to it.
- REAL-colon has a centre–class confound: study 002 is mostly hyperplastic (31 vs 7) and study 004 mostly neoplastic (17 vs 2). Report it, and consider a leave-one-study-out check.

---

## 9. Limitations (state these in the paper)

1. **Oracle lesion boxes.** Inputs are cropped with ground-truth boxes at test time. On full frames, the lesion-crop models fall to near chance (binary 0.52–0.57). A deployable system needs a detector in front, or full-frame training; the full-frame cluster runs are approved in the plan but not launched.
2. **No untouched hold-out set.** Freeze policy, LR, modality dropout and lesion crop were tuned on the same 641 patients, before the nested protocol. The nested CV removes checkpoint-selection leakage, not design-choice leakage. The backbones for seeds 123/7 were originally chosen on seed-42 outer-fold scores; all 7 are now complete at 3 seeds, which removes that particular bias.
3. **Single-centre dominance.** 537 of 641 patients are from Center-1, so the results say little about other hospitals. An external-centre split is possible (`--external-test-centres`, Centres 2–6 = 104 patients) but not yet run.
4. **Class imbalance.** Leiomyoma has 24 patients and NET G2 has 44, so minority-class CIs are wide. NET G2 recall is about 0.09.
5. **EUS for 421 of 641 patients only.** Fusion can only be evaluated on that subset, and the gated design falls back to WLE.
6. **Whole-dataset normalisation stats** for the CNN towers (`MODALITY_STATS`): a minor, non-label leak of the pixel distribution.
7. **Saliency is supporting evidence only.** The causal claim rests on the counterfactual gate (§6.1).
8. **Mixed hardware.** 34 of the 42 ablation runs were trained on the laptop and 8 on the cluster. The cross-machine check (§7) shows the difference is within seed noise.

---

## 10. Figure list (suggested paper figures first)

| # | File | Suggested caption |
|---|---|---|
| F1 | `diagrams/04_two_tower_efficientnet_td.mmd` (re-draw for DINOv3) | Two-tower architecture: per-modality towers, attention pooling over frames, gated fusion |
| F2 | `plots/two_tower_ablation/two_tower_ablation_binary_by_modality.png` and `..._subtype_by_modality.png` | Backbone ablation: balanced accuracy, AUROC and F1 per input view, mean over 3 seeds, fold-SD whiskers |
| F3 | `plots/two_tower_stats/bootstrap_forest.png` | Balanced accuracy with 95% patient-level bootstrap CIs, seed 42 and 3-seed ensembles |
| F4 | `plots/two_tower_confusion/cm_subtype_fused_ens_grid.png` (+ binary) | Confusion matrices, 3-seed ensembles, all 7 backbones |
| F5 | `plots/two_tower_confusion/cm_subtype_fusion_delta_s42_grid.png` | What EUS changes: fused minus WLE-only counts on the 421 patients with EUS |
| F6 | `plots/cross_dataset/cadsel_fusion_strategies.png` | End-to-end fusion vs late fusion of separately trained towers |
| F7 | `plots/two_tower_explain/counterfactual_retention.png` | Counterfactual inputs: lesion pixels are necessary, the crop is necessary |
| F8 | `plots/two_tower_explain/gallery_dinov3_vits16_subtype.png` | Grad-CAM examples (contains patient images; the dataset is CC BY, cite it) |
| F9 | `plots/two_tower_stats/roc_binary.png`, `calibration_selective.png` | ROC curves and operating points; calibration and refer-to-expert curves |
| F10 | `plots/cross_dataset/normalised_balanced_accuracy.png` | Same protocol on CAD-SEL and LIMUC (REAL-colon pending) |
| S1 | `plots/two_tower_loss/loss_*_grid.png`, `selectbal_*_grid.png` | Training curves for all runs |
| S2 | `plots/two_tower_explain/box_energy.png`, `sanity_randomisation.png`, `embedding_knn_decodability.png`, `vit_cls_attention_*.png` | Saliency checks |
| S3 | `plots/two_tower_ablation/two_tower_ablation_*_{fused,wle_only,eus_only,fused_paired}.png` | Full metric panels per view |
| S4 | `plots/two_tower_confusion/cm_*_s42_grid.png`, `single/` | Seed-42 and per-modality confusion matrices |

---

## 11. Claims ledger (what the evidence supports)

| Claim | Evidence | Strength |
|---|---|---|
| DINOv3 towers give the best binary performance | 3-seed mean 0.774, ensemble 0.806; significantly beats ResNet-50 / ConvNeXt ensembles | moderate: top-4 differences are within noise |
| Self-supervised DINOv3 also leads on subtype on average | 3-seed mean 0.455; but the MobileViT ensemble is 0.509 (n.s.) | weak |
| Learned WLE+EUS fusion adds little over WLE | Fused − WLE ≤ 0.02 on average; on paired subtype, fused < WLE-only for DINOv3 | strong |
| The EUS tower is under-used by end-to-end training | Dedicated EUS model ≫ the two-tower EUS view; late fusion ≥ two-tower | strong at seed 42; confirm at 3 seeds |
| Late fusion of separate towers beats end-to-end fusion on subtype | DINOv3 +0.105 [+0.037, +0.177]; EfficientNet ≈ 0 | moderate (one seed, one backbone significant) |
| Predictions rely on lesion pixels | Counterfactual gate: lesion removed → chance | strong |
| The result needs ground-truth lesion boxes | Full frame → near chance | strong; the main deployment caveat |
| No centre shortcut in the DINOv3 embedding | Centre 5-NN bal 0.52 | moderate (Center-1 dominates) |
| The pipeline transfers to another dataset | LIMUC QWK 0.847, remission bal 0.923 | moderate (different task) |

---

## 12. Open items before submission

1. REAL-colon runs. The data is downloading; then 2 runs need approval in the cluster session.
2. Access requests for PIBAdb, PICCOLO and CPC-Paired. Drafts are in the session scratchpad; the user sends them.
3. Optional, cheap, high-value:
   - seeds 123/7 for the 8 CAD-SEL single-modality runs, to confirm §5.2
   - the approved-but-unlaunched follow-ups: full-frame training (removes the oracle-box caveat), the external-centre split, symmetric fusion, DINOv3 ViT-B/L
4. Update `report/main.tex` and the review deck, which still quote the old non-nested 0.81 / 0.54.
