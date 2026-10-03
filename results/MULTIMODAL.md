# CAD-SEL multimodal (WLE + EUS)

### Shared-split unimodal baselines

Re-run on the joint (leak-free) split, because the completed grid in `runs/` split each modality independently and is therefore not a valid comparison for a model that sees both.

| Model | Modality | Task | Protocol | AP@0.5 | F1 | epoch |
| --- | --- | --- | --- | --- | --- | --- |
| Fasterrcnn_Mobilenet | EUS | binary | patient | 0.5040 | 0.6184 | 25 |
| Fasterrcnn_Mobilenet | WLE | binary | patient | 0.6566 | 0.7167 | 14 |
| Fasterrcnn_Mobilenet | EUS | multiclass | patient | 0.4481 | 0.5929 | 23 |
| Fasterrcnn_Mobilenet | WLE | multiclass | patient | 0.3750 | 0.5953 | 14 |

### Joint WLE+EUS detector — per-modality AP@0.5

`Δ` compares against the shared-split unimodal baseline for the same model, task and protocol. Per FINDINGS §8.3 the seed-to-seed spread on this dataset is ~0.02 AP, so treat anything smaller as noise.

| Model | Task | Protocol | Cond. | HN | AP WLE | Δ WLE | AP EUS | Δ EUS | macro AP | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fasterrcnn_Mobilenet | binary | patient | film | 30% | 0.6402 | -0.0164 | 0.4329 | -0.0710 | 0.5366 | 0.6615 |
| Fasterrcnn_Mobilenet | binary | patient | none | 30% | 0.6540 | -0.0026 | 0.4559 | -0.0481 | 0.5550 | 0.6558 |
| Fasterrcnn_Mobilenet | multiclass | patient | film | 30% | 0.3863 | +0.0113 | 0.4134 | -0.0347 | 0.3999 | 0.5487 |
| Fasterrcnn_Mobilenet | multiclass | patient | none | 30% | 0.3525 | -0.0225 | 0.4116 | -0.0365 | 0.3820 | 0.5614 |

### Per-class AP@0.5, multi-class task

| Tested model | Modality | NET | Leiomyoma | Lipoma | NonNeoplasm |
| --- | --- | --- | --- | --- | --- |
| Fasterrcnn_Mobilenet | film/patient | 0.6948 | — | — | — |
| Fasterrcnn_Mobilenet | none/patient | 0.7045 | — | — | — |
| Fasterrcnn_Mobilenet | film/patient | 0.6321 | 0.3226 | 0.4302 | 0.1234 |
| Fasterrcnn_Mobilenet | none/patient | 0.6491 | 0.2581 | 0.4093 | 0.1116 |

### Patient-level fusion head

5-fold patient-level cross-validation, classes: NET_G1, NET_G2, Leiomyoma, Lipoma, NonNeoplasm.

**patient-level cross-validation; NOT comparable to Table 4/5 AP** — this scores a patient, not an image.

| Ablation | Balanced acc. | Macro F1 | Accuracy |
| --- | --- | --- | --- |
| wle_only | 0.4258 ± 0.0547 | 0.3160 ± 0.0377 | 0.3904 ± 0.0863 |
| eus_only | 0.3890 ± 0.0458 | 0.2216 ± 0.0511 | 0.2960 ± 0.0925 |
| fused | 0.4190 ± 0.0482 | 0.3178 ± 0.0346 | 0.3933 ± 0.0832 |
| fused_moddrop | 0.4163 ± 0.0432 | 0.3461 ± 0.0543 | 0.4760 ± 0.0934 |
