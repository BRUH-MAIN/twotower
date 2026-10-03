# CAD-SEL reproduction — results

8 completed runs. Held-out validation, patient-level split (no patient appears in both train and val).

### Table 4 — Multi-class detection (reproduction vs. paper)

| Tested model | Modality | AP@0.5 | Precision | Recall | F1-score | paper AP@0.5 | paper P | paper R | paper F1 | ΔAP | ΔF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fasterrcnn_Mobilenet | WLE | 0.4098 | 0.5050 | 0.5993 | 0.5481 | 0.8817 | 0.8014 | 0.9067 | 0.8508 | -0.4719 | -0.3027 |
| Fasterrcnn_Mobilenet | EUS | 0.4514 | 0.5693 | 0.4946 | 0.5293 | 0.5768 | 0.7764 | 0.7561 | 0.7661 | -0.1254 | -0.2368 |
| Fasterrcnn_Mobilenet_320 | WLE | 0.3809 | 0.5845 | 0.6582 | 0.6192 | 0.8641 | 0.8661 | 0.8922 | 0.8790 | -0.4832 | -0.2598 |
| Fasterrcnn_Mobilenet_320 | EUS | 0.3476 | 0.6947 | 0.6215 | 0.6561 | 0.5320 | 0.7730 | 0.6840 | 0.7258 | -0.1844 | -0.0697 |

### Table 5 — Binary detection, NET vs non-NET (reproduction vs. paper)

| Tested model | Modality | AP@0.5 (NET) | AP@0.5 (non-NET) | paper NET | paper non-NET | Δ NET | Δ non-NET |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fasterrcnn_Mobilenet | WLE | 0.8203 | 0.5015 | 0.8851 | 0.8784 | -0.0648 | -0.3769 |
| Fasterrcnn_Mobilenet | EUS | 0.7456 | 0.2780 | 0.8143 | 0.3393 | -0.0687 | -0.0613 |
| Fasterrcnn_Mobilenet_320 | WLE | 0.8108 | 0.4809 | 0.9023 | 0.8259 | -0.0915 | -0.3450 |
| Fasterrcnn_Mobilenet_320 | EUS | 0.6688 | 0.1876 | 0.7257 | 0.3383 | -0.0569 | -0.1507 |

### WLE vs EUS (the paper's headline qualitative claim)

| Model | Task | WLE AP@0.5 | EUS AP@0.5 | WLE better? |
| --- | --- | --- | --- | --- |
| fasterrcnn_mobilenet | multiclass | 0.4098 | 0.4514 | no |
| fasterrcnn_mobilenet | binary | 0.6609 | 0.5118 | yes |
| fasterrcnn_mobilenet_320 | multiclass | 0.3809 | 0.3476 | yes |
| fasterrcnn_mobilenet_320 | binary | 0.6458 | 0.4282 | yes |

### Binary task — overall detection metrics

| Run | AP@0.5 (mean) | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fasterrcnn_mobilenet_320__EUS__binary | 0.4282 | 0.6763 | 0.6065 | 0.6395 | 282 | 135 | 183 |
| fasterrcnn_mobilenet_320__WLE__binary | 0.6458 | 0.7252 | 0.7508 | 0.7378 | 446 | 169 | 148 |
| fasterrcnn_mobilenet__EUS__binary | 0.5118 | 0.6511 | 0.6903 | 0.6701 | 321 | 172 | 144 |
| fasterrcnn_mobilenet__WLE__binary | 0.6609 | 0.6372 | 0.7896 | 0.7053 | 469 | 267 | 125 |

### Per-class detail (best epoch of each run)

| Run | Class | AP@0.5 | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fasterrcnn_mobilenet_320__EUS__binary | NET | 0.6688 | 0.7676 | 0.6905 | 0.7270 | 261 | 79 | 117 |
| fasterrcnn_mobilenet_320__EUS__binary | non-NET | 0.1876 | 0.2727 | 0.2414 | 0.2561 | 21 | 56 | 66 |
| fasterrcnn_mobilenet_320__EUS__multiclass | NET | 0.6763 | 0.7743 | 0.7169 | 0.7445 | 271 | 79 | 107 |
| fasterrcnn_mobilenet_320__EUS__multiclass | Leiomyoma | 0.2535 | 0.3077 | 0.2353 | 0.2667 | 4 | 9 | 13 |
| fasterrcnn_mobilenet_320__EUS__multiclass | Lipoma | 0.4232 | 0.5833 | 0.3784 | 0.4590 | 14 | 10 | 23 |
| fasterrcnn_mobilenet_320__EUS__multiclass | NonNeoplasm | 0.0373 | 0.0000 | 0.0000 | 0.0000 | 0 | 29 | 33 |
| fasterrcnn_mobilenet_320__WLE__binary | NET | 0.8108 | 0.7598 | 0.8937 | 0.8213 | 370 | 117 | 44 |
| fasterrcnn_mobilenet_320__WLE__binary | non-NET | 0.4809 | 0.5938 | 0.4222 | 0.4935 | 76 | 52 | 104 |
| fasterrcnn_mobilenet_320__WLE__multiclass | NET | 0.7461 | 0.7617 | 0.7874 | 0.7743 | 326 | 102 | 88 |
| fasterrcnn_mobilenet_320__WLE__multiclass | Leiomyoma | 0.2595 | 0.4286 | 0.3333 | 0.3750 | 9 | 12 | 18 |
| fasterrcnn_mobilenet_320__WLE__multiclass | Lipoma | 0.4290 | 0.4444 | 0.4124 | 0.4278 | 40 | 50 | 57 |
| fasterrcnn_mobilenet_320__WLE__multiclass | NonNeoplasm | 0.0890 | 0.1231 | 0.2857 | 0.1720 | 16 | 114 | 40 |
| fasterrcnn_mobilenet__EUS__binary | NET | 0.7456 | 0.8040 | 0.7381 | 0.7697 | 279 | 68 | 99 |
| fasterrcnn_mobilenet__EUS__binary | non-NET | 0.2780 | 0.2877 | 0.4828 | 0.3605 | 42 | 104 | 45 |
| fasterrcnn_mobilenet__EUS__multiclass | NET | 0.7384 | 0.8553 | 0.5159 | 0.6436 | 195 | 33 | 183 |
| fasterrcnn_mobilenet__EUS__multiclass | Leiomyoma | 0.6078 | 0.5625 | 0.5294 | 0.5455 | 9 | 7 | 8 |
| fasterrcnn_mobilenet__EUS__multiclass | Lipoma | 0.4004 | 0.2222 | 0.4865 | 0.3051 | 18 | 63 | 19 |
| fasterrcnn_mobilenet__EUS__multiclass | NonNeoplasm | 0.0588 | 0.1013 | 0.2424 | 0.1429 | 8 | 71 | 25 |
| fasterrcnn_mobilenet__WLE__binary | NET | 0.8203 | 0.7650 | 0.8647 | 0.8118 | 358 | 110 | 56 |
| fasterrcnn_mobilenet__WLE__binary | non-NET | 0.5015 | 0.4142 | 0.6167 | 0.4955 | 111 | 157 | 69 |
| fasterrcnn_mobilenet__WLE__multiclass | NET | 0.7512 | 0.7734 | 0.6594 | 0.7119 | 273 | 80 | 141 |
| fasterrcnn_mobilenet__WLE__multiclass | Leiomyoma | 0.2801 | 0.2200 | 0.4074 | 0.2857 | 11 | 39 | 16 |
| fasterrcnn_mobilenet__WLE__multiclass | Lipoma | 0.4214 | 0.3917 | 0.4845 | 0.4332 | 47 | 73 | 50 |
| fasterrcnn_mobilenet__WLE__multiclass | NonNeoplasm | 0.1864 | 0.1374 | 0.4464 | 0.2101 | 25 | 157 | 31 |

### Training detail

| Run | Epochs run | Best epoch | Batch | Img | Train imgs / patients | Val imgs / patients | Wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fasterrcnn_mobilenet_320__EUS__binary | 34 | 19 | 16 | 320 | 1660 / 336 | 435 / 85 | 22 min |
| fasterrcnn_mobilenet_320__EUS__multiclass | 33 | 18 | 16 | 320 | 1660 / 336 | 435 / 85 | 26 min |
| fasterrcnn_mobilenet_320__WLE__binary | 23 | 8 | 16 | 320 | 2249 / 513 | 568 / 128 | 20 min |
| fasterrcnn_mobilenet_320__WLE__multiclass | 32 | 17 | 16 | 320 | 2249 / 513 | 568 / 128 | 54 min |
| fasterrcnn_mobilenet__EUS__binary | 26 | 13 | 8 | 640 | 1660 / 336 | 435 / 85 | 18 min |
| fasterrcnn_mobilenet__EUS__multiclass | 23 | 8 | 8 | 640 | 1660 / 336 | 435 / 85 | 20 min |
| fasterrcnn_mobilenet__WLE__binary | 24 | 9 | 8 | 640 | 2249 / 513 | 568 / 128 | 32 min |
| fasterrcnn_mobilenet__WLE__multiclass | 26 | 11 | 8 | 640 | 2249 / 513 | 568 / 128 | 62 min |
