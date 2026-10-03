# CAD-SEL technical validation — reproduction

### Table 4 — Multi-class detection (reproduction vs. paper)

| Tested model | Modality | AP@0.5 | Precision | Recall | F1-score | paper AP@0.5 | paper P | paper R | paper F1 | ΔAP | ΔF1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fasterrcnn_Resnet | WLE | 0.2809 | 0.4032 | 0.5926 | 0.4799 | 0.8593 | 0.6824 | 0.9146 | 0.7816 | -0.5784 | -0.3017 |
| Fasterrcnn_Resnet50_FPN_V2 | WLE | 0.2594 | 0.2687 | 0.2357 | 0.2511 | 0.8792 | 0.6772 | 0.9290 | 0.7834 | -0.6198 | -0.5323 |
| Fasterrcnn_Mobilenet | WLE | 0.4098 | 0.5050 | 0.5993 | 0.5481 | 0.8817 | 0.8014 | 0.9067 | 0.8508 | -0.4719 | -0.3027 |
| Fasterrcnn_Mobilenet | EUS | 0.4514 | 0.5693 | 0.4946 | 0.5293 | 0.5768 | 0.7764 | 0.7561 | 0.7661 | -0.1254 | -0.2368 |
| Fasterrcnn_Mobilenet_320 | WLE | 0.3809 | 0.5845 | 0.6582 | 0.6192 | 0.8641 | 0.8661 | 0.8922 | 0.8790 | -0.4832 | -0.2598 |
| Fasterrcnn_Mobilenet_320 | EUS | 0.3163 | 0.4492 | 0.1140 | 0.1818 | 0.5320 | 0.7730 | 0.6840 | 0.7258 | -0.2157 | -0.5440 |

### Per-class AP@0.5, multi-class task

| Tested model | Modality | NET | Leiomyoma | Lipoma | NonNeoplasm |
| --- | --- | --- | --- | --- | --- |
| Fasterrcnn_Resnet | WLE | 0.6116 | 0.0947 | 0.3246 | 0.0926 |
| Fasterrcnn_Resnet50_FPN_V2 | WLE | 0.6077 | 0.0673 | 0.2500 | 0.1127 |
| Fasterrcnn_Mobilenet | WLE | 0.7512 | 0.2801 | 0.4214 | 0.1864 |
| Fasterrcnn_Mobilenet | EUS | 0.7384 | 0.6078 | 0.4004 | 0.0588 |
| Fasterrcnn_Mobilenet_320 | WLE | 0.7461 | 0.2595 | 0.4290 | 0.0890 |
| Fasterrcnn_Mobilenet_320 | EUS | 0.5814 | 0.1946 | 0.3964 | 0.0928 |

### Table 5 — Binary detection, NET vs non-NET (reproduction vs. paper)

| Tested model | Modality | AP@0.5 (NET) | AP@0.5 (non-NET) | paper NET | paper non-NET | Δ NET | Δ non-NET |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Fasterrcnn_Mobilenet | WLE | 0.8203 | 0.5015 | 0.8851 | 0.8784 | -0.0648 | -0.3769 |
| Fasterrcnn_Mobilenet | EUS | 0.7456 | 0.2780 | 0.8143 | 0.3393 | -0.0687 | -0.0613 |
| Fasterrcnn_Mobilenet_320 | WLE | 0.8108 | 0.4809 | 0.9023 | 0.8259 | -0.0915 | -0.3450 |
| Fasterrcnn_Mobilenet_320 | EUS | 0.6688 | 0.1876 | 0.7257 | 0.3383 | -0.0569 | -0.1507 |
