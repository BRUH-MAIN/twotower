# CAD-SEL dataset characteristics (reproduced)

### Table 2 — Overview of the data records

| Category | Patients (ours) | Images (ours) | Patients (paper) | Images (paper) | Match |
| --- | --- | --- | --- | --- | --- |
| WLE set | 641 | 2817 | 641 | 2817 | yes |
| WLE · NET | 455 | 1945 | 455 | 1945 | yes |
| WLE · NET_G1 | 411 | 1749 | 411 | 1749 | yes |
| WLE · NET_G2 | 44 | 196 | 44 | 196 | yes |
| WLE · Lipoma | 96 | 453 | 96 | 453 | yes |
| WLE · Leiomyoma | 24 | 137 | 24 | 137 | yes |
| WLE · NonNeoplasm | 66 | 282 | 66 | 282 | yes |
| EUS set | 421 | 2095 | 421 | 2095 | yes |
| EUS · NET | 325 | 1602 | 325 | 1602 | yes |
| EUS · NET_G1 | 299 | 1444 | 299 | 1444 | yes |
| EUS · NET_G2 | 26 | 158 | 26 | 158 | yes |
| EUS · Lipoma | 49 | 254 | 49 | 254 | yes |
| EUS · Leiomyoma | 14 | 79 | 14 | 79 | yes |
| EUS · NonNeoplasm | 33 | 160 | 33 | 160 | yes |

Total images: **4912** (paper: 4,912)

### Bounding boxes per annotation class

| Modality | NET (0) | Leiomyoma (1) | Lipoma (2) | NonNeoplasm (3) | Total |
| --- | --- | --- | --- | --- | --- |
| WLE | 2025 | 139 | 480 | 299 | 2943 |
| EUS | 1731 | 81 | 282 | 177 | 2271 |

### Fig 3 — Clinical data distribution

- Patients: 641 (paper: 641)
- Mean age: 52.0 y (paper: 52.0)
- Age: {'30-60': 70.8, '<30': 4.1, '>60': 25.1} (paper: 68% aged 30-60)
- Sex: {'female': 48.7, 'male': 51.3} (paper: 51% male / 49% female)
- Size (cm): {'1-2': 15.0, '<0.5': 19.9, '0.5-1': 59.8, '>2': 5.3} (paper: <0.5 20%, 0.5-1 54%, 1-2 20%, >2 6%)
- Location: rectum 81.4%, transverse colon 5.1%, sigmoid 5.0%, ascending colon 3.4%, ileolcecal region 2.8%, descending colon 2.2%
  (paper: rectum 81%, sigmoid 5%, ascending 5%, descending 3%, transverse 3%, ileocecal 3%)
