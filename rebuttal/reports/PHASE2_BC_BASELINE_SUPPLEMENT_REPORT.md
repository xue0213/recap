# Phase 2 B/C Baseline Completion Supplement

Date: 2026-07-26

Status: **PASS**

This user-revised confirmatory supplement adds UNPrompt and
AnomalyGFM-ZS to Settings B and C. Original ARC/IA-GGAD and RECAP
artifacts are reused only for reporting; no original run was modified.
All values are mean±population standard deviation over seeds 0/1/2,
in percent.

## Setting B — Leave-Social-Domain-Out

| Method | Flickr | BlogCatalog | Facebook | Weibo | Reddit | Social Macro |
|---|---:|---:|---:|---:|---:|---:|
| AnomalyGFM-ZS | 53.15±2.03 / 7.68±0.39 | 55.81±2.29 / 7.12±0.58 | 48.04±3.49 / 2.88±0.31 | 43.19±8.46 / 21.54±3.21 | 48.01±1.54 / 2.95±0.11 | 49.64±2.22 / 8.43±0.61 |
| IA-GGAD | 71.95±1.17 / 37.75±0.25 | 74.47±0.58 / 34.94±0.27 | 70.85±0.40 / 6.54±0.78 | 90.64±0.07 / 70.51±0.25 | 56.30±2.93 / 3.95±0.46 | 72.84±0.69 / 30.74±0.22 |
| ARC | 74.91±0.15 / 38.42±0.14 | 73.99±0.29 / 34.95±0.16 | 67.32±1.44 / 5.87±0.63 | 88.73±0.37 / 64.02±1.63 | 58.84±0.30 / 4.20±0.05 | 72.76±0.23 / 29.49±0.25 |
| UNPrompt | 68.37±0.12 / 16.98±2.76 | 67.77±0.78 / 16.72±3.06 | 52.04±9.81 / 2.75±0.81 | 45.36±1.70 / 18.10±4.51 | 56.04±0.65 / 3.89±0.14 | 57.92±2.13 / 11.69±1.21 |

## Setting C — Citation-only Source Transfer

| Method | BlogCatalog | Flickr | Reddit | Amazon | Questions | Dataset-Macro | Domain-Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnomalyGFM-ZS | 58.80±0.94 / 7.55±0.20 | 54.58±1.76 / 8.03±0.21 | 41.06±2.75 / 2.58±0.14 | 54.00±4.29 / 6.89±0.74 | 46.73±0.67 / 2.79±0.06 | 51.03±0.79 / 5.57±0.12 | 50.74±1.33 / 5.24±0.22 |
| IA-GGAD | 74.40±0.84 / 35.23±0.52 | 71.29±1.23 / 37.77±0.10 | 54.86±1.09 / 3.80±0.21 | 57.70±0.18 / 8.01±0.12 | 56.92±0.18 / 3.89±0.06 | 63.03±0.44 / 17.74±0.16 | 60.49±0.22 / 12.50±0.10 |
| ARC | 74.38±0.18 / 36.06±0.09 | 74.75±0.26 / 38.91±0.25 | 58.80±0.94 / 4.27±0.16 | 65.43±2.67 / 11.08±2.24 | 57.52±0.30 / 3.97±0.09 | 66.18±0.74 / 18.86±0.47 | 64.09±1.06 / 13.82±0.74 |
| UNPrompt | 66.44±1.32 / 18.17±1.78 | 69.42±0.42 / 20.78±0.71 | 53.59±0.30 / 3.65±0.08 | 63.33±6.43 / 11.59±3.02 | 43.45±2.17 / 2.65±0.27 | 59.25±1.49 / 11.37±0.41 | 56.64±2.64 / 9.48±0.90 |

## Dataset-macro comparison with RECAP

| Setting | RECAP | ARC | IA-GGAD | UNPrompt | AnomalyGFM-ZS |
|---|---:|---:|---:|---:|---:|
| B | 67.75±0.22 / 21.98±0.32 | 72.76±0.23 / 29.49±0.25 | 72.84±0.69 / 30.74±0.22 | 57.92±2.13 / 11.69±1.21 | 49.64±2.22 / 8.43±0.61 |
| C | 67.31±0.43 / 17.50±0.09 | 66.18±0.74 / 18.86±0.47 | 63.03±0.44 / 17.74±0.16 | 59.25±1.49 / 11.37±0.41 | 51.03±0.79 / 5.57±0.12 |

## Supplement artifact audit

- Training runs: 12/12
- Recomputed evaluations: 60/60
- Frozen score files: 60
- Label-audit events: 168
- Maximum checkpoint reload difference: 3.3378601e-06
- Audit problems: 0

## AnomalyGFM source-only calibration locks

| Lock | Selected weight | Seed | Grid size |
|---|---:|---:|---:|
| anomalygfm_zs__setting_B | 2 | 0 | 6 |
| anomalygfm_zs__setting_C | 4 | 0 | 6 |

## Supplement resource totals

| Method | Runs | Preparation (s) | Training (s) | Evaluation (s) | Peak GPU GiB |
|---|---:|---:|---:|---:|---:|
| AnomalyGFM-ZS | 6 | 7.98 | 52.65 | 0.55 | 0.88 |
| UNPrompt | 6 | 5.15 | 903.11 | 1.73 | 3.18 |
