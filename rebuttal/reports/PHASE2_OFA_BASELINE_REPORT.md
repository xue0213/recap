# Phase 2 RECAP-OFA Baseline Reproduction Report

Date: 2026-07-26

Status: **PASS**

All values are mean±population standard deviation over seeds 0/1/2,
in percent. Target labels were read only after score freeze, except
ARC's protocol-authorized 10 labeled-normal target contexts.

## Table 3 baseline rows — Setting A AUROC (%)

| Method | Cora | CiteSeer | ACM | BlogCatalog | Facebook | Weibo | Reddit | Amazon | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AnomalyGFM-ZS | 56.75±2.16 | 55.36±3.29 | 55.17±0.99 | 57.01±1.46 | 40.30±1.06 | 71.49±8.20 | 42.03±1.28 | 46.46±3.44 | 53.07±0.30 |
| IA-GGAD | 86.93±0.91 | 91.63±0.11 | 78.97±0.28 | 73.73±0.84 | 67.81±0.85 | 90.27±0.39 | 56.87±2.62 | 71.57±11.71 | 77.22±1.16 |
| ARC | 87.53±0.95 | 90.80±0.19 | 80.07±0.39 | 74.17±0.02 | 67.98±0.46 | 88.70±0.20 | 59.48±0.59 | 80.77±0.17 | 78.69±0.03 |
| UNPrompt | 48.36±2.01 | 54.67±2.88 | 70.29±1.27 | 67.96±0.47 | 50.23±1.20 | 51.42±1.95 | 53.55±0.86 | 67.72±6.34 | 58.02±1.41 |

## Table 4 baseline rows — Setting A AUPRC (%)

| Method | Cora | CiteSeer | ACM | BlogCatalog | Facebook | Weibo | Reddit | Amazon | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AnomalyGFM-ZS | 6.57±0.44 | 5.49±0.52 | 4.10±0.13 | 7.24±0.16 | 1.89±0.07 | 30.46±5.36 | 2.60±0.06 | 5.85±0.41 | 8.02±0.53 |
| IA-GGAD | 49.02±1.15 | 48.75±0.22 | 40.53±0.08 | 35.07±0.26 | 8.14±2.25 | 70.02±0.58 | 3.99±0.39 | 28.56±15.71 | 35.51±2.12 |
| ARC | 49.31±1.78 | 48.34±0.77 | 40.69±0.02 | 35.41±0.22 | 7.89±1.18 | 63.90±0.64 | 4.24±0.17 | 42.13±4.78 | 36.49±0.69 |
| UNPrompt | 5.33±0.42 | 5.23±0.36 | 9.53±1.40 | 21.54±2.97 | 2.44±0.21 | 19.56±3.10 | 3.67±0.10 | 14.51±5.25 | 10.22±1.42 |

## Table 5 baseline rows — Source-target robustness

| Method | Setting A Dataset-Macro | Setting B Social-Macro | Setting C Dataset-Macro | Setting C Domain-Macro |
|---|---:|---:|---:|---:|
| ARC | 78.69±0.03 / 36.49±0.69 | 72.76±0.23 / 29.49±0.25 | 66.18±0.74 / 18.86±0.47 | 64.09±1.06 / 13.82±0.74 |
| IA-GGAD | 77.22±1.16 / 35.51±2.12 | 72.84±0.69 / 30.74±0.22 | 63.03±0.44 / 17.74±0.16 | 60.49±0.22 / 12.50±0.10 |

## Table 6 baseline rows — Leave-Social-Domain-Out

| Method | Flickr | BlogCatalog | Facebook | Weibo | Reddit | Social Macro |
|---|---:|---:|---:|---:|---:|---:|
| ARC | 74.91±0.15 / 38.42±0.14 | 73.99±0.29 / 34.95±0.16 | 67.32±1.44 / 5.87±0.63 | 88.73±0.37 / 64.02±1.63 | 58.84±0.30 / 4.20±0.05 | 72.76±0.23 / 29.49±0.25 |
| IA-GGAD | 71.95±1.17 / 37.75±0.25 | 74.47±0.58 / 34.94±0.27 | 70.85±0.40 / 6.54±0.78 | 90.64±0.07 / 70.51±0.25 | 56.30±2.93 / 3.95±0.46 | 72.84±0.69 / 30.74±0.22 |

## Table 7 baseline rows — Citation-only Source Transfer

| Method | BlogCatalog | Flickr | Reddit | Amazon | Questions | Dataset-Macro | Domain-Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| ARC | 74.38±0.18 / 36.06±0.09 | 74.75±0.26 / 38.91±0.25 | 58.80±0.94 / 4.27±0.16 | 65.43±2.67 / 11.08±2.24 | 57.52±0.30 / 3.97±0.09 | 66.18±0.74 / 18.86±0.47 | 64.09±1.06 / 13.82±0.74 |
| IA-GGAD | 74.40±0.84 / 35.23±0.52 | 71.29±1.23 / 37.77±0.10 | 54.86±1.09 / 3.80±0.21 | 57.70±0.18 / 8.01±0.12 | 56.92±0.18 / 3.89±0.06 | 63.03±0.44 / 17.74±0.16 | 60.49±0.22 / 12.50±0.10 |

## Independent artifact audit

- Training runs: 24/24
- Recomputed evaluations: 156/156
- Frozen score files verified: 156
- Label-audit events checked: 462
- Maximum checkpoint reload difference: 9.6559525e-06
- Audit problems: 0

The failed pre-recovery UNPrompt attempt and pre-amendment ARC results
are retained outside the formal `runs/` directory and are excluded
from every aggregate above.

## Relation to the locked RECAP results

This comparison is descriptive: the baselines use source labels;
RECAP uses no source labels, target context, or target tuning.

| Setting | RECAP | ARC | IA-GGAD |
|---|---:|---:|---:|
| A | 74.65±0.23 / 27.04±0.34 | 78.69±0.03 / 36.49±0.69 | 77.22±1.16 / 35.51±2.12 |
| B | 67.75±0.22 / 21.98±0.32 | 72.76±0.23 / 29.49±0.25 | 72.84±0.69 / 30.74±0.22 |
| C | 67.31±0.43 / 17.50±0.09 | 66.18±0.74 / 18.86±0.47 | 63.03±0.44 / 17.74±0.16 |

RECAP trails ARC and IA-GGAD in Settings A/B. Under citation-only
sources (C), RECAP has higher dataset-macro AUROC than both while
its AUPRC remains slightly below both. In Setting A, RECAP exceeds
the target-context-free UNPrompt and AnomalyGFM-ZS reproductions
on both dataset-macro metrics.

## Source-only calibration locks

| Lock | Selected weight | Seed | Grid size |
|---|---:|---:|---:|
| anomalygfm_zs__setting_A | 6 | 0 | 6 |
| ia_ggad__setting_A | 0.1 | 0 | 15 |
| ia_ggad__setting_B | 0.3 | 0 | 15 |
| ia_ggad__setting_C | 0.5 | 0 | 15 |

## Formal resource totals

Times are summed once per formal training run; peak memory is the
maximum observed allocation for that method.

| Method | Runs | Preparation (s) | Training (s) | Evaluation (s) | Peak GPU GiB |
|---|---:|---:|---:|---:|---:|
| ARC | 9 | 14.18 | 30.40 | 0.63 | 5.38 |
| AnomalyGFM-ZS | 3 | 5.59 | 32.86 | 0.28 | 0.90 |
| IA-GGAD | 9 | 13.48 | 126.37 | 3.02 | 5.01 |
| UNPrompt | 3 | 3.55 | 725.53 | 1.58 | 3.20 |

Method-native settings and every compatibility adaptation are frozen
in `rebuttal/BASELINE_OFA_REPROTOCOL.md`; dense/sparse and deterministic
equivalence evidence is recorded in the Phase 2 gate report.
