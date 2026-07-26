# RECAP Experiment Protocol Completion Report

## Conclusion

**PASS. No requested training or inference cell is missing, and no rerun is required.** The revised scope contains 450 successful training runs and 720 final evaluations. All consolidated values below were recomputed from unrounded seed-level or seed-pair-level records using population standard deviation (`ddof=0`).

## Completion matrix

| Requirement | Expected | Actual | Status |
| --- | --- | --- | --- |
| 12 datasets load and have immutable raw hashes | 12 datasets | 12 | PASS |
| RECAP-OFO | 12 datasets × 3 seeds | 36 evaluations / 36 training runs | PASS |
| RECAP-OFA A/B/C | 3 settings × 3 seeds | 54 evaluations / 9 training runs | PASS |
| 10 OFO baselines | 10 methods × 12 datasets × 3 seeds | 360 evaluations / 360 training runs | PASS |
| 5 OFA baselines in A/B/C | 5 methods × (8+5+5) targets × 3 seeds | 270 evaluations / 45 training runs | PASS |
| Community stability | 30 scopes × 3 seed pairs | 90 pair records | PASS |
| Training diagnostics | 486 rows after Questions addendum | 486 | PASS |
| Checkpoints and reload gates | 225 checkpoints / 45 final reload gates | 225 / 45 | PASS |
| Protocol Tables 1–9 | all cells backed by raw records | consolidated and recomputed | PASS |

## One-for-One 12-dataset results

### AUROC (%)

| Method | PubMed | Cora | CiteSeer | ACM | Flickr | BlogCatalog | Facebook | Weibo | Reddit | Questions | YelpChi | Amazon | Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | 80.91 ± 0.45 | 75.13 ± 0.67 | 79.53 ± 1.53 | 77.97 ± 1.97 | 79.87 ± 2.77 | 76.23 ± 0.91 | 99.16 ± 0.50 | 99.30 ± 0.13 | 64.02 ± 5.22 | 73.79 ± 1.22 | 86.13 ± 0.67 | 86.25 ± 0.49 | 81.52 ± 0.47 |
| GAT | 89.48 ± 2.02 | 79.21 ± 3.97 | 82.53 ± 1.50 | 77.75 ± 0.65 | 78.34 ± 3.31 | 75.33 ± 3.25 | 99.23 ± 0.61 | 99.09 ± 0.23 | 66.23 ± 2.26 | 72.86 ± 0.22 | 89.09 ± 0.35 | 87.88 ± 1.50 | 83.09 ± 0.89 |
| BWGNN | 82.56 ± 3.68 | 58.23 ± 8.09 | 65.24 ± 10.31 | 54.32 ± 6.34 | 62.12 ± 2.71 | 68.54 ± 2.88 | 82.12 ± 3.47 | 98.77 ± 0.47 | 61.38 ± 1.40 | 75.49 ± 1.13 | 86.79 ± 0.29 | 96.97 ± 0.59 | 74.38 ± 0.81 |
| XGBGraph | 95.14 ± 0.55 | 89.06 ± 0.98 | 94.70 ± 0.94 | 80.45 ± 1.85 | 84.56 ± 3.09 | 95.20 ± 0.10 | 99.23 ± 0.73 | 99.67 ± 0.09 | 64.48 ± 1.96 | 73.73 ± 0.46 | 92.92 ± 0.22 | 96.92 ± 0.62 | 88.84 ± 0.29 |
| DOMINANT | 83.67 ± 0.00 | 66.84 ± 0.04 | 68.94 ± 0.00 | 67.29 ± 0.00 | 54.28 ± 0.00 | 56.17 ± 0.00 | 7.19 ± 0.07 | 37.51 ± 1.38 | 56.15 ± 0.29 | 56.13 ± 0.27 | 27.17 ± 1.09 | 58.51 ± 0.18 | 53.32 ± 0.18 |
| AnomalyDAE | 78.27 ± 0.25 | 68.65 ± 0.07 | 71.59 ± 0.20 | 69.91 ± 0.00 | 48.06 ± 0.40 | 25.76 ± 0.10 | 67.66 ± 0.40 | 38.62 ± 1.65 | 56.01 ± 0.16 | 65.18 ± 1.44 | 27.60 ± 0.05 | 57.56 ± 4.14 | 56.24 ± 0.33 |
| CoLA | 74.84 ± 0.40 | 69.21 ± 2.49 | 75.82 ± 1.67 | 51.23 ± 1.77 | 41.41 ± 0.57 | 40.02 ± 0.12 | 92.99 ± 0.30 | 14.67 ± 0.89 | 53.13 ± 1.37 | 42.96 ± 0.34 | 40.77 ± 1.10 | 55.30 ± 0.50 | 54.36 ± 0.19 |
| ADA-GAD | 70.75 ± 0.51 | 66.89 ± 0.06 | 67.29 ± 0.20 | 65.34 ± 0.10 | 65.36 ± 0.02 | 59.24 ± 0.02 | 28.88 ± 1.16 | 79.74 ± 0.87 | 44.46 ± 2.79 | 48.06 ± 1.81 | 49.65 ± 1.82 | 45.25 ± 4.13 | 57.57 ± 0.75 |
| DiffGAD | 82.72 ± 0.00 | 67.37 ± 0.00 | 69.05 ± 0.00 | 67.62 ± 0.00 | 54.27 ± 0.01 | 56.22 ± 0.01 | 29.91 ± 18.13 | 38.92 ± 13.75 | 52.30 ± 5.13 | 57.12 ± 0.00 | 29.12 ± 0.00 | 55.77 ± 0.13 | 55.03 ± 2.17 |
| GUIDE | 92.98 ± 0.03 | 97.75 ± 0.03 | 98.67 ± 0.06 | 87.53 ± 0.05 | 73.90 ± 0.01 | 74.45 ± 0.16 | 45.87 ± 1.07 | 83.02 ± 0.77 | 56.64 ± 0.21 | 59.31 ± 0.05 | 48.86 ± 0.50 | 83.33 ± 0.98 | 75.19 ± 0.14 |
| RECAP-OFO | 82.55 ± 0.33 | 82.72 ± 0.26 | 89.86 ± 0.34 | 79.01 ± 0.53 | 74.15 ± 0.36 | 73.87 ± 0.18 | 57.88 ± 0.64 | 78.79 ± 0.72 | 58.03 ± 1.43 | 63.47 ± 0.05 | 42.60 ± 0.67 | 70.06 ± 4.65 | 71.08 ± 0.35 |

### AUPRC (%)

| Method | PubMed | Cora | CiteSeer | ACM | Flickr | BlogCatalog | Facebook | Weibo | Reddit | Questions | YelpChi | Amazon | Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GCN | 16.30 ± 4.23 | 42.57 ± 5.35 | 45.06 ± 4.94 | 40.11 ± 0.96 | 57.18 ± 1.69 | 48.51 ± 3.54 | 73.26 ± 4.85 | 96.86 ± 0.20 | 6.26 ± 1.18 | 15.27 ± 0.88 | 37.10 ± 0.95 | 35.44 ± 1.46 | 42.83 ± 0.66 |
| GAT | 54.62 ± 5.68 | 31.16 ± 10.38 | 33.99 ± 4.98 | 34.90 ± 3.95 | 51.20 ± 3.52 | 33.84 ± 10.17 | 73.85 ± 22.14 | 94.36 ± 1.35 | 5.67 ± 0.40 | 13.57 ± 0.50 | 42.40 ± 0.98 | 50.47 ± 13.07 | 43.34 ± 4.73 |
| BWGNN | 21.26 ± 4.72 | 11.92 ± 0.47 | 15.31 ± 5.32 | 13.18 ± 3.40 | 19.74 ± 1.52 | 24.07 ± 2.17 | 7.97 ± 1.47 | 95.14 ± 0.31 | 6.29 ± 0.42 | 19.75 ± 2.38 | 35.45 ± 1.76 | 86.17 ± 1.81 | 29.69 ± 0.78 |
| XGBGraph | 72.55 ± 1.78 | 63.18 ± 4.26 | 76.28 ± 3.99 | 55.33 ± 1.19 | 61.45 ± 4.42 | 53.33 ± 2.22 | 87.63 ± 11.13 | 97.97 ± 0.39 | 5.46 ± 0.34 | 20.33 ± 1.13 | 62.43 ± 0.81 | 87.84 ± 0.80 | 61.98 ± 1.61 |
| DOMINANT | 10.65 ± 0.00 | 25.02 ± 0.06 | 33.48 ± 0.03 | 9.51 ± 0.00 | 6.61 ± 0.00 | 7.16 ± 0.00 | 1.26 ± 0.00 | 7.59 ± 0.15 | 3.76 ± 0.02 | 4.80 ± 0.02 | 3.20 ± 0.05 | 11.78 ± 0.04 | 10.40 ± 0.01 |
| AnomalyDAE | 9.87 ± 0.16 | 22.22 ± 0.54 | 34.39 ± 0.04 | 10.02 ± 0.00 | 5.92 ± 0.04 | 3.75 ± 0.01 | 4.38 ± 0.20 | 8.04 ± 0.24 | 3.76 ± 0.01 | 6.47 ± 0.08 | 3.22 ± 0.00 | 7.59 ± 0.58 | 9.97 ± 0.11 |
| CoLA | 25.62 ± 0.48 | 17.69 ± 0.92 | 23.01 ± 1.54 | 12.84 ± 1.29 | 6.28 ± 0.22 | 4.97 ± 0.09 | 26.63 ± 2.23 | 6.11 ± 0.12 | 4.01 ± 0.26 | 2.38 ± 0.02 | 3.97 ± 0.07 | 9.04 ± 0.27 | 11.88 ± 0.24 |
| ADA-GAD | 5.41 ± 0.06 | 9.34 ± 0.02 | 8.57 ± 0.05 | 5.78 ± 0.02 | 8.88 ± 0.00 | 7.77 ± 0.00 | 1.54 ± 0.02 | 21.73 ± 0.97 | 2.75 ± 0.14 | 3.34 ± 0.29 | 5.30 ± 0.36 | 5.65 ± 0.42 | 7.17 ± 0.06 |
| DiffGAD | 10.47 ± 0.00 | 25.58 ± 0.00 | 33.53 ± 0.00 | 9.56 ± 0.00 | 6.61 ± 0.00 | 7.16 ± 0.00 | 2.09 ± 0.98 | 8.38 ± 2.24 | 3.43 ± 0.44 | 4.91 ± 0.00 | 3.29 ± 0.00 | 12.30 ± 0.15 | 10.61 ± 0.24 |
| GUIDE | 56.18 ± 0.07 | 79.58 ± 0.16 | 81.20 ± 0.06 | 27.94 ± 0.09 | 37.42 ± 0.05 | 33.68 ± 0.01 | 2.09 ± 0.06 | 35.29 ± 0.76 | 4.36 ± 0.15 | 4.21 ± 0.01 | 4.82 ± 0.04 | 30.28 ± 1.83 | 33.09 ± 0.09 |
| RECAP-OFO | 28.81 ± 0.69 | 41.59 ± 0.69 | 44.96 ± 0.78 | 37.68 ± 0.23 | 35.17 ± 0.26 | 33.41 ± 0.16 | 4.90 ± 0.55 | 30.51 ± 0.31 | 4.52 ± 0.25 | 4.78 ± 0.05 | 4.45 ± 0.07 | 12.50 ± 1.41 | 23.61 ± 0.26 |

## One-for-All Setting A results

### AUROC (%)

| Method | Cora | CiteSeer | ACM | BlogCatalog | Facebook | Weibo | Reddit | Amazon | Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | 87.53 ± 0.95 | 90.80 ± 0.19 | 80.07 ± 0.39 | 74.17 ± 0.02 | 67.98 ± 0.46 | 88.70 ± 0.20 | 59.48 ± 0.59 | 80.77 ± 0.17 | 78.69 ± 0.03 |
| IA-GGAD | 86.93 ± 0.91 | 91.63 ± 0.11 | 78.97 ± 0.28 | 73.73 ± 0.84 | 67.81 ± 0.85 | 90.27 ± 0.39 | 56.87 ± 2.62 | 71.57 ± 11.71 | 77.22 ± 1.16 |
| UNPrompt | 48.36 ± 2.01 | 54.67 ± 2.88 | 70.29 ± 1.27 | 67.96 ± 0.47 | 50.23 ± 1.20 | 51.42 ± 1.95 | 53.55 ± 0.86 | 67.72 ± 6.34 | 58.02 ± 1.41 |
| AnomalyGFM-ZS | 56.75 ± 2.16 | 55.36 ± 3.29 | 55.17 ± 0.99 | 57.01 ± 1.46 | 40.30 ± 1.06 | 71.49 ± 8.20 | 42.03 ± 1.28 | 46.46 ± 3.44 | 53.07 ± 0.30 |
| OWLEYE | 79.88 ± 0.35 | 85.35 ± 0.75 | 76.99 ± 0.18 | 74.99 ± 0.21 | 64.63 ± 0.62 | 86.70 ± 0.54 | 55.61 ± 0.19 | 84.20 ± 1.35 | 76.04 ± 0.05 |
| RECAP | 82.58 ± 0.06 | 90.48 ± 0.13 | 78.51 ± 0.05 | 73.03 ± 0.27 | 59.79 ± 0.54 | 80.46 ± 1.24 | 59.23 ± 0.08 | 73.13 ± 1.15 | 74.65 ± 0.23 |

### AUPRC (%)

| Method | Cora | CiteSeer | ACM | BlogCatalog | Facebook | Weibo | Reddit | Amazon | Macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | 49.31 ± 1.78 | 48.34 ± 0.77 | 40.69 ± 0.02 | 35.41 ± 0.22 | 7.89 ± 1.18 | 63.90 ± 0.64 | 4.24 ± 0.17 | 42.13 ± 4.78 | 36.49 ± 0.69 |
| IA-GGAD | 49.02 ± 1.15 | 48.75 ± 0.22 | 40.53 ± 0.08 | 35.07 ± 0.26 | 8.14 ± 2.25 | 70.02 ± 0.58 | 3.99 ± 0.39 | 28.56 ± 15.71 | 35.51 ± 2.12 |
| UNPrompt | 5.33 ± 0.42 | 5.23 ± 0.36 | 9.53 ± 1.40 | 21.54 ± 2.97 | 2.44 ± 0.21 | 19.56 ± 3.10 | 3.67 ± 0.10 | 14.51 ± 5.25 | 10.22 ± 1.42 |
| AnomalyGFM-ZS | 6.57 ± 0.44 | 5.49 ± 0.52 | 4.10 ± 0.13 | 7.24 ± 0.16 | 1.89 ± 0.07 | 30.46 ± 5.36 | 2.60 ± 0.06 | 5.85 ± 0.41 | 8.02 ± 0.53 |
| OWLEYE | 43.08 ± 0.91 | 44.28 ± 1.47 | 38.21 ± 0.24 | 33.74 ± 0.23 | 5.28 ± 0.63 | 56.81 ± 1.88 | 3.76 ± 0.03 | 60.20 ± 5.02 | 35.67 ± 0.52 |
| RECAP | 42.04 ± 0.71 | 46.53 ± 0.47 | 37.22 ± 0.09 | 33.02 ± 0.24 | 5.63 ± 1.34 | 33.87 ± 0.42 | 4.69 ± 0.02 | 13.36 ± 0.49 | 27.04 ± 0.34 |

## OFA robustness summary

| Method | Setting A dataset macro | Setting B social macro | Setting C dataset macro | Setting C domain macro |
| --- | --- | --- | --- | --- |
| ARC | 78.69 ± 0.03 / 36.49 ± 0.69 | 72.76 ± 0.23 / 29.49 ± 0.25 | 66.18 ± 0.74 / 18.86 ± 0.47 | 64.09 ± 1.06 / 13.82 ± 0.74 |
| IA-GGAD | 77.22 ± 1.16 / 35.51 ± 2.12 | 72.84 ± 0.69 / 30.74 ± 0.22 | 63.03 ± 0.44 / 17.74 ± 0.16 | 60.49 ± 0.22 / 12.50 ± 0.10 |
| UNPrompt | 58.02 ± 1.41 / 10.22 ± 1.42 | 57.92 ± 2.13 / 11.69 ± 1.21 | 59.25 ± 1.49 / 11.37 ± 0.41 | 56.64 ± 2.64 / 9.48 ± 0.90 |
| AnomalyGFM-ZS | 53.07 ± 0.30 / 8.02 ± 0.53 | 49.64 ± 2.22 / 8.43 ± 0.61 | 51.03 ± 0.79 / 5.57 ± 0.12 | 50.74 ± 1.33 / 5.24 ± 0.22 |
| OWLEYE | 76.04 ± 0.05 / 35.67 ± 0.52 | 71.79 ± 0.20 / 28.66 ± 0.22 | 60.33 ± 0.68 / 18.01 ± 0.09 | 56.24 ± 1.07 / 12.34 ± 0.25 |
| RECAP | 74.65 ± 0.23 / 27.04 ± 0.34 | 67.75 ± 0.22 / 21.98 ± 0.32 | 67.31 ± 0.43 / 17.50 ± 0.09 | 67.11 ± 0.51 / 13.58 ± 0.08 |

## Setting B: leave-Social-domain-out

| Method | Flickr | BlogCatalog | Facebook | Weibo | Reddit | Dataset macro |
| --- | --- | --- | --- | --- | --- | --- |
| ARC | 74.91 ± 0.15 / 38.42 ± 0.14 | 73.99 ± 0.29 / 34.95 ± 0.16 | 67.32 ± 1.44 / 5.87 ± 0.63 | 88.73 ± 0.37 / 64.02 ± 1.63 | 58.84 ± 0.30 / 4.20 ± 0.05 | 72.76 ± 0.23 / 29.49 ± 0.25 |
| IA-GGAD | 71.95 ± 1.17 / 37.75 ± 0.25 | 74.47 ± 0.58 / 34.94 ± 0.27 | 70.85 ± 0.40 / 6.54 ± 0.78 | 90.64 ± 0.07 / 70.51 ± 0.25 | 56.30 ± 2.93 / 3.95 ± 0.46 | 72.84 ± 0.69 / 30.74 ± 0.22 |
| UNPrompt | 68.37 ± 0.12 / 16.98 ± 2.76 | 67.77 ± 0.78 / 16.72 ± 3.06 | 52.04 ± 9.81 / 2.75 ± 0.81 | 45.36 ± 1.70 / 18.10 ± 4.51 | 56.04 ± 0.65 / 3.89 ± 0.14 | 57.92 ± 2.13 / 11.69 ± 1.21 |
| AnomalyGFM-ZS | 53.15 ± 2.03 / 7.68 ± 0.39 | 55.81 ± 2.29 / 7.12 ± 0.58 | 48.04 ± 3.49 / 2.88 ± 0.31 | 43.19 ± 8.46 / 21.54 ± 3.21 | 48.01 ± 1.54 / 2.95 ± 0.11 | 49.64 ± 2.22 / 8.43 ± 0.61 |
| OWLEYE | 75.41 ± 0.92 / 38.02 ± 0.42 | 74.88 ± 0.14 / 34.82 ± 0.12 | 64.95 ± 1.01 / 5.61 ± 1.63 | 87.70 ± 0.11 / 61.01 ± 0.81 | 56.01 ± 0.82 / 3.86 ± 0.12 | 71.79 ± 0.20 / 28.66 ± 0.22 |
| RECAP | 72.27 ± 0.45 / 34.57 ± 0.15 | 72.93 ± 0.33 / 33.27 ± 0.28 | 59.46 ± 0.93 / 5.68 ± 1.77 | 75.10 ± 1.70 / 31.71 ± 0.46 | 58.95 ± 0.78 / 4.64 ± 0.14 | 67.75 ± 0.22 / 21.98 ± 0.32 |

## Setting C: citation-only transfer

| Method | BlogCatalog | Flickr | Reddit | Amazon | Questions | Dataset macro | Domain macro |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ARC | 74.38 ± 0.18 / 36.06 ± 0.09 | 74.75 ± 0.26 / 38.91 ± 0.25 | 58.80 ± 0.94 / 4.27 ± 0.16 | 65.43 ± 2.67 / 11.08 ± 2.24 | 57.52 ± 0.30 / 3.97 ± 0.09 | 66.18 ± 0.74 / 18.86 ± 0.47 | 64.09 ± 1.06 / 13.82 ± 0.74 |
| IA-GGAD | 74.40 ± 0.84 / 35.23 ± 0.52 | 71.29 ± 1.23 / 37.77 ± 0.10 | 54.86 ± 1.09 / 3.80 ± 0.21 | 57.70 ± 0.18 / 8.01 ± 0.12 | 56.92 ± 0.18 / 3.89 ± 0.06 | 63.03 ± 0.44 / 17.74 ± 0.16 | 60.49 ± 0.22 / 12.50 ± 0.10 |
| UNPrompt | 66.44 ± 1.32 / 18.17 ± 1.78 | 69.42 ± 0.42 / 20.78 ± 0.71 | 53.59 ± 0.30 / 3.65 ± 0.08 | 63.33 ± 6.43 / 11.59 ± 3.02 | 43.45 ± 2.17 / 2.65 ± 0.27 | 59.25 ± 1.49 / 11.37 ± 0.41 | 56.64 ± 2.64 / 9.48 ± 0.90 |
| AnomalyGFM-ZS | 58.80 ± 0.94 / 7.55 ± 0.20 | 54.58 ± 1.76 / 8.03 ± 0.21 | 41.06 ± 2.75 / 2.58 ± 0.14 | 54.00 ± 4.29 / 6.89 ± 0.74 | 46.73 ± 0.67 / 2.79 ± 0.06 | 51.03 ± 0.79 / 5.57 ± 0.12 | 50.74 ± 1.33 / 5.24 ± 0.22 |
| OWLEYE | 74.89 ± 0.19 / 36.80 ± 0.40 | 74.36 ± 1.07 / 39.42 ± 0.13 | 50.11 ± 0.67 / 3.32 ± 0.02 | 55.16 ± 3.14 / 7.48 ± 0.81 | 47.10 ± 0.01 / 3.02 ± 0.21 | 60.33 ± 0.68 / 18.01 ± 0.09 | 56.24 ± 1.07 / 12.34 ± 0.25 |
| RECAP | 72.51 ± 1.40 / 32.23 ± 0.91 | 71.90 ± 0.78 / 33.40 ± 0.14 | 58.42 ± 0.14 / 4.52 ± 0.04 | 71.86 ± 1.18 / 12.72 ± 0.53 | 61.86 ± 0.56 / 4.63 ± 0.02 | 67.31 ± 0.43 / 17.50 ± 0.09 | 67.11 ± 0.51 / 13.58 ± 0.08 |

## Corrected community-stability summary

| Setting | Scope | NMI | ARI | Soft co-assignment | Score Spearman | C_eff |
| --- | --- | --- | --- | --- | --- | --- |
| OFO | 12 datasets | 0.372 ± 0.008 | 0.415 ± 0.017 | 0.650 ± 0.012 | 0.899 ± 0.006 | 24.91 ± 1.49 |
| A | 8 datasets | 0.265 ± 0.037 | 0.364 ± 0.094 | 0.576 ± 0.078 | 0.878 ± 0.019 | 16.20 ± 2.33 |
| B | 5 datasets | 0.172 ± 0.005 | 0.257 ± 0.056 | 0.573 ± 0.040 | 0.735 ± 0.006 | 22.34 ± 1.29 |
| C | 5 datasets | 0.091 ± 0.019 | 0.047 ± 0.096 | 0.331 ± 0.079 | 0.663 ± 0.018 | 16.15 ± 0.49 |

## Corrected RECAP timing summary

| Setting | Source graphs | Target graphs | Preparation (s) | Training (s) | Diagnostics (s) | Inference/target (s) |
| --- | --- | --- | --- | --- | --- | --- |
| OFO | 1 per model | 12 | 0.22 ± 0.10 | 3.99 ± 0.02 | 0.13 ± 0.00 | 0.02 ± 0.00 |
| A | 4 | 8 | 0.84 ± 0.06 | 25.92 ± 0.23 | 0.70 ± 0.01 | 0.01 ± 0.00 |
| B | 4 | 5 | 0.64 ± 0.02 | 24.68 ± 0.09 | 0.67 ± 0.02 | 0.01 ± 0.00 |
| C | 4 | 5 | 0.64 ± 0.02 | 11.66 ± 0.10 | 0.32 ± 0.01 | 0.05 ± 0.02 |

## Consistency audit

| Check | Result |
| --- | --- |
| all required cells have seeds 0 1 2 | PASS |
| no duplicate required cells | PASS |
| all metrics finite and in unit interval | PASS |
| all six source artifact audits passed | PASS |
| three baseline extension 81 of 81 runs | PASS |
| three baseline extension 126 of 126 evaluations | PASS |
| three baseline metric recomputation exact | PASS |
| three baseline label and score freeze audit passed | PASS |
| dataset macros recomputed seed first | PASS |
| setting c domain macros recomputed seed first | PASS |
| stability recomputed pair macro first | PASS |
| timing recomputed seed first | PASS |
| cross setting overall average omitted | PASS |
| evaluation population strata explicit | PASS |

### Corrected reporting gaps

- The original protocol text names eight OFO targets; later user instructions superseded it with all 12 datasets. Consolidated OFO tables therefore use 12 datasets.
- The earlier stability summary averaged per-dataset means and did not report macro standard deviations. The consolidated table uses seed-pair-first aggregation and reports population std.
- The earlier timing summary did not provide the protocol Table 9 seed-first mean±std view. The consolidated table now does.
- OFA results were split between a primary report and a B/C supplement. They are now combined from raw records.

### Interpretation boundaries

- Supervised OFO baselines report a held-out stratified 40% test population; unsupervised baselines and RECAP report the full graph.
- ARC excludes 10 labeled-normal target contexts from evaluation; IA-GGAD excludes 10 randomly sampled unlabeled internal reference nodes; UNPrompt, AnomalyGFM-ZS, OWLEYE, and RECAP score the full target graph. OWLEYE's 10 unlabeled target patterns remain in the evaluation population.
- Settings A, B, and C use different target sets and are not averaged into one cross-setting number.

### Evaluation strata

| Paradigm | Method | Source labels | Target context/reference | Evaluation population |
| --- | --- | --- | --- | --- |
| OFO | GCN | yes | none | stratified_test_40pct |
| OFO | GAT | yes | none | stratified_test_40pct |
| OFO | BWGNN | yes | none | stratified_test_40pct |
| OFO | XGBGraph | yes | none | stratified_test_40pct |
| OFO | DOMINANT | no | none | full_graph |
| OFO | AnomalyDAE | no | none | full_graph |
| OFO | CoLA | no | none | full_graph |
| OFO | ADA-GAD | no | none | full_graph |
| OFO | DiffGAD | no | none | full_graph |
| OFO | GUIDE | no | none | full_graph |
| OFO | RECAP-OFO | no | none | full_graph |
| OFA | ARC | yes | 10 labeled-normal target nodes | all target nodes except the 10 contexts |
| OFA | IA-GGAD | yes | 10 unlabeled random internal references | all target nodes except the 10 references |
| OFA | UNPrompt | yes | none | full target graph |
| OFA | AnomalyGFM-ZS | yes | none | full target graph |
| OFA | OWLEYE | yes | 10 unlabeled target patterns | full target graph |
| OFA | RECAP | no | none | full target graph |

## Machine-readable evidence

- `completion_matrix.csv` and `completion_matrix.json`
- `consistency_audit.json`
- `consolidated_ofo12_by_dataset.csv` and `consolidated_ofo12_macro.csv`
- `consolidated_ofa_abc_by_dataset.csv` and `consolidated_ofa_abc_macro.csv`
- `recap_stability_pair_macros.csv` and `recap_stability_summary.csv`
- `recap_timing_by_seed.csv` and `recap_timing_summary.csv`
- `evaluation_strata.csv`
- `THREE_BASELINE_EXTENSION_REPORT.md`
- Three-baseline audit and summaries under `rebuttal/artifacts/three_baseline_extension/analysis/`
