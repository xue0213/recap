# GUIDE/DiffGAD Source-Only OFA Adaptation Report

All numbers are dataset-macro AUROC/AUPRC in percent, mean +/- population standard deviation over seeds 0/1/2.

| Method | Setting A | Setting B | Setting C |
|---|---:|---:|---:|
| RECAP-OFA | 74.65 +/- 0.23 / 27.04 +/- 0.34 | 67.75 +/- 0.22 / 21.98 +/- 0.32 | 67.31 +/- 0.43 / 17.50 +/- 0.09 |
| GUIDE-OFA-adapted | 63.86 +/- 0.24 / 27.74 +/- 0.04 | 50.66 +/- 0.24 / 6.90 +/- 0.05 | 55.68 +/- 0.14 / 7.01 +/- 0.04 |
| DiffGAD-OFA-adapted | 66.94 +/- 0.00 / 21.11 +/- 0.02 | 59.61 +/- 0.00 / 18.47 +/- 0.00 | 69.61 +/- 0.01 / 17.99 +/- 0.00 |

## Audit

- Runs: 18/18
- Evaluations: 108/108
- Maximum metric recomputation difference: 0
- Maximum checkpoint reload score difference: 9.53674316406e-07

## Setting A per-target AUROC/AUPRC (%)

| Method | cora | citeseer | ACM | BlogCatalog | Facebook | weibo | Reddit | Amazon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GUIDE-OFA-adapted | 92.39 +/- 0.38 / 65.75 +/- 0.41 | 94.85 +/- 0.20 / 68.41 +/- 0.27 | 89.03 +/- 0.12 / 56.24 +/- 0.71 | 68.81 +/- 0.29 / 10.82 +/- 0.30 | 23.69 +/- 0.44 / 1.46 +/- 0.00 | 40.71 +/- 1.62 / 8.55 +/- 0.38 | 52.36 +/- 1.12 / 3.51 +/- 0.08 | 49.01 +/- 0.38 / 7.20 +/- 0.01 |
| DiffGAD-OFA-adapted | 76.90 +/- 0.00 / 31.17 +/- 0.02 | 83.53 +/- 0.01 / 31.20 +/- 0.17 | 80.20 +/- 0.00 / 33.05 +/- 0.02 | 77.97 +/- 0.00 / 33.98 +/- 0.00 | 39.87 +/- 0.00 / 2.39 +/- 0.00 | 46.41 +/- 0.00 / 19.07 +/- 0.00 | 56.18 +/- 0.00 / 4.05 +/- 0.01 | 74.43 +/- 0.00 / 14.00 +/- 0.00 |

## Setting B per-target AUROC/AUPRC (%)

| Method | Flickr | BlogCatalog | Facebook | weibo | Reddit |
|---|---:|---:|---:|---:|---:|
| GUIDE-OFA-adapted | 67.40 +/- 0.23 / 9.75 +/- 0.16 | 68.79 +/- 0.12 / 11.01 +/- 0.11 | 20.65 +/- 0.42 / 1.42 +/- 0.01 | 41.53 +/- 1.92 / 8.47 +/- 0.34 | 54.94 +/- 0.51 / 3.84 +/- 0.05 |
| DiffGAD-OFA-adapted | 77.02 +/- 0.00 / 32.76 +/- 0.00 | 77.92 +/- 0.00 / 33.93 +/- 0.00 | 39.80 +/- 0.01 / 2.39 +/- 0.00 | 47.03 +/- 0.01 / 19.20 +/- 0.00 | 56.26 +/- 0.01 / 4.06 +/- 0.00 |

## Setting C per-target AUROC/AUPRC (%)

| Method | BlogCatalog | Flickr | Reddit | Amazon | questions |
|---|---:|---:|---:|---:|---:|
| GUIDE-OFA-adapted | 68.98 +/- 0.06 / 11.06 +/- 0.05 | 63.40 +/- 1.01 / 9.97 +/- 0.12 | 50.80 +/- 0.54 / 3.35 +/- 0.02 | 52.07 +/- 1.06 / 7.92 +/- 0.31 | 43.13 +/- 0.37 / 2.77 +/- 0.04 |
| DiffGAD-OFA-adapted | 78.03 +/- 0.00 / 33.97 +/- 0.00 | 77.09 +/- 0.00 / 32.80 +/- 0.00 | 56.07 +/- 0.00 / 4.04 +/- 0.00 | 74.90 +/- 0.00 / 14.25 +/- 0.00 | 61.98 +/- 0.04 / 4.86 +/- 0.00 |
