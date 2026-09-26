
All numbers are dataset-macro AUROC/AUPRC in percent, mean +/- population standard deviation over seeds 0/1/2.

| Method | Setting A | Setting B | Setting C |
|---|---:|---:|---:|
| RECAP-OFA | 74.65 +/- 0.23 / 27.04 +/- 0.34 | 67.75 +/- 0.22 / 21.98 +/- 0.32 | 67.31 +/- 0.43 / 17.50 +/- 0.09 |
| DiffGAD-OFA-adapted | 66.94 +/- 0.00 / 21.11 +/- 0.02 | 59.61 +/- 0.00 / 18.47 +/- 0.00 | 69.61 +/- 0.01 / 17.99 +/- 0.00 |

## Audit

- Runs: 18/18
- Evaluations: 108/108
- Maximum metric recomputation difference: 0
- Maximum checkpoint reload score difference: 9.53674316406e-07

## Setting A per-target AUROC/AUPRC (%)

| Method | cora | citeseer | ACM | BlogCatalog | Facebook | weibo | Reddit | Amazon |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffGAD-OFA-adapted | 76.90 +/- 0.00 / 31.17 +/- 0.02 | 83.53 +/- 0.01 / 31.20 +/- 0.17 | 80.20 +/- 0.00 / 33.05 +/- 0.02 | 77.97 +/- 0.00 / 33.98 +/- 0.00 | 39.87 +/- 0.00 / 2.39 +/- 0.00 | 46.41 +/- 0.00 / 19.07 +/- 0.00 | 56.18 +/- 0.00 / 4.05 +/- 0.01 | 74.43 +/- 0.00 / 14.00 +/- 0.00 |

## Setting B per-target AUROC/AUPRC (%)

| Method | Flickr | BlogCatalog | Facebook | weibo | Reddit |
|---|---:|---:|---:|---:|---:|
| DiffGAD-OFA-adapted | 77.02 +/- 0.00 / 32.76 +/- 0.00 | 77.92 +/- 0.00 / 33.93 +/- 0.00 | 39.80 +/- 0.01 / 2.39 +/- 0.00 | 47.03 +/- 0.01 / 19.20 +/- 0.00 | 56.26 +/- 0.01 / 4.06 +/- 0.00 |

## Setting C per-target AUROC/AUPRC (%)

| Method | BlogCatalog | Flickr | Reddit | Amazon | questions |
|---|---:|---:|---:|---:|---:|
| DiffGAD-OFA-adapted | 78.03 +/- 0.00 / 33.97 +/- 0.00 | 77.09 +/- 0.00 / 32.80 +/- 0.00 | 56.07 +/- 0.00 / 4.04 +/- 0.00 | 74.90 +/- 0.00 / 14.25 +/- 0.00 | 61.98 +/- 0.04 / 4.86 +/- 0.00 |
