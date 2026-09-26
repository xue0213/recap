# RECAP Three-Baseline Extension Report

Status: **PASS**

DiffGAD and GUIDE are unsupervised one-for-one methods evaluated on the full
graph. OWLEYE is source-label supervised and target-label-free zero-shot; its
ten uniformly sampled unlabeled target pattern nodes remain in the full-target
evaluation population.

All values are mean ± population standard deviation over seeds 0/1/2, in
percent. Dataset macros are computed within each seed before the three-seed
mean and standard deviation.

## OFO AUROC (%)

| Method | PubMed | Cora | CiteSeer | ACM | Flickr | BlogCatalog | Facebook | Weibo | Reddit | Questions | YelpChi | Amazon | Macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DiffGAD | 82.72 ± 0.00 | 67.37 ± 0.00 | 69.05 ± 0.00 | 67.62 ± 0.00 | 54.27 ± 0.01 | 56.22 ± 0.01 | 29.91 ± 18.13 | 38.92 ± 13.75 | 52.30 ± 5.13 | 57.12 ± 0.00 | 29.12 ± 0.00 | 55.77 ± 0.13 | 55.03 ± 2.17 |
| GUIDE | 92.98 ± 0.03 | 97.75 ± 0.03 | 98.67 ± 0.06 | 87.53 ± 0.05 | 73.90 ± 0.01 | 74.45 ± 0.16 | 45.87 ± 1.07 | 83.02 ± 0.77 | 56.64 ± 0.21 | 59.31 ± 0.05 | 48.86 ± 0.50 | 83.33 ± 0.98 | 75.19 ± 0.14 |

## OFO AUPRC (%)

| Method | PubMed | Cora | CiteSeer | ACM | Flickr | BlogCatalog | Facebook | Weibo | Reddit | Questions | YelpChi | Amazon | Macro |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DiffGAD | 10.47 ± 0.00 | 25.58 ± 0.00 | 33.53 ± 0.00 | 9.56 ± 0.00 | 6.61 ± 0.00 | 7.16 ± 0.00 | 2.09 ± 0.98 | 8.38 ± 2.24 | 3.43 ± 0.44 | 4.91 ± 0.00 | 3.29 ± 0.00 | 12.30 ± 0.15 | 10.61 ± 0.24 |
| GUIDE | 56.18 ± 0.07 | 79.58 ± 0.16 | 81.20 ± 0.06 | 27.94 ± 0.09 | 37.42 ± 0.05 | 33.68 ± 0.01 | 2.09 ± 0.06 | 35.29 ± 0.76 | 4.36 ± 0.15 | 4.21 ± 0.01 | 4.82 ± 0.04 | 30.28 ± 1.83 | 33.09 ± 0.09 |

## OWLEYE Setting A (AUROC / AUPRC, %)

| Method | Cora | CiteSeer | ACM | BlogCatalog | Facebook | Weibo | Reddit | Amazon | Dataset macro |
|---|---|---|---|---|---|---|---|---|---|
| OWLEYE | 79.88 ± 0.35 / 43.08 ± 0.91 | 85.35 ± 0.75 / 44.28 ± 1.47 | 76.99 ± 0.18 / 38.21 ± 0.24 | 74.99 ± 0.21 / 33.74 ± 0.23 | 64.63 ± 0.62 / 5.28 ± 0.63 | 86.70 ± 0.54 / 56.81 ± 1.88 | 55.61 ± 0.19 / 3.76 ± 0.03 | 84.20 ± 1.35 / 60.20 ± 5.02 | 76.04 ± 0.05 / 35.67 ± 0.52 |

## OWLEYE Setting B (AUROC / AUPRC, %)

| Method | Flickr | BlogCatalog | Facebook | Weibo | Reddit | Dataset macro |
|---|---|---|---|---|---|---|
| OWLEYE | 75.41 ± 0.92 / 38.02 ± 0.42 | 74.88 ± 0.14 / 34.82 ± 0.12 | 64.95 ± 1.01 / 5.61 ± 1.63 | 87.70 ± 0.11 / 61.01 ± 0.81 | 56.01 ± 0.82 / 3.86 ± 0.12 | 71.79 ± 0.20 / 28.66 ± 0.22 |

## OWLEYE Setting C (AUROC / AUPRC, %)

| Method | BlogCatalog | Flickr | Reddit | Amazon | Questions | Dataset macro | Domain macro |
|---|---|---|---|---|---|---|---|
| OWLEYE | 74.89 ± 0.19 / 36.80 ± 0.40 | 74.36 ± 1.07 / 39.42 ± 0.13 | 50.11 ± 0.67 / 3.32 ± 0.02 | 55.16 ± 3.14 / 7.48 ± 0.81 | 47.10 ± 0.01 / 3.02 ± 0.21 | 60.33 ± 0.68 / 18.01 ± 0.09 | 56.24 ± 1.07 / 12.34 ± 0.25 |

## Timing and resources

| Method | Runs | Preprocess sum (s) | Train sum (s) | Inference sum (s) | Total sum (s) | Mean/run (s) | Max GPU allocated (GiB) | Peak RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DiffGAD | 36 | 5.89 | 608.76 | 40.51 | 695.81 | 19.33 | 45.76 | 2.57 |
| GUIDE | 36 | 38.71 | 196.28 | 0.99 | 237.11 | 6.59 | 3.63 | 3.74 |
| OWLEYE | 9 | 8.63 | 181.01 | 7.77 | 206.09 | 22.90 | 3.77 | 2.12 |

## Independent audit

- Training runs: 81/81
- Recomputed evaluations: 126/126
- Frozen score files verified: 126
- Label-audit events checked: 288
- Maximum metric recomputation difference: 0
- Maximum checkpoint reload score difference: 8.7916851e-06
- Problems: 0

## Fidelity notes

- DiffGAD removes the released target-label selection over autoencoder trials
  and 500 diffusion levels. It uses the locked ten-level label-free average and
  exact non-quadratic structure loss.
- GUIDE uses exact ORCA order-four node orbits. The mapping was validated
  against independent induced-subgraph enumeration.
- OWLEYE uses source labels and must not be described as fully unsupervised.
  Target labels are unavailable until every target's full-node score vector is
  frozen.
- No weak or high-variance result was replaced, tuned, or selectively rerun.
