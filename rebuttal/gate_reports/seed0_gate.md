# Phase 1 Seed-0 Gate Report

Date: 2026-07-26

Status: **PASS**

## OFO Cora seed 0

- AUROC: 0.828746
- AUPRC: 0.420978
- First/last loss: 0.682219 / 0.281920
- Training diagnostics: 6 rows at epochs 1, 10, 25, 50, 75, 100
- Community matrix: 2708 x 36, float32
- Checkpoint reload maximum absolute score difference: 2.38e-7
- Training cache provenance: exact KNN, reused, fully recorded

An earlier tooling attempt completed training but failed during the
post-training checkpoint audit because of PyTorch's `weights_only=True`
default and then exposed an RNG-state device issue during recovery. Both
attempts were excluded from formal results and retained under
`artifacts/phase1/failed_attempts/`. The clean formal rerun reproduced the same
metrics.

## OFA Setting A seed 0

Sources: PubMed, Flickr, Questions, YelpChi.

| Target | AUROC | AUPRC |
|---|---:|---:|
| Cora | 0.826669 | 0.428743 |
| CiteSeer | 0.906510 | 0.464378 |
| ACM | 0.785630 | 0.373362 |
| BlogCatalog | 0.730441 | 0.332886 |
| Facebook | 0.603106 | 0.072960 |
| Weibo | 0.787219 | 0.341121 |
| Reddit | 0.591281 | 0.047181 |
| Amazon | 0.715663 | 0.126726 |

- Dataset-macro AUROC: 0.743315
- Dataset-macro AUPRC: 0.273419
- First/last four-source macro loss: 0.556675 / 0.199947
- Training time: 26.21 seconds
- Diagnostic time: 0.72 seconds
- Peak allocated GPU memory: 38,170.74 MiB
- Diagnostic rows: 30
- Checkpoint reload maximum absolute score difference: 2.38e-7

## Manuscript comparison

The manuscript reports a three-seed Setting A dataset-macro AUROC of about
0.7448 and AUPRC of about 0.2658 in the module-removal summary. The seed-0 gate
is close on AUROC and is 0.0076 higher on AUPRC, within a reasonable macro
seed-level fluctuation relative to the reported standard deviation.

Per-dataset seed-0 values are broadly aligned with the manuscript three-seed
means. Facebook AUPRC is notably higher than the submitted mean and will be
re-audited after seeds 1 and 2; it is not used to tune or change any setting.

## Decision

Proceed with the remaining locked manifest. Do not change any paper
hyperparameter. Recheck Setting A mean and population standard deviation after
all three seeds before final aggregation.
