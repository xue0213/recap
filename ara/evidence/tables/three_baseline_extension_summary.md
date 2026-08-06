# Three-Baseline Extension Summary

## Coverage and audit

| Scope | Training runs | Final evaluations | Status |
|---|---:|---:|---|
| DiffGAD OFO | 36 | 36 | PASS |
| GUIDE OFO | 36 | 36 | PASS |
| OWLEYE OFA A/B/C | 9 | 54 | PASS |
| Total | 81 | 126 | PASS |

The independent audit verified 126 frozen score files, 288 label events, and
all checkpoint reloads. Metric recomputation had maximum difference zero;
maximum reload score difference was `8.7916851e-06`.

## Macro results

All values are percentages, mean ± population standard deviation over seeds.

| Method/scope | AUROC | AUPRC |
|---|---:|---:|
| DiffGAD OFO, 12-dataset macro | 55.03 ± 2.17 | 10.61 ± 0.24 |
| GUIDE OFO, 12-dataset macro | 75.19 ± 0.14 | 33.09 ± 0.09 |
| OWLEYE OFA A, dataset macro | 76.04 ± 0.05 | 35.67 ± 0.52 |
| OWLEYE OFA B, dataset macro | 71.79 ± 0.20 | 28.66 ± 0.22 |
| OWLEYE OFA C, dataset macro | 60.33 ± 0.68 | 18.01 ± 0.09 |
| OWLEYE OFA C, domain macro | 56.24 ± 1.07 | 12.34 ± 0.25 |

DiffGAD and GUIDE use the full graph without labels. OWLEYE explicitly uses
source labels but freezes every full-target score vector before target-label
access; its ten unlabeled target pattern nodes remain in the query population.
