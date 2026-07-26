# Protocol Completion Summary

## Coverage

| Scope | Training runs | Final evaluations | Status |
|---|---:|---:|---|
| RECAP | 45 | 90 | PASS |
| OFA baselines | 45 | 270 | PASS |
| OFO baselines | 360 | 360 | PASS |
| Total | 450 | 720 | PASS |

Every required method-setting-dataset cell has seeds 0, 1, and 2. All six
source artifact audits pass; no missing experiment or rerun remains.

## Consolidated macros

| Scope/method | AUROC | AUPRC |
|---|---:|---:|
| RECAP-OFO, 12-dataset macro | 0.710826 ± 0.003545 | 0.236056 ± 0.002612 |
| RECAP-OFA A, dataset macro | 0.746515 ± 0.002293 | 0.270445 ± 0.003428 |
| RECAP-OFA B, dataset macro | 0.677450 ± 0.002182 | 0.219756 ± 0.003185 |
| RECAP-OFA C, dataset macro | 0.673100 ± 0.004322 | 0.175007 ± 0.000924 |
| RECAP-OFA C, domain macro | 0.671091 ± 0.005091 | 0.135780 ± 0.000781 |
| DiffGAD OFO, 12-dataset macro | 0.5503 ± 0.0217 | 0.1061 ± 0.0024 |
| GUIDE OFO, 12-dataset macro | 0.7519 ± 0.0014 | 0.3309 ± 0.0009 |
| OWLEYE OFA A, dataset macro | 0.7604 ± 0.0005 | 0.3567 ± 0.0052 |
| OWLEYE OFA B, dataset macro | 0.7179 ± 0.0020 | 0.2866 ± 0.0022 |
| OWLEYE OFA C, dataset macro | 0.6033 ± 0.0068 | 0.1801 ± 0.0009 |

## Corrected aggregation evidence

The 12-dataset RECAP-OFO stability macro, computed by averaging datasets
within each seed pair before reporting population standard deviation, is NMI
0.372432 ± 0.007804, ARI 0.415125 ± 0.016583, soft co-assignment
0.650046 ± 0.012397, and score Spearman 0.898862 ± 0.006192.

The protocol Table 9 timing view was recomputed by averaging target graphs
within each seed before reporting three-seed mean ± population standard
deviation. RECAP-OFO training is 3.9940 ± 0.0165 seconds; shared OFA training
is 25.9179 ± 0.2251, 24.6826 ± 0.0866, and 11.6583 ± 0.1011 seconds for
Settings A/B/C.

## Interpretation boundary

Supervised OFO baselines use a held-out stratified 40% target-test population;
unsupervised OFO baselines and RECAP use the full graph. ARC excludes 10
labeled-normal target contexts, while IA-GGAD excludes 10 randomly sampled
unlabeled internal reference nodes. OWLEYE uses source labels and ten unlabeled
target pattern nodes, but scores the full target population before target-label
access. These strata are reported together only with explicit annotations.
