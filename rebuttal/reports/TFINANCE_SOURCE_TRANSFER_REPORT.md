# RECAP T-Finance Single-Source Transfer Report

Date: 2026-07-28

## Design and evidence boundary

This experiment reuses the three already completed and independently audited
label-free T-Finance epoch-100 checkpoints. Repeating the identical training
would add no information; checkpoint SHA-256 values are locked in
`TFINANCE_SOURCE_TRANSFER_PROTOCOL.md`.

All targets use their canonical independently aligned 32-dimensional
features. T-Finance uses exact top-64 candidates; DGraph-Fin and T-Social use
their locked ANN top-64 candidates. All 27 score vectors were finite and
globally frozen before any target label was loaded. Score inversion and
target-metric model selection are forbidden.

T-Finance is an unsupervised same-graph training/evaluation result.
DGraph-Fin and T-Social are cross-graph, target-label-free transfer results.
They are not combined into a macro average.

## Results

Mean ± population standard deviation over seeds 0/1/2:

| Target | Full RECAP AUROC / AUPRC | Adhesion-only AUROC / AUPRC | Context-only AUROC / AUPRC |
|---|---:|---:|---:|
| T-Finance | 0.311602 ± 0.018624 / 0.030188 ± 0.000772 | 0.313401 ± 0.022597 / 0.030330 ± 0.000934 | 0.266782 ± 0.006764 / 0.029891 ± 0.000421 |
| DGraph-Fin | 0.364853 ± 0.001316 / 0.009024 ± 0.000016 | 0.376467 ± 0.000429 / 0.009187 ± 0.000008 | 0.386155 ± 0.004815 / 0.009575 ± 0.000076 |
| T-Social | 0.438492 ± 0.003158 / 0.025703 ± 0.000275 | 0.478123 ± 0.002638 / 0.028783 ± 0.000287 | 0.438791 ± 0.005540 / 0.025634 ± 0.000307 |

Primary full-score comparison against the original Setting-A checkpoints:

| Target | Original Setting-A AUROC / AUPRC | T-Finance-trained AUROC / AUPRC | Change |
|---|---:|---:|---:|
| T-Finance | 0.255656 / 0.028012 | 0.311602 / 0.030188 | +0.055946 / +0.002176 |
| DGraph-Fin | 0.367466 / 0.009044 | 0.364853 / 0.009024 | -0.002612 / -0.000020 |
| T-Social | 0.437851 / 0.025952 | 0.438492 / 0.025703 | +0.000641 / -0.000249 |

The AUPRC random references (evaluation anomaly prevalence) are 0.04584,
0.01265 and 0.03015. The primary score remains below both random-ranking
references on all three targets.

## Conclusion

Large-graph T-Finance training improves the same-graph result but does not
produce useful anomaly ranking. More importantly, it does not improve
DGraph-Fin despite the shared broad financial domain, and it is effectively
unchanged on the cross-domain T-Social negative control. This rejects the
hypothesis that choosing a large financial source alone resolves the transfer
failure. Feature semantics and graph/anomaly mechanisms differ enough that
domain identity is not a sufficient source-selection rule.

The independent audit passed 27 score hashes and 27 AUROC/AUPRC
recomputations with maximum difference 0. Audit SHA-256:
`95a083a823edbae120f5c12ff0f66927f9ba7640e5dd410a3b36b64ed60f97d2`.
