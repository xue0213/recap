# Matched 2x2 Ablation

| Representation | Conventional detector | RECAP community learning/scoring |
|---|---:|---:|
| Non-residual | 0.6029±0.0019 / 0.1373±0.0031 | 0.5339±0.0019 / 0.1060±0.0005 |
| Residual | 0.6782±0.0034 / 0.1823±0.0045 | 0.7448±0.0025 / 0.2658±0.0057 |

Values are five-seed dataset-macro AUROC/AUPRC over the eight Setting-A target graphs.

## Paired macro effects

| Effect | AUROC difference | AUPRC difference |
|---|---:|---:|
| Residual representation effect with KMeans | +0.0753 | +0.0450 |
| Residual representation effect with RECAP | +0.2109 | +0.1598 |
| Community framework effect with non-residual features | -0.0690 | -0.0313 |
| Community framework effect with residual features | +0.0666 | +0.0836 |
| Difference-in-differences interaction | +0.1356 | +0.1148 |

## Validation

- Complete records: 160/160.
- Failed runs: 0.
- Residual + KMeans maximum absolute difference versus the accepted formal record: 8.49e-06 (tolerance: 1e-05).
