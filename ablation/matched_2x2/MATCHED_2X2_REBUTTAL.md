#### 3. Matched \(2\times2\) attribution study

To separate the contribution of residual encoding from that of RECAP's
community framework, we conduct a matched factorial comparison using the same
eight Setting-A targets, hyperparameters, and five random seeds:

| Representation | Conventional detector | RECAP community learning/scoring |
|---|---:|---:|
| Non-residual propagated features | 0.6029±0.0019 / 0.1373±0.0031 | 0.5339±0.0019 / 0.1060±0.0005 |
| Residual features | 0.6782±0.0034 / 0.1823±0.0045 | **0.7448±0.0025 / 0.2658±0.0057** |

Values are five-seed dataset-macro AUROC/AUPRC. The conventional detector is
KMeans with \(C=36\). `RECAP w/o Residual` replaces residual hop differences
with the matched concatenation of propagated-hop features while retaining all
other RECAP components.

The comparison gives two controlled conclusions. First, under the same KMeans
detector, residual encoding improves AUROC/AUPRC by **7.53/4.50 percentage
points**, confirming that the ARC-derived representation is useful. Second,
with the residual representation fixed, replacing KMeans with RECAP's
community learning and scoring improves AUROC/AUPRC by **6.66/8.36 points**.
Thus, residual features alone do not explain RECAP's performance.

The table also reveals a positive difference-in-differences interaction of
**13.56/11.48 points**. RECAP's community framework is therefore not
representation-agnostic: it is specifically effective in the residual pattern
space, while its non-residual variant does not outperform non-residual KMeans.
We will revise our wording accordingly. The evidence supports a
**residual-aware community-learning contribution and a strong interaction
between the two components**, rather than claiming that they are independent
additive modules.
