# Context-Neighborhood Ablation

## Controlled intervention

The trained checkpoint, residual embeddings, soft community assignments, target-induced prototypes, prototype-adhesion scores, `k=64`, `tau_s=0.3`, `beta=0.02`, and score normalization are fixed. Only the inference-time context graph is changed from residual-space cosine KNN to exact cosine KNN over the aligned 32-dimensional input features.

## Three-seed, eight-target dataset-macro results

| Scoring method | AUROC | AUPRC |
|---|---:|---:|
| Prototype only | 0.7343±0.0011 | 0.2666±0.0037 |
| + Residual-KNN context | 0.7465±0.0023 | 0.2704±0.0034 |
| + Aligned-feature-KNN context | 0.7494±0.0002 | 0.2721±0.0038 |

## Paired context effects relative to prototype-only scoring

| Context graph | ΔAUROC | ΔAUPRC | Positive seeds |
|---|---:|---:|---:|
| Residual KNN | +0.0122±0.0021 | +0.0038±0.0005 | 3/3 AUROC, 3/3 AUPRC |
| Aligned-feature KNN | +0.0151±0.0013 | +0.0055±0.0008 | 3/3 AUROC, 3/3 AUPRC |

## Per-target means

| Target | Prototype only | Residual-KNN context | Aligned-feature-KNN context |
|---|---:|---:|---:|
| Cora | 0.8259/0.4200 | 0.8258/0.4204 | 0.8254/0.4199 |
| CiteSeer | 0.9041/0.4633 | 0.9048/0.4653 | 0.9045/0.4647 |
| ACM | 0.7917/0.3706 | 0.7851/0.3722 | 0.7885/0.3694 |
| BlogCatalog | 0.7298/0.3374 | 0.7303/0.3302 | 0.7312/0.3304 |
| Facebook | 0.5989/0.0565 | 0.5979/0.0563 | 0.5975/0.0563 |
| Weibo | 0.6987/0.3047 | 0.8046/0.3387 | 0.8277/0.3572 |
| Reddit | 0.5926/0.0470 | 0.5923/0.0469 | 0.5920/0.0468 |
| Amazon | 0.7326/0.1336 | 0.7313/0.1336 | 0.7283/0.1324 |

## Validation

- Status: **pass**.
- Saved residual score identity maximum absolute error: `0`.
- Residual-KNN metrics checked against Phase 1 raw records: 24; maximum AUROC/AUPRC errors `0`/`0`.

## Interpretation

Aligned-feature-KNN context improves the dataset-macro AUROC and AUPRC over prototype-only scoring for every training seed. Therefore, the community-context aggregation remains useful when its neighborhood is not residual-derived. The gain is dataset-dependent and is largest on Weibo, so this result supports an average complementary effect rather than a claim of uniform improvement on every target graph.
