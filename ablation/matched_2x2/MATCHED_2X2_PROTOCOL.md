# Matched 2x2 Residual/Community Ablation Protocol

## Question

Separate the effect of ARC-derived residual encoding from the effect of
RECAP's community learning and scoring.

## Factorial design

| Representation | Conventional detector | RECAP community learning/scoring |
|---|---|---|
| Non-residual propagated hops | Non-residual + KMeans | RECAP w/o Residual |
| Residual hop differences | Residual + KMeans | Full RECAP |

The conventional-detector column uses target-specific KMeans only. The RECAP
column uses the paper's label-free source-training/target-inference protocol.
The factorial comparison therefore isolates the representation change within
each downstream paradigm; it does not claim that KMeans and RECAP have the same
deployment protocol.

## Locked inputs

- Source graphs for RECAP cells: PubMed, Flickr, Questions, YelpChi.
- Target graphs: Facebook, Cora, CiteSeer, ACM, BlogCatalog, Weibo, Reddit,
  Amazon.
- Seeds: 0, 1, 2, 3, 4.
- Aligned dimension: 32.
- Propagation depth: 4.
- Community/KMeans count: 36.
- Metrics: full-target AUROC and AUPRC; labels are used only for final metric
  computation.

## Representations

- Residual: `[X^(1)-X^(0) || ... || X^(4)-X^(0)]`.
- Non-residual: `[X^(1) || ... || X^(4)]`.

For the KMeans cells, the selected representation is standardized
coordinate-wise on each target graph. KMeans uses Euclidean distance,
`algorithm=lloyd`, `n_init=20`, and the anomaly score is distance to the
nearest centroid.

For the RECAP cells, the non-residual variant replaces both the learned and
initial residual embeddings with the corresponding propagated-hop
concatenations while leaving all other model settings unchanged.

## Reuse and acceptance gates

- Full RECAP and RECAP w/o Residual are reused from the accepted five-seed
  module-removal run.
- Residual + KMeans is rerun together with the missing Non-residual + KMeans
  cell.
- The rerun is accepted only if every Residual + KMeans dataset/seed metric
  reproduces the accepted formal record within an absolute tolerance of
  `1e-5`. This tolerance covers CPU-versus-GPU sparse-propagation rounding;
  the accepted formal Residual + KMeans records remain the values reported in
  the final table.
- All four cells must contain all 8 targets and all 5 seeds before aggregation.
- Macro means are computed per seed across the 8 targets, followed by
  mean/population-standard-deviation across the 5 seeds.
