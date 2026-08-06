# RECAP T-Finance to T-Social Shared-Schema Transfer Protocol

Status: **locked before feature preparation, training, or scoring**

Date: 2026-07-27

This exploratory addendum tests whether independent graph-wise PCA coordinates
harm transfer between T-Finance and T-Social, whose released raw features have
the same 10-dimensional account-behavior schema.

- Fit robust median/IQR statistics and PCA axes on T-Finance without labels.
- Reproduce the accepted T-Finance aligned features within `1e-4`.
- If the strict pointwise maximum gate fails, the route cannot be called
  confirmatory. It may continue only as an explicitly exploratory mechanism
  check when mean absolute error is below `1e-6` and the 99.99th percentile
  is below `5e-5`; the failed maximum and its location remain reported.
- Apply the frozen T-Finance axes to T-Social, followed by label-free
  T-Social coordinate-wise z-scoring and zero-padding to 32 dimensions.
- Train RECAP label-free on the full T-Finance graph for 100 epochs and seeds
  0/1/2 with the paper configuration.
- Infer every T-Social node without target training or labels.
- Because the coordinates change, construct a new target-only FAISS-IVFPQ
  index with `nlist=4096`, `nprobe=64`, retrieve/rerank 512, and top-64.
- Freeze and hash all three paper-score vectors before loading T-Social labels.
- Report the paper score as primary and adhesion/context component ablations
  separately. Never invert a score or select a seed using target metrics.

This route is not applied to DGraph-Fin because its 17 raw features do not
share the T-Finance/T-Social schema.
