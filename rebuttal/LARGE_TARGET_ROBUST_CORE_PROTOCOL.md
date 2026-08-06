# RECAP Large-Target Robust-Core Adaptation Protocol

Status: **locked after the uniform target-adaptation results and before any
robust-core score is generated**

Date: 2026-07-28

Classification: **exploratory, label-free and potentially deployable only
after independent confirmation**. Target labels have already been observed
for earlier experiments, so this addendum cannot be promoted to confirmatory
evidence even though its implementation is label-isolated.

## Hypothesis

Uniform target adaptation may teach RECAP communities to absorb anomalous
nodes. Training on a fixed, label-free central subset should make prototype
adhesion more selective and improve full-target anomaly ranking.

## Fixed normal-core rule

For every target, before loading labels:

1. Use the released aligned feature matrix before graph propagation.
2. Compute each node's Euclidean feature norm.
3. Compute the absolute robust deviation of `log1p(unweighted_degree)` from
   its median, scaled by its interquartile range.
4. Convert both quantities to deterministic stable ordinal percentile ranks
   and average them with weights `0.5/0.5`.
5. Keep the lowest-scoring 90% of nodes as the robust core.
6. Train on all retained T-Finance nodes. For DGraph-Fin and T-Social, take
   the same fixed-seed (`20260727`) uniform 131,072-node sample from the core.

The selection rule, fraction and weights are identical for all three targets
and are not changed after metrics are known.

## Model and evaluation

- Keep the paper-locked RECAP model, optimizer, 100 epochs, K=64 and all
  original loss hyperparameters.
- Build fixed, label-blind training candidates on the selected core sample.
- Reuse the immutable full-target exact/ANN candidates for inference.
- Run seeds 0/1/2.
- Freeze and hash paper, adhesion-only and context-only score vectors for all
  seeds before the first label access in each target process.
- Use the original evaluation populations and report AUROC/AUPRC mean and
  population standard deviation.
- Never invert scores. The paper score remains primary; component routes are
  named ablations.

## Interpretation

All results are reported as exploratory because this mechanism was proposed
after earlier target metrics were observed. Success would justify a future
pre-registered replication; failure would rule out simple label-free
normal-core filtering as the main missing ingredient.
