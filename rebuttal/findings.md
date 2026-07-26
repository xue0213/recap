# RECAP Phase 1 Findings

## Research Question

Does RECAP remain effective and community-stable under the locked OFO and OFA
evaluation protocol?

## Current Understanding

The protocol is computationally feasible on the available RTX PRO 6000
Blackwell server. A clean timing benchmark using the paper Git baseline showed
that the largest OFO-excluded graph, Questions, trains for 100 epochs in about
12.7 seconds and uses about 37.2 GiB peak GPU memory. The main risk is therefore
scientific and bookkeeping correctness rather than raw compute capacity.

Formal results have not started. The implementation, label-isolation,
checkpoint, cache-equivalence, and output-schema gates must pass first.

## Key Results

No formal metric results yet.

## Patterns and Insights

- Exact KNN construction is fast enough for all Phase 1 datasets on the current
  GPU.
- Fixed feature-alignment and KNN-candidate caches can be reused across seeds
  without changing model outputs when full cache keys match.
- Soft co-assignment stability must use the exact `C x C` identity rather than
  materializing an `N x N` matrix.

## Lessons and Constraints

- The dirty ANN working tree is not a valid source for paper-setting Phase 1
  because it drifted from the committed `tau_c`, entropy-constraint, and
  `lambda_E=0` implementation.
- Target labels must never be read inside training or checkpoint-selection
  paths. Intermediate target metrics are post-processed.
- Questions is excluded only from OFO; removing it from OFA would violate the
  locked protocol.
- Amazon contains 10,224 nodes in the actual dataset; the paper's 10,244 count
  is treated as a documented likely typo.

## Open Questions

- Does the first formal seed-0 gate reproduce the expected performance range?
- Are all final community assignments stable across seeds under every setting?
- Are any dataset-specific failures caused by preprocessing/cache version
  differences?

## Optimization Trajectory

No confirmatory runs yet.
