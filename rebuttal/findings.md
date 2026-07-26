# RECAP Phase 1 Findings

## Research Question

Does RECAP remain effective and community-stable under the locked OFO and OFA
evaluation protocol?

## Current Understanding

The locked Phase 1 is complete and computationally feasible on the available
RTX PRO 6000 Blackwell server. All implementation, label-isolation,
cache-equivalence, data-manifest, checkpoint-reload, raw-record, and artifact
gates passed. The complete accounted data preparation, training, diagnostics,
and inference time was 313.59 seconds; this excludes orchestration pauses and
tooling fixes.

## Key Results

- OFO 11-dataset macro: AUROC 0.717747 ± 0.003883 and AUPRC
  0.253175 ± 0.002850.
- The user-requested OFO Questions addendum achieved AUROC
  0.634693 ± 0.000468 and AUPRC 0.047751 ± 0.000522. Combining it with the
  immutable 11-dataset records gives a separately labelled 12-dataset OFO
  macro of 0.710826 ± 0.003545 AUROC and 0.236056 ± 0.002612 AUPRC.
- OFA Setting A macro: AUROC 0.746515 ± 0.002293 and AUPRC
  0.270445 ± 0.003428.
- OFA Setting B macro: AUROC 0.677450 ± 0.002182 and AUPRC
  0.219756 ± 0.003185.
- OFA Setting C dataset macro: AUROC 0.673100 ± 0.004322 and AUPRC
  0.175007 ± 0.000924; domain macro is 0.671091/0.135780.
- Setting A closely reproduces the manuscript: per-dataset mean absolute
  differences are 0.37 AUROC percentage points and 0.65 AUPRC points.
- The full 12-dataset preflight completed in 27.87 seconds. All file hashes and
  actual graph statistics are locked in `data_manifest.json`.

## Patterns and Insights

- Exact KNN construction is fast enough for all Phase 1 datasets on the current
  GPU.
- Fixed feature-alignment and KNN-candidate caches can be reused across seeds
  without changing model outputs when full cache keys match.
- Soft co-assignment stability must use the exact `C x C` identity rather than
  materializing an `N x N` matrix.
- The clean Phase 1 worktree contains no ANN module, so approximate KNN cannot
  accidentally affect the confirmatory datasets.
- Setting A remains aligned with the manuscript after all three seeds.
- Final anomaly rankings are substantially more stable than community
  identities. The setting-level macro score Spearman values are 0.890, 0.878,
  0.735, and 0.663 for OFO/A/B/C; the corresponding soft co-assignment values
  are 0.632, 0.576, 0.573, and 0.331.
- Negative evidence must be retained: OFO YelpChi AUROC is 0.4260, and Setting
  C BlogCatalog score Spearman is only 0.370.
- Questions OFO is highly stable in final anomaly ranking across seeds
  (Spearman 0.993) and reasonably stable in soft community structure (0.848),
  even though its AUPRC is low because of the difficult, imbalanced target.

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
- The three historical failure-log entries were post-training tooling defects,
  not accepted scientific runs. They remain preserved for provenance.
- Questions was added only after the original 11-dataset Phase 1 completed.
  The combined 12-dataset macro is therefore a transparent post hoc scope
  addition and must not be presented as part of the original locked scope.

## Open Questions

- Why is OFO below random on YelpChi under the paper-locked configuration?
- Why do cross-domain Setting C communities change substantially across seeds
  even when final score rankings are moderately stable?
- These are follow-up research questions, not reasons to alter the completed
  confirmatory Phase 1.

## Optimization Trajectory

The trajectory moved from infrastructure validation to two seed-0 gates, then
broadened to all 42 locked runs. No metric-driven hyperparameter changes were
made. Exact feature/KNN cache reuse and the exact C×C soft co-assignment
identity reduced redundant work without changing the experiment definition or
outputs.
