# RECAP Phase 1–2 Findings

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

The locked Phase 2 supervised OFA baseline reproduction is also complete:
24 training runs, 156 final evaluations, and an independent audit of every
frozen score, metric, checkpoint, label event, source-only calibration, and
protocol hash. Its accounted preparation, training, and evaluation time was
957.48 seconds.

The user-revised B/C completion supplement is complete as a separate immutable
artifact: 12 additional training runs and 60 evaluations for UNPrompt and
AnomalyGFM-ZS. Its independently audited accounted time was 971.17 seconds.

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
- Setting A supervised baseline macros (AUROC/AUPRC) are ARC
  0.7869/0.3649, IA-GGAD 0.7722/0.3551, UNPrompt 0.5802/0.1022, and
  AnomalyGFM-ZS 0.5307/0.0802.
- Setting B macros are ARC 0.7276/0.2949 and IA-GGAD 0.7284/0.3074.
- Setting C dataset macros are ARC 0.6618/0.1886 and IA-GGAD
  0.6303/0.1774.
- The completed Setting B zero-context baseline macros are UNPrompt
  0.5792/0.1169 and AnomalyGFM-ZS 0.4964/0.0843.
- The completed Setting C zero-context baseline macros are UNPrompt
  0.5925/0.1137 and AnomalyGFM-ZS 0.5103/0.0557.

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
- ARC and IA-GGAD lead label-free RECAP in Settings A/B. With citation-only
  sources in Setting C, RECAP has higher dataset-macro AUROC than both, while
  its AUPRC is slightly below both.
- In Setting A, RECAP exceeds the target-context-free UNPrompt and
  AnomalyGFM-ZS reproductions on both macro metrics. This negative baseline
  evidence is retained without target-driven tuning.
- The same ordering holds in Settings B/C: RECAP exceeds both UNPrompt and
  AnomalyGFM-ZS on dataset-macro AUROC and AUPRC. This completes the symmetric
  four-baseline A/B/C comparison requested by the user.
- IA-GGAD's Setting-A Amazon result is high variance
  (AUROC standard deviation 0.1171 and AUPRC standard deviation 0.1571);
  it must not be summarized as uniformly stable.

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
- ARC alone uses 10 labeled-normal target contexts. IA-GGAD's 10 internal
  target references are sampled without labels; UNPrompt and AnomalyGFM-ZS
  use no target context.
- IA-GGAD's released target-specific weights were not used. Source-only seed-0
  selection froze weights 0.1/0.3/0.5 for A/B/C. AnomalyGFM-ZS froze weight
  6.0 for A and independently froze source-only weights 2.0/4.0 for B/C.
- The rejected UNPrompt CUDA sparse-reduction attempt and the three
  pre-amendment ARC runs remain preserved outside the accepted formal run
  directory. They are excluded from every reported aggregate.

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

Phase 2 followed the same pattern: upstream/data/environment provenance, dense
versus sparse equivalence gates, four method smokes, a deterministic UNPrompt
recovery gate, then the immutable 24-run manifest and an independent 156-score
recomputation. Direct top-8 ARPACK SVD and feature-cache reuse removed redundant
work without changing the retained feature subspace or formal objectives.

The B/C completion was pre-registered as a user-revised supplement rather than
rewriting the original manifest. Ten tests and four method-setting smokes
preceded its 12 formal runs; all 60 metrics were recomputed independently.
