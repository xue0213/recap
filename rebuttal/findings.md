# RECAP Phase 1–2 Findings

## Research Question

Does RECAP remain effective and community-stable under the locked OFO and OFA
evaluation protocol, including the expanded DiffGAD, GUIDE, and OWLEYE
comparison?

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

The user-requested 12-dataset OFO baseline reproduction is complete: eight
methods, twelve datasets, three seeds, 288 training-and-inference runs, and 576
independently recomputed AUROC/AUPRC values. All score/mask/data hashes, label
audits, and checkpoint reloads passed. Its accounted formal-run time is
2344.41 seconds, excluding smokes, orchestration, independent auditing, and
report generation.

The three-baseline extension is complete: 81 training runs and 126 final
evaluations for DiffGAD, GUIDE, and OWLEYE. The independent audit verified all
126 frozen score vectors, 288 label events, checkpoint reloads, and metric
recomputations with zero metric discrepancy.

The final protocol-wide audit is complete. It jointly covers 450 training runs
and 720 final evaluations: 45/90 for RECAP, 45/270 for the OFA baselines, and
360/360 for the OFO baselines. Every required cell has seeds 0/1/2, all six
source artifact audits pass, and no missing experiment or rerun remains.

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
- The completed supervised OFO baseline macros (AUROC/AUPRC) are GCN
  0.81525/0.42827, GAT 0.83085/0.43336, BWGNN 0.74378/0.29687, and
  XGBGraph 0.88840/0.61981. These methods use a stratified 40% target-test
  population after train/validation supervision.
- The completed full-graph unsupervised OFO baseline macros are DOMINANT
  0.53321/0.10402, AnomalyDAE 0.56238/0.09969, CoLA 0.54363/0.11881,
  and ADA-GAD 0.57575/0.07172. RECAP-OFO's separately labelled post hoc
  12-dataset macro is 0.71083/0.23606 on the same full-graph, label-free
  evaluation regime.
- The added full-graph OFO macros are DiffGAD 0.5503 ± 0.0217 AUROC /
  0.1061 ± 0.0024 AUPRC and GUIDE 0.7519 ± 0.0014 /
  0.3309 ± 0.0009.
- OWLEYE's OFA dataset macros are 0.7604/0.3567 in Setting A,
  0.7179/0.2866 in Setting B, and 0.6033/0.1801 in Setting C; its Setting-C
  domain macro is 0.5624/0.1234.
- Under the protocol-correct seed-pair-first aggregation, the 12-dataset
  RECAP-OFO stability macro is NMI 0.3724 ± 0.0078, ARI
  0.4151 ± 0.0166, soft co-assignment 0.6500 ± 0.0124, and score Spearman
  0.8989 ± 0.0062. Effective community count is 24.91 ± 1.49.
- Under the protocol-correct seed-first timing aggregation, RECAP-OFO averages
  3.99 ± 0.02 seconds training and 0.020 ± 0.000 seconds inference per target.
  Shared OFA training averages 25.92 ± 0.23, 24.68 ± 0.09, and
  11.66 ± 0.10 seconds for Settings A/B/C, respectively.

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
- XGBGraph is the strongest reproduced supervised OFO baseline but also the
  main runtime bottleneck: 1306.99 of 2344.41 accounted seconds. The locked
  100-tree configuration was retained.
- RECAP exceeds the original DOMINANT/AnomalyDAE/CoLA/ADA-GAD full-graph
  unsupervised set and DiffGAD, but GUIDE exceeds RECAP on both expanded
  12-dataset macro metrics (0.7519/0.3309 versus 0.7108/0.2361). The earlier
  leadership claim is therefore valid only for its original four-baseline
  scope.
- OWLEYE must not be described as fully unsupervised: its official zero-shot
  path uses source normal/anomaly labels, while target labels remain sealed
  until the full target score vector is frozen.
- GAT has materially higher macro AUPRC seed variation (0.04731) than the
  other supervised baselines. The three fixed seeds were retained without
  result-dependent reruns.
- The earlier human-readable stability table omitted the protocol-required
  macro standard deviations and averaged per-dataset summaries instead of
  forming seed-pair macros first. The corrected consolidation changes only the
  reporting aggregation, not any model output.
- The earlier timing report contained the raw components and counts but not
  the protocol Table 9 seed-first mean±standard-deviation view. The corrected
  consolidation fills that reporting gap without rerunning a model.

## Lessons and Constraints

- The dirty ANN working tree is not a valid source for paper-setting Phase 1
  because it drifted from the committed `tau_c`, entropy-constraint, and
  `lambda_E=0` implementation.
- Target labels must never be read inside training or checkpoint-selection
  paths. Intermediate target metrics are post-processed.
- The original locked Phase 1 excluded Questions only from its initial
  11-dataset OFO manifest; the later RECAP addendum and the current
  12-dataset baseline study include Questions. OFA always retained it where
  specified by Settings A/B/C.
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
- The 12-dataset OFO study intentionally has two evaluation populations:
  supervised GCN/GAT/BWGNN/XGBGraph use the stratified test mask, whereas
  unsupervised DOMINANT/AnomalyDAE/CoLA/ADA-GAD and RECAP use every node.
- DOMINANT's quadratic structure term was replaced by an exact algebraic
  identity. AnomalyDAE and ADA-GAD use the pre-registered 1:1 non-edge
  sampling with inverse-probability weighting; CoLA uses the documented PyGOD
  random-neighbor context adapter. These adaptations and their equivalence or
  smoke gates are part of the final report.
- DiffGAD's released target-label selection over autoencoder trials and 500
  diffusion levels was removed. Formal inference uses one preregistered
  ten-level average and an exact non-quadratic structure identity.
- GUIDE's nested motif enumeration was replaced by exact ORCA order-four node
  orbits, verified against independent induced-subgraph enumeration.
- OWLEYE uses official hash-matched 64-dimensional feature caches. Its
  released tau=1 pair-distance multiplier cancels exactly, and chunked target
  inference was numerically verified against the full-query path.

## Open Questions

- Why is OFO below random on YelpChi under the paper-locked configuration?
- Why do cross-domain Setting C communities change substantially across seeds
  even when final score rankings are moderately stable?
- Why does DiffGAD have high seed variance on Facebook and Weibo?
- Which exact GUIDE motif families account for its gain over RECAP on the
  expanded unsupervised OFO macro?
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

The 12-dataset OFO baseline phase proceeded method by method with an audit
after every 12-run seed block and a reflection after every 36-run method.
XGBGraph's native Booster persistence and the scale-aware CUDA reload threshold
were locked at the smoke stage. Formal runs then completed without scope
reduction, parallel timing contamination, metric-driven tuning, or selective
reruns; a final independent analyzer regenerated all 576 reported metrics from
the frozen score vectors.

The three-baseline extension began with upstream supervision and leakage
inspection, then locked exact scalable replacements and passed numerical and
end-to-end smokes before formal execution. Its analyzer independently
recomputed all 126 evaluations, after which the protocol-wide consolidator was
regenerated and deterministically verified at 450 training runs and 720 final
evaluations.

## Large-Target Inference Findings

The confirmatory target-side full-graph inference phase is complete: zero new
training runs and nine evaluations from the accepted RECAP-OFA Setting-A
checkpoints. Every run produced a finite score for every target node. The
independent audit verified all data, checkpoint, candidate, score and mask
hashes, confirmed that scores froze before labels were unlocked, and
recomputed all 18 metrics with no discrepancy.

- T-Finance exact-primary AUROC/AUPRC is
  0.25566 ± 0.02002 / 0.02801 ± 0.00077.
- DGraph-Fin ANN-primary AUROC/AUPRC is
  0.36747 ± 0.00651 / 0.00904 ± 0.00009.
- T-Social ANN-primary AUROC/AUPRC is
  0.43785 ± 0.00515 / 0.02595 ± 0.00012.
- Shared cold setup takes 9.71, 503.73 and 841.69 seconds, respectively.
  Warm per-checkpoint inference averages 0.16, 6.61 and 10.51 seconds.
- Peak GPU allocation is 0.98, 48.10 and 75.13 GiB; peak reserved memory on
  T-Social is 80.66 GiB against the 95.59-GiB device.
- ANN recall@64 is 0.8934 over all T-Finance nodes and 0.5735/0.4338 over the
  fixed 512-query DGraph-Fin/T-Social samples. The million-node approximation
  is therefore computationally useful but only moderately faithful.

The accepted conclusion is deliberately narrow. These results establish that
the isolated adapter can complete target-side inference on graphs up to
5.78 million nodes and 146.21 million stored adjacency entries under the
recorded hardware. They do not test large-target training. All three mean
AUROCs are below 0.5 and all mean AUPRCs are below their anomaly-prevalence
references, so the experiment does not support predictive effectiveness of
the unadapted Setting-A checkpoints on these large targets. No score inversion,
target-label tuning, or selective rerun was used to hide that negative result.

## Large-Target Predictive Optimization (In Progress)

The first outer-loop test scanned all 45 accepted RECAP checkpoints on
T-Finance: OFA A/B/C and all 12 OFO source families, each with seeds 0/1/2.
Every checkpoint produced and hashed the paper score, adhesion-only score and
context-only score before any label was loaded. Three label-blind,
seed-aligned ensembles were frozen at the same boundary. An independent audit
verified 162 score hashes and recomputed 162 metric rows with zero difference.

The result rules out source selection among the existing small graphs as the
main fix. OFA-A remains the best three-seed full-score family at
0.25566 AUROC / 0.02801 AUPRC. The best individual full-score checkpoint is
only 0.30770 AUROC. Context alone is less anti-correlated but the best
three-seed family, OFO-BlogCatalog, is still only 0.37073 AUROC. No score was
inverted.

A T-Finance-fitted shared coordinate cache for T-Social was prepared without
labels. The strict source reproduction maximum gate failed because one
extreme coordinate differs by 3.18e-4; mean absolute difference is 8.67e-8
and the 99.99th percentile is 1.19e-5. The route is therefore retained only
as exploratory mechanism evidence. Its T-Social feature SHA-256 is
`6f6e36ec03a1aa85e245693887fef4f781f0b36d485712a57235986c74709966`.

The target-adaptation implementation uses deterministic label-free node
samples, full-graph propagated features, fixed ANN candidates and a two-pass
chunked quotient derivative. Its community loss is algebraically identical
to the original symmetrized fixed-KNN loss on the numerical gate (difference
zero). Residual and community-weight gradients also match the native loss
within the locked `2e-6` absolute/relative tolerance. The default CPU runtime
over-expanded a tiny backward test across the host's large thread pool; with
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, the gradient test completes in
0.009 seconds and the four-test suite in 0.287 seconds. Formal launches must
therefore cap CPU threads explicitly. The actual CPU training smoke could not
pass the former no-card container's 2 GiB cgroup limit; DGraph-Fin and
T-Social also could not load their five propagated feature tensors in that
mode. The experiment has now migrated to a new instance with the same
95.59-GiB RTX PRO 6000 GPU class and a 120-GiB cgroup limit. All three
canonical bundles, fixed KNN candidates, 45 accepted checkpoints and
optimization source files match their previously recorded SHA-256 values,
and the four numerical/protocol gates pass. Formal DGraph-Fin source scanning
was started with explicitly capped CPU threads.

The DGraph-Fin and T-Social source scans are now complete. Together with
T-Finance, the independent scanner reverified 486 frozen score hashes and
recomputed all 486 AUROC/AUPRC rows with maximum difference zero. DGraph-Fin's
best three-seed full-score family is OFA-C at 0.38389/0.00931. T-Social's is
OFO-Facebook at 0.48451/0.02932. The predeclared social-domain ensemble crosses
random only for the separately named adhesion-only ablation
(0.50222 AUROC), while post-hoc selection of OFO-weibo raises that ablation to
0.51255 and is therefore oracle evidence.

Full-graph T-Finance target adaptation completed for three seeds. The
label-free objective converged smoothly and the independent metric audit
passed with zero difference. The paper score improves from 0.25566/0.02801 to
0.31160/0.03019, and adhesion-only reaches 0.31340/0.03033, but both remain
below random. Target adaptation therefore helps modestly without resolving
the T-Finance feature/label mismatch. DGraph-Fin target adaptation is running.
