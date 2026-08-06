# RECAP Experiment Protocol Completion Audit

Status: locked before the consolidated audit  
Scope: the user-revised experiment scope through the 12-dataset OFO and
RECAP-OFA A/B/C baseline addenda

## 1. Purpose

This audit does not introduce a new model configuration or rerun a completed
experiment. It verifies that every requested cell has raw seed-level evidence,
recomputes all consolidated tables from those raw records, and identifies any
remaining mismatch in datasets, seeds, label access, evaluation populations,
aggregation, timing, or artifact counts.

## 2. Authoritative scope

The later user instructions supersede the original eight-dataset OFO scope:

- RECAP-OFO: 12 datasets, three seeds.
- OFO baselines: GCN, GAT, BWGNN, XGBGraph, DOMINANT, AnomalyDAE, CoLA,
  and ADA-GAD on the same 12 datasets and three seeds.
- RECAP-OFA: Settings A, B, and C, three seeds.
- OFA baselines: ARC, IA-GGAD, UNPrompt, and AnomalyGFM-ZS in all three
  settings and three seeds.

The target counts are therefore:

- RECAP: 45 training runs and 90 final target evaluations.
- OFA baselines: 36 training runs and 216 final target evaluations.
- OFO baselines: 288 training/evaluation runs.
- Overall: 369 training runs and 594 final evaluations.

## 3. Authoritative raw inputs

The audit must use unrounded seed-level or seed-pair-level files:

- Phase 1 RECAP raw results, pairwise stability results, timing rows, artifact
  validation, and run manifests.
- Questions OFO addendum raw results and pairwise stability results.
- Phase 2 OFA baseline primary and B/C supplement raw records and artifact
  audits.
- Twelve-dataset OFO baseline run records and global audit.

Human-readable report numbers are comparison targets, not calculation inputs.

## 4. Locked aggregation

All standard deviations are population standard deviations (`ddof=0`).

### 4.1 Predictive metrics

- Dataset cells: mean and standard deviation over seeds 0, 1, and 2.
- Dataset macro: first average datasets within each seed, then report mean and
  standard deviation over the three seed-level macro values.
- Setting C domain macro: first average datasets within each domain and seed,
  then average domains within each seed, then report mean and standard
  deviation over seeds.
- No aggregate is allowed to mix OFO with OFA or Settings A, B, and C.

### 4.2 Stability

- For NMI, ARI, soft co-assignment similarity, and score Spearman correlation:
  first average datasets within each of the three seed pairs `(0,1)`, `(0,2)`,
  and `(1,2)`, then report mean and standard deviation over those three
  pair-level macro values.
- For effective community count: first average datasets within each seed, then
  report mean and standard deviation over the three seed-level macro values.
- Questions is included in the revised 12-dataset OFO aggregate.

### 4.3 Timing

- RECAP-OFO: for each seed, average the 12 dataset-specific values for data
  preparation, training, diagnostics, and inference; then report mean and
  standard deviation over seeds.
- RECAP-OFA: shared training/preparation/diagnostic time is counted once per
  seed; inference is averaged across targets within a seed; then mean and
  standard deviation are reported over seeds.
- Baseline timing remains method-specific and is never added to RECAP timing.

## 5. Consistency checks

The audit fails if any of the following holds:

- a required method/setting/dataset/seed cell is absent or duplicated;
- a predictive metric is non-finite or outside `[0,1]`;
- a method has an unexpected target set or seed set;
- a reported macro disagrees with recomputation from seed-level records beyond
  floating-point tolerance;
- an artifact audit reports failure;
- the revised RECAP-OFO stability evidence does not cover 12 datasets and all
  three seed pairs;
- RECAP diagnostics/checkpoints/community outputs do not match the expected
  revised counts;
- results with different label rights or evaluation populations are presented
  as directly equivalent without an explicit stratum annotation.

## 6. Interpretation strata

The consolidated OFO comparison must preserve these annotations:

- GCN, GAT, BWGNN, and XGBGraph are supervised and use their recorded
  train/validation/test split; their final metric population is the held-out
  test subset.
- DOMINANT, AnomalyDAE, CoLA, ADA-GAD, and RECAP-OFO are unsupervised and are
  evaluated on the full graph.
- RECAP remains label-free during training and model selection.
- OFA baselines must retain their recorded query/context population rather than
  being silently relabeled as identical to RECAP.

Consequently, the table may report all requested methods side by side, but the
audit report must state that supervised-test and unsupervised-full-graph
columns are different evaluation strata.

## 7. Required outputs

- machine-readable completion matrix;
- machine-readable consistency audit;
- consolidated 12-dataset OFO dataset and macro tables;
- consolidated OFA A/B/C dataset and macro tables;
- corrected RECAP stability table;
- corrected RECAP timing table;
- one human-readable completion report linking every protocol table to its
  evidence and explicitly listing any caveat or corrected reporting gap.

