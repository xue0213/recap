# Phase 2 B/C Baseline Completion Addendum

Status: **locked before supplementary formal results**

Date: 2026-07-26

Provenance: **user-revised scope**. After the original 24-run Phase 2 primary
track completed, the user explicitly required UNPrompt and AnomalyGFM-ZS in
Settings B and C as well. This addendum expands only the required-method scope.
It does not modify or replace any accepted primary run.

## 1. Confirmatory supplementary scope

| Setting | Sources | Targets | Added methods |
|---|---|---|---|
| B | PubMed, Cora, Questions, YelpChi | Flickr, BlogCatalog, Facebook, Weibo, Reddit | UNPrompt, AnomalyGFM-ZS |
| C | PubMed, Cora, CiteSeer, ACM | BlogCatalog, Flickr, Reddit, Amazon, Questions | UNPrompt, AnomalyGFM-ZS |

Seeds are `0`, `1`, and `2`. Each method uses one shared checkpoint per
setting and seed. The supplement therefore contains exactly 12 training runs
and 60 method/target/seed evaluations.

The original primary artifacts under
`rebuttal/artifacts/phase2_baselines/` are immutable. Supplementary artifacts
use `rebuttal/artifacts/phase2_bc_supplement/`.

## 2. Label and aggregation boundary

1. Source labels are permitted.
2. No target node, label, prompt, tuning, checkpoint selection, calibration,
   or early stopping is permitted for either method.
3. Target labels may be loaded only after the final score vector and immutable
   all-node query mask have been saved and hashed.
4. AUROC and AUPRC are recomputed from frozen scores.
5. Dataset macros first average the five targets within each seed, then report
   the three-seed mean and population standard deviation (`ddof=0`).
6. Setting C domain macro follows the already locked equal-domain weighting.

## 3. Method settings

### UNPrompt

Use the same official-compatible configuration as the accepted Setting-A
primary runs:

- exact rank-8 SVD plus non-affine batch normalization;
- four source graphs cycled in listed order;
- GRACE pretraining for 200 epochs with full all-node negatives evaluated in
  exact deterministic row blocks;
- edge-drop 0.2, feature-drop 0.3, learning rate `1e-3`, weight decay `1e-5`;
- 10 unified prompts, hidden size 128, prompt training for 900 epochs,
  learning rate `1e-3`, no prompt weight decay;
- deterministic CPU sparse aggregation for frozen target inference only.

There is no target context or calibration.

### AnomalyGFM-ZS

Use the same official-compatible configuration as the accepted Setting-A
primary runs:

- deterministic sparse ARPACK rank-8 SVD and official row normalization;
- two-layer residual GCN, hidden size 400, learned normal/anomalous prototypes;
- four source graphs cycled in listed order;
- official deterministic 30% source-node training split;
- 301 epochs, Adam learning rate `1e-4`, no weight decay;
- all target nodes remain in the query population.

The prototype score weight is selected independently for each setting on
**setting seed-0 source scores only** from `{0, .5, 1, 2, 4, 6}` using source
dataset-macro AUROC, then AUPRC, then smaller weight as deterministic
tie-breaks. The resulting B and C weights are frozen across all five targets
and seeds 0/1/2. No Setting-A result or weight is reused because the source
composition changes.

## 4. Cache and compatibility policy

Existing feature caches may be reused only when their version and raw dataset
SHA-256 match. Cache reuse does not reuse a checkpoint, score, calibration,
random sample, or result. All compatibility replacements and their numerical
equivalence tests remain those in `BASELINE_OFA_REPROTOCOL.md`.

## 5. Gates and completion

Before formal execution:

- the supplementary manifest must contain exactly 12 runs and 60 evaluations;
- all eight existing compatibility/label/statistics tests must pass;
- new supplement scope and aggregation tests must pass;
- one-epoch B and C smokes must pass for both added methods;
- every smoke must pass score freeze, label audit, and checkpoint reload.

Completion requires:

- 12/12 supplementary training runs;
- 60/60 final evaluations;
- three seeds for each method/setting;
- one source-only AnomalyGFM calibration lock per setting;
- zero early target-label access;
- independent score/hash/checkpoint/metric validation;
- B/C per-dataset and macro tables, timing, and all negative/high-variance
  results retained without selective tuning.
