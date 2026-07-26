# RECAP Phase 2 OFA Baseline Reproduction Lock

Status: **locked before formal baseline results**

Date: 2026-07-26

This document governs the Phase 2 reproduction of the supervised generalist
baselines required by RECAP-OFA Settings A, B, and C. It supplements
`RECAP_EXPERIMENT_PROTOCOL.md`; the source/target splits and aggregation rules
in that document remain authoritative.

## 1. Confirmatory scope

The primary, protocol-comparable track contains:

| Setting | Sources | Targets | Required methods |
|---|---|---|---|
| A | PubMed, Flickr, Questions, YelpChi | Cora, CiteSeer, ACM, BlogCatalog, Facebook, Weibo, Reddit, Amazon | ARC, UNPrompt, AnomalyGFM-ZS, IA-GGAD |
| B | PubMed, Cora, Questions, YelpChi | Flickr, BlogCatalog, Facebook, Weibo, Reddit | ARC, IA-GGAD |
| C | PubMed, Cora, CiteSeer, ACM | BlogCatalog, Flickr, Reddit, Amazon, Questions | ARC, IA-GGAD |

Every method uses one shared checkpoint per setting and seed. Seeds are
`0`, `1`, and `2`. The confirmatory scope therefore contains 24 training runs
and 156 final method/target/seed evaluations.

An additional Setting A released-code-fidelity check may be run when an
official release conflicts with the label-safe protocol below. Such checks are
diagnostic only and must not replace the protocol-comparable table.

## 2. Common label and evaluation boundary

1. Source anomaly labels may be used because all four baselines are supervised
   generalist methods.
2. Target labels must not enter feature alignment, propagation, optimization,
   checkpoint selection, hyperparameter selection, score calibration, or
   failure recovery.
3. Target labels may be loaded only after the final score vector and immutable
   evaluation mask have been saved.
4. No target fine-tuning or target-label early stopping is allowed.
5. AUROC and AUPRC are computed from the saved final scores. Standard
   deviations use NumPy's population convention (`ddof=0`).
6. Dataset macros first average over target datasets within each seed and then
   report mean and population standard deviation over the three seeds.
7. Setting C domain macros follow the locked domain weighting in
   `RECAP_EXPERIMENT_PROTOCOL.md`.
8. Each run saves the checkpoint, score vectors, query masks, context indices,
   resolved configuration, environment, timing, memory, dataset hashes, and a
   label-access audit.

## 3. Method-native settings

### 3.1 ARC

- Upstream source: official ARC repository at the pinned revision recorded in
  `baselines/upstream_manifest.json`.
- Feature alignment: smoothness-ordered PCA to 64 dimensions, with the official
  Gaussian random projection path when the raw dimension is below 64.
- Encoder/scorer: official 2-hop residual encoder and cross-attentive context
  scorer.
- Optimization: 40 epochs, Adam, learning rate `1e-5`, weight decay `5e-5`,
  hidden size 1024, four linear layers, ELU, and 10 source prompts.
- Target information: exactly 10 target nodes known to be normal, sampled after
  seeding. These labeled normal context nodes are used for scoring and excluded
  from the query metric. There is no target optimization.

### 3.2 IA-GGAD

- Upstream source: official IA-GGAD repository at the pinned revision.
- Feature alignment and invariant branch: official 64-dimensional
  smoothness-ordered PCA, residual encoder, invariant codebook, and 40-epoch
  optimization.
- Affinity branch: preserve the official edge-wise cosine-affinity objective and
  score, but compute it directly on sparse edges rather than materializing an
  `N x N` dense matrix.
- Internal references: sample 10 target nodes uniformly without reading their
  labels, as in the released code. These nodes are an internal inference
  sample, not externally supplied labeled context, and are excluded from the
  immutable query mask exactly as in the release.
- Fusion weight: the released repository contains target-name-specific weights,
  which conflict with the paper's statement that hyperparameters remain fixed
  across target graphs. The primary track therefore selects one fusion weight
  on Setting/seed-0 **source scores only** from the locked grid
  `{0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1}` by source
  dataset-macro AUROC (AUPRC then smaller weight as deterministic tie-breaks).
  That single weight is frozen for every target and all three seeds of the
  setting. Target-specific released weights may be evaluated only in the
  diagnostic Setting A fidelity check.

### 3.3 UNPrompt

- Upstream source: official UNPrompt repository at the pinned revision.
- The official single-source training loop already accepts a list of source
  graphs. For protocol equivalence, optimizer steps cycle through the four
  locked source graphs in listed order within each epoch.
- Feature alignment: official rank-8 SVD projection followed by non-affine
  batch normalization.
- Encoder pretraining: official GRACE objective for 200 epochs, edge-drop 0.2,
  feature-drop 0.3, learning rate `1e-3`, and weight decay `1e-5`.
- Prompt training: 10 unified prompts, 128-dimensional hidden representation,
  900 epochs, Adam learning rate `1e-3`, and no weight decay.
- Source labels enter only the official completion loss. No target node,
  target prompt, target label, target tuning, or target calibration is used.
- The exact all-negative contrastive loss may be evaluated in deterministic
  row blocks to control memory. Blocking must preserve the full denominator;
  negative subsampling is not allowed in the formal run.

### 3.4 AnomalyGFM-ZS

- Upstream source: official AnomalyGFM zero-shot release at the pinned revision.
- For protocol equivalence, optimizer steps cycle through the four source
  graphs in listed order within each epoch.
- Feature alignment: rank-8 SVD followed by the official row normalization.
- Model: two-layer residual GCN, 400-dimensional hidden representation,
  learnable normal/anomalous prototypes, BCE plus prototype alignment.
- Optimization: 301 epochs, Adam learning rate `1e-4`, no weight decay. The
  official deterministic 30% source-node training split is used independently
  for every source graph.
- Target inference is zero-shot and uses the entire target query population.
- Prototype score weight: because the paper and released code disagree on
  target-dependent/default values, the primary track selects one value on
  Setting-A/seed-0 source scores only from `{0, .5, 1, 2, 4, 6}` by source
  dataset-macro AUROC (AUPRC then smaller weight as tie-breaks), and freezes it
  for every Setting A target and seed.

## 4. Compatibility adaptations

The server GPU is an NVIDIA RTX PRO 6000 Blackwell (compute capability 12.0).
The official CUDA 11-era environments cannot execute reliably on it. Formal
runs use the existing verified environment:

- Python 3.12.3
- PyTorch 2.11.0+cu128
- PyG 2.7.0
- NumPy 2.1.3
- scikit-learn 1.8.0

Compatibility changes are limited to:

1. current public API equivalents;
2. sparse matrix multiplication in place of algebraically identical dense/DGL
   propagation;
3. edge-wise cosine aggregation in place of dense masked cosine matrices;
4. deterministic exact row blocking for full contrastive denominators;
5. direct top-8 sparse ARPACK SVD in place of computing and discarding the
   full dense singular basis for AnomalyGFM;
6. resumable artifact, audit, and logging infrastructure.

Before formal runs, small-graph tests must show numerical equivalence for every
replaced dense operation within `atol=1e-5, rtol=1e-5`. A compatibility change
that alters the objective, samples, label rights, or evaluation population
requires a new protocol commit.

## 5. Gates and acceptance

Formal execution starts only after all of the following pass:

- upstream archive SHA-256 and dataset SHA-256 verification;
- environment import/GPU probe;
- preprocessing shape/finite-value checks for all A/B/C datasets;
- dense-versus-sparse equivalence tests;
- label-access guard tests;
- one seed-0 smoke run per method;
- checkpoint reload and score-reproduction audit.

Completion requires:

- 24/24 primary training runs;
- 156/156 final evaluations;
- three successful seeds per required method/setting;
- zero target-label access before score freeze, except ARC's explicitly allowed
  10 labeled-normal context samples;
- complete raw artifacts, method/dataset tables, macros, timing, failure log,
  upstream provenance, and a report of every compatibility or protocol
  deviation.
