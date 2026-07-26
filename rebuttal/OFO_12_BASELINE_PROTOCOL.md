# RECAP Phase 2 — 12-Dataset OFO Baseline Reproduction Lock

Status: **locked before formal results**

Date: 2026-07-26

This document governs the user-requested reproduction of the eight one-for-one
(OFO) baselines on all twelve graphs used by RECAP-OFO. It supplements
`RECAP_EXPERIMENT_PROTOCOL.md` and does not modify any completed RECAP or OFA
baseline artifact.

## 1. Confirmatory scope

The methods are:

- supervised OFO: GCN, GAT, BWGNN, XGBGraph;
- unsupervised OFO: DOMINANT, AnomalyDAE, CoLA, ADA-GAD.

The datasets are:

- Citation: PubMed, Cora, CiteSeer, ACM;
- Social: Flickr, BlogCatalog, Facebook, Weibo, Reddit;
- Q&A: Questions;
- E-commerce: YelpChi, Amazon.

Every method is trained independently from scratch on every dataset with seeds
`0`, `1`, and `2`. The locked scope is therefore:

- 8 methods × 12 datasets × 3 seeds = **288 training runs**;
- one final score vector and one final evaluation per run = **288 evaluations**.

Questions is included. No result from a different graph, seed, target split, or
previous RECAP/OFA experiment may replace a missing run.

## 2. Data identity and preprocessing

1. Use the same twelve MAT graph files as the completed RECAP experiments under
   `/root/autodl-tmp/recap/dataset`.
2. Read adjacency, node attributes, and binary anomaly labels from those files.
   Do not substitute similarly named GADBench datasets because several have
   different node/edge counts.
3. Symmetrize the adjacency with a binary union, remove duplicate edges, and
   add self-loops only inside methods whose released convolution does so.
4. Row-normalize raw node attributes. Sparse input is retained until a model
   operation requires a dense tensor. There is no label-dependent PCA,
   feature selection, or normalization.
5. Cache only deterministic, hash-keyed preprocessing. A cache mismatch must
   invalidate the cache rather than reuse it.

The paper's main experimental text defines supervised OFO as label-supervised
target-graph training and unsupervised OFO as single-target-graph
self-supervised training. Appendix C.1's statement that OFO methods use a
pre-train-only transfer protocol conflicts with that definition, the OFO name,
and the user-requested train-plus-inference experiment. This lock follows the
main OFO definition.

## 3. Supervised OFO protocol

### 3.1 Splits and label rights

For each dataset and seed, create a deterministic stratified node split:

- 40% train;
- 20% validation;
- 40% test.

Each class must be represented in all three partitions. The seed controls both
the split and model randomness. Labels may be read for the train and validation
nodes during fitting and checkpoint selection. Test labels remain inaccessible
until the final test score vector, mask, and SHA-256 hashes are frozen.

The test mask is the evaluation population. Class weighting is the
normal-to-anomaly ratio in the training split. Checkpoint selection uses
validation AUPRC only; test AUROC/AUPRC must never select an epoch or
hyperparameter.

### 3.2 GCN, GAT, and BWGNN

These methods follow the default fully supervised GADBench training recipe:

- 200 maximum epochs and patience 50;
- Adam, learning rate `0.01`, zero weight decay;
- weighted two-class cross entropy;
- hidden dimension 32, two message-passing layers, no dropout;
- best validation-AUPRC checkpoint;
- GCN uses symmetric GCN normalization;
- GAT uses two attention heads and averages the final class-head logits;
- BWGNN uses the released beta-wavelet polynomial filters with order 2 and a
  two-layer classifier.

The DGL operations in the releases are expressed with current PyTorch/PyG sparse
operators. Dense-versus-sparse and polynomial reference tests are required
before formal execution.

### 3.3 XGBGraph

XGBGraph follows GADBench's parameter-free GIN feature construction:

- concatenate raw features with two successive mean-GIN aggregations;
- `init_eps=-1`, matching the released neighbor-only first aggregation;
- XGBoost 3.0.2, histogram tree method, 100 estimators, maximum depth 6,
  learning rate 0.3, one deterministic training thread, seed equal to the run
  seed;
- training-split class weights; no test-label tuning.

The model is saved in native XGBoost JSON form and must reproduce the frozen
test scores after reload.

## 4. Unsupervised OFO protocol

All four methods train transductively on the full graph using attributes and
structure only. Labels are inaccessible until the full-node score vector and
all hashes are frozen. The full node set is the evaluation population.

### 4.1 DOMINANT

- released two-layer shared GCN encoder, two-layer attribute decoder, and
  dot-product structure decoder;
- hidden dimension 64, dropout 0.3, 100 epochs, Adam learning rate `0.005`;
- score weight 0.8 for attribute reconstruction and 0.2 for structure
  reconstruction.

The dense `ZZᵀ-A` row error is evaluated exactly without materializing `N×N`:

`sum_j (z_i^T z_j - A_ij)^2 =
 z_i^T(Z^T Z)z_i - 2 sum_(i,j in E) z_i^T z_j + degree_i`.

This is an algebraic replacement, not negative sampling. A dense-reference
equivalence gate is mandatory.

### 4.2 AnomalyDAE

- released GAT structure encoder and dual node/attribute embeddings;
- hidden and embedding dimensions 64;
- 100 epochs, Adam learning rate `0.001`, no dropout;
- fixed cross-dataset parameters `alpha=0.7`, `eta=5`, `theta=40`.

The original dense structure decoder is quadratic. Formal runs retain every
positive edge and use deterministic uniform non-edge samples at a 1:1 ratio.
Inverse sampling-probability weighting estimates the original dense
reconstruction term. Training negatives are resampled from the run RNG each
epoch; the final node score averages four fixed negative-sampling rounds. This
is the only AnomalyDAE objective approximation and must be reported as such.

### 4.3 CoLA

- official contrastive node/context discriminator;
- hidden dimension 64, four-layer GCN encoder, 100 epochs;
- Adam learning rate `0.001`, no weight decay or dropout;
- one negative per positive;
- random-neighbor context sampling, which is the documented scalable PyGOD
  1.1.0 implementation of CoLA;
- final score averages 64 deterministic inference rounds.

### 4.4 ADA-GAD

ADA-GAD preserves the released two-stage design:

1. pretrain a GCN autoencoder on three progressively anomaly-denoised,
   label-free graph views;
2. initialize from the averaged pretrained encoder, freeze it, and retrain the
   reconstruction decoder on the original graph for detection.

The generic cross-dataset configuration is frozen to the official small-graph
defaults: hidden dimension 32, two GCN layers, Adam learning rate `0.001`,
weight decay `2e-4`, 20 pretraining epochs per view, and 20 detector epochs.
Denoised views remove 0%, 5%, and 10% of the highest label-free joint
degree/feature-deviation candidates from message passing. Attribute and
structure scores are combined with equal weight.

As for AnomalyDAE, the structure term uses all positive edges and deterministic
1:1 non-edge sampling with inverse-probability weighting, and the final score
averages four fixed rounds. This compatibility adapter is a faithful
large-graph implementation of the published two-stage mechanism, not a claim
that the legacy repository runs unmodified on all twelve unsupported datasets.

## 5. Versions and upstream provenance

The formal environment is isolated at
`/root/autodl-tmp/envs/recap-ofo-baselines`:

- Python 3.12.3;
- PyTorch 2.11.0+cu128;
- PyG 2.7.0;
- NumPy 2.1.3;
- SciPy 1.17.1;
- scikit-learn 1.8.0;
- PyGOD 1.1.0;
- XGBoost 3.0.2;
- pandas 2.3.1;
- PyYAML 6.0.2.

Pinned repository revisions and archive hashes are recorded in
`ofo_baselines/upstream_manifest.json`. Formal adapters may use only current API
substitutions and the scalability adaptations explicitly locked above.

## 6. Artifacts, timing, and acceptance gates

Every run must save:

- resolved configuration and upstream revision identifiers;
- raw data SHA-256;
- split masks where applicable;
- checkpoint/model;
- frozen score vector and evaluation mask;
- score and mask SHA-256;
- train, inference, and total wall-clock times;
- peak GPU allocated/reserved memory and peak process RSS;
- epoch/loss trace;
- label-access audit and environment metadata;
- checkpoint-reload score-difference audit.

Formal execution may start only after:

- protocol and 288-run manifest are committed;
- upstream archive and dataset hash verification;
- environment import/GPU probe;
- all twelve preprocessing shape/finite checks;
- split balance/coverage tests;
- dense-versus-sparse equivalence tests;
- one small-graph and one Questions smoke per method;
- label isolation and checkpoint reload tests.

During execution, audit after each completed method-seed block (12 runs), and
perform an outer-loop reflection after each method (36 runs). A failed run is
retried only after recording its failure and cause. Weak or high-variance
results are retained without target-test tuning.

Completion requires:

- 288/288 successful runs and evaluations;
- three seeds for every method-dataset cell;
- zero unauthorized test/unsupervised label access before score freeze;
- independent metric recomputation from frozen scores;
- per-dataset AUROC/AUPRC tables with population standard deviation;
- dataset and domain macro summaries;
- complete training/inference timing and resource tables;
- a final report of every implementation and protocol deviation.
