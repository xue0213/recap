# Source-Only OFA Adaptations of GUIDE and DiffGAD

Status: **locked before formal execution**

Date: 2026-08-04

## 1. Purpose and method names

This experiment tests whether the target-specific unsupervised detectors GUIDE
and DiffGAD transfer when target-side optimization is prohibited. The variants
are named **GUIDE-OFA-adapted** and **DiffGAD-OFA-adapted** because neither
released repository defines a heterogeneous-feature, multi-source OFA path.
They must not be described as official OFA implementations.

## 2. Splits, seeds, and target rights

The variants use the exact RECAP Settings A/B/C and seeds `0/1/2`:

| Setting | Source graphs | Target graphs |
|---|---|---|
| A | PubMed, Flickr, Questions, YelpChi | Cora, CiteSeer, ACM, BlogCatalog, Facebook, Weibo, Reddit, Amazon |
| B | PubMed, Cora, Questions, YelpChi | Flickr, BlogCatalog, Facebook, Weibo, Reddit |
| C | PubMed, Cora, CiteSeer, ACM | BlogCatalog, Flickr, Reddit, Amazon, Questions |

For every method-setting-seed cell, one model is trained only on the four
source graphs and frozen. Target attributes and structure are visible only for
deterministic preprocessing and forward inference. Target labels, target
optimization, target checkpoint selection, target early stopping, reference
nodes, prompts, and target-dependent hyperparameter selection are forbidden.
All target score vectors in a run are saved and hashed before any target label
is loaded.

The formal scope is 2 methods x 3 settings x 3 seeds = **18 training runs** and
54 + 54 = **108 target evaluations**.

## 3. Common feature compatibility layer

The released OFO checkpoints cannot be transferred directly because their
attribute encoders and decoders are tied to graph-specific raw dimensions
(25--12,047 in this benchmark). Both adapted variants therefore use the same
label-free 32-dimensional alignment already frozen for RECAP:

1. coordinate-wise median/IQR normalization;
2. per-graph PCA to 32 dimensions;
3. coordinate-wise post-projection z-score normalization.

Only the numeric `feat` and `alignment_version` members of each frozen cache
are loaded. Embedded MAT payloads or labels are never deserialized. The cache
version must equal `robust_pca_post_zscore_v1`, and every cache is recorded by
SHA-256. The variants share this compatibility layer with RECAP, but retain
their own model-specific transformations below.

All graphs use the locked binary undirected adjacency union, with raw
self-loops removed before method-specific self-loop handling. Source graphs
are equally weighted: every epoch accumulates one mean node loss per source,
divides each by four, and performs one optimizer step.

## 4. GUIDE-OFA-adapted

GUIDE retains its released architecture, exact six structural inputs
`[degree, M31, M32, M41, M42, M43]`, and settings:

- embedding 32 and hidden dimensions 128/64;
- dropout 0.3 and attention negative slope 0.3;
- attribute/structure score weight 0.3/0.7;
- Adam learning rate 0.005 and weight decay 5e-4;
- 200 source-only epochs.

Aligned attributes and exact ORCA motif columns receive per-graph column-wise
min-max normalization. Target motifs and normalization statistics are
permitted deterministic target preprocessing; no parameter is updated.

## 5. DiffGAD-OFA-adapted

DiffGAD retains the locked leakage-corrected OFO mechanism and settings:

- shared four-layer graph autoencoder, hidden dimension 32;
- autoencoder attribute weight 0.8, 300 source-only epochs, Adam 0.01 with
  weight decay 0.01 and StepLR(100, 0.5);
- shared unconditional and prototype-conditioned EDM-style diffusion models,
  dimension 64, Adam 0.004, at most 800 epochs, training-loss patience 100,
  and StepLR(100, 0.5);
- prototype coefficient 0.1 and classifier-free weight 1.0;
- 500 forward levels, 50 reverse steps, and the preregistered label-free mean
  over levels `[49,99,149,199,249,299,349,399,449,499]`.

The autoencoder is trained with equal source-graph weighting. The shared
unconditional diffusion model is likewise optimized using a macro source loss.
It maintains one evolving reconstruction prototype per source; the transferable
prototype is the arithmetic mean of the four source prototypes at the best
macro training-loss epoch. The conditional model is trained on all sources
using this fixed source-only prototype. No target embedding participates in
training or prototype construction.

## 6. Reporting and acceptance

For every target, report full-graph AUROC and average precision. Compute the
dataset macro within each seed, then report mean and population standard
deviation over seeds. Compare against the existing RECAP-OFA values from the
same frozen Settings A/B/C artifacts.

Every run must save the resolved configuration, source/target lists, data and
feature-cache hashes, loss traces, checkpoint, frozen target scores, label
audit, timing, resource use, and checkpoint-reload score agreement.

Formal execution requires:

- a manifest of exactly 18 unique runs;
- feature-cache and graph-shape validation on all twelve datasets;
- a source-only smoke run for each adapted method;
- finite scores on every target;
- no target label event before all run targets are frozen;
- independent metric recomputation difference at most `1e-12`;
- checkpoint reload difference at most
  `1e-5 + 5e-6 * max(abs(score))` for every target.
