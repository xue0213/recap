# RECAP Three-Baseline Extension — Locked Reproduction Protocol

Status: **locked before formal results**

Date: 2026-07-26

This document governs the addition of DiffGAD, GUIDE, and OWLEYE to the
completed RECAP rebuttal experiment record. It supplements
`RECAP_EXPERIMENT_PROTOCOL.md`, `OFO_12_BASELINE_PROTOCOL.md`, and
`BASELINE_OFA_REPROTOCOL.md`. Existing runs and frozen score vectors are
immutable.

## 1. Confirmatory scope and method classification

### 1.1 One-for-one unsupervised baselines

DiffGAD and GUIDE are trained independently and transductively on each of the
twelve RECAP graphs:

- Citation: PubMed, Cora, CiteSeer, ACM;
- Social: Flickr, BlogCatalog, Facebook, Weibo, Reddit;
- Q&A: Questions;
- E-commerce: YelpChi, Amazon.

Each method-dataset cell uses seeds `0`, `1`, and `2`:

- 2 methods × 12 datasets × 3 seeds = **72 training runs**;
- one full-node frozen score vector per run = **72 final evaluations**.

Attributes and graph structure are model-visible. Anomaly labels are not read
until the final full-node score vector and its SHA-256 are frozen. There is no
label-dependent early stopping, checkpoint selection, hyperparameter tuning,
thresholding, or retry.

### 1.2 Source-supervised, target-zero-shot baseline

OWLEYE is not classified as fully unsupervised. Its released training loss and
pattern extraction explicitly use normal/anomaly labels on the source graphs.
It is therefore recorded as **source-label supervised and target-label-free
zero-shot**.

OWLEYE is run on the existing RECAP OFA splits:

| Setting | Source graphs | Target graphs |
|---|---|---|
| A | PubMed, Flickr, Questions, YelpChi | Cora, CiteSeer, ACM, BlogCatalog, Facebook, Weibo, Reddit, Amazon |
| B | PubMed, Cora, Questions, YelpChi | Flickr, BlogCatalog, Facebook, Weibo, Reddit |
| C | PubMed, Cora, CiteSeer, ACM | BlogCatalog, Flickr, Reddit, Amazon, Questions |

Each setting uses seeds `0`, `1`, and `2`:

- 3 settings × 3 seeds = **9 training runs**;
- 54 target graph score vectors and **54 final evaluations**.

Source labels are model-visible. Target labels are inaccessible until each
target graph's full-node score vector and SHA-256 are frozen. The released
zero-shot path uniformly samples ten unlabeled target nodes as target-side
patterns; these nodes remain in the evaluation population, matching the
released demo. The optional 10-shot code is outside this extension.

The complete extension is therefore **81 training runs and 126 final
evaluations**.

## 2. Data identity and common reporting

1. All raw MAT files are the same hash-locked twelve files used by the
   completed RECAP experiments under `/root/autodl-tmp/recap/dataset`.
2. Adjacency is converted to a binary undirected union with duplicate edges and
   self-loops removed before method-specific self-loop handling.
3. Dataset names, node order, anomaly labels, and evaluation populations are
   unchanged.
4. AUROC and average precision (reported as AUPRC) are recomputed independently
   from frozen score vectors. Tables report the mean and population standard
   deviation over seeds 0/1/2.
5. No result from a different split, graph version, seed, or upstream demo may
   replace a missing run.

## 3. DiffGAD

### 3.1 Released mechanism retained

The compatibility implementation retains the released four-layer graph
autoencoder, unconditional EDM-style diffusion model, prototype-conditioned
diffusion model, classifier-free reconstruction, and node-wise joint
attribute/structure reconstruction score.

The following single result-independent configuration is frozen for all twelve
previously unsupported target graphs:

- hidden dimension 32 and diffusion time-embedding dimension 64;
- autoencoder dropout `0.3`, learning rate `0.01`, weight decay `0.01`;
- autoencoder attribute weight `0.8`, 300 epochs, StepLR(100, 0.5);
- diffusion learning rate `0.004`, zero weight decay, at most 800 epochs,
  patience 100 on training loss, StepLR(100, 0.5);
- prototype coefficient `0.1`, classifier-free weight `1.0`;
- 500 forward diffusion levels and 50 reverse sampling steps.

The autoencoder final epoch is used. The two diffusion checkpoints are selected
only by their own training loss.

### 3.2 Mandatory leakage correction

The released script evaluates anomaly labels after every autoencoder trial and
selects the trial with maximum AUROC. It also evaluates all 500 diffusion
levels and reports a label-selected maximum. Both choices violate the
unsupervised protocol and are disabled.

Formal inference instead uses the preregistered fixed grid
`[49, 99, 149, 199, 249, 299, 349, 399, 449, 499]` and averages the ten
node-score vectors. The inference random stream is derived only from the run
seed. Labels are loaded after this mean vector is frozen.

### 3.3 Exact large-graph structure loss

The released implementation materializes both the dense adjacency and
`ZZᵀ`. Formal runs use the exact identity

`sum_j (A_ij - z_i^T z_j)^2 =
 degree_i - 2 sum_(i,j in E) z_i^T z_j + z_i^T(Z^T Z)z_i`.

The structure decoder itself is unchanged; only the mathematically identical
row-error calculation is replaced. Dense-reference forward and gradient gates
must pass before formal execution.

## 4. GUIDE

### 4.1 Released mechanism and settings

GUIDE retains the released attribute GCN autoencoder, graph-node-attention
structural autoencoder, six structural inputs
`[degree, M31, M32, M41, M42, M43]`, and joint node score.

The released generic settings are frozen for every graph:

- hidden embedding 32, intermediate dimensions 128 and 64;
- dropout `0.3`, attention negative slope `0.3`;
- attribute/structure score weight `beta=0.3`;
- Adam learning rate `0.005`, weight decay `5e-4`;
- exactly 200 epochs with no label-based checkpoint selection;
- independent seeds 0/1/2.

Attributes and motif columns receive the released column-wise min-max
normalization. The adjacency receives released symmetric normalization after
adding self-loops.

### 4.2 Exact scalable motif preprocessing

The released Python/NetworkX nested enumeration is replaced by ORCA's exact
induced graphlet-orbit counter up to order four. The six GUIDE values are
obtained from the standard node orbits:

- degree: orbit 0;
- M31 triangle: orbit 3;
- M32 three-node path: orbits 1 + 2;
- M41 four-clique: orbit 14;
- M42 diamond: orbits 12 + 13;
- M43 chordless four-cycle: orbit 8.

Motifs are cached by raw adjacency SHA-256 and ORCA revision. Exact equality
against the released `motiffeature` routine is mandatory on a graph suite and
on Cora before formal runs.

The released training script reads labels each epoch only to print metrics.
Those reads are removed; the fixed epoch-200 checkpoint is evaluated once
after score freeze.

## 5. OWLEYE

### 5.1 Released mechanism and settings

OWLEYE retains residual representations over four propagated feature hops,
four learned structural hops, normal source pattern extraction, source-label
contrastive/triplet training, domain similarity, cross-graph attention, and the
released ten-unlabeled-node target pattern augmentation.

The released best configuration is frozen:

- official 64-dimensional preprocessing caches;
- 100 epochs, four propagation hops, three MLP layers;
- hidden dimension 512, ELU, dropout `0.2`;
- Adam learning rate `3.12964801067075e-5`;
- weight decay `2.848035868435802e-4`;
- up to 2,000 source support nodes;
- attention temperature `0.001`, structural score weight `0.01`;
- `tau=1`, mask-ratio metadata `0.7`.

The raw MAT payloads bundled by OWLEYE must have the same SHA-256 as the RECAP
files before its official `*_64.npz` features may be used. Only the numeric
`feat` member is loaded from those caches; embedded labels are never exposed to
the adapter.

### 5.2 Safe large-graph execution

Under the released `tau=1` configuration, the pair-distance terms in
`normalization()` cancel algebraically and its final multiplier is exactly one;
the effective operation is division by mean node L2 norm. Formal code performs
that effective operation without constructing `N×N` distances. A small-graph
gate compares the multiplier with the released function.

Target queries are processed in fixed node-order chunks. Chunking changes
neither attention supports nor softmax axes and must match the released
full-query result within floating-point tolerance. Final source normal-pattern
indices and the ten target pattern indices are saved and reused for checkpoint
reload verification.

## 6. Upstream provenance and environment

Pinned revisions and archive SHA-256 values are stored in
`new_baselines/upstream_manifest.json`. DiffGAD, GUIDE, and OWLEYE currently
publish no repository license file; their code is therefore not redistributed.
The tracked adapter is an independently written current-PyTorch compatibility
implementation. ORCA is invoked under its bundled MIT license.

Formal runs use the existing isolated environment at
`/root/autodl-tmp/envs/recap-ofo-baselines`:

- Python 3.12.3;
- PyTorch 2.11.0+cu128;
- PyG 2.7.0;
- NumPy 2.1.3;
- SciPy 1.17.1;
- scikit-learn 1.8.0;
- PyGOD 1.1.0.

The newer environment is an API compatibility environment. Pinned upstream
revisions, effective model equations, and every intentional scalability or
leakage correction are the fidelity boundary.

## 7. Artifacts and acceptance gates

Every run must save:

- resolved configuration, protocol/upstream/data hashes, and environment;
- checkpoint and optimizer-independent reload material;
- frozen score vector, evaluation mask, and SHA-256;
- source/target sampled indices where applicable;
- loss trace;
- preprocessing, training, inference, and total wall-clock times;
- peak GPU allocation/reservation and process RSS;
- label-access audit;
- checkpoint-reload maximum score difference.

Formal execution may start only after:

- this protocol and the 81-run manifest are committed;
- upstream archive, OWLEYE raw-data, and twelve RECAP file hashes pass;
- environment import and GPU probes pass;
- exact DiffGAD dense-loss forward/gradient gates pass;
- exact GUIDE motif equality and sparse-layer gates pass;
- OWLEYE normalization and query-chunk equivalence gates pass;
- Cora and Questions smoke runs pass for DiffGAD and GUIDE;
- one Setting-A OWLEYE smoke passes;
- all label-isolation and score-freeze tests pass.

Completion requires:

- 81/81 successful training runs and 126/126 evaluations;
- three seeds in every method-dataset or method-setting cell;
- no unauthorized target/unsupervised label access before score freeze;
- independent metric recomputation from frozen scores;
- checkpoint-reload agreement within
  `1e-5 + 5e-6 * max(abs(score))`;
- complete AUROC/AUPRC, timing, and resource tables;
- an explicit report of every departure from released scripts.
