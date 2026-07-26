# RECAP Large-Target Full-Graph Inference Scalability Protocol

Status: **locked before numerical gates and formal inference**

Date: 2026-07-26

This protocol tests only **target-side inference scalability**. It does not
claim that RECAP can be trained end to end on million-node targets. The
accepted RECAP-OFA Setting-A checkpoints remain immutable and no large target
is used for training, checkpoint selection, or parameter tuning.

## 1. Confirmatory question and scope

Can the already trained RECAP-OFA model produce target-label-free, full-graph
anomaly scores on T-Finance, T-Social, and DGraph-Fin with bounded memory,
finite runtime, and auditable accuracy?

The primary Cartesian product is:

- checkpoints: accepted Setting-A seeds `0`, `1`, and `2`;
- targets: T-Finance, T-Social, and DGraph-Fin;
- **0 new training runs and 9 primary full-graph evaluations**.

T-Finance uses exact cosine KNN in the primary table because its 39,357-node
candidate search is feasible. T-Social and DGraph-Fin use the locked
FAISS-IVFPQ approximate KNN route. T-Finance additionally receives three
auxiliary ANN score vectors for a paired approximation-fidelity analysis;
these auxiliary vectors never replace its exact primary results.

## 2. Immutable checkpoints

| Seed | Accepted checkpoint | SHA-256 |
|---:|---|---|
| 0 | `rebuttal/artifacts/phase1/runs/ofa__a__seed0/checkpoints/final.pt` | `ca111ebd17e53d63c4b722c4aa47caff069f81a9525b969070f4d5bf2ee3fe44` |
| 1 | `rebuttal/artifacts/phase1/runs/ofa__a__seed1/checkpoints/final.pt` | `702d4b65f5b7b85fea0286d5980954ea8fdb34978cf24c1ac212405da42d6c6d` |
| 2 | `rebuttal/artifacts/phase1/runs/ofa__a__seed2/checkpoints/final.pt` | `def7d06e558a0897c105b021924b2a7fbd1e1e52b9b547a15c01b68218311cdd` |

Every checkpoint must declare the accepted Setting-A sources
`PubMed/Flickr/Questions/YelpChi`, 100 epochs, the paper-locked model
configuration, and the matching seed. State dictionaries are loaded strictly.

## 3. Dataset interpretation

The canonical bundles under `dataset/{tfinance,tsocial,dgraphfin}` contain
memory-mappable features and labels plus CSR adjacency. Their raw-source
provenance, graph statistics, and component SHA-256 values are recorded during
preflight.

- T-Finance and T-Social retain the released BWGNN topology. Their CSR files
  contain both directions; no second symmetrization is performed.
- DGraph-Fin is evaluated as a static symmetrized graph. Edge timestamps and
  types are outside this claim.
- DGraph-Fin labels 0/1 form the evaluation population. Labels 2/3 remain
  message-passing context and are excluded through the frozen evaluation mask.
- All datasets use the already accepted 32-dimensional
  `robust_sampled_pca_post_zscore_v1` cache and four-hop propagation.

## 4. KNN and scoring routes

The standard RECAP model and checkpoints are not modified. A separate
inference adapter computes the paper-equivalent score from the model's
residual embeddings.

### 4.1 Exact primary route

T-Finance uses the paper implementation's blockwise exact cosine top-64
candidate search. The score is evaluated with a memory-bounded formulation
that must match the native exact scorer on a small graph before formal use.

### 4.2 ANN primary route

T-Social and DGraph-Fin use one deterministic target-only FAISS index:

- backend: `IndexIVFPQ`, inner-product search after L2 normalization;
- `nlist=4096`, `nprobe=16`, `pq_m=16`, 8-bit PQ;
- deterministic evenly spaced training set, at most 262,144 nodes;
- query batch 4,096 and add batch 262,144;
- retrieve at most 256 candidates and rerank them by exact cosine similarity;
- retain top 64 non-self candidates;
- ANN seed 0, shared by all three checkpoint seeds.

Parameters are frozen and are not changed in response to anomaly labels,
AUROC, AUPRC, or individual dataset performance.

### 4.3 Approximation-fidelity reporting

Approximation quality is descriptive and cannot select formal parameters:

1. T-Finance exact versus ANN top-64 neighbor recall;
2. paired exact/ANN score Spearman and AUROC/AUPRC difference for all three
   checkpoints;
3. top-64 recall on 512 deterministic, label-blind query nodes for T-Social
   and DGraph-Fin, using blocked exact cosine search only for those queries.

No fidelity result may trigger target-label tuning or replacement of a weak
formal result.

## 5. Label isolation and evaluation

The inference graph contains propagated features and provenance only. It never
contains anomaly labels or the DGraph-Fin evaluation mask.

For each checkpoint-target pair:

1. load the checkpoint strictly;
2. produce the complete full-node score vector;
3. verify finiteness and node order;
4. atomically save and SHA-256 hash scores and the declared query mask;
5. append a `scores_frozen` audit event;
6. only then load labels and compute AUROC/AUPRC.

T-Finance and T-Social query all nodes. DGraph-Fin scores every node but
computes metrics only on its official foreground mask. Metrics use unrounded
scores and population standard deviation (`ddof=0`) over checkpoint seeds.

## 6. Scalability measurements

For each target, record the shared cold-target phases:

- canonical bundle load and adjacency normalization;
- four-hop propagation;
- initial residual construction;
- exact/ANN candidate construction and cache serialization;
- peak GPU allocated/reserved memory and peak process RSS.

For each checkpoint, record:

- checkpoint load;
- residual model forward;
- score construction;
- score serialization and hash;
- label-unlock metric computation;
- total warm-checkpoint inference;
- peak GPU allocated/reserved memory and process RSS.

The report separates shared cold-target setup from per-checkpoint inference.
It reports node/edge throughput, cache size, full-graph completion, and
estimated cold end-to-end latency (`setup + checkpoint inference`).

## 7. Acceptance gates

Before formal inference:

1. all three canonical bundles match expected shape, binary-evaluation
   semantics, finite features, and immutable component hashes;
2. all three checkpoint hashes and embedded configurations match this
   protocol;
3. the external exact scorer matches the native RECAP scorer within
   `atol=1e-5, rtol=1e-5`;
4. ANN candidates cover every node, contain no invalid/self IDs, and are
   deterministic for a fixed seed;
5. the original exact-KNN tests for the twelve standard datasets still pass;
6. a T-Finance seed-0 end-to-end smoke freezes scores before labels and
   independently recomputes identical metrics.

Formal acceptance requires 9/9 primary score vectors, finite full-node scores,
complete label-audit events, exact independent metric recomputation, matching
data/checkpoint/score/mask hashes, and no missing or selectively rerun cell.

## 8. Artifacts

Formal artifacts are written to
`rebuttal/artifacts/large_target_inference/` and include:

- preflight data/checkpoint/environment manifests;
- shared target setup records and immutable KNN caches;
- one result directory per checkpoint-target pair;
- frozen scores, hashes, resource/timing records, and label audit;
- approximation-fidelity records;
- independent global audit and regenerated Markdown/CSV/JSON tables.

Large arrays and caches remain on the experiment server. The protocol,
adapters, compact analysis records, and final report are tracked in Git.
