# RECAP Phase 1 — Locked Execution Protocol

Status: **CONFIRMATORY / LOCKED BEFORE FORMAL RUNS**

Locked on: 2026-07-26 (Asia/Shanghai)

Scientific baseline: manuscript settings plus Git commit
`c94c4d7985d2cb1438c430173ad868d68d0c1efe`.

The full reporting and evaluation requirements are defined in
`RECAP_EXPERIMENT_PROTOCOL.md`. This file records the user-approved execution
amendments and removes ambiguity before any formal result is generated.

## 1. Locked scope

### OFO

Train and infer independently on each of the following 11 datasets:

1. PubMed
2. Cora
3. CiteSeer
4. ACM
5. Flickr
6. BlogCatalog
7. Facebook
8. Weibo
9. Reddit
10. YelpChi
11. Amazon

Questions is explicitly excluded from OFO by user decision because of its
size. Seeds are 0, 1, and 2. This yields 33 OFO training runs and 33 final
evaluations.

### OFA

OFA Settings A, B, and C are unchanged from the full protocol, including all
source and target datasets. In particular, Questions remains present wherever
the original OFA protocol specifies it.

- Setting A sources: PubMed, Flickr, Questions, YelpChi.
- Setting B sources: PubMed, Cora, Questions, YelpChi.
- Setting C sources: PubMed, Cora, CiteSeer, ACM.

There are 9 OFA training runs and 54 final target evaluations.

### Total

- Formal training runs: 42.
- Final model-dataset-seed evaluations: 87.
- Dataset-setting stability scopes: 29.
- Pairwise seed stability comparisons: 87.

## 2. Locked model and training settings

The main result uses the paper settings:

- epochs: 100, with no target-label early stopping;
- dimensions: 32;
- propagation hops: 4;
- communities: 36;
- exact KNN candidates: 64;
- learning rate: 5e-5;
- weight decay: 5e-5;
- `tau_s`: 0.3;
- `tau_c`: 0.3;
- `lambda_H`: 0.1;
- `lambda_usage_entropy`: 0.1;
- `beta`: 0.02;
- `lambda_E` / `lambda_emb`: 0;
- seeds: 0, 1, 2.

Architecture fields not expanded in the manuscript are taken from the released
`params/recap_auprc_best.json` and recorded in every resolved run config.

Any later parameter change is exploratory and cannot silently replace the
confirmatory paper-setting results.

## 3. Label isolation

- Training code must not read target labels.
- Target labels must not affect optimization, checkpoint selection, early
  stopping, hyperparameter choice, caching, or failure handling.
- Fixed-epoch target metrics are computed post hoc from saved checkpoints or
  saved label-free scores.
- Target labels may be loaded only inside the metric-evaluation boundary.

## 4. Result-preserving efficiency rules

The following optimizations are permitted because they are deterministic or
algebraically exact:

- reuse versioned feature-alignment and exact-KNN candidate caches across seeds
  when their full cache keys match;
- avoid duplicate final inference when the exact final scores are already
  available, while retaining checkpoint-reload equivalence audits;
- save full node-level community outputs at the final epoch and compact
  aggregate diagnostics at intermediate epochs;
- compute soft co-assignment similarity without materializing an `N x N`
  matrix, using
  `inner(HH^T, GG^T) = ||H^T G||_F^2`.

Approximate KNN is not permitted for the 12 Phase 1 datasets. The ANN route for
T-Social and DGraph-Fin remains isolated from this protocol.

## 5. Data-statistics decision

Amazon is run using the actual available dataset with 10,224 nodes. The
manuscript value 10,244 is treated as a likely table typo and is documented,
not silently rewritten. Edge statistics record both adjacency nonzeros and
unique undirected-edge counts.

## 6. Aggregation and acceptance

- Standard deviation uses population convention (`ddof=0`).
- All tables must be reproducible from raw per-seed records.
- Formal completion requires 42/42 successful training runs, 87/87 final
  evaluations, 87/87 pairwise stability rows, no duplicate run IDs, no NaN or
  Inf metrics, complete configs/logs/timings, and a passing label-isolation
  audit.

## 7. Reflection checkpoints

An outer-loop review is required:

- after infrastructure and preflight;
- after the first formal seed-0 gate;
- after every 5–10 completed training runs;
- immediately after any failure, unexpected metric, NaN, cache mismatch, or
  reproducibility discrepancy;
- before final aggregation.

Every review updates `research-state.yaml`, `research-log.md`, and
`findings.md`.
