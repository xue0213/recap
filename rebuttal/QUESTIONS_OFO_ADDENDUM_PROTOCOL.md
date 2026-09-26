# RECAP OFO–Questions Addendum — Locked Protocol

Status: **CONFIRMATORY ADDENDUM / LOCKED BEFORE RUNS**

Requested and locked on: 2026-07-26 (Asia/Shanghai)

This addendum restores Questions to the one-for-one evaluation after the
original user-approved 11-dataset Phase 1 excluded it for runtime caution. It
does not modify or replace the completed 42-run Phase 1 results.

## Scope

- Dataset: Questions only.
- Paradigm: `RECAP-OFO`, unsupervised transductive one-for-one training.
- Training seeds: 0, 1, and 2.
- Formal training runs: 3.
- Final evaluations: 3.
- Cross-seed stability comparisons: `(0,1)`, `(0,2)`, and `(1,2)`.

## Locked configuration

Use the same clean paper baseline and exact configuration as Phase 1:

- scientific base commit:
  `c94c4d7985d2cb1438c430173ad868d68d0c1efe`;
- 100 epochs, final checkpoint, no early stopping;
- aligned dimension 32, 4 hops, 36 communities;
- exact KNN with `K=64`; no ANN;
- learning rate and weight decay `5e-5`;
- `tau_s=0.3`, `tau_c=0.3`, `tau_e=1`;
- `lambda_H=0.1`, `lambda_usage_entropy=0.1`, `lambda_bal=0.1`;
- `lambda_E=0`, `beta=0.02`, `gamma=0.01`;
- population standard deviation (`ddof=0`).

Questions must be trained independently from a fresh model for every seed. No
OFA checkpoint or model state may be reused. Versioned feature and exact-KNN
caches may be reused only when their full cache keys match; this does not
permit approximate neighbors.

## Label boundary

Questions anomaly labels are evaluation-only. The model-facing PyG `Data`
object must contain no `ano_labels`. Labels may be accessed only after final
anomaly scores are fixed and converted to CPU arrays.

## Outputs and acceptance

The addendum uses a separate artifact root:

`rebuttal/artifacts/questions_ofo_addendum`

Acceptance requires:

- 3/3 complete 100-epoch runs and 3/3 final inference records;
- epoch 25/50/75/100 resume checkpoints plus final checkpoint for every seed;
- fixed diagnostics at epochs 1/10/25/50/75/100;
- 3/3 checkpoint-reload equivalence audits;
- exact-KNN provenance and finite AUROC/AUPRC for every seed;
- final node-order-preserving community outputs for all three seeds;
- NMI, ARI, exact soft co-assignment similarity, score Spearman, and effective
  community count;
- a separate Questions report plus a combined 12-dataset OFO macro computed
  from the original 11-dataset raw records and this addendum.

The original 11-dataset report remains immutable. Any 12-dataset number must be
labelled as the post hoc user-requested combined OFO result.
