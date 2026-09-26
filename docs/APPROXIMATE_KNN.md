# Approximate KNN for large-graph RECAP inference

## Scope

Only `tsocial` and `dgraphfin` use the approximate route. Every other dataset,
including `tfinance`, retains `_select_knn_candidates`, its exact blockwise
cosine search, and the original `recap_knn_candidates_v1` cache key.

The large route is selected from the model's dataset cache key. It uses a
separate `recap_ann_knn_candidates_v1` cache namespace containing all
accuracy-affecting ANN parameters.

## Algorithm

1. L2-normalize the initial residual embeddings.
2. Train a deterministic FAISS IVF-PQ index on evenly spaced rows.
3. Add all nodes in batches.
4. Probe `nprobe` inverted lists and retrieve `k × rerank_factor` candidates.
5. Recompute exact cosine similarities only within that candidate pool.
6. Remove self ids, retain the exact top `k`, and save the fixed candidate
   matrix through RECAP's existing disk-cache mechanism.

This changes candidate construction from all-pairs quadratic work to an IVF
index plus bounded candidate reranking. Differentiable edge weighting remains
unchanged after candidate selection.

## Pinned dependency

`requirements-large.txt` pins `faiss-cpu==1.14.3`. It is optional and imported
lazily, so ordinary datasets do not require FAISS.

## Default parameters

- `nlist=4096`
- `nprobe=16`
- `pq_m=16`
- `train_size=262144`
- `rerank_factor=32`
- `max_rerank_candidates=256`
- `query_batch_size=4096`
- `seed=0`

All are exposed on `ModelConfig`. Changing an accuracy-affecting value creates
a different cache key.

## Prebuild a cache

The `knn_k` value must match the checkpoint/model configuration.

```bash
python tools/build_large_knn_cache.py \
  --dataset tsocial --dims 32 --num-hops 4 --knn-k 64

python tools/build_large_knn_cache.py \
  --dataset dgraphfin --dims 32 --num-hops 4 --knn-k 64
```

The first run builds and saves candidates. Later inference loads the same
`N × k` tensor and does not retrain or search the FAISS index.

The repository's current `params/recap_auprc_best.json` uses `knn_k=64`, so
commands targeting that configuration must pass `--knn-k 64`. The CLI requires
this value explicitly to prevent a cache/config mismatch.

## Acceptance performed

On 12,000 deterministic synthetic 32-D vectors, IVF-PQ plus exact candidate
reranking achieved Recall@5 of 0.89 with no self-neighbors. A routing test
confirmed that `cora` produced exactly the same indices as the original exact
function, while only `tsocial` and `dgraphfin` selected ANN.

Additional acceptance used the repository's current `knn_k=64`: a 12,000-node
model ran from `recap.forward()` through all score components successfully.
With a fixed candidate matrix, the large blockwise scoring path matched the
original implementation for total, adhesion, scale, raw scores, and neighbor
context within `2e-5` numerical tolerance.

ANN solves candidate-search scaling, but it does not by itself solve memory
used by very wide learned embeddings. For the same two large datasets, scoring
therefore also uses a dedicated memory-bounded path: candidate weights and
symmetrized community aggregation are processed in blocks, and centroid
distances use the squared-distance identity rather than materializing an
`N × clusters × embedding_dim` tensor. Other datasets retain the original
scoring functions.
