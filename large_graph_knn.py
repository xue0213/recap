"""FAISS-based approximate KNN used only by RECAP's large-graph route."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F


def _largest_divisor_at_most(value: int, upper_bound: int) -> int:
    for candidate in range(min(value, upper_bound), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _normalized_cpu_array(embeddings: torch.Tensor) -> np.ndarray:
    normalized = F.normalize(
        embeddings.detach().float(), p=2, dim=1
    ).cpu().contiguous()
    return normalized.numpy()


def select_faiss_ivfpq_candidates(
    embeddings: torch.Tensor,
    k: int,
    *,
    nlist: int = 4096,
    nprobe: int = 16,
    pq_m: int = 16,
    train_size: int = 262_144,
    query_batch_size: int = 4_096,
    add_batch_size: int = 262_144,
    rerank_factor: int = 32,
    max_rerank_candidates: int = 256,
    seed: int = 0,
) -> torch.Tensor:
    """Return approximate cosine-neighbor ids as an ``N x k`` CPU tensor.

    Vectors are L2-normalized before indexing, making inner product equivalent
    to cosine similarity. The index is trained on deterministic evenly spaced
    rows and queried in batches. Self ids are removed from every result row.
    """
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "Large-graph approximate KNN requires faiss-cpu. "
            "Install the pinned optional dependency before inference."
        ) from exc

    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding, got {embeddings.shape}")
    num_nodes, dims = embeddings.shape
    k = min(int(k), max(num_nodes - 1, 0))
    if k == 0:
        return torch.empty(num_nodes, 0, dtype=torch.long)

    started = time.time()
    vectors = _normalized_cpu_array(embeddings)
    actual_train_size = min(num_nodes, max(256, int(train_size)))
    # FAISS recommends enough training examples per IVF centroid.
    actual_nlist = min(int(nlist), max(1, actual_train_size // 64))
    actual_nprobe = min(max(1, int(nprobe)), actual_nlist)
    actual_pq_m = _largest_divisor_at_most(dims, max(1, int(pq_m)))

    faiss.omp_set_num_threads(max(1, min(32, torch.get_num_threads())))
    quantizer = faiss.IndexFlatIP(dims)
    index = faiss.IndexIVFPQ(
        quantizer,
        dims,
        actual_nlist,
        actual_pq_m,
        8,
        faiss.METRIC_INNER_PRODUCT,
    )
    index.nprobe = actual_nprobe
    index.cp.seed = int(seed)
    index.pq.cp.seed = int(seed)

    train_idx = np.linspace(
        0, num_nodes - 1, num=actual_train_size, dtype=np.int64
    )
    print(
        "ANN KNN: "
        f"N={num_nodes}, d={dims}, k={k}, nlist={actual_nlist}, "
        f"nprobe={actual_nprobe}, pq_m={actual_pq_m}, "
        f"train={actual_train_size}",
        flush=True,
    )
    index.train(np.ascontiguousarray(vectors[train_idx]))
    for start in range(0, num_nodes, max(1, int(add_batch_size))):
        stop = min(start + int(add_batch_size), num_nodes)
        index.add_with_ids(
            np.ascontiguousarray(vectors[start:stop]),
            np.arange(start, stop, dtype=np.int64),
        )

    # Retrieve a wider compressed candidate pool, then rerank it using exact
    # cosine scores from the original normalized vectors. This removes most PQ
    # distance error without returning to an all-pairs search.
    search_k = min(
        num_nodes,
        max(
            k + 16,
            min(
                k * max(1, int(rerank_factor)) + 1,
                max(k + 1, int(max_rerank_candidates)),
            ),
        ),
    )
    output = np.empty((num_nodes, k), dtype=np.int64)
    for start in range(0, num_nodes, max(1, int(query_batch_size))):
        stop = min(start + int(query_batch_size), num_nodes)
        _, found = index.search(
            np.ascontiguousarray(vectors[start:stop]), search_k
        )
        valid = found >= 0
        safe_found = np.where(valid, found, 0)
        candidate_vectors = vectors[safe_found]
        exact_scores = np.einsum(
            "bd,brd->br",
            vectors[start:stop],
            candidate_vectors,
            optimize=True,
        )
        row_ids = np.arange(start, stop, dtype=np.int64)[:, None]
        exact_scores[~valid | (found == row_ids)] = -np.inf
        finite_count = np.isfinite(exact_scores).sum(axis=1)
        if np.any(finite_count < k):
            bad = int(np.flatnonzero(finite_count < k)[0] + start)
            raise RuntimeError(
                f"FAISS returned fewer than {k} non-self candidates for "
                f"node {bad}; increase nprobe."
            )
        top_partition = np.argpartition(
            exact_scores, kth=exact_scores.shape[1] - k, axis=1
        )[:, -k:]
        top_scores = np.take_along_axis(exact_scores, top_partition, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top_positions = np.take_along_axis(top_partition, order, axis=1)
        output[start:stop] = np.take_along_axis(
            found, top_positions, axis=1
        )

    print(
        f"ANN KNN finished in {time.time() - started:.2f}s",
        flush=True,
    )
    return torch.from_numpy(output)
