"""Deterministic exact and FAISS candidate construction for large targets."""

from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rebuttal.large_target_inference.common import sha256_file


def _largest_divisor_at_most(value: int, upper_bound: int) -> int:
    for candidate in range(min(value, upper_bound), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _temporary_npy(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.tmp.{os.getpid()}.npy")


def validate_candidates(path: Path, node_count: int, k: int) -> dict:
    candidates = np.load(path, mmap_mode="r")
    if candidates.shape != (node_count, k):
        raise ValueError(
            f"Candidate shape {candidates.shape} != {(node_count, k)}"
        )
    minimum = int(candidates.min())
    maximum = int(candidates.max())
    if minimum < 0 or maximum >= node_count:
        raise ValueError(f"Invalid candidate IDs: min={minimum}, max={maximum}")
    duplicate_rows = 0
    self_rows = 0
    block = 100_000
    for start in range(0, node_count, block):
        stop = min(start + block, node_count)
        current = np.asarray(candidates[start:stop])
        self_rows += int(
            np.any(
                current
                == np.arange(start, stop, dtype=np.int64)[:, None],
                axis=1,
            ).sum()
        )
        duplicate_rows += int(
            np.any(np.diff(np.sort(current, axis=1), axis=1) == 0, axis=1).sum()
        )
    if self_rows or duplicate_rows:
        raise ValueError(
            f"Candidate validation failed: self_rows={self_rows}, "
            f"duplicate_rows={duplicate_rows}"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "shape": [node_count, k],
        "dtype": str(candidates.dtype),
        "minimum_id": minimum,
        "maximum_id": maximum,
        "self_neighbor_rows": self_rows,
        "duplicate_rows": duplicate_rows,
    }


@torch.no_grad()
def build_exact_candidates(
    embeddings: torch.Tensor,
    *,
    k: int,
    output_path: Path,
    query_batch_size: int = 256,
    search_dtype: torch.dtype | None = None,
) -> dict:
    node_count = int(embeddings.shape[0])
    k = min(int(k), max(node_count - 1, 0))
    if k <= 0:
        raise ValueError("Exact candidate construction requires at least 2 nodes")
    started = time.perf_counter()
    if search_dtype is None:
        search_dtype = (
            torch.bfloat16
            if embeddings.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
            else torch.float32
        )
    normalized = F.normalize(embeddings.detach().float(), p=2, dim=1)
    normalized = normalized.to(search_dtype)
    temporary = _temporary_npy(output_path)
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.int32,
        shape=(node_count, k),
    )
    for start in range(0, node_count, query_batch_size):
        stop = min(start + query_batch_size, node_count)
        scores = torch.mm(normalized[start:stop], normalized.t())
        rows = torch.arange(stop - start, device=scores.device)
        ids = torch.arange(start, stop, device=scores.device)
        scores[rows, ids] = -torch.inf
        output[start:stop] = (
            scores.topk(k, dim=1).indices.detach().cpu().numpy().astype(np.int32)
        )
    output.flush()
    del output, normalized
    os.replace(temporary, output_path)
    validation = validate_candidates(output_path, node_count, k)
    validation.update(
        {
            "route": "exact",
            "seconds": time.perf_counter() - started,
            "query_batch_size": query_batch_size,
            "search_dtype": str(search_dtype).replace("torch.", ""),
        }
    )
    return validation


@torch.no_grad()
def build_faiss_candidates(
    embeddings: torch.Tensor,
    *,
    k: int,
    output_path: Path,
    nlist: int,
    nprobe: int,
    pq_m: int,
    train_size: int,
    query_batch_size: int,
    add_batch_size: int,
    rerank_factor: int,
    max_rerank_candidates: int,
    seed: int,
) -> dict:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required for large-target ANN") from exc

    node_count, dims = (int(value) for value in embeddings.shape)
    k = min(int(k), max(node_count - 1, 0))
    if k <= 0:
        raise ValueError("ANN candidate construction requires at least 2 nodes")
    started = time.perf_counter()
    normalized_gpu = F.normalize(
        embeddings.detach().float(), p=2, dim=1
    )
    vectors = normalized_gpu.cpu().contiguous().numpy()
    del normalized_gpu
    if embeddings.device.type == "cuda":
        torch.cuda.empty_cache()

    actual_train_size = min(node_count, max(256, int(train_size)))
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
    train_indices = np.linspace(
        0, node_count - 1, num=actual_train_size, dtype=np.int64
    )
    index.train(np.ascontiguousarray(vectors[train_indices]))
    for start in range(0, node_count, max(1, int(add_batch_size))):
        stop = min(start + int(add_batch_size), node_count)
        index.add_with_ids(
            np.ascontiguousarray(vectors[start:stop]),
            np.arange(start, stop, dtype=np.int64),
        )

    search_k = min(
        node_count,
        max(
            k + 16,
            min(
                k * max(1, int(rerank_factor)) + 1,
                max(k + 1, int(max_rerank_candidates)),
            ),
        ),
    )
    temporary = _temporary_npy(output_path)
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.int32,
        shape=(node_count, k),
    )
    for start in range(0, node_count, max(1, int(query_batch_size))):
        stop = min(start + int(query_batch_size), node_count)
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
                f"FAISS returned fewer than {k} non-self candidates at {bad}"
            )
        partition = np.argpartition(
            exact_scores, kth=exact_scores.shape[1] - k, axis=1
        )[:, -k:]
        partition_scores = np.take_along_axis(
            exact_scores, partition, axis=1
        )
        order = np.argsort(-partition_scores, axis=1)
        positions = np.take_along_axis(partition, order, axis=1)
        output[start:stop] = np.take_along_axis(
            found, positions, axis=1
        ).astype(np.int32)
    output.flush()
    del output, index, quantizer, vectors
    gc.collect()
    os.replace(temporary, output_path)
    validation = validate_candidates(output_path, node_count, k)
    validation.update(
        {
            "route": "faiss_ivfpq",
            "seconds": time.perf_counter() - started,
            "dims": dims,
            "nlist": actual_nlist,
            "nprobe": actual_nprobe,
            "pq_m": actual_pq_m,
            "train_size": actual_train_size,
            "query_batch_size": int(query_batch_size),
            "add_batch_size": int(add_batch_size),
            "search_k": int(search_k),
            "seed": int(seed),
        }
    )
    return validation


@torch.no_grad()
def exact_neighbors_for_queries(
    embeddings: torch.Tensor,
    query_indices: np.ndarray,
    *,
    k: int,
    database_batch_size: int = 131_072,
    query_batch_size: int = 64,
) -> np.ndarray:
    """Blocked exact cosine top-k for a fixed, label-blind query sample."""
    device = embeddings.device
    node_count = int(embeddings.shape[0])
    query_indices = np.asarray(query_indices, dtype=np.int64)
    search_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    normalized = F.normalize(embeddings.detach().float(), p=2, dim=1).to(
        search_dtype
    )
    query_tensor = torch.from_numpy(query_indices).to(device=device)
    output = np.empty((len(query_indices), k), dtype=np.int32)
    for query_start in range(0, len(query_indices), query_batch_size):
        query_stop = min(
            query_start + query_batch_size, len(query_indices)
        )
        current_ids = query_tensor[query_start:query_stop]
        queries = normalized[current_ids]
        best_scores = torch.full(
            (len(current_ids), k),
            -torch.inf,
            dtype=search_dtype,
            device=device,
        )
        best_ids = torch.full(
            (len(current_ids), k),
            -1,
            dtype=torch.long,
            device=device,
        )
        for database_start in range(
            0, node_count, database_batch_size
        ):
            database_stop = min(
                database_start + database_batch_size, node_count
            )
            scores = torch.mm(
                queries, normalized[database_start:database_stop].t()
            )
            relative = current_ids - database_start
            inside = (relative >= 0) & (relative < scores.shape[1])
            if torch.any(inside):
                rows = torch.nonzero(inside, as_tuple=False).reshape(-1)
                scores[rows, relative[inside]] = -torch.inf
            local_k = min(k, scores.shape[1])
            local_scores, local_ids = scores.topk(local_k, dim=1)
            local_ids = local_ids + database_start
            merged_scores = torch.cat((best_scores, local_scores), dim=1)
            merged_ids = torch.cat((best_ids, local_ids), dim=1)
            best_scores, positions = merged_scores.topk(k, dim=1)
            best_ids = torch.gather(merged_ids, 1, positions)
        output[query_start:query_stop] = (
            best_ids.detach().cpu().numpy().astype(np.int32)
        )
    del normalized
    return output


def candidate_recall(
    exact: np.ndarray, approximate: np.ndarray
) -> dict[str, float | int]:
    exact = np.asarray(exact)
    approximate = np.asarray(approximate)
    if exact.shape != approximate.shape:
        raise ValueError(
            f"Recall shape mismatch: {exact.shape} vs {approximate.shape}"
        )
    per_query = np.empty(exact.shape[0], dtype=np.float64)
    for index in range(exact.shape[0]):
        per_query[index] = (
            len(set(exact[index].tolist()) & set(approximate[index].tolist()))
            / exact.shape[1]
        )
    return {
        "queries": int(exact.shape[0]),
        "k": int(exact.shape[1]),
        "mean_recall": float(per_query.mean()),
        "median_recall": float(np.median(per_query)),
        "minimum_recall": float(per_query.min()),
        "p05_recall": float(np.quantile(per_query, 0.05)),
    }
