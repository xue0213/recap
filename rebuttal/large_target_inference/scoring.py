"""Memory-bounded score computation equivalent to RECAP's fixed-KNN scorer."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _standardize(score: torch.Tensor, eps: float) -> torch.Tensor:
    return (score - score.mean()) / score.std(
        unbiased=False
    ).clamp(min=eps)


@torch.no_grad()
def compute_score_components_chunked(
    *,
    residual: torch.Tensor,
    cluster,
    candidates: np.ndarray,
    score_batch_size: int = 1024,
) -> dict[str, np.ndarray | float]:
    """Compute exact RECAP components without N×K×D or N×C×D tensors."""
    node_count = int(residual.shape[0])
    candidates = np.asarray(candidates)
    if candidates.shape != (node_count, cluster.knn_k):
        raise ValueError(
            f"Candidate shape {candidates.shape} does not match "
            f"{(node_count, cluster.knn_k)}"
        )
    device = residual.device
    batch = max(1, int(score_batch_size))
    assignments = cluster.cluster(residual)
    denominator = assignments.sum(dim=0).clamp(min=cluster.eps)
    centroids = torch.mm(assignments.t(), residual) / denominator[:, None]
    centroid_norm = (centroids * centroids).sum(dim=1)[None, :]

    adhesion_raw = residual.new_empty(node_count)
    for start in range(0, node_count, batch):
        stop = min(start + batch, node_count)
        current = residual[start:stop]
        distances = (
            (current * current).sum(dim=1, keepdim=True)
            + centroid_norm
            - 2.0 * torch.mm(current, centroids.t())
        ).clamp_min_(0)
        adhesion_raw[start:stop] = (
            assignments[start:stop] * distances
        ).sum(dim=1) / max(cluster.tau_e, cluster.eps)

    neighbor_assignments = torch.zeros_like(assignments)
    degree = residual.new_zeros(node_count)
    scale = max(cluster.tau_s, cluster.eps)
    for start in range(0, node_count, batch):
        stop = min(start + batch, node_count)
        columns = torch.from_numpy(
            np.asarray(candidates[start:stop], dtype=np.int64)
        ).to(device=device)
        query_normalized = F.normalize(
            residual[start:stop], p=2, dim=1
        )
        candidate_normalized = residual[columns]
        candidate_norm = candidate_normalized.norm(
            p=2, dim=2, keepdim=True
        ).clamp(min=cluster.eps)
        candidate_normalized.div_(candidate_norm)
        cosine = (
            candidate_normalized * query_normalized[:, None, :]
        ).sum(dim=2)
        weights = F.softmax(cosine / scale, dim=1) * 0.5

        neighbor_assignments[start:stop] += (
            assignments[columns] * weights[:, :, None]
        ).sum(dim=1)
        degree[start:stop] += weights.sum(dim=1)

        flat_columns = columns.reshape(-1)
        reverse_values = (
            assignments[start:stop, None, :]
            * weights[:, :, None]
        ).reshape(-1, assignments.shape[1])
        neighbor_assignments.index_add_(
            0, flat_columns, reverse_values
        )
        degree.index_add_(0, flat_columns, weights.reshape(-1))
        del (
            columns,
            query_normalized,
            candidate_normalized,
            candidate_norm,
            cosine,
            weights,
            flat_columns,
            reverse_values,
        )

    neighbor_assignments = neighbor_assignments / degree.clamp(
        min=cluster.eps
    )[:, None]
    neighbor_assignments = neighbor_assignments / neighbor_assignments.sum(
        dim=1, keepdim=True
    ).clamp(min=cluster.eps)
    midpoint = 0.5 * (assignments + neighbor_assignments)
    kl_self = (
        assignments
        * (
            torch.log(assignments + cluster.eps)
            - torch.log(midpoint + cluster.eps)
        )
    ).sum(dim=1)
    kl_neighbor = (
        neighbor_assignments
        * (
            torch.log(neighbor_assignments + cluster.eps)
            - torch.log(midpoint + cluster.eps)
        )
    ).sum(dim=1)
    context_raw = 0.5 * (kl_self + kl_neighbor)
    context_raw = context_raw / math.log(2.0)

    adhesion = _standardize(adhesion_raw, cluster.eps)
    context = _standardize(context_raw, cluster.eps)
    total = adhesion + cluster.beta * context
    usage = assignments.mean(dim=0)
    effective = torch.exp(
        -(usage * torch.log(usage + cluster.eps)).sum()
    )
    output: dict[str, np.ndarray | float] = {
        "total": total.float().cpu().numpy(),
        "adhesion": adhesion.float().cpu().numpy(),
        "context": context.float().cpu().numpy(),
        "adhesion_raw": adhesion_raw.float().cpu().numpy(),
        "context_raw": context_raw.float().cpu().numpy(),
        "hard_assignments": assignments.argmax(dim=1).short().cpu().numpy(),
        "usage": usage.double().cpu().numpy(),
        "effective_communities": float(effective.item()),
    }
    floating = [
        value
        for value in output.values()
        if isinstance(value, np.ndarray) and value.dtype.kind == "f"
    ]
    if any(not np.all(np.isfinite(value)) for value in floating):
        raise FloatingPointError("Non-finite chunked RECAP score component")
    return output
