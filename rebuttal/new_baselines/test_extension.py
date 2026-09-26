"""Acceptance gates for the three-baseline extension."""

from __future__ import annotations

import argparse
import itertools
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from rebuttal.new_baselines.common import atomic_json
from rebuttal.new_baselines.diffgad import (
    dense_structure_squared_error,
    exact_structure_squared_error,
)
from rebuttal.new_baselines.owleye import (
    OWLEYEConfig,
    OWLEYEGraph,
    OWLEYEModel,
    effective_normalization,
    load_official_features,
)
from rebuttal.new_baselines.protocol import (
    build_manifest,
    expected_evaluations,
    validate_manifest,
)


def _edge_index(adjacency: np.ndarray) -> torch.Tensor:
    row, col = np.nonzero(adjacency)
    return torch.tensor(np.vstack([row, col]), dtype=torch.long)


def gate_diffgad_exact_loss() -> dict[str, float]:
    generator = np.random.default_rng(17)
    adjacency = generator.random((11, 11)) < 0.25
    adjacency = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, False)
    edge_index = _edge_index(adjacency)
    z_exact = torch.randn(11, 7, dtype=torch.float64, requires_grad=True)
    exact = exact_structure_squared_error(z_exact, edge_index)
    exact.sum().backward()
    exact_gradient = z_exact.grad.detach().clone()
    z_dense = z_exact.detach().clone().requires_grad_(True)
    dense = dense_structure_squared_error(z_dense, edge_index)
    dense.sum().backward()
    dense_gradient = z_dense.grad.detach().clone()
    forward_difference = float(torch.max(torch.abs(exact - dense)))
    gradient_difference = float(
        torch.max(torch.abs(exact_gradient - dense_gradient))
    )
    if forward_difference > 1e-10 or gradient_difference > 1e-10:
        raise AssertionError(
            f"DiffGAD exact gate failed: {forward_difference}, "
            f"{gradient_difference}"
        )
    return {
        "forward_max_abs_difference": forward_difference,
        "gradient_max_abs_difference": gradient_difference,
    }








def gate_owleye_chunking() -> dict[str, float]:
    torch.manual_seed(31)
    config = OWLEYEConfig(
        hidden_dim=8,
        hops=2,
        layers=2,
        dropout=0.0,
        query_chunk_size=3,
    )
    node_count = 13
    adjacency_np = np.eye(node_count, dtype=np.float32)
    for index in range(node_count - 1):
        adjacency_np[index, index + 1] = 1
        adjacency_np[index + 1, index] = 1
    degree = adjacency_np.sum(axis=1)
    normalized = (
        adjacency_np
        / np.sqrt(degree[:, None])
        / np.sqrt(degree[None, :])
    )
    row, col = np.nonzero(normalized)
    adjacency = torch.sparse_coo_tensor(
        torch.tensor(np.vstack([row, col]), dtype=torch.long),
        torch.tensor(normalized[row, col]),
        (node_count, node_count),
    ).coalesce()
    feature = torch.randn(node_count, 64)
    propagated = [feature]
    for _ in range(config.hops):
        propagated.append(torch.sparse.mm(adjacency, propagated[-1]))
    graph = OWLEYEGraph(
        name="gate",
        features=feature,
        propagated=tuple(propagated),
        adjacency=adjacency,
        raw_sha256="gate",
    )
    model = OWLEYEModel(config)
    model.eval()
    with torch.no_grad():
        feature_embedding, structure_embedding = model.embeddings(graph)
        feature_patterns = [
            feature_embedding[[0, 2, 4, 6, 8]],
            feature_embedding[[1, 3, 5, 7, 9]],
        ]
        structure_patterns = [
            structure_embedding[[0, 2, 4, 6, 8]],
            structure_embedding[[1, 3, 5, 7, 9]],
        ]
        full, _ = model.anomaly_scores(
            feature_embedding,
            structure_embedding,
            feature_patterns,
            structure_patterns,
            chunk_size=node_count,
        )
        chunked, _ = model.anomaly_scores(
            feature_embedding,
            structure_embedding,
            feature_patterns,
            structure_patterns,
            chunk_size=3,
        )
    difference = float(torch.max(torch.abs(full - chunked)))
    if difference > 2e-6:
        raise AssertionError(f"OWLEYE chunk mismatch: {difference}")
    return {"max_abs_difference": difference}


def gate_owleye_normalization(cache_path: Path) -> dict[str, float | list[int]]:
    features = load_official_features(cache_path)
    subset = features[: min(128, len(features))]
    normalized = effective_normalization(subset, 1.0)
    expected = subset / np.linalg.norm(subset, axis=1).mean()
    difference = float(np.max(np.abs(normalized - expected)))
    # In the released routine, the second distance pass reads stale x_list.
    # Therefore dist_normalized == dist_original and the locked tau=1 scale is 1.
    distances = torch.cdist(
        torch.from_numpy(subset[:32]), torch.from_numpy(subset[:32])
    ).mean().item()
    multiplier = float(np.sqrt(distances * distances / (distances * distances)))
    if difference != 0 or abs(multiplier - 1.0) > 1e-12:
        raise AssertionError("OWLEYE normalization cancellation gate failed")
    return {
        "cache_shape": list(features.shape),
        "effective_normalization_max_abs_difference": difference,
        "released_multiplier": multiplier,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = build_manifest()
    validate_manifest(manifest)
    gates = {
        "manifest": {
            "training_runs": len(manifest),
            "evaluations": expected_evaluations(),
        },
        "diffgad_exact_structure": gate_diffgad_exact_loss(),
        "owleye_chunking": gate_owleye_chunking(),
        "owleye_normalization": gate_owleye_normalization(
            args.vendor_root / "owleye" / "dataset" / "cora_64.npz"
        ),
    }
    report = {
        "format": "recap_three_baseline_gate_report_v1",
        "passed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "gates": gates,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
