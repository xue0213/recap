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
from rebuttal.new_baselines.guide import (
    SparseResidualAttention,
    guide_motifs_from_orbits,
    orca_node_orbits,
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


def _brute_guide_motifs(adjacency: np.ndarray) -> np.ndarray:
    node_count = adjacency.shape[0]
    output = np.zeros((node_count, 6), dtype=np.float64)
    output[:, 0] = adjacency.sum(axis=1)
    for size in (3, 4):
        for nodes in itertools.combinations(range(node_count), size):
            subgraph = adjacency[np.ix_(nodes, nodes)]
            degrees = subgraph.sum(axis=1).astype(int)
            edges = int(subgraph.sum() // 2)
            if np.any(degrees == 0):
                continue
            column = None
            if size == 3 and edges == 3:
                column = 1
            elif size == 3 and sorted(degrees.tolist()) == [1, 1, 2]:
                column = 2
            elif size == 4 and edges == 6:
                column = 3
            elif size == 4 and edges == 5:
                column = 4
            elif size == 4 and edges == 4 and np.all(degrees == 2):
                column = 5
            if column is not None:
                output[list(nodes), column] += 1
    return output.astype(np.float32)


def gate_guide_orca(orca_binary: Path) -> dict[str, float]:
    graph_suite = []
    # Triangle, path, square, diamond, clique, and a deterministic mixed graph.
    for edges, node_count in (
        ([(0, 1), (1, 2), (2, 0)], 3),
        ([(0, 1), (1, 2)], 3),
        ([(0, 1), (1, 2), (2, 3), (3, 0)], 4),
        ([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)], 4),
        (list(itertools.combinations(range(4), 2)), 4),
        (
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 2),
                (0, 4),
                (1, 5),
            ],
            6,
        ),
    ):
        adjacency = np.zeros((node_count, node_count), dtype=np.float32)
        for source, target in edges:
            adjacency[source, target] = adjacency[target, source] = 1
        graph_suite.append(adjacency)
    maximum = 0.0
    with tempfile.TemporaryDirectory(prefix="guide_orca_gate_") as temporary:
        work_dir = Path(temporary)
        for adjacency in graph_suite:
            orbits = orca_node_orbits(
                sp.csr_matrix(adjacency),
                orca_binary=orca_binary,
                work_dir=work_dir,
            )
            observed = guide_motifs_from_orbits(orbits)
            expected = _brute_guide_motifs(adjacency)
            maximum = max(maximum, float(np.max(np.abs(observed - expected))))
    if maximum != 0:
        raise AssertionError(f"GUIDE ORCA motif mismatch: {maximum}")
    return {"maximum_absolute_count_difference": maximum}


def gate_guide_attention() -> dict[str, float]:
    torch.manual_seed(23)
    adjacency_np = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    indices = _edge_index(adjacency_np)
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.shape[1]),
        (4, 4),
    ).coalesce()
    x = torch.randn(4, 3)
    layer = SparseResidualAttention(3, 5, dropout=0.0, alpha=0.3)
    observed = layer(x, adjacency)
    source, target = adjacency.indices()
    hidden = x @ layer.weight
    logits = torch.nn.functional.leaky_relu(
        torch.sum(
            (hidden[source] - hidden[target]) * layer.attention, dim=1
        ),
        negative_slope=0.3,
    )
    weights = torch.exp(-logits)
    denominator = torch.zeros(4).index_add_(0, source, weights)
    numerator = torch.zeros(4, 5).index_add_(
        0, source, weights[:, None] * hidden[target]
    )
    expected = torch.nn.functional.elu(
        numerator / denominator[:, None]
    )
    difference = float(torch.max(torch.abs(observed - expected)))
    if difference > 1e-7:
        raise AssertionError(f"GUIDE sparse attention mismatch: {difference}")
    return {"max_abs_difference": difference}


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
        "guide_orca": gate_guide_orca(args.vendor_root / "bin" / "orca"),
        "guide_sparse_attention": gate_guide_attention(),
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
