import os
from pathlib import Path

import pytest
from rebuttal.large_target_optimization.scan import discover_checkpoints
from rebuttal.large_target_optimization.train import _edge_terms


def test_protocol_is_locked():
    protocol = (
        Path(__file__).resolve().parents[1]
        / "LARGE_TARGET_OPTIMIZATION_PROTOCOL.md"
    )
    text = protocol.read_text(encoding="utf-8")
    assert "no post-hoc inversion" in text.lower()
    assert "45 accepted existing checkpoints" in text
    assert "oracle source-sensitivity" in text


def test_checkpoint_inventory():
    root = os.environ.get("RECAP_PHASE1_ROOT")
    if not root:
        pytest.skip("RECAP_PHASE1_ROOT is not set")
    records = discover_checkpoints(Path(root))
    assert len(records) == 45
    assert {record["seed"] for record in records} == {0, 1, 2}
    assert len({record["source_family"] for record in records}) == 15


def test_chunked_edge_algebra_matches_native():
    import numpy as np
    import torch
    from model import EgoCluster

    torch.manual_seed(7)
    residual = torch.randn(19, 11, requires_grad=True)
    cluster = EgoCluster(
        embed_dim=11,
        num_clusters=5,
        knn_k=4,
        tau_s=0.3,
        tau_c=0.3,
    )
    candidates = (
        cluster._select_knn_candidates(residual.detach()).cpu().numpy()
    )
    assignments = cluster.cluster(residual)
    edge_index, edge_weight = cluster.build_ego_graph(
        residual, E_init=residual.detach()
    )
    native = cluster._compute_con_loss(
        assignments, edge_index, edge_weight
    )
    numerator = residual.new_tensor(0.0)
    denominator = residual.new_tensor(0.0)
    for start in range(0, len(residual), 6):
        current = _edge_terms(
            residual,
            assignments,
            candidates,
            start,
            min(start + 6, len(residual)),
            cluster.tau_s,
            cluster.eps,
        )
        numerator = numerator + current[0]
        denominator = denominator + current[1]
    observed = numerator / denominator
    assert np.isclose(
        float(native.detach()), float(observed.detach()), atol=1e-6
    )
