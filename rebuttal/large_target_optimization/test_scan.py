import os
import unittest
from pathlib import Path

import numpy as np
import torch

from model import EgoCluster
from rebuttal.large_target_optimization.scan import discover_checkpoints
from rebuttal.large_target_optimization.train import (
    _edge_terms,
    chunked_loss_value_and_backward,
)


class LargeTargetOptimizationTests(unittest.TestCase):
    def test_protocol_is_locked(self):
        protocol = (
            Path(__file__).resolve().parents[1]
            / "LARGE_TARGET_OPTIMIZATION_PROTOCOL.md"
        )
        text = protocol.read_text(encoding="utf-8")
        self.assertIn("no post-hoc inversion", text.lower())
        self.assertIn("45 accepted existing checkpoints", text)
        self.assertIn("oracle source-sensitivity", text)

    @unittest.skipUnless(
        os.environ.get("RECAP_PHASE1_ROOT"),
        "RECAP_PHASE1_ROOT is not set",
    )
    def test_checkpoint_inventory(self):
        records = discover_checkpoints(
            Path(os.environ["RECAP_PHASE1_ROOT"])
        )
        self.assertEqual(len(records), 45)
        self.assertEqual({record["seed"] for record in records}, {0, 1, 2})
        self.assertEqual(
            len({record["source_family"] for record in records}), 15
        )

    def test_chunked_edge_algebra_matches_native(self):
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
        self.assertTrue(
            np.isclose(
                float(native.detach()),
                float(observed.detach()),
                atol=1e-6,
            )
        )

    def test_chunked_loss_gradients_match_native(self):
        torch.manual_seed(19)
        initial = torch.randn(23, 13)
        native_residual = torch.randn(23, 13, requires_grad=True)
        chunked_residual = native_residual.detach().clone().requires_grad_(True)
        native_cluster = EgoCluster(
            embed_dim=13,
            num_clusters=6,
            knn_k=5,
            tau_s=0.3,
            tau_c=0.3,
            lambda_H=0.1,
            lambda_bal=0.1,
            lambda_usage_entropy=0.1,
            assignment_entropy_lower=0.45,
            assignment_entropy_upper=0.85,
            usage_entropy_lower=0.65,
            usage_entropy_upper=0.9,
        )
        chunked_cluster = EgoCluster(
            embed_dim=13,
            num_clusters=6,
            knn_k=5,
            tau_s=0.3,
            tau_c=0.3,
            lambda_H=0.1,
            lambda_bal=0.1,
            lambda_usage_entropy=0.1,
            assignment_entropy_lower=0.45,
            assignment_entropy_upper=0.85,
            usage_entropy_lower=0.65,
            usage_entropy_upper=0.9,
        )
        chunked_cluster.load_state_dict(native_cluster.state_dict())
        candidates = (
            native_cluster._select_knn_candidates(initial).cpu().numpy()
        )

        assignments = native_cluster.cluster(native_residual)
        edge_index, edge_weight = native_cluster.build_ego_graph(
            native_residual, E_init=initial
        )
        native_l_con = native_cluster._compute_con_loss(
            assignments, edge_index, edge_weight
        )
        native_l_h, _, _ = native_cluster._compute_H_loss(assignments)
        native_loss = native_l_con + native_cluster.lambda_H * native_l_h
        native_loss.backward()

        observed = chunked_loss_value_and_backward(
            residual=chunked_residual,
            cluster=chunked_cluster,
            candidates=candidates,
            batch_size=7,
        )
        self.assertAlmostEqual(
            float(native_loss.detach()), observed["total"], places=6
        )
        self.assertTrue(
            torch.allclose(
                native_residual.grad,
                chunked_residual.grad,
                atol=2e-6,
                rtol=2e-6,
            ),
            msg=str(
                (
                    native_residual.grad - chunked_residual.grad
                ).abs().max().item()
            ),
        )
        self.assertTrue(
            torch.allclose(
                native_cluster.W.weight.grad,
                chunked_cluster.W.weight.grad,
                atol=2e-6,
                rtol=2e-6,
            ),
            msg=str(
                (
                    native_cluster.W.weight.grad
                    - chunked_cluster.W.weight.grad
                ).abs().max().item()
            ),
        )


if __name__ == "__main__":
    unittest.main()
