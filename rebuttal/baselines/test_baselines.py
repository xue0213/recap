from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

from rebuttal.baselines.baseline_common import (
    LabelVault,
    atomic_npz,
    sha256_array,
)
from rebuttal.baselines.baseline_models import (
    SparseGraphConv,
    UNPromptGCN,
    UNPromptGrace,
    dense_affinity_message_reference,
    sparse_affinity_message,
)
from rebuttal.baselines.baseline_protocol import (
    build_manifest,
    expected_evaluations,
    validate_manifest,
)


class BaselineProtocolTests(unittest.TestCase):
    def test_manifest_scope(self) -> None:
        specs = build_manifest()
        validate_manifest(specs)
        self.assertEqual(len(specs), 24)
        self.assertEqual(expected_evaluations(), 156)
        setting_methods = {
            setting: {spec.method for spec in specs if spec.setting == setting}
            for setting in ("A", "B", "C")
        }
        self.assertEqual(
            setting_methods["A"],
            {"ARC", "UNPrompt", "AnomalyGFM-ZS", "IA-GGAD"},
        )
        self.assertEqual(setting_methods["B"], {"ARC", "IA-GGAD"})
        self.assertEqual(setting_methods["C"], {"ARC", "IA-GGAD"})

    def test_affinity_sparse_dense_equivalence(self) -> None:
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(9, 5, generator=generator)
        dense = torch.zeros(9, 9)
        for node in range(9):
            dense[node, node] = 1
            dense[node, (node + 1) % 9] = 1
            dense[(node + 1) % 9, node] = 1
        edge_index = dense.nonzero(as_tuple=False).T.contiguous()
        dense_loss, dense_message = dense_affinity_message_reference(
            features, dense
        )
        sparse_loss, sparse_message = sparse_affinity_message(
            features, edge_index, features.shape[0]
        )
        self.assertTrue(
            torch.allclose(dense_loss, sparse_loss, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                dense_message, sparse_message, atol=1e-6, rtol=1e-6
            )
        )

    def test_sparse_graph_conv_matches_dense(self) -> None:
        generator = torch.Generator().manual_seed(11)
        raw = torch.rand(7, 7, generator=generator)
        raw = (raw + raw.T) / 2
        raw = raw + torch.eye(7)
        degree = raw.sum(dim=1)
        normalized = (
            degree.rsqrt()[:, None] * raw * degree.rsqrt()[None, :]
        )
        sparse = normalized.to_sparse().coalesce()
        features = torch.randn(7, 4, generator=generator)
        conv = SparseGraphConv(4, 6)
        expected = normalized @ (features @ conv.weight) + conv.bias
        actual = conv(sparse, features)
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
        )

    def test_unprompt_blocked_loss_matches_full(self) -> None:
        generator = torch.Generator().manual_seed(13)
        first = torch.randn(17, 8, generator=generator, requires_grad=True)
        second = torch.randn(17, 8, generator=generator, requires_grad=True)
        grace = UNPromptGrace(UNPromptGCN(), hidden_features=128)
        blocked = grace.exact_blocked_loss(first, second, block_size=5)

        a = torch.nn.functional.normalize(first, dim=1)
        b = torch.nn.functional.normalize(second, dim=1)

        def directional(left, right):
            reflected = torch.exp(left @ left.T / 0.5)
            between = torch.exp(left @ right.T / 0.5)
            return -torch.log(
                between.diag()
                / (
                    reflected.sum(1)
                    + between.sum(1)
                    - reflected.diag()
                )
            )

        full = 0.5 * (
            directional(a, b).mean() + directional(b, a).mean()
        )
        self.assertTrue(
            torch.allclose(blocked, full, atol=1e-6, rtol=1e-6)
        )

    def test_label_vault_blocks_early_target_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sio.savemat(
                root / "cora.mat",
                {
                    "Network": sp.eye(4),
                    "Attributes": np.ones((4, 2), dtype=np.float32),
                    "Label": np.array([[0], [1], [0], [0]], dtype=np.float32),
                },
            )
            vault = LabelVault(root)
            with self.assertRaises(PermissionError):
                vault.load_target_for_evaluation("cora")
            score_path = root / "score.npz"
            scores = np.array([0.2, 0.8, 0.1, 0.3], dtype=np.float32)
            query = np.ones(4, dtype=np.bool_)
            atomic_npz(score_path, scores=scores, query_mask=query)
            vault.mark_score_frozen(
                "cora",
                score_path=score_path,
                score_sha256=sha256_array(scores),
                query_mask_sha256=sha256_array(query),
            )
            labels = vault.load_target_for_evaluation("cora")
            np.testing.assert_array_equal(labels, np.array([0, 1, 0, 0]))
            self.assertTrue(vault.audit()["passed"])


if __name__ == "__main__":
    unittest.main()
