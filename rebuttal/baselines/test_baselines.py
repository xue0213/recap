from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.utils.extmath import svd_flip

from rebuttal.baselines.baseline_analysis import aggregate
from rebuttal.baselines.baseline_bc_protocol import (
    build_supplement_manifest,
    expected_supplement_evaluations,
    validate_supplement_manifest,
)
from rebuttal.baselines.baseline_common import (
    LabelVault,
    atomic_npz,
    load_anomalygfm_features,
    row_normalize_features,
    sha256_array,
)
from rebuttal.baselines.baseline_models import (
    SparseGraphConv,
    UNPromptGCN,
    UNPromptGrace,
    dense_affinity_message_reference,
    sparse_affinity_message,
    unprompt_deterministic_neighbor_eval,
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

    def test_bc_supplement_manifest_scope(self) -> None:
        specs = build_supplement_manifest()
        validate_supplement_manifest(specs)
        self.assertEqual(len(specs), 12)
        self.assertEqual(expected_supplement_evaluations(), 60)
        self.assertEqual({spec.setting for spec in specs}, {"B", "C"})
        self.assertEqual(
            {spec.method for spec in specs},
            {"UNPrompt", "AnomalyGFM-ZS"},
        )
        self.assertTrue(
            all(len(spec.source_graphs) == 4 for spec in specs)
        )
        self.assertTrue(
            all(len(spec.target_graphs) == 5 for spec in specs)
        )

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

    def test_unprompt_deterministic_eval_matches_dense(self) -> None:
        generator = torch.Generator().manual_seed(19)
        dense = torch.eye(8)
        dense[torch.arange(7), torch.arange(1, 8)] = 0.5
        dense[torch.arange(1, 8), torch.arange(7)] = 0.5
        features = torch.randn(8, 8, generator=generator)
        encoder = UNPromptGCN()
        encoder.eval()
        actual = unprompt_deterministic_neighbor_eval(
            encoder, features, dense.to_sparse().coalesce()
        )
        expected = encoder.activation(
            encoder.batch_norm(
                dense @ encoder.linear(features) + encoder.bias
            )
        )
        repeated = unprompt_deterministic_neighbor_eval(
            encoder, features, dense.to_sparse().coalesce()
        )
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(torch.equal(actual, repeated))

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

    def test_anomalygfm_rank8_sparse_svd_matches_full(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(17)
            attributes = rng.normal(size=(30, 12)).astype(np.float32)
            attributes[np.abs(attributes) < 0.4] = 0
            sio.savemat(
                root / "cora.mat",
                {
                    "Network": sp.eye(30),
                    "Attributes": sp.csr_matrix(attributes),
                    "Label": np.zeros((30, 1), dtype=np.float32),
                },
            )
            actual = load_anomalygfm_features(
                root, root / "cache", "cora"
            )
            left, singular, right = np.linalg.svd(
                attributes, full_matrices=False
            )
            left, _ = svd_flip(
                left[:, :8], right[:8], u_based_decision=True
            )
            expected = row_normalize_features(
                left * singular[:8]
            ).astype(np.float32)
            np.testing.assert_allclose(
                actual, expected, atol=2e-4, rtol=2e-4
            )

    def test_analysis_uses_three_seed_population_statistics(self) -> None:
        records = []
        for spec in build_manifest():
            for target_index, target in enumerate(spec.target_graphs):
                value = 0.5 + 0.01 * spec.seed + 0.001 * target_index
                records.append(
                    {
                        "setting": spec.setting,
                        "method": spec.method,
                        "seed": spec.seed,
                        "target_graph": target,
                        "domain": {
                            "Flickr": "Social",
                            "BlogCatalog": "Social",
                            "Facebook": "Social",
                            "weibo": "Social",
                            "Reddit": "Social",
                            "Amazon": "E-commerce",
                            "questions": "Q&A",
                        }.get(target, "Citation"),
                        "auroc": value,
                        "auprc": value / 2,
                    }
                )
        dataset_rows, macro_rows = aggregate(records)
        self.assertEqual(len(dataset_rows), 52)
        self.assertEqual(len(macro_rows), 10)
        setting_a_arc = next(
            row
            for row in macro_rows
            if row["setting"] == "A"
            and row["method"] == "ARC"
            and row["aggregation"] == "dataset_macro"
        )
        self.assertAlmostEqual(
            setting_a_arc["auroc_std"], np.std([0.0, 0.01, 0.02], ddof=0)
        )
        self.assertEqual(set(setting_a_arc["seed_values"]), {"0", "1", "2"})

    def test_bc_supplement_aggregation_scope(self) -> None:
        records = []
        for spec in build_supplement_manifest():
            for target_index, target in enumerate(spec.target_graphs):
                records.append(
                    {
                        "setting": spec.setting,
                        "method": spec.method,
                        "seed": spec.seed,
                        "target_graph": target,
                        "domain": {
                            "Flickr": "Social",
                            "BlogCatalog": "Social",
                            "Facebook": "Social",
                            "weibo": "Social",
                            "Reddit": "Social",
                            "Amazon": "E-commerce",
                            "questions": "Q&A",
                        }[target],
                        "auroc": 0.4
                        + 0.01 * spec.seed
                        + 0.001 * target_index,
                        "auprc": 0.2
                        + 0.005 * spec.seed
                        + 0.0005 * target_index,
                    }
                )
        dataset_rows, macro_rows = aggregate(records)
        self.assertEqual(len(dataset_rows), 20)
        self.assertEqual(len(macro_rows), 6)
        self.assertEqual(
            {
                (row["setting"], row["method"], row["aggregation"])
                for row in macro_rows
            },
            {
                ("B", "UNPrompt", "dataset_macro"),
                ("B", "AnomalyGFM-ZS", "dataset_macro"),
                ("C", "UNPrompt", "dataset_macro"),
                ("C", "UNPrompt", "domain_macro"),
                ("C", "AnomalyGFM-ZS", "dataset_macro"),
                ("C", "AnomalyGFM-ZS", "domain_macro"),
            },
        )


if __name__ == "__main__":
    unittest.main()
