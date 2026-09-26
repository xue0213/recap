from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

from rebuttal.ofo_baselines.common import (
    OFOLabelVault,
    load_graph_without_labels,
    sample_rowwise_nonedges,
    stratified_split,
)
from rebuttal.ofo_baselines.models import (
    CoLAModel,
    beta_wavelet_coefficients,
    exact_dot_structure_error,
    polynomial_filter,
)
from rebuttal.ofo_baselines.protocol import (
    build_manifest,
    validate_manifest,
)


class OFOBaselineTests(unittest.TestCase):
    def test_manifest_is_complete_cartesian_product(self) -> None:
        manifest = build_manifest()
        validate_manifest(manifest)
        self.assertEqual(len(manifest), 288)
        self.assertEqual(len({spec.run_id for spec in manifest}), 288)

    def test_stratified_split_is_disjoint_and_seeded(self) -> None:
        labels = np.array([0] * 90 + [1] * 30, dtype=np.int64)
        first = stratified_split(labels, 7)
        repeated = stratified_split(labels, 7)
        different = stratified_split(labels, 8)
        for key in first:
            np.testing.assert_array_equal(first[key], repeated[key])
            self.assertEqual(set(np.unique(labels[first[key]])), {0, 1})
        self.assertFalse(np.array_equal(first["train"], different["train"]))
        self.assertTrue(
            np.all(first["train"] | first["validation"] | first["test"])
        )

    def test_unsupervised_label_vault_blocks_early_label_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sio.savemat(
                root / "cora.mat",
                {
                    "Network": sp.eye(6),
                    "Attributes": sp.eye(6),
                    "Label": np.array([[0], [0], [0], [1], [0], [1]]),
                },
            )
            vault = OFOLabelVault(root, "cora", False, 0)
            with self.assertRaises(PermissionError):
                vault.supervised_partitions()
            with self.assertRaises(PermissionError):
                vault.evaluation_labels()

    def test_label_free_loader_does_not_require_label_variable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adjacency = sp.csr_matrix(
                np.array(
                    [[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.float32
                )
            )
            attributes = sp.csr_matrix(
                np.array([[1, 1], [0, 2], [0, 0]], dtype=np.float32)
            )
            sio.savemat(
                root / "cora.mat",
                {"Network": adjacency, "Attributes": attributes},
            )
            loaded_adjacency, loaded_features = load_graph_without_labels(
                root, "cora"
            )
            self.assertEqual(loaded_adjacency.nnz, 4)
            np.testing.assert_allclose(
                loaded_features.toarray(),
                np.array([[0.5, 0.5], [0.0, 1.0], [0.0, 0.0]]),
            )

    def test_exact_dominant_structure_error_matches_dense(self) -> None:
        generator = torch.Generator().manual_seed(11)
        latent = torch.randn(9, 5, generator=generator)
        adjacency = torch.zeros(9, 9)
        for node in range(9):
            adjacency[node, node] = 1
            adjacency[node, (node + 1) % 9] = 1
        edge_index = adjacency.nonzero(as_tuple=False).T.contiguous()
        actual = exact_dot_structure_error(latent, edge_index)
        expected = torch.sqrt(
            torch.sum((latent @ latent.T - adjacency).square(), dim=1)
        )
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        )

    def test_beta_polynomial_sparse_filter_matches_dense(self) -> None:
        generator = torch.Generator().manual_seed(13)
        normalized = torch.rand(7, 7, generator=generator)
        normalized = (normalized + normalized.T) / 2
        features = torch.randn(7, 4, generator=generator)
        coefficients = beta_wavelet_coefficients(2)[1]
        actual = polynomial_filter(
            features, normalized.to_sparse().coalesce(), coefficients
        )
        laplacian = torch.eye(7) - normalized
        expected = coefficients[0] * features
        power = features
        for coefficient in coefficients[1:]:
            power = laplacian @ power
            expected = expected + coefficient * power
        self.assertTrue(
            torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
        )

    def test_rowwise_negative_sampler_returns_only_nonedges(self) -> None:
        adjacency = sp.eye(10, format="csr")
        adjacency[0, 1] = 1
        adjacency[1, 0] = 1
        positive = sp.coo_matrix(adjacency)
        rows, cols, weights = sample_rowwise_nonedges(
            adjacency.tocsr(),
            positive.row,
            np.random.default_rng(17),
        )
        self.assertEqual(rows.shape, cols.shape)
        self.assertTrue(np.all(np.asarray(adjacency[rows, cols]).reshape(-1) == 0))
        self.assertTrue(np.all(weights[np.bincount(rows, minlength=10) > 0] > 0))

    def test_cola_adapter_matches_pygod_base_forward(self) -> None:
        try:
            from pygod.nn import CoLABase
        except ImportError:
            self.skipTest("PyGOD is only installed in the formal environment")
        generator = torch.Generator().manual_seed(19)
        features = torch.randn(12, 6, generator=generator)
        edge_index = torch.tensor(
            [
                list(range(12)) + list(range(12)),
                list(range(1, 12)) + [0] + list(range(11, -1, -1)),
            ],
            dtype=torch.long,
        )
        adapter = CoLAModel(6, hidden_features=8, num_layers=2)
        upstream = CoLABase(6, hid_dim=8, num_layers=2)
        upstream.load_state_dict(adapter.state_dict())
        permutation = torch.randperm(12, generator=generator)
        positive, negative = adapter.logits(
            features, edge_index, permutation
        )
        upstream_embedding = upstream.encoder(features, edge_index)
        expected_positive = upstream.discriminator(
            features, upstream_embedding
        ).squeeze(1)
        expected_negative = upstream.discriminator(
            features[permutation], upstream_embedding
        ).squeeze(1)
        self.assertTrue(torch.equal(positive, expected_positive))
        self.assertTrue(torch.equal(negative, expected_negative))


if __name__ == "__main__":
    unittest.main()
