"""Unit and numerical tests for the large-target inference extension."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from model import EgoCluster
from rebuttal.large_target_inference.common import LabelVault, atomic_npy
from rebuttal.large_target_inference.protocol import build_manifest
from rebuttal.large_target_inference.scoring import (
    compute_score_components_chunked,
)


class LargeTargetInferenceTests(unittest.TestCase):
    def test_manifest_is_exact_three_by_three_product(self) -> None:
        manifest = build_manifest()
        self.assertEqual(len(manifest), 9)
        self.assertEqual(
            {(item.target, item.seed) for item in manifest},
            {
                (target, seed)
                for target in ("tfinance", "dgraphfin", "tsocial")
                for seed in (0, 1, 2)
            },
        )

    def test_label_vault_blocks_early_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
            labels_path = root / "labels.npy"
            atomic_npy(labels_path, labels)
            vault = LabelVault(
                labels_path=labels_path,
                evaluation_mask_path=None,
                node_count=4,
                events_path=root / "events.jsonl",
            )
            with self.assertRaises(AssertionError):
                vault.unlock(("exact",))
            scores_path = root / "scores.npy"
            mask_path = root / "mask.npy"
            atomic_npy(scores_path, np.arange(4, dtype=np.float32))
            atomic_npy(mask_path, np.ones(4, dtype=np.bool_))
            vault.freeze(
                route="exact",
                scores_path=scores_path,
                scores_sha256="score",
                mask_path=mask_path,
                mask_sha256="mask",
            )
            observed_labels, observed_mask = vault.unlock(("exact",))
            np.testing.assert_array_equal(observed_labels, labels)
            self.assertTrue(observed_mask.all())

    def test_chunked_scoring_matches_native_exact_path(self) -> None:
        torch.manual_seed(9)
        residual = torch.randn(83, 13)
        cluster = EgoCluster(
            embed_dim=13,
            num_clusters=6,
            knn_k=7,
            tau_s=0.3,
            tau_c=0.3,
            tau_e=1.0,
            beta=0.02,
            sim_block_size=19,
            knn_cache_enabled=False,
        )
        candidates = (
            cluster._select_knn_candidates(residual)
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        native = cluster.compute_score_components(
            residual, E_init=residual, cache_key=None
        )
        observed = compute_score_components_chunked(
            residual=residual,
            cluster=cluster,
            candidates=candidates,
            score_batch_size=17,
        )
        np.testing.assert_allclose(
            observed["total"],
            native["total"].detach().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            observed["adhesion_raw"],
            native["adhesion_raw"].detach().numpy(),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            observed["context_raw"],
            native["scale_raw"].detach().numpy(),
            atol=1e-6,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
