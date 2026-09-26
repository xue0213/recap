"""Unit tests for Phase 1 protocol invariants and exact optimizations."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from config import ModelConfig
from model import recap
from rebuttal.phase1_analysis import exact_soft_coassignment_similarity
from rebuttal.phase1_protocol import build_manifest
from rebuttal.phase1_runner import (
    CHECKPOINT_SCORE_ATOL,
    CHECKPOINT_SCORE_RTOL,
    DEFAULT_CONFIG_PATH,
    label_free_graph,
    load_manifest,
    load_model_config,
    restore_rng_state,
    rng_state,
)


class ProtocolTests(unittest.TestCase):
    def test_locked_manifest_counts_and_questions_scope(self):
        manifest = build_manifest()
        self.assertEqual(len(manifest), 42)
        self.assertEqual(sum(len(run.target_graphs) for run in manifest), 87)
        ofo = [run for run in manifest if run.setting == "OFO"]
        ofa = [run for run in manifest if run.setting != "OFO"]
        self.assertEqual(len(ofo), 33)
        self.assertEqual(len(ofa), 9)
        self.assertFalse(any("questions" in run.source_graphs for run in ofo))
        self.assertTrue(
            any(
                "questions" in (*run.source_graphs, *run.target_graphs)
                for run in ofa
            )
        )

    def test_locked_config_has_explicit_paper_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = load_model_config(DEFAULT_CONFIG_PATH, Path(temp_dir))
        self.assertEqual(config.lambda_E, 0.0)
        self.assertEqual(config.tau_e, 1.0)
        self.assertEqual(config.lambda_H, 0.1)
        self.assertEqual(config.tau_c, 0.3)
        self.assertEqual(config.knn_k, 64)
        self.assertEqual(config.num_clusters, 36)

    def test_persisted_manifest_matches_declarative_manifest(self):
        manifest_path = Path(__file__).with_name("phase1_manifest.json")
        loaded = load_manifest(manifest_path)
        self.assertEqual(loaded, build_manifest())

    def test_questions_ofo_addendum_manifest_is_isolated_and_three_seed(self):
        from rebuttal.questions_ofo_addendum import load_addendum_manifest

        specs = load_addendum_manifest(
            Path(__file__).with_name("questions_ofo_manifest.json")
        )
        self.assertEqual(len(specs), 3)
        self.assertEqual({spec.seed for spec in specs}, {0, 1, 2})
        self.assertEqual({spec.source_graphs for spec in specs}, {("questions",)})
        self.assertEqual({spec.target_graphs for spec in specs}, {("questions",)})
        self.assertTrue(all(spec.paradigm == "one-for-one" for spec in specs))


class ExactOptimizationTests(unittest.TestCase):
    def test_soft_coassignment_identity_matches_naive_matrix(self):
        rng = np.random.default_rng(7)
        first = rng.random((17, 5), dtype=np.float64)
        second = rng.random((17, 5), dtype=np.float64)
        first /= first.sum(axis=1, keepdims=True)
        second /= second.sum(axis=1, keepdims=True)
        first_matrix = first @ first.T
        second_matrix = second @ second.T
        naive = float(
            np.sum(first_matrix * second_matrix)
            / (
                np.linalg.norm(first_matrix, ord="fro")
                * np.linalg.norm(second_matrix, ord="fro")
            )
        )
        optimized = exact_soft_coassignment_similarity(first, second)
        self.assertAlmostEqual(naive, optimized, places=12)

    def test_explicit_zero_lambda_e_preserves_legacy_default_model(self):
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            explicit = json.load(handle)
        legacy = dict(explicit)
        for key in ("lambda_E", "lambda_bal", "tau_e", "gamma"):
            legacy.pop(key)
        torch.manual_seed(11)
        explicit_model = recap(**explicit)
        torch.manual_seed(11)
        legacy_model = recap(**legacy)
        for key, value in explicit_model.state_dict().items():
            self.assertTrue(torch.equal(value, legacy_model.state_dict()[key]), key)
        self.assertEqual(explicit_model.lambda_E, legacy_model.lambda_E)
        self.assertEqual(explicit_model.tau_e, legacy_model.tau_e)

    def test_tau_c_assignment_matches_formula(self):
        cluster_config = ModelConfig(
            dims=2,
            h_feats=2,
            num_layers=0,
            num_hops=1,
            num_clusters=3,
            knn_k=1,
            tau_c=0.3,
        )
        model = recap(**cluster_config.to_dict())
        residual = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
        cluster = model.ego_clusters[0]
        expected = torch.softmax(cluster.W(residual) / 0.3, dim=1)
        actual = cluster.cluster(residual)
        self.assertTrue(torch.allclose(expected, actual, atol=0.0, rtol=0.0))


class LabelIsolationTests(unittest.TestCase):
    def test_sanitized_graph_excludes_labels_and_runs_model(self):
        original = Data(
            x_list=[
                torch.randn(8, 4),
                torch.randn(8, 4),
            ],
            ano_labels=torch.tensor([0, 1, 0, 0, 0, 1, 0, 0]),
            dataset_name="synthetic",
            feature_alignment_version="test",
            feature_dims=4,
            adjacency_version="test",
        )
        clean = label_free_graph(original, "a" * 64)
        self.assertNotIn("ano_labels", clean)
        model = recap(
            dims=4,
            h_feats=4,
            num_layers=1,
            num_hops=1,
            num_clusters=3,
            knn_k=2,
            knn_search_dtype="float32",
            knn_cache_enabled=False,
            lambda_E=0.0,
        )
        model(clean)
        loss = model.get_cluster_loss()
        self.assertTrue(torch.isfinite(loss))


class ResumeTests(unittest.TestCase):
    def test_rng_state_round_trip(self):
        torch.manual_seed(123)
        state = rng_state()
        expected = torch.rand(4)
        restore_rng_state(state)
        actual = torch.rand(4)
        self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_score_tolerance_is_float32_scale_only(self):
        self.assertEqual(CHECKPOINT_SCORE_ATOL, 1e-5)
        self.assertEqual(CHECKPOINT_SCORE_RTOL, 1e-5)
        reference = np.asarray([0.0, 1.0, -2.0], dtype=np.float32)
        harmless = reference + np.asarray([2e-6, -2e-6, 2e-6], dtype=np.float32)
        material = reference + np.asarray([2e-4, 0.0, 0.0], dtype=np.float32)
        self.assertTrue(
            np.allclose(
                reference,
                harmless,
                atol=CHECKPOINT_SCORE_ATOL,
                rtol=CHECKPOINT_SCORE_RTOL,
            )
        )
        self.assertFalse(
            np.allclose(
                reference,
                material,
                atol=CHECKPOINT_SCORE_ATOL,
                rtol=CHECKPOINT_SCORE_RTOL,
            )
        )


if __name__ == "__main__":
    unittest.main()
