"""Shared data, split, provenance, and label-boundary utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split

from rebuttal.baselines.baseline_common import (
    atomic_json,
    atomic_npz,
    atomic_torch_save,
    environment_metadata,
    sha256_array,
    sha256_file,
    symmetric_normalize,
    utc_now,
)
from rebuttal.baselines.baseline_protocol import DATASETS


PREPROCESS_VERSION = "ofo12_raw_row_norm_binary_sym_v1"
SPLIT_VERSION = "ofo12_stratified_40_20_40_v1"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dataset_path(dataset_dir: Path, name: str) -> Path:
    return dataset_dir / DATASETS[name]["file"]


def _load_variables(path: Path, variables: tuple[str, ...]) -> dict[str, Any]:
    return sio.loadmat(path, variable_names=list(variables))


def load_graph_without_labels(
    dataset_dir: Path, name: str
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Read only graph structure and attributes from a MAT file."""

    raw = _load_variables(
        dataset_path(dataset_dir, name), ("Network", "A", "Attributes", "X")
    )
    adjacency_value = raw["Network"] if "Network" in raw else raw["A"]
    feature_value = raw["Attributes"] if "Attributes" in raw else raw["X"]

    adjacency = sp.csr_matrix(adjacency_value, dtype=np.float32)
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1.0
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    features = sp.csr_matrix(feature_value, dtype=np.float32)
    if adjacency.shape[0] != features.shape[0]:
        raise ValueError(f"{name}: adjacency/feature node mismatch")

    row_sum = np.asarray(features.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = np.abs(row_sum) > 1e-12
    inverse[nonzero] = 1.0 / row_sum[nonzero]
    features = sp.diags(inverse).dot(features).tocsr()
    features.eliminate_zeros()
    if not np.isfinite(features.data).all():
        raise ValueError(f"{name}: non-finite normalized features")
    return adjacency, features


def load_labels(dataset_dir: Path, name: str) -> np.ndarray:
    raw = _load_variables(dataset_path(dataset_dir, name), ("Label", "gnd"))
    value = raw["Label"] if "Label" in raw else raw["gnd"]
    labels = np.asarray(value, dtype=np.int64).reshape(-1)
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name}: expected binary labels, found {unique}")
    return labels


def stratified_split(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(labels.shape[0], dtype=np.int64)
    train, remainder = train_test_split(
        indices,
        train_size=0.4,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    validation, test = train_test_split(
        remainder,
        train_size=1.0 / 3.0,
        random_state=seed + 100_003,
        shuffle=True,
        stratify=labels[remainder],
    )
    masks = {}
    for key, values in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        mask = np.zeros(labels.shape[0], dtype=np.bool_)
        mask[np.asarray(values, dtype=np.int64)] = True
        masks[key] = mask
        if set(np.unique(labels[mask]).tolist()) != {0, 1}:
            raise ValueError(f"seed {seed}: {key} does not contain both classes")
    if np.any(masks["train"] & masks["validation"]):
        raise ValueError("train/validation overlap")
    if np.any(masks["train"] & masks["test"]):
        raise ValueError("train/test overlap")
    if np.any(masks["validation"] & masks["test"]):
        raise ValueError("validation/test overlap")
    if not np.all(masks["train"] | masks["validation"] | masks["test"]):
        raise ValueError("split does not cover all nodes")
    return masks


def torch_sparse(matrix: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = sp.coo_matrix(matrix, dtype=np.float32)
    indices = np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
    return (
        torch.sparse_coo_tensor(
            torch.from_numpy(indices),
            torch.from_numpy(coo.data),
            size=coo.shape,
            dtype=torch.float32,
        )
        .coalesce()
        .to(device)
    )


def adjacency_tensors(
    adjacency: sp.csr_matrix, device: torch.device
) -> dict[str, torch.Tensor]:
    node_count = adjacency.shape[0]
    with_loop = adjacency + sp.eye(node_count, dtype=np.float32, format="csr")
    with_loop.data[:] = 1.0
    with_loop.eliminate_zeros()
    normalized_loop = symmetric_normalize(with_loop)
    normalized_no_loop = symmetric_normalize(adjacency)
    edge_no_loop = sp.coo_matrix(adjacency)
    edge_with_loop = sp.coo_matrix(with_loop)
    return {
        "adj_norm_loop": torch_sparse(normalized_loop, device),
        "adj_norm_no_loop": torch_sparse(normalized_no_loop, device),
        "edge_index_no_loop": torch.from_numpy(
            np.vstack([edge_no_loop.row, edge_no_loop.col]).astype(np.int64)
        ).to(device),
        "edge_index_with_loop": torch.from_numpy(
            np.vstack([edge_with_loop.row, edge_with_loop.col]).astype(np.int64)
        ).to(device),
        "edge_weight_with_loop": torch.from_numpy(
            edge_with_loop.data.astype(np.float32)
        ).to(device),
    }


def sample_rowwise_nonedges(
    adjacency_with_loop: sp.csr_matrix,
    positive_rows: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample one non-edge per positive edge, preserving source-row counts."""

    node_count = adjacency_with_loop.shape[0]
    rows = np.asarray(positive_rows, dtype=np.int64)
    cols = rng.integers(0, node_count, size=rows.shape[0], dtype=np.int64)
    invalid = np.asarray(adjacency_with_loop[rows, cols]).reshape(-1) != 0
    attempts = 0
    while invalid.any():
        cols[invalid] = rng.integers(
            0, node_count, size=int(invalid.sum()), dtype=np.int64
        )
        invalid = np.asarray(adjacency_with_loop[rows, cols]).reshape(-1) != 0
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Unable to sample non-edges after 100 attempts")

    positive_count = np.bincount(rows, minlength=node_count).astype(np.float64)
    zero_count = node_count - np.asarray(
        adjacency_with_loop.getnnz(axis=1), dtype=np.float64
    )
    row_weight = np.zeros(node_count, dtype=np.float32)
    valid = positive_count > 0
    row_weight[valid] = (zero_count[valid] / positive_count[valid]).astype(
        np.float32
    )
    return rows, cols, row_weight


@dataclass
class PreparedGraph:
    name: str
    adjacency: sp.csr_matrix
    features: sp.csr_matrix
    raw_sha256: str

    @property
    def node_count(self) -> int:
        return self.adjacency.shape[0]

    @property
    def feature_count(self) -> int:
        return self.features.shape[1]


def prepare_graph(dataset_dir: Path, name: str) -> PreparedGraph:
    adjacency, features = load_graph_without_labels(dataset_dir, name)
    return PreparedGraph(
        name=name,
        adjacency=adjacency,
        features=features,
        raw_sha256=sha256_file(dataset_path(dataset_dir, name)),
    )


class OFOLabelVault:
    """Auditable label boundary for supervised and unsupervised OFO."""

    def __init__(
        self, dataset_dir: Path, name: str, supervised: bool, seed: int
    ) -> None:
        self.dataset_dir = dataset_dir
        self.name = name
        self.supervised = supervised
        self.seed = seed
        self.events: list[dict[str, Any]] = []
        self._labels: np.ndarray | None = None
        self._masks: dict[str, np.ndarray] | None = None
        self._frozen: dict[str, Any] | None = None

    def _event(self, action: str, **payload: Any) -> None:
        self.events.append({"at": utc_now(), "action": action, **payload})

    def supervised_partitions(
        self,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if not self.supervised:
            raise PermissionError("unsupervised method requested training labels")
        if self._labels is None:
            self._labels = load_labels(self.dataset_dir, self.name)
            self._masks = stratified_split(self._labels, self.seed)
            self._event(
                "create_stratified_split",
                allowed=True,
                purpose="split construction and train/validation supervision",
                split_version=SPLIT_VERSION,
                score_frozen=False,
            )
        assert self._masks is not None
        return self._labels.copy(), {
            key: value.copy() for key, value in self._masks.items()
        }

    def freeze(
        self,
        *,
        score_path: Path,
        scores: np.ndarray,
        query_mask: np.ndarray,
    ) -> dict[str, str]:
        if self._frozen is not None:
            raise ValueError("score already frozen")
        self._frozen = {
            "score_path": str(score_path),
            "score_sha256": sha256_array(scores),
            "query_mask_sha256": sha256_array(query_mask),
            "frozen_at": utc_now(),
        }
        self._event(
            "freeze_scores",
            allowed=True,
            score_frozen=True,
            **self._frozen,
        )
        return {
            "score_sha256": self._frozen["score_sha256"],
            "query_mask_sha256": self._frozen["query_mask_sha256"],
        }

    def evaluation_labels(self) -> np.ndarray:
        if self._frozen is None:
            raise PermissionError("evaluation labels requested before score freeze")
        if self._labels is None:
            self._labels = load_labels(self.dataset_dir, self.name)
            origin = "post_freeze_mat_read"
        else:
            origin = "supervised_split_cache"
        self._event(
            "load_evaluation_labels",
            allowed=True,
            score_frozen=True,
            origin=origin,
        )
        return self._labels.copy()

    def audit(self) -> dict[str, Any]:
        invalid = [event for event in self.events if not event.get("allowed")]
        return {
            "format": "recap_ofo12_label_audit_v1",
            "dataset": self.name,
            "supervised": self.supervised,
            "seed": self.seed,
            "passed": self._frozen is not None and not invalid,
            "events": self.events,
            "frozen": self._frozen,
            "invalid_events": invalid,
        }


__all__ = [
    "PREPROCESS_VERSION",
    "SPLIT_VERSION",
    "PreparedGraph",
    "OFOLabelVault",
    "adjacency_tensors",
    "atomic_json",
    "atomic_npz",
    "atomic_torch_save",
    "environment_metadata",
    "prepare_graph",
    "sample_rowwise_nonedges",
    "set_seed",
    "sha256_array",
    "torch_sparse",
    "utc_now",
]
