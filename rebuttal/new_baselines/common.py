"""Shared data, artifact, timing, and label-boundary helpers."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from rebuttal.baselines.baseline_common import (
    LabelVault,
    atomic_json,
    atomic_npz,
    atomic_torch_save,
    environment_metadata,
    scipy_to_torch_sparse,
    sha256_array,
    sha256_file,
    symmetric_normalize,
    utc_now,
)
from rebuttal.ofo_baselines.common import OFOLabelVault, set_seed
from rebuttal.ofo_baselines.protocol import DATASETS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = PROJECT_ROOT / "rebuttal" / "THREE_BASELINE_EXTENSION_PROTOCOL.md"
UPSTREAM_MANIFEST_PATH = (
    PROJECT_ROOT / "rebuttal" / "new_baselines" / "upstream_manifest.json"
)
DEFAULT_DATASET_DIR = Path("/root/autodl-tmp/recap/dataset")
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "rebuttal" / "artifacts" / "three_baseline_extension"
)
DEFAULT_VENDOR_ROOT = Path("/root/autodl-tmp/recap_three_baselines/vendor")


@dataclass
class RawGraph:
    name: str
    adjacency: sp.csr_matrix
    features: sp.csr_matrix
    raw_sha256: str

    @property
    def node_count(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.features.shape[1])


def dataset_path(dataset_dir: Path, name: str) -> Path:
    return dataset_dir / DATASETS[name]["file"]


def load_raw_graph(
    dataset_dir: Path,
    name: str,
    *,
    undirected: bool,
) -> RawGraph:
    path = dataset_path(dataset_dir, name)
    raw = sio.loadmat(
        path,
        variable_names=["Network", "A", "Attributes", "X"],
    )
    adjacency_value = raw["Network"] if "Network" in raw else raw["A"]
    feature_value = raw["Attributes"] if "Attributes" in raw else raw["X"]
    adjacency = sp.csr_matrix(adjacency_value, dtype=np.float32)
    if undirected:
        adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1.0
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    features = sp.csr_matrix(feature_value, dtype=np.float32)
    if adjacency.shape[0] != features.shape[0]:
        raise ValueError(f"{name}: adjacency/feature node mismatch")
    if not np.isfinite(features.data).all():
        raise ValueError(f"{name}: non-finite attributes")
    return RawGraph(
        name=name,
        adjacency=adjacency,
        features=features,
        raw_sha256=sha256_file(path),
    )


def row_normalize_features(features: sp.csr_matrix) -> sp.csr_matrix:
    rowsum = np.asarray(features.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(rowsum, dtype=np.float32)
    valid = np.abs(rowsum) > 1e-12
    inverse[valid] = 1.0 / rowsum[valid]
    output = sp.diags(inverse).dot(features).tocsr()
    output.eliminate_zeros()
    return output


def dense_features(
    features: sp.csr_matrix,
    device: torch.device,
    *,
    row_normalize: bool,
) -> torch.Tensor:
    value = row_normalize_features(features) if row_normalize else features
    dense = np.asarray(value.toarray(), dtype=np.float32)
    if not np.isfinite(dense).all():
        raise ValueError("non-finite dense features")
    return torch.from_numpy(dense).to(device)


def edge_index_from_adjacency(
    adjacency: sp.csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    coo = adjacency.tocoo()
    indices = np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
    return torch.from_numpy(indices).to(device)


def normalized_adjacency_with_loops(
    adjacency: sp.csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    with_loops = adjacency + sp.eye(
        adjacency.shape[0], dtype=np.float32, format="csr"
    )
    with_loops.data[:] = 1.0
    with_loops.eliminate_zeros()
    return scipy_to_torch_sparse(
        symmetric_normalize(with_loops), device
    ).coalesce()


def raw_normalized_adjacency(
    adjacency: sp.csr_matrix,
    device: torch.device,
) -> torch.Tensor:
    return scipy_to_torch_sparse(
        symmetric_normalize(adjacency), device
    ).coalesce()


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    if not np.isfinite(scores).all():
        raise ValueError("non-finite anomaly scores")
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
    }


def current_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    divisor = 1024.0
    if os.uname().sysname == "Darwin":
        divisor *= 1024.0
    return float(usage.ru_maxrss / divisor)


def gpu_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0}
    return {
        "allocated_mb": float(torch.cuda.max_memory_allocated(device) / 2**20),
        "reserved_mb": float(torch.cuda.max_memory_reserved(device) / 2**20),
    }


def reload_tolerance(scores: np.ndarray) -> float:
    return float(1e-5 + 5e-6 * np.max(np.abs(scores)))


def save_unsupervised_scores(
    *,
    directory: Path,
    scores: np.ndarray,
    vault: OFOLabelVault,
) -> tuple[Path, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    mask = np.ones(scores.shape[0], dtype=np.bool_)
    path = directory / "scores.npz"
    atomic_npz(path, scores=scores, evaluation_mask=mask)
    vault.freeze(score_path=path, scores=scores, query_mask=mask)
    return path, mask


def save_target_scores(
    *,
    directory: Path,
    name: str,
    scores: np.ndarray,
    pattern_indices: np.ndarray,
    vault: LabelVault,
) -> tuple[Path, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    mask = np.ones(scores.shape[0], dtype=np.bool_)
    path = directory / "scores" / f"{name}.npz"
    atomic_npz(
        path,
        scores=scores,
        evaluation_mask=mask,
        target_pattern_indices=np.asarray(pattern_indices, dtype=np.int64),
    )
    vault.mark_score_frozen(
        name,
        score_path=path,
        score_sha256=sha256_array(scores),
        query_mask_sha256=sha256_array(mask),
    )
    return path, mask


def base_metadata(
    *,
    run: dict[str, Any],
    dataset_dir: Path,
    vendor_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "format": "recap_three_baseline_run_v1",
        "run": run,
        "started_at": utc_now(),
        "dataset_dir": str(dataset_dir.resolve()),
        "vendor_root": str(vendor_root.resolve()),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "upstream_manifest_path": str(UPSTREAM_MANIFEST_PATH),
        "upstream_manifest_sha256": sha256_file(UPSTREAM_MANIFEST_PATH),
        "environment": environment_metadata(device),
    }


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def timed(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function()
    return value, time.perf_counter() - started


__all__ = [
    "DEFAULT_DATASET_DIR",
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_VENDOR_ROOT",
    "LabelVault",
    "OFOLabelVault",
    "PROTOCOL_PATH",
    "RawGraph",
    "UPSTREAM_MANIFEST_PATH",
    "atomic_json",
    "atomic_npz",
    "atomic_torch_save",
    "base_metadata",
    "current_rss_mb",
    "dataset_path",
    "dense_features",
    "edge_index_from_adjacency",
    "gpu_memory",
    "load_raw_graph",
    "normalized_adjacency_with_loops",
    "raw_normalized_adjacency",
    "reload_tolerance",
    "row_normalize_features",
    "run_command",
    "save_target_scores",
    "save_unsupervised_scores",
    "score_metrics",
    "set_seed",
    "sha256_array",
    "sha256_file",
    "timed",
    "utc_now",
]
