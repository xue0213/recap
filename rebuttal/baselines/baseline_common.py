"""Shared data, provenance, and label-isolation utilities for Phase 2."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

from .baseline_protocol import DATASETS


ARC_ALIGNMENT_VERSION = "official_arc_smooth_pca64_cache_v1"
UNPROMPT_ALIGNMENT_VERSION = "official_unprompt_torch_svd8_bn_v1"
ANOMALYGFM_ALIGNMENT_VERSION = "official_anomalygfm_numpy_svd8_rownorm_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".pt",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def _load_mat_variables(path: Path, names: tuple[str, ...]) -> dict[str, Any]:
    return sio.loadmat(path, variable_names=list(names))


def load_adjacency(dataset_dir: Path, name: str) -> sp.csr_matrix:
    """Load graph structure without materializing any label variable."""

    raw = _load_mat_variables(dataset_path(dataset_dir, name), ("Network", "A"))
    value = raw["Network"] if "Network" in raw else raw["A"]
    adjacency = sp.csr_matrix(value, dtype=np.float32)
    adjacency.eliminate_zeros()
    return adjacency


def load_raw_features(dataset_dir: Path, name: str) -> sp.csr_matrix:
    """Load node attributes without materializing any label variable."""

    raw = _load_mat_variables(dataset_path(dataset_dir, name), ("Attributes", "X"))
    value = raw["Attributes"] if "Attributes" in raw else raw["X"]
    return sp.csr_matrix(value, dtype=np.float32)


def _load_labels_only(dataset_dir: Path, name: str) -> np.ndarray:
    raw = _load_mat_variables(dataset_path(dataset_dir, name), ("Label", "gnd"))
    value = raw["Label"] if "Label" in raw else raw["gnd"]
    labels = np.asarray(value, dtype=np.float32).reshape(-1)
    unique = set(np.unique(labels).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"{name}: expected binary labels, found {sorted(unique)}")
    return labels


class LabelVault:
    """Audited boundary between model-facing data and anomaly labels."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.events: list[dict[str, Any]] = []
        self._source_cache: dict[str, np.ndarray] = {}
        self._arc_context_cache: dict[str, np.ndarray] = {}
        self._frozen_scores: dict[str, dict[str, Any]] = {}

    def _event(self, **payload: Any) -> None:
        self.events.append({"at": utc_now(), **payload})

    def load_source(self, name: str) -> np.ndarray:
        if name not in self._source_cache:
            self._source_cache[name] = _load_labels_only(self.dataset_dir, name)
        self._event(
            dataset=name,
            action="load_source_labels",
            allowed=True,
            score_frozen=name in self._frozen_scores,
        )
        return self._source_cache[name].copy()

    def load_arc_context(self, name: str) -> np.ndarray:
        if name not in self._arc_context_cache:
            self._arc_context_cache[name] = _load_labels_only(
                self.dataset_dir, name
            )
        self._event(
            dataset=name,
            action="load_target_labels_for_arc_normal_context_only",
            allowed=True,
            score_frozen=name in self._frozen_scores,
        )
        return self._arc_context_cache[name].copy()

    def mark_score_frozen(
        self,
        name: str,
        *,
        score_path: Path,
        score_sha256: str,
        query_mask_sha256: str,
    ) -> None:
        if name in self._frozen_scores:
            raise ValueError(f"{name}: score was already frozen")
        self._frozen_scores[name] = {
            "score_path": str(score_path),
            "score_sha256": score_sha256,
            "query_mask_sha256": query_mask_sha256,
            "frozen_at": utc_now(),
        }
        self._event(
            dataset=name,
            action="mark_target_score_frozen",
            allowed=True,
            score_frozen=True,
            score_path=str(score_path),
        )

    def load_target_for_evaluation(self, name: str) -> np.ndarray:
        if name not in self._frozen_scores:
            raise PermissionError(
                f"{name}: target labels requested before immutable score freeze"
            )
        if name in self._arc_context_cache:
            labels = self._arc_context_cache[name]
            origin = "arc_context_cache"
        else:
            labels = _load_labels_only(self.dataset_dir, name)
            origin = "post_score_mat_read"
        self._event(
            dataset=name,
            action="load_target_labels_for_metric",
            allowed=True,
            score_frozen=True,
            origin=origin,
        )
        return labels.copy()

    def audit(self) -> dict[str, Any]:
        invalid = [event for event in self.events if not event.get("allowed", False)]
        return {
            "format": "recap_phase2_label_audit_v1",
            "passed": not invalid,
            "events": self.events,
            "frozen_scores": self._frozen_scores,
            "invalid_events": invalid,
        }


def scipy_to_torch_sparse(
    matrix: sp.spmatrix, device: torch.device | str
) -> torch.Tensor:
    coo = sp.coo_matrix(matrix, dtype=np.float32)
    indices = np.vstack([coo.row, coo.col]).astype(np.int64, copy=False)
    tensor = torch.sparse_coo_tensor(
        torch.from_numpy(indices),
        torch.from_numpy(coo.data),
        size=coo.shape,
        dtype=torch.float32,
    ).coalesce()
    return tensor.to(device)


def symmetric_normalize(adjacency: sp.spmatrix) -> sp.coo_matrix:
    adjacency = sp.coo_matrix(adjacency, dtype=np.float32)
    rowsum = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = np.power(rowsum, -0.5, where=rowsum != 0)
    inverse[rowsum == 0] = 0.0
    degree = sp.diags(inverse)
    return adjacency.dot(degree).transpose().dot(degree).tocoo()


def row_normalize(adjacency: sp.spmatrix) -> sp.coo_matrix:
    adjacency = sp.coo_matrix(adjacency, dtype=np.float32)
    rowsum = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = np.power(rowsum, -1.0, where=rowsum != 0)
    inverse[rowsum == 0] = 0.0
    return sp.diags(inverse).dot(adjacency).tocoo()


def row_normalize_features(features: np.ndarray) -> np.ndarray:
    rowsum = np.asarray(features.sum(axis=1)).reshape(-1)
    inverse = np.power(rowsum, -1.0, where=rowsum != 0)
    inverse[rowsum == 0] = 0.0
    return features * inverse[:, None]


@dataclass
class ArcGraph:
    name: str
    x: torch.Tensor
    propagated: tuple[torch.Tensor, ...]
    adj_norm: torch.Tensor
    affinity_adj_norm: torch.Tensor
    affinity_edge_index: torch.Tensor
    node_count: int
    raw_sha256: str


@dataclass
class UNPromptGraph:
    name: str
    x: torch.Tensor
    adjacency_with_loop_raw: sp.coo_matrix
    adjacency_with_loop_norm: torch.Tensor
    adjacency_without_loop_norm: torch.Tensor
    node_count: int
    raw_sha256: str


@dataclass
class AnomalyGFMGraph:
    name: str
    x: torch.Tensor
    gcn_adjacency: torch.Tensor
    neighbor_adjacency: torch.Tensor
    node_count: int
    raw_sha256: str


def load_arc_features(dataset_dir: Path, name: str) -> np.ndarray:
    """Load the official ARC 64-D cache without loading its embedded labels."""

    cache_path = dataset_dir / f"{name}_64.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing official ARC preprocessing cache: {cache_path}"
        )
    with np.load(cache_path, allow_pickle=True) as cache:
        features = np.asarray(cache["feat"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 64:
        raise ValueError(f"{name}: invalid ARC cache shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{name}: non-finite ARC features")
    return features


def prepare_arc_graph(
    dataset_dir: Path,
    name: str,
    device: torch.device | str,
    *,
    num_hops: int = 2,
) -> ArcGraph:
    adjacency = load_adjacency(dataset_dir, name)
    features = load_arc_features(dataset_dir, name)
    if adjacency.shape[0] != features.shape[0]:
        raise ValueError(f"{name}: adjacency/feature node mismatch")

    invariant_base = adjacency
    if name not in {"YelpChi", "Facebook"}:
        invariant_base = adjacency + sp.eye(adjacency.shape[0], dtype=np.float32)
    invariant_norm = symmetric_normalize(invariant_base)

    # DGL GraphConv in the IA-GGAD affinity branch removes then adds self-loops
    # and applies its default symmetric normalization.
    affinity_base = adjacency.copy().tolil()
    affinity_base.setdiag(0)
    affinity_base = affinity_base.tocsr()
    affinity_base.eliminate_zeros()
    affinity_base = affinity_base + sp.eye(adjacency.shape[0], dtype=np.float32)
    affinity_norm = symmetric_normalize(affinity_base)
    affinity_coo = sp.coo_matrix(affinity_base)

    x = torch.from_numpy(features).to(device)
    adj_norm = scipy_to_torch_sparse(invariant_norm, device)
    affinity_adj_norm = scipy_to_torch_sparse(affinity_norm, device)
    edge_index = torch.from_numpy(
        np.vstack([affinity_coo.row, affinity_coo.col]).astype(np.int64)
    ).to(device)

    propagated = [x]
    for _ in range(num_hops):
        propagated.append(torch.sparse.mm(adj_norm, propagated[-1]))

    return ArcGraph(
        name=name,
        x=x,
        propagated=tuple(propagated),
        adj_norm=adj_norm,
        affinity_adj_norm=affinity_adj_norm,
        affinity_edge_index=edge_index,
        node_count=features.shape[0],
        raw_sha256=sha256_file(dataset_path(dataset_dir, name)),
    )


def _unprompt_cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / "features" / "unprompt" / f"{name}_svd8_bn.npz"


def load_unprompt_features(
    dataset_dir: Path,
    cache_dir: Path,
    name: str,
    device: torch.device | str,
) -> np.ndarray:
    """Official rank-8 torch SVD and non-affine BatchNorm preprocessing."""

    raw_path = dataset_path(dataset_dir, name)
    raw_hash = sha256_file(raw_path)
    cache_path = _unprompt_cache_path(cache_dir, name)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            version = str(cache["alignment_version"].item())
            cached_hash = str(cache["raw_sha256"].item())
            features = np.asarray(cache["features"], dtype=np.float32)
        if version == UNPROMPT_ALIGNMENT_VERSION and cached_hash == raw_hash:
            return features

    raw = load_raw_features(dataset_dir, name).toarray().astype(
        np.float32, copy=False
    )
    values = torch.from_numpy(raw).to(device)
    # ``full_matrices=False`` preserves the leading singular triplets used by
    # the release while avoiding the unused N x N tail for Questions.
    left, singular, _ = torch.linalg.svd(values, full_matrices=False)
    reduced = left[:, :8] * singular[:8]
    batch_norm = torch.nn.BatchNorm1d(8, affine=False).to(device)
    batch_norm.train()
    normalized = batch_norm(reduced).detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name}: non-finite UNPrompt aligned features")
    atomic_npz(
        cache_path,
        features=normalized,
        alignment_version=np.array(UNPROMPT_ALIGNMENT_VERSION),
        raw_sha256=np.array(raw_hash),
    )
    return normalized


def prepare_unprompt_graph(
    dataset_dir: Path,
    cache_dir: Path,
    name: str,
    device: torch.device | str,
) -> UNPromptGraph:
    adjacency = load_adjacency(dataset_dir, name)
    features = load_unprompt_features(dataset_dir, cache_dir, name, device)
    diagonal_present = bool(np.all(adjacency.diagonal() > 0))
    if diagonal_present:
        with_loop = sp.coo_matrix(adjacency, dtype=np.float32)
        without_loop = adjacency - sp.eye(
            adjacency.shape[0], dtype=np.float32
        )
    else:
        with_loop = sp.coo_matrix(
            adjacency + sp.eye(adjacency.shape[0], dtype=np.float32),
            dtype=np.float32,
        )
        without_loop = adjacency
    with_loop_norm = row_normalize(with_loop)
    without_loop_norm = row_normalize(without_loop)
    return UNPromptGraph(
        name=name,
        x=torch.from_numpy(features).to(device),
        adjacency_with_loop_raw=with_loop,
        adjacency_with_loop_norm=scipy_to_torch_sparse(
            with_loop_norm, device
        ),
        adjacency_without_loop_norm=scipy_to_torch_sparse(
            without_loop_norm, device
        ),
        node_count=features.shape[0],
        raw_sha256=sha256_file(dataset_path(dataset_dir, name)),
    )


def _anomalygfm_cache_path(cache_dir: Path, name: str) -> Path:
    return cache_dir / "features" / "anomalygfm" / f"{name}_svd8_rownorm.npz"


def load_anomalygfm_features(
    dataset_dir: Path, cache_dir: Path, name: str
) -> np.ndarray:
    raw_path = dataset_path(dataset_dir, name)
    raw_hash = sha256_file(raw_path)
    cache_path = _anomalygfm_cache_path(cache_dir, name)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            version = str(cache["alignment_version"].item())
            cached_hash = str(cache["raw_sha256"].item())
            features = np.asarray(cache["features"], dtype=np.float32)
        if version == ANOMALYGFM_ALIGNMENT_VERSION and cached_hash == raw_hash:
            return features

    raw = load_raw_features(dataset_dir, name).toarray()
    left, singular, _ = np.linalg.svd(raw, full_matrices=False)
    reduced = left[:, :8] * singular[:8]
    normalized = row_normalize_features(reduced).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name}: non-finite AnomalyGFM aligned features")
    atomic_npz(
        cache_path,
        features=normalized,
        alignment_version=np.array(ANOMALYGFM_ALIGNMENT_VERSION),
        raw_sha256=np.array(raw_hash),
    )
    return normalized


def prepare_anomalygfm_graph(
    dataset_dir: Path,
    cache_dir: Path,
    name: str,
    device: torch.device | str,
) -> AnomalyGFMGraph:
    adjacency = load_adjacency(dataset_dir, name)
    features = load_anomalygfm_features(dataset_dir, cache_dir, name)
    adjacency_without_loop = adjacency.copy().tolil()
    adjacency_without_loop.setdiag(0)
    adjacency_without_loop = adjacency_without_loop.tocsr()
    adjacency_without_loop.eliminate_zeros()

    # Released zero-shot code first normalizes A symmetrically and then adds I.
    gcn_adjacency = symmetric_normalize(adjacency) + sp.eye(
        adjacency.shape[0], dtype=np.float32
    )
    neighbor_adjacency = row_normalize(adjacency_without_loop)
    return AnomalyGFMGraph(
        name=name,
        x=torch.from_numpy(features).to(device),
        gcn_adjacency=scipy_to_torch_sparse(gcn_adjacency, device),
        neighbor_adjacency=scipy_to_torch_sparse(
            neighbor_adjacency, device
        ),
        node_count=features.shape[0],
        raw_sha256=sha256_file(dataset_path(dataset_dir, name)),
    )


def environment_metadata(device: torch.device | str) -> dict[str, Any]:
    gpu_query = ""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        gpu_query = (result.stdout or result.stderr).strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        gpu_query = f"unavailable: {exc}"
    return {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_query": gpu_query,
    }
