"""Memory-bounded canonical large-graph loading and propagation."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from rebuttal.large_target_inference.common import measured_phase
from rebuttal.large_target_inference.protocol import DATASETS


FEATURE_ALIGNMENT_VERSION = "robust_sampled_pca_post_zscore_v1"
ADJACENCY_VERSION = "symmetric_normalized_with_self_loops_v1"


@dataclass
class LargeTargetContext:
    name: str
    graph: Data
    node_count: int
    adjacency_nnz: int
    aligned_dims: int
    labels_path: Path
    evaluation_mask_path: Path | None
    metadata: dict
    phase_records: list[dict]


def canonical_paths(dataset_root: Path, name: str, dims: int) -> dict[str, Path]:
    base = dataset_root / name
    return {
        "base": base,
        "metadata": base / "metadata.json",
        "adjacency": base / "adjacency.npz",
        "features": base / "features.npy",
        "aligned_features": (
            base
            / f"features_aligned_{dims}_{FEATURE_ALIGNMENT_VERSION}.npy"
        ),
        "labels": base / "labels.npy",
        "evaluation_mask": base / "evaluation_mask.npy",
    }


def _normalized_sparse_tensor(adjacency: sp.csr_matrix) -> torch.Tensor:
    adjacency = adjacency.astype(np.float32, copy=False)
    adjacency = adjacency + sp.eye(
        adjacency.shape[0], dtype=np.float32, format="csr"
    )
    rowsum = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    positive = rowsum > 0
    inverse_sqrt[positive] = np.power(rowsum[positive], -0.5)
    diagonal = sp.diags(inverse_sqrt, format="csr")
    normalized = (diagonal @ adjacency @ diagonal).tocsr()
    normalized.sort_indices()
    coo = normalized.tocoo(copy=False)
    indices = torch.stack(
        (
            torch.from_numpy(coo.row.astype(np.int64, copy=False)),
            torch.from_numpy(coo.col.astype(np.int64, copy=False)),
        ),
        dim=0,
    )
    values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        values,
        size=normalized.shape,
        dtype=torch.float32,
        is_coalesced=True,
    )


def load_and_propagate(
    *,
    dataset_root: Path,
    name: str,
    dims: int,
    num_hops: int,
    device: str,
) -> LargeTargetContext:
    if name not in DATASETS:
        raise KeyError(f"Unknown locked large target: {name}")
    paths = canonical_paths(dataset_root, name, dims)
    required = [
        paths["metadata"],
        paths["adjacency"],
        paths["features"],
        paths["aligned_features"],
        paths["labels"],
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete canonical bundle: {missing}")
    phase_records: list[dict] = []
    with paths["metadata"].open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    with measured_phase("canonical_load_and_normalize", device) as record:
        adjacency = sp.load_npz(paths["adjacency"]).tocsr()
        features_memmap = np.load(paths["aligned_features"], mmap_mode="r")
        expected = DATASETS[name]
        if adjacency.shape != (expected["nodes"], expected["nodes"]):
            raise ValueError(
                f"{name}: adjacency shape {adjacency.shape} is not locked"
            )
        if int(adjacency.nnz) != int(expected["adjacency_nnz"]):
            raise ValueError(
                f"{name}: adjacency nnz {adjacency.nnz} is not locked"
            )
        if features_memmap.shape != (expected["nodes"], dims):
            raise ValueError(
                f"{name}: aligned feature shape {features_memmap.shape}"
            )
        adjacency_tensor = _normalized_sparse_tensor(adjacency)
        features_cpu = torch.from_numpy(np.asarray(features_memmap))
        record["nodes"] = int(features_cpu.shape[0])
        record["adjacency_nnz_without_self_loops"] = int(adjacency.nnz)
        record["aligned_dims"] = int(features_cpu.shape[1])
    phase_records.append(dict(record))

    resolved_device = torch.device(device)
    with measured_phase("four_hop_propagation", device) as record:
        adjacency_gpu = adjacency_tensor.to(resolved_device)
        current = features_cpu.to(resolved_device)
        x_list = [current]
        for _ in range(num_hops):
            current = torch.sparse.mm(adjacency_gpu, current)
            x_list.append(current)
        if any(not torch.isfinite(value).all() for value in x_list):
            raise FloatingPointError(f"{name}: non-finite propagated features")
        record["num_hops"] = int(num_hops)
        record["x_list_shapes"] = [list(value.shape) for value in x_list]
    phase_records.append(dict(record))

    del adjacency, adjacency_tensor, adjacency_gpu, features_cpu, features_memmap
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    graph = Data(
        x_list=x_list,
        dataset_name=name,
        feature_alignment_version=FEATURE_ALIGNMENT_VERSION,
        feature_dims=dims,
        adjacency_version=ADJACENCY_VERSION,
    )
    if "ano_labels" in graph or "evaluation_mask" in graph:
        raise AssertionError("Large-target inference graph contains labels/mask")
    return LargeTargetContext(
        name=name,
        graph=graph,
        node_count=int(DATASETS[name]["nodes"]),
        adjacency_nnz=int(DATASETS[name]["adjacency_nnz"]),
        aligned_dims=dims,
        labels_path=paths["labels"],
        evaluation_mask_path=(
            paths["evaluation_mask"] if paths["evaluation_mask"].exists() else None
        ),
        metadata=metadata,
        phase_records=phase_records,
    )


def initial_residual(x_list: list[torch.Tensor]) -> torch.Tensor:
    base = x_list[0]
    return torch.hstack([current - base for current in x_list[1:]])
