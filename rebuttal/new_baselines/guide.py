"""Exact-motif, sparse current-PyTorch implementation of GUIDE."""

from __future__ import annotations

import hashlib
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GUIDEConfig:
    embedding_dim: int = 32
    hidden1: int = 128
    hidden2: int = 64
    dropout: float = 0.3
    attention_alpha: float = 0.3
    attribute_weight: float = 0.3
    learning_rate: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 200


def minmax_columns(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    minimum = np.min(value, axis=0)
    maximum = np.max(value, axis=0)
    scale = maximum - minimum
    output = np.zeros_like(value, dtype=np.float32)
    valid = scale > 0
    output[:, valid] = (
        value[:, valid] - minimum[valid]
    ) / scale[valid]
    return output


def adjacency_digest(adjacency: sp.csr_matrix) -> str:
    matrix = adjacency.tocsr()
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode())
    digest.update(matrix.indptr.astype(np.int64).tobytes())
    digest.update(matrix.indices.astype(np.int64).tobytes())
    return digest.hexdigest()


def orca_node_orbits(
    adjacency: sp.csr_matrix,
    *,
    orca_binary: Path,
    work_dir: Path,
) -> np.ndarray:
    """Run pinned ORCA and return the 15 induced node orbits through order 4."""

    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    upper = sp.triu(adjacency, k=1).tocoo()
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=work_dir,
        prefix="orca_graph_",
        suffix=".in",
        delete=False,
    ) as input_handle:
        input_path = Path(input_handle.name)
        input_handle.write(f"{adjacency.shape[0]} {upper.nnz}\n")
        for source, target in zip(upper.row.tolist(), upper.col.tolist()):
            input_handle.write(f"{source} {target}\n")
    with tempfile.NamedTemporaryFile(
        dir=work_dir,
        prefix="orca_orbits_",
        suffix=".out",
        delete=False,
    ) as output_handle:
        output_path = Path(output_handle.name)
    try:
        completed = subprocess.run(
            [
                str(orca_binary),
                "node",
                "4",
                str(input_path),
                str(output_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr)
        orbits = np.loadtxt(output_path, dtype=np.int64, ndmin=2)
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
    if orbits.shape != (adjacency.shape[0], 15):
        raise ValueError(f"unexpected ORCA output shape {orbits.shape}")
    if (orbits < 0).any():
        raise ValueError("negative ORCA orbit count")
    return orbits


def guide_motifs_from_orbits(orbits: np.ndarray) -> np.ndarray:
    """Map standard ORCA node orbits to the six released GUIDE columns."""

    orbits = np.asarray(orbits, dtype=np.int64)
    if orbits.ndim != 2 or orbits.shape[1] < 15:
        raise ValueError(f"invalid orbit matrix {orbits.shape}")
    columns = (
        orbits[:, 0],
        orbits[:, 3],
        orbits[:, 1] + orbits[:, 2],
        orbits[:, 14],
        orbits[:, 12] + orbits[:, 13],
        orbits[:, 8],
    )
    return np.column_stack(columns).astype(np.float32)


def cached_guide_motifs(
    adjacency: sp.csr_matrix,
    *,
    orca_binary: Path,
    cache_dir: Path,
    orca_commit: str,
) -> tuple[np.ndarray, Path, str]:
    graph_hash = adjacency_digest(adjacency)
    key = hashlib.sha256(f"{graph_hash}:{orca_commit}:guide6_v1".encode()).hexdigest()
    path = cache_dir / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            motifs = np.asarray(archive["motifs"], dtype=np.float32)
            cached_hash = str(archive["adjacency_sha256"].item())
            cached_commit = str(archive["orca_commit"].item())
        if cached_hash != graph_hash or cached_commit != orca_commit:
            raise ValueError(f"motif cache provenance mismatch: {path}")
        return motifs, path, "reused"
    orbits = orca_node_orbits(
        adjacency,
        orca_binary=orca_binary,
        work_dir=cache_dir / "tmp",
    )
    motifs = guide_motifs_from_orbits(orbits)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=cache_dir,
        prefix=f".{key}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary,
            motifs=motifs,
            adjacency_sha256=np.asarray(graph_hash),
            orca_commit=np.asarray(orca_commit),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return motifs, path, "new"


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.weight.shape[1])
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        output = torch.sparse.mm(adjacency, x @ self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class SparseResidualAttention(nn.Module):
    """Sparse algebraic form of GUIDE's released graph-node attention."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        dropout: float,
        alpha: float,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
        self.attention = nn.Parameter(torch.zeros(1, output_dim))
        nn.init.xavier_normal_(self.weight, gain=1.414)
        nn.init.xavier_normal_(self.attention, gain=1.414)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        adjacency = adjacency.coalesce()
        source, target = adjacency.indices()
        node_count = x.shape[0]
        hidden = x @ self.weight
        difference = hidden[source] - hidden[target]
        logits = F.leaky_relu(
            torch.sum(difference * self.attention, dim=1),
            negative_slope=self.alpha,
        )
        edge_weight = torch.exp(-logits)
        denominator = torch.zeros(
            node_count, dtype=hidden.dtype, device=hidden.device
        ).index_add_(0, source, edge_weight)
        dropped = self.dropout(edge_weight)
        numerator = torch.zeros_like(hidden).index_add_(
            0, source, dropped[:, None] * hidden[target]
        )
        output = numerator / torch.clamp(denominator[:, None], min=1e-12)
        return F.elu(output)


class GUIDEModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        motif_dim: int,
        config: GUIDEConfig,
    ) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.attr1 = GraphConvolution(feature_dim, config.hidden1)
        self.attr2 = GraphConvolution(config.hidden1, config.hidden2)
        self.attr3 = GraphConvolution(config.hidden2, config.embedding_dim)
        self.struct1 = SparseResidualAttention(
            motif_dim,
            config.hidden1,
            dropout=config.dropout,
            alpha=config.attention_alpha,
        )
        self.struct2 = SparseResidualAttention(
            config.hidden1,
            config.hidden2,
            dropout=config.dropout,
            alpha=config.attention_alpha,
        )
        self.struct3 = SparseResidualAttention(
            config.hidden2,
            config.embedding_dim,
            dropout=config.dropout,
            alpha=config.attention_alpha,
        )
        self.attr_decoder = GraphConvolution(
            config.embedding_dim, feature_dim
        )
        self.struct_decoder = SparseResidualAttention(
            config.embedding_dim,
            motif_dim,
            dropout=config.dropout,
            alpha=config.attention_alpha,
        )

    def forward(
        self,
        attributes: torch.Tensor,
        motifs: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.attr1(attributes, adjacency))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.attr2(x, adjacency))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.attr3(x, adjacency))

        structure = F.relu(self.struct1(motifs, adjacency))
        structure = F.dropout(
            structure, self.dropout, training=self.training
        )
        structure = F.relu(self.struct2(structure, adjacency))
        structure = F.dropout(
            structure, self.dropout, training=self.training
        )
        structure = F.relu(self.struct3(structure, adjacency))

        attributes_hat = F.relu(self.attr_decoder(x, adjacency))
        motifs_hat = F.relu(self.struct_decoder(structure, adjacency))
        return attributes_hat, motifs_hat


def guide_score(
    attributes: torch.Tensor,
    attributes_hat: torch.Tensor,
    motifs: torch.Tensor,
    motifs_hat: torch.Tensor,
    attribute_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    attribute_error = torch.sqrt(
        torch.clamp(torch.sum((attributes - attributes_hat) ** 2, dim=1), min=0)
    )
    structure_error = torch.sqrt(
        torch.clamp(torch.sum((motifs - motifs_hat) ** 2, dim=1), min=0)
    )
    score = (
        attribute_weight * attribute_error
        + (1.0 - attribute_weight) * structure_error
    )
    return score, attribute_error.mean(), structure_error.mean()


__all__ = [
    "GUIDEConfig",
    "GUIDEModel",
    "SparseResidualAttention",
    "adjacency_digest",
    "cached_guide_motifs",
    "guide_motifs_from_orbits",
    "guide_score",
    "minmax_columns",
    "orca_node_orbits",
]
