"""Method-faithful model components and sparse compatibility operations."""

from __future__ import annotations

import importlib.util
import math
import random
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ArcCrossAttention(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.Wq = nn.Linear(embedding_dim, embedding_dim)
        self.Wk = nn.Linear(embedding_dim, embedding_dim)

    def cross_attention(
        self, query_features: torch.Tensor, support_features: torch.Tensor
    ) -> torch.Tensor:
        queries = self.Wq(query_features)
        keys = self.Wk(support_features)
        attention = torch.matmul(queries, keys.T) / math.sqrt(self.embedding_dim)
        weights = F.softmax(attention, dim=1)
        return torch.matmul(weights, support_features)

    def training_loss(
        self, embeddings: torch.Tensor, labels: torch.Tensor, num_prompt: int
    ) -> torch.Tensor:
        anomaly_indices = torch.nonzero(labels == 1).squeeze(1).tolist()
        all_normal_indices = torch.nonzero(labels == 0).squeeze(1).tolist()
        normal_indices = random.sample(all_normal_indices, len(anomaly_indices))

        anomaly = embeddings[anomaly_indices]
        normal = embeddings[normal_indices]
        query = torch.vstack([anomaly, normal])
        remaining = list(set(all_normal_indices) - set(normal_indices))
        if len(remaining) < num_prompt:
            raise ValueError("Not enough source normal nodes for ARC prompts")
        support_indices = torch.tensor(
            random.sample(remaining, num_prompt),
            device=labels.device,
            dtype=torch.long,
        )
        support = embeddings[support_indices]
        updated = self.cross_attention(query, support)
        updated_anomaly = updated[: len(anomaly_indices)]
        updated_normal = updated[len(anomaly_indices) :]
        positive = torch.ones(len(normal_indices), device=labels.device)
        negative = -torch.ones(len(anomaly_indices), device=labels.device)
        normal_loss = F.cosine_embedding_loss(normal, updated_normal, positive)
        anomaly_loss = F.cosine_embedding_loss(
            anomaly, updated_anomaly, negative
        )
        return torch.mean(anomaly_loss + normal_loss)

    def target_score(
        self,
        embeddings: torch.Tensor,
        context_indices: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        support = embeddings[context_indices]
        query = embeddings[query_mask]
        updated = self.cross_attention(query, support)
        return torch.sqrt(torch.sum((query - updated) ** 2, dim=1))


class ArcModel(nn.Module):
    """Official ARC residual encoder and context scorer."""

    def __init__(
        self,
        *,
        in_feats: int = 64,
        hidden_feats: int = 1024,
        num_layers: int = 4,
        num_hops: int = 2,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_feats, hidden_feats)])
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(hidden_feats, hidden_feats))
        self.activation = nn.ELU()
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.cross_attention = ArcCrossAttention(hidden_feats * num_hops)

    def forward(self, propagated: tuple[torch.Tensor, ...]) -> torch.Tensor:
        values = list(propagated)
        for index, layer in enumerate(self.layers):
            if index:
                values = [self.dropout(value) for value in values]
            values = [layer(value) for value in values]
            if index != len(self.layers) - 1:
                values = [self.activation(value) for value in values]
        origin = values[0]
        return torch.hstack([value - origin for value in values[1:]])


class SparseGraphConv(nn.Module):
    """DGL GraphConv(norm='both') equivalent for pre-normalized sparse A."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, normalized_adjacency: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        return torch.sparse.mm(
            normalized_adjacency, torch.matmul(features, self.weight)
        ) + self.bias


class SparseAffinityEncoder(nn.Module):
    """Two-layer affinity GCN used by IA-GGAD."""

    def __init__(self, in_features: int = 64, hidden_features: int = 128) -> None:
        super().__init__()
        self.conv1 = SparseGraphConv(in_features, 2 * hidden_features)
        self.conv2 = SparseGraphConv(2 * hidden_features, hidden_features)

    def forward(
        self, normalized_adjacency: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        hidden = F.relu(self.conv1(normalized_adjacency, features))
        return F.relu(self.conv2(normalized_adjacency, hidden))


def sparse_affinity_message(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    node_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact edge-wise form of IA-GGAD's dense masked cosine operation."""

    normalized = F.normalize(features, p=2, dim=-1)
    source, target = edge_index
    similarities = torch.sum(normalized[source] * normalized[target], dim=1)
    similarities = torch.nan_to_num(similarities)

    message_sum = torch.zeros(
        node_count, device=features.device, dtype=features.dtype
    )
    message_sum.scatter_add_(0, source, similarities)
    column_degree = torch.zeros_like(message_sum)
    column_degree.scatter_add_(0, target, torch.ones_like(similarities))
    inverse_degree = torch.where(
        column_degree > 0,
        column_degree.reciprocal(),
        torch.zeros_like(column_degree),
    )
    message = message_sum * inverse_degree
    return -torch.sum(message), message


def dense_affinity_message_reference(
    features: torch.Tensor, dense_adjacency: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unoptimized released-code expression, used only by equivalence tests."""

    normalized = features / torch.norm(features, dim=-1, keepdim=True)
    similarities = torch.mm(normalized, normalized.T) * dense_adjacency
    similarities = torch.nan_to_num(similarities)
    row_sum = torch.sum(dense_adjacency, dim=0)
    inverse = torch.where(
        row_sum > 0, row_sum.reciprocal(), torch.zeros_like(row_sum)
    )
    message = torch.sum(similarities, dim=1) * inverse
    return -torch.sum(message), message


def load_ia_vendor_model(
    vendor_root: Path,
    *,
    codebook_size: int = 2048,
    topk: int = 15,
) -> nn.Module:
    """Load the pinned official invariant branch without importing legacy DGL."""

    model_path = vendor_root / "IA-GGAD" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    module_name = "recap_phase2_ia_vendor_model"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(code_size=codebook_size, topk=topk)
    return module.GCN(
        args,
        in_feats=64,
        h_feats=1024,
        num_layers=4,
        dropout_rate=0.0,
        activation="ELU",
        num_hops=2,
    )


def ia_forward(
    model: nn.Module, propagated: tuple[torch.Tensor, ...], dataset_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    container = SimpleNamespace(x_list=list(propagated))
    descriptor = SimpleNamespace(name=dataset_name)
    residual, code_loss, quantized, codebook = model(container, descriptor)
    return residual, code_loss, quantized, codebook


def ia_training_loss(
    model: nn.Module,
    residual: torch.Tensor,
    quantized: torch.Tensor,
    codebook: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_prompt: int = 10,
) -> torch.Tensor:
    return model.cross_attn.get_train_loss(
        residual, quantized, codebook, labels, num_prompt
    )


def ia_invariant_score(
    model: nn.Module,
    residual: torch.Tensor,
    source_codebook: torch.Tensor,
    reference_mask: torch.Tensor,
) -> torch.Tensor:
    # The released function's ``y`` argument is unused.
    dummy = torch.empty(0, device=residual.device)
    return model.cross_attn.get_test_score(
        residual, source_codebook, reference_mask, dummy
    )


def minmax_anomaly_from_affinity(message: torch.Tensor) -> torch.Tensor:
    minimum = torch.min(message)
    maximum = torch.max(message)
    denominator = maximum - minimum
    if float(denominator) == 0.0:
        return torch.zeros_like(message)
    return 1.0 - (message - minimum) / denominator

