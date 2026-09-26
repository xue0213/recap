"""Target-label-free OWLEYE compatibility implementation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RawGraph, raw_normalized_adjacency


@dataclass(frozen=True)
class OWLEYEConfig:
    feature_dim: int = 64
    hidden_dim: int = 512
    hops: int = 4
    layers: int = 3
    dropout: float = 0.2
    activation: str = "ELU"
    learning_rate: float = 3.12964801067075e-5
    weight_decay: float = 2.848035868435802e-4
    epochs: int = 100
    support_count: int = 2000
    target_pattern_count: int = 10
    temperature: float = 0.001
    structure_weight: float = 0.01
    tau: float = 1.0
    mask_ratio_metadata: float = 0.7
    query_chunk_size: int = 512


@dataclass
class OWLEYEGraph:
    name: str
    features: torch.Tensor
    propagated: tuple[torch.Tensor, ...]
    adjacency: torch.Tensor
    raw_sha256: str

    @property
    def node_count(self) -> int:
        return int(self.features.shape[0])


def load_official_features(cache_path: Path) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as archive:
        features = np.asarray(archive["feat"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 64:
        raise ValueError(f"invalid OWLEYE cache {cache_path}: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"non-finite OWLEYE cache {cache_path}")
    return features


def effective_normalization(features: np.ndarray, tau: float) -> np.ndarray:
    """Released effective normalization for the locked tau=1 configuration."""

    if tau != 1.0:
        raise ValueError("pair-distance cancellation is locked only for tau=1")
    value = np.asarray(features, dtype=np.float32)
    mean_norm = float(np.linalg.norm(value, axis=1).mean())
    if not math.isfinite(mean_norm) or mean_norm <= 0:
        raise ValueError("invalid OWLEYE mean feature norm")
    return value / mean_norm


def prepare_owleye_graph(
    graph: RawGraph,
    *,
    cache_path: Path,
    config: OWLEYEConfig,
    device: torch.device,
) -> OWLEYEGraph:
    features = effective_normalization(
        load_official_features(cache_path), config.tau
    )
    if features.shape[0] != graph.node_count:
        raise ValueError(f"{graph.name}: cache/raw node mismatch")
    x = torch.from_numpy(features).to(device)
    adjacency = raw_normalized_adjacency(graph.adjacency, device)
    propagated = [x]
    for _ in range(config.hops):
        propagated.append(torch.sparse.mm(adjacency, propagated[-1]))
    return OWLEYEGraph(
        name=graph.name,
        features=x,
        propagated=tuple(propagated),
        adjacency=adjacency,
        raw_sha256=graph.raw_sha256,
    )


class DomainSimilarity(nn.Module):
    def __init__(self, structure_dim: int) -> None:
        super().__init__()
        self.structure_projection = nn.Linear(structure_dim, structure_dim)

    def forward(
        self,
        structure_patterns: list[torch.Tensor],
        structure_query: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        for pattern in structure_patterns:
            projected = self.structure_projection(pattern)
            logits = projected @ structure_query.T
            values.append(F.softmax(logits, dim=0).max(dim=0).values)
        return torch.stack(values, dim=0)


class OWLEYEModel(nn.Module):
    def __init__(self, config: OWLEYEConfig) -> None:
        super().__init__()
        self.config = config
        activation = getattr(nn, config.activation)
        self.activation = activation()
        feature_layers: list[nn.Module] = [
            nn.Linear(config.feature_dim, config.hidden_dim)
        ]
        feature_layers.extend(
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(1, config.layers)
        )
        self.feature_layers = nn.ModuleList(feature_layers)
        self.structure_layers = nn.ModuleList(
            nn.Linear(10, 10) for _ in range(config.hops + 1)
        )
        self.dropout = (
            nn.Dropout(config.dropout)
            if config.dropout > 0
            else nn.Identity()
        )
        embedding_dim = config.hidden_dim * config.hops
        structure_dim = 10 * config.hops
        self.embedding_dim = embedding_dim
        self.feature_query = nn.Linear(embedding_dim, embedding_dim // 2)
        self.feature_key = nn.Linear(embedding_dim, embedding_dim // 2)
        self.structure_query = nn.Linear(structure_dim, structure_dim)
        self.structure_key = nn.Linear(structure_dim, structure_dim)
        self.domain_similarity = DomainSimilarity(structure_dim)
        self.triplet = nn.TripletMarginLoss(
            margin=0.2, p=2.0, eps=1e-6, swap=True
        )
        self.structure_triplet = nn.TripletMarginLoss(
            margin=0.1, p=2.0, eps=1e-6, swap=True
        )

    def embeddings(
        self, graph: OWLEYEGraph
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature_list = list(graph.propagated)
        for index, layer in enumerate(self.feature_layers):
            if index:
                feature_list = [self.dropout(value) for value in feature_list]
            feature_list = [layer(value) for value in feature_list]
            if index != len(self.feature_layers) - 1:
                feature_list = [
                    self.activation(value) for value in feature_list
                ]
        first_feature = feature_list[0]
        feature_embedding = torch.hstack(
            [value - first_feature for value in feature_list[1:]]
        )

        structure = torch.ones(
            (graph.node_count, 10),
            dtype=graph.features.dtype,
            device=graph.features.device,
        )
        structure_list = []
        for index, layer in enumerate(self.structure_layers):
            structure = torch.sparse.mm(graph.adjacency, layer(structure))
            if index != len(self.structure_layers) - 1:
                structure = self.activation(structure)
            structure_list.append(structure)
        first_structure = structure_list[0]
        structure_embedding = torch.hstack(
            [value - first_structure for value in structure_list[1:]]
        )
        return feature_embedding, structure_embedding

    def cross_attention(
        self,
        query: torch.Tensor,
        supports: list[torch.Tensor],
        domain_similarity: torch.Tensor,
        query_projection: nn.Linear,
        key_projection: nn.Linear,
    ) -> torch.Tensor:
        projected_query = F.leaky_relu(query_projection(query))
        output = torch.zeros_like(query)
        denominator = math.sqrt(float(self.embedding_dim))
        for index, support in enumerate(supports):
            projected_key = F.leaky_relu(key_projection(support))
            attention = projected_query @ projected_key.T / denominator
            remove_count = int(attention.shape[1] * 0.1)
            if remove_count:
                remove = torch.topk(
                    attention, remove_count, dim=1, largest=False
                ).indices
                attention = attention.scatter(
                    1,
                    remove,
                    torch.full_like(remove, float("-inf"), dtype=attention.dtype),
                )
            weights = F.softmax(
                attention / self.config.temperature, dim=1
            )
            attended = weights @ support
            output = output + domain_similarity[index, :, None] * attended
        return output / len(supports)

    def training_loss(
        self,
        feature_embedding: torch.Tensor,
        structure_embedding: torch.Tensor,
        labels: torch.Tensor,
        feature_patterns: list[torch.Tensor],
        structure_patterns: list[torch.Tensor],
        *,
        support_count: int,
    ) -> torch.Tensor:
        anomaly = torch.nonzero(labels == 1, as_tuple=False).reshape(-1).tolist()
        normal_all = torch.nonzero(labels == 0, as_tuple=False).reshape(-1).tolist()
        count = min(support_count, len(anomaly), len(normal_all))
        normal_indices = random.sample(normal_all, count)
        if len(anomaly) > count:
            anomaly = random.sample(anomaly, count)
        normal = feature_embedding[normal_indices]
        abnormal = feature_embedding[anomaly]
        normal_structure = structure_embedding[normal_indices]
        abnormal_structure = structure_embedding[anomaly]

        normal_domain = self.domain_similarity(
            structure_patterns, normal_structure
        )
        abnormal_domain = self.domain_similarity(
            structure_patterns, abnormal_structure
        )
        adapted_normal = self.cross_attention(
            normal,
            feature_patterns,
            normal_domain,
            self.feature_query,
            self.feature_key,
        )
        adapted_abnormal = self.cross_attention(
            abnormal,
            feature_patterns,
            abnormal_domain,
            self.feature_query,
            self.feature_key,
        )
        adapted_structure = self.cross_attention(
            normal_structure,
            structure_patterns,
            normal_domain,
            self.structure_query,
            self.structure_key,
        )
        positive = torch.ones(count, device=labels.device)
        negative = -torch.ones(count, device=labels.device)
        loss = F.cosine_embedding_loss(normal, adapted_normal, positive)
        loss = loss + F.cosine_embedding_loss(
            abnormal, adapted_abnormal, negative
        )
        loss = loss + F.cosine_embedding_loss(
            normal, adapted_abnormal, negative
        )
        loss = loss + self.triplet(adapted_normal, normal, abnormal)
        loss = loss + self.config.structure_weight * self.structure_triplet(
            adapted_structure, normal_structure, abnormal_structure
        )
        return loss

    @torch.no_grad()
    def anomaly_scores(
        self,
        feature_embedding: torch.Tensor,
        structure_embedding: torch.Tensor,
        feature_patterns: list[torch.Tensor],
        structure_patterns: list[torch.Tensor],
        *,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = []
        domains = []
        for start in range(0, feature_embedding.shape[0], chunk_size):
            stop = min(start + chunk_size, feature_embedding.shape[0])
            feature_query = feature_embedding[start:stop]
            structure_query = structure_embedding[start:stop]
            domain = self.domain_similarity(
                structure_patterns, structure_query
            )
            adapted_feature = self.cross_attention(
                feature_query,
                feature_patterns,
                domain,
                self.feature_query,
                self.feature_key,
            )
            adapted_structure = self.cross_attention(
                structure_query,
                structure_patterns,
                domain,
                self.structure_query,
                self.structure_key,
            )
            score = torch.linalg.vector_norm(
                adapted_feature - feature_query, dim=1
            )
            score = score + self.config.structure_weight * torch.linalg.vector_norm(
                adapted_structure - structure_query, dim=1
            )
            scores.append(score)
            domains.append(domain)
        return torch.cat(scores), torch.cat(domains, dim=1)


def sample_normal_indices(
    labels: torch.Tensor,
    count: int,
) -> list[int]:
    candidates = torch.nonzero(labels == 0, as_tuple=False).reshape(-1).tolist()
    if len(candidates) <= count:
        return candidates
    return random.sample(candidates, count)


def sample_target_indices(node_count: int, count: int) -> list[int]:
    if node_count < count:
        raise ValueError(f"target has only {node_count} nodes")
    return random.sample(list(range(node_count)), count)


__all__ = [
    "OWLEYEConfig",
    "OWLEYEGraph",
    "OWLEYEModel",
    "effective_normalization",
    "load_official_features",
    "prepare_owleye_graph",
    "sample_normal_indices",
    "sample_target_indices",
]
