"""Current-PyTorch implementations of the eight locked OFO baselines."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCN


class SparseGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adjacency, self.linear(features))


class GCNClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 32) -> None:
        super().__init__()
        self.conv1 = SparseGraphConv(in_features, hidden_features)
        self.conv2 = SparseGraphConv(hidden_features, 2)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv1(adjacency, features))
        return self.conv2(adjacency, hidden)


class GATClassifier(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int = 32, heads: int = 2
    ) -> None:
        super().__init__()
        self.conv1 = GATConv(
            in_features,
            hidden_features,
            heads=heads,
            concat=True,
            dropout=0.0,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            hidden_features * heads,
            2,
            heads=1,
            concat=False,
            dropout=0.0,
            add_self_loops=True,
        )

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = F.elu(self.conv1(features, edge_index))
        return self.conv2(hidden, edge_index)


def beta_wavelet_coefficients(order: int) -> list[list[float]]:
    coefficients = []
    for index in range(order + 1):
        left = np.polynomial.polynomial.polypow([0.0, 0.5], index)
        right = np.polynomial.polynomial.polypow(
            [1.0, -0.5], order - index
        )
        polynomial = np.polynomial.polynomial.polymul(left, right)
        polynomial = polynomial / scipy.special.beta(
            index + 1, order + 1 - index
        )
        padded = np.zeros(order + 1, dtype=np.float64)
        padded[: polynomial.shape[0]] = polynomial
        coefficients.append(padded.tolist())
    return coefficients


def polynomial_filter(
    features: torch.Tensor,
    normalized_adjacency: torch.Tensor,
    coefficients: Iterable[float],
) -> torch.Tensor:
    coefficients = tuple(coefficients)
    value = features
    output = coefficients[0] * value
    for coefficient in coefficients[1:]:
        value = value - torch.sparse.mm(normalized_adjacency, value)
        output = output + coefficient * value
    return output


class BWGNNClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 32,
        order: int = 2,
    ) -> None:
        super().__init__()
        self.input_linear = nn.Linear(in_features, hidden_features)
        self.hidden_linear = nn.Linear(hidden_features, hidden_features)
        self.filters = beta_wavelet_coefficients(order)
        self.output = nn.Sequential(
            nn.Linear(hidden_features * len(self.filters), hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 2),
        )

    def forward(
        self, features: torch.Tensor, normalized_adjacency: torch.Tensor
    ) -> torch.Tensor:
        hidden = F.relu(self.input_linear(features))
        hidden = F.relu(self.hidden_linear(hidden))
        filtered = [
            polynomial_filter(hidden, normalized_adjacency, coefficients)
            for coefficients in self.filters
        ]
        return self.output(torch.cat(filtered, dim=1))


class DOMINANTModel(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int = 64, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.enc1 = SparseGraphConv(in_features, hidden_features)
        self.enc2 = SparseGraphConv(hidden_features, hidden_features)
        self.attr1 = SparseGraphConv(hidden_features, hidden_features)
        self.attr2 = SparseGraphConv(hidden_features, in_features)
        self.struct = SparseGraphConv(hidden_features, hidden_features)
        self.dropout = dropout

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.enc1(adjacency, features))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = F.relu(self.enc2(adjacency, hidden))

        reconstructed = F.relu(self.attr1(adjacency, hidden))
        reconstructed = F.dropout(
            reconstructed, self.dropout, training=self.training
        )
        reconstructed = F.relu(self.attr2(adjacency, reconstructed))

        structure_latent = F.relu(self.struct(adjacency, hidden))
        structure_latent = F.dropout(
            structure_latent, self.dropout, training=self.training
        )
        return reconstructed, structure_latent


def exact_dot_structure_error(
    latent: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact rowwise ||ZZ^T-A||_2 without forming ZZ^T."""

    if edge_weight is None:
        edge_weight = torch.ones(
            edge_index.shape[1], device=latent.device, dtype=latent.dtype
        )
    source, target = edge_index
    gram = latent.T @ latent
    predicted_square_sum = torch.sum((latent @ gram) * latent, dim=1)
    edge_dot = torch.sum(latent[source] * latent[target], dim=1)
    cross = torch.zeros(
        latent.shape[0], device=latent.device, dtype=latent.dtype
    )
    target_square = torch.zeros_like(cross)
    cross.scatter_add_(0, source, edge_dot * edge_weight)
    target_square.scatter_add_(0, source, edge_weight.square())
    squared = predicted_square_sum - 2.0 * cross + target_square
    return torch.sqrt(torch.clamp(squared, min=1e-12))


def dominant_scores(
    features: torch.Tensor,
    reconstructed: torch.Tensor,
    latent: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    alpha: float = 0.8,
) -> torch.Tensor:
    attribute = torch.sqrt(
        torch.clamp(torch.sum((reconstructed - features).square(), dim=1), 1e-12)
    )
    structure = exact_dot_structure_error(latent, edge_index, edge_weight)
    return alpha * attribute + (1.0 - alpha) * structure


class AnomalyDAEModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        node_count: int,
        embedding_features: int = 64,
        hidden_features: int = 64,
    ) -> None:
        super().__init__()
        self.node_count = node_count
        self.structure_input = nn.Linear(in_features, embedding_features)
        self.structure_attention = GATConv(
            embedding_features,
            hidden_features,
            heads=1,
            concat=False,
            add_self_loops=True,
        )
        self.attribute_input = nn.Linear(node_count, embedding_features)
        self.attribute_hidden = nn.Linear(
            embedding_features, hidden_features
        )

    def encode(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_hidden = F.relu(self.structure_input(features))
        node_embedding = self.structure_attention(node_hidden, edge_index)
        attribute_hidden = F.relu(self.attribute_input(features.T))
        attribute_embedding = self.attribute_hidden(attribute_hidden)
        return node_embedding, attribute_embedding

    def forward(
        self, features: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_embedding, attribute_embedding = self.encode(
            features, edge_index
        )
        reconstructed = node_embedding @ attribute_embedding.T
        return reconstructed, node_embedding


def weighted_attribute_error(
    features: torch.Tensor,
    reconstructed: torch.Tensor,
    positive_penalty: float,
) -> torch.Tensor:
    multiplier = torch.where(
        features != 0,
        torch.as_tensor(
            positive_penalty, device=features.device, dtype=features.dtype
        ),
        torch.ones((), device=features.device, dtype=features.dtype),
    )
    return torch.sqrt(
        torch.clamp(
            torch.sum(((reconstructed - features) * multiplier).square(), dim=1),
            min=1e-12,
        )
    )


def sampled_sigmoid_structure_error(
    latent: torch.Tensor,
    positive_edge_index: torch.Tensor,
    negative_edge_index: torch.Tensor,
    negative_row_weight: torch.Tensor,
    positive_penalty: float,
) -> torch.Tensor:
    node_count = latent.shape[0]
    positive_source, positive_target = positive_edge_index
    negative_source, negative_target = negative_edge_index
    positive_prediction = torch.sigmoid(
        torch.sum(latent[positive_source] * latent[positive_target], dim=1)
    )
    negative_prediction = torch.sigmoid(
        torch.sum(latent[negative_source] * latent[negative_target], dim=1)
    )
    row_squared = torch.zeros(
        node_count, device=latent.device, dtype=latent.dtype
    )
    row_squared.scatter_add_(
        0,
        positive_source,
        ((positive_prediction - 1.0) * positive_penalty).square(),
    )
    row_squared.scatter_add_(
        0,
        negative_source,
        negative_prediction.square()
        * negative_row_weight[negative_source],
    )
    return torch.sqrt(torch.clamp(row_squared, min=1e-12))


class CoLAModel(nn.Module):
    """PyGOD 1.1 CoLABase architecture with explicit full-graph control."""

    def __init__(
        self, in_features: int, hidden_features: int = 64, num_layers: int = 4
    ) -> None:
        super().__init__()
        self.encoder = GCN(
            in_channels=in_features,
            hidden_channels=hidden_features,
            num_layers=num_layers,
            out_channels=hidden_features,
            dropout=0.0,
            act=F.relu,
        )
        self.discriminator = nn.Bilinear(in_features, hidden_features, 1)

    def logits(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        permutation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(features, edge_index)
        positive = self.discriminator(features, embedding).squeeze(1)
        negative = self.discriminator(
            features[permutation], embedding
        ).squeeze(1)
        return positive, negative


class ADAGADAutoencoder(nn.Module):
    def __init__(
        self, in_features: int, hidden_features: int = 32
    ) -> None:
        super().__init__()
        self.enc1 = SparseGraphConv(in_features, hidden_features)
        self.enc2 = SparseGraphConv(hidden_features, hidden_features)
        self.attr1 = SparseGraphConv(hidden_features, hidden_features)
        self.attr2 = SparseGraphConv(hidden_features, in_features)
        self.struct = nn.Linear(hidden_features, hidden_features)

    def encode(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        hidden = F.relu(self.enc1(adjacency, features))
        return F.relu(self.enc2(adjacency, hidden))

    def decode(
        self, hidden: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reconstructed = F.relu(self.attr1(adjacency, hidden))
        reconstructed = self.attr2(adjacency, reconstructed)
        latent = F.relu(self.struct(hidden))
        return reconstructed, latent

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode(self.encode(features, adjacency), adjacency)

    def encoder_state(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        return {
            key: value.detach().clone()
            for key, value in state.items()
            if key.startswith("enc")
        }

    def load_encoder_state(
        self, state: dict[str, torch.Tensor], *, freeze: bool
    ) -> None:
        own = self.state_dict()
        own.update(state)
        self.load_state_dict(own)
        for name, parameter in self.named_parameters():
            if name.startswith("enc"):
                parameter.requires_grad = not freeze


def average_states(
    states: list[dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("No states to average")
    return {
        key: torch.stack([state[key] for state in states]).mean(dim=0)
        for key in states[0]
    }


def label_free_candidate_score(
    features: torch.Tensor, adjacency: torch.Tensor
) -> torch.Tensor:
    neighbor_mean = torch.sparse.mm(adjacency, features)
    feature_deviation = torch.linalg.vector_norm(
        features - neighbor_mean, dim=1
    )
    degree_proxy = torch.sparse.sum(adjacency, dim=1).to_dense()

    def rank_unit(value: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(value)
        ranks = torch.empty_like(value)
        ranks[order] = torch.arange(
            value.shape[0], device=value.device, dtype=value.dtype
        )
        return ranks / max(value.shape[0] - 1, 1)

    return rank_unit(feature_deviation) + rank_unit(degree_proxy)
