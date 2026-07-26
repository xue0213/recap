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
from torch.utils.checkpoint import checkpoint


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


class UNPromptGCN(nn.Module):
    def __init__(self, in_features: int = 8, hidden_features: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, hidden_features, bias=False)
        self.batch_norm = nn.BatchNorm1d(hidden_features)
        self.activation = nn.PReLU()
        self.bias = nn.Parameter(torch.zeros(hidden_features))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor | None
    ) -> torch.Tensor:
        output = self.linear(features)
        if adjacency is not None:
            output = torch.sparse.mm(adjacency, output)
        output = output + self.bias
        return self.activation(self.batch_norm(output))


class UNPromptGrace(nn.Module):
    def __init__(
        self,
        encoder: UNPromptGCN,
        hidden_features: int = 128,
        projection_features: int = 256,
        tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc1 = nn.Linear(hidden_features, projection_features)
        self.fc2 = nn.Linear(projection_features, hidden_features)

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        representation = self.encoder(features, adjacency)
        return self.fc2(F.elu(self.fc1(representation)))

    def exact_blocked_loss(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        *,
        block_size: int = 2048,
    ) -> torch.Tensor:
        """Exact official GRACE denominator with checkpointed row blocks."""

        first_normalized = F.normalize(first, p=2, dim=1)
        second_normalized = F.normalize(second, p=2, dim=1)
        node_count = first.shape[0]

        def block_loss(
            left: torch.Tensor,
            right: torch.Tensor,
            start_tensor: torch.Tensor,
            end_tensor: torch.Tensor,
        ) -> torch.Tensor:
            start = int(start_tensor.item())
            end = int(end_tensor.item())

            def directional(
                query: torch.Tensor,
                same: torch.Tensor,
                other: torch.Tensor,
            ) -> torch.Tensor:
                reflected = torch.exp(
                    torch.mm(query, same.T) / self.tau
                )
                between = torch.exp(
                    torch.mm(query, other.T) / self.tau
                )
                row_index = torch.arange(
                    end - start, device=query.device
                )
                column_index = torch.arange(start, end, device=query.device)
                numerator = between[row_index, column_index]
                reflected_diagonal = reflected[row_index, column_index]
                denominator = (
                    reflected.sum(dim=1)
                    + between.sum(dim=1)
                    - reflected_diagonal
                )
                return -torch.log(numerator / denominator).sum()

            loss_forward = directional(
                left[start:end], left, right
            )
            loss_reverse = directional(
                right[start:end], right, left
            )
            return 0.5 * (loss_forward + loss_reverse)

        total = first.new_zeros(())
        for start in range(0, node_count, block_size):
            end = min(start + block_size, node_count)
            start_tensor = torch.tensor(start, device=first.device)
            end_tensor = torch.tensor(end, device=first.device)
            total = total + checkpoint(
                block_loss,
                first_normalized,
                second_normalized,
                start_tensor,
                end_tensor,
                use_reentrant=False,
            )
        return total / node_count


class UNPromptPrompts(nn.Module):
    def __init__(self, in_features: int = 8, prompt_count: int = 10) -> None:
        super().__init__()
        self.prompts = nn.Parameter(torch.empty(prompt_count, in_features))
        self.attention = nn.Linear(in_features, prompt_count)
        nn.init.xavier_uniform_(self.prompts)
        self.attention.reset_parameters()

    def add(self, features: torch.Tensor) -> torch.Tensor:
        weight = F.softmax(self.attention(features), dim=1)
        return features + torch.mm(weight, self.prompts)


class UNPromptProjection(nn.Module):
    def __init__(self, hidden_features: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_features, hidden_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def unprompt_completion_loss(
    neighbor: torch.Tensor,
    self_embedding: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    neighbor = F.normalize(neighbor, p=2, dim=-1)
    self_embedding = F.normalize(self_embedding, p=2, dim=-1)
    difference = -torch.sum(neighbor * self_embedding, dim=1)
    modified = torch.where(labels == 0, difference, -difference)
    return torch.mean(modified)


def unprompt_anomaly_score(
    neighbor: torch.Tensor, self_embedding: torch.Tensor
) -> torch.Tensor:
    similarity = torch.sum(
        F.normalize(neighbor, p=2, dim=-1)
        * F.normalize(self_embedding, p=2, dim=-1),
        dim=1,
    )
    minimum = torch.min(similarity)
    maximum = torch.max(similarity)
    if float(maximum - minimum) == 0.0:
        return torch.zeros_like(similarity)
    return 1.0 - (similarity - minimum) / (maximum - minimum)


def unprompt_deterministic_neighbor_eval(
    encoder: UNPromptGCN,
    features: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    """Evaluate sparse aggregation deterministically on CPU.

    CUDA sparse reductions may change the last few bits between identical
    calls. UNPrompt's graph-level min-max score can amplify those differences.
    This function changes only the evaluation backend, not the linear map,
    normalization, activation, graph, or arithmetic expression.
    """

    if encoder.training:
        raise ValueError("Deterministic UNPrompt evaluation requires eval mode")
    projected = encoder.linear(features)
    aggregated = torch.sparse.mm(
        adjacency.cpu(), projected.cpu()
    ).to(features.device)
    output = aggregated + encoder.bias
    return encoder.activation(encoder.batch_norm(output))


class AnomalyGFMGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.PReLU()
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(
        self, features: torch.Tensor, adjacency: torch.Tensor
    ) -> torch.Tensor:
        output = torch.sparse.mm(adjacency, self.linear(features))
        return self.activation(output + self.bias)


class AnomalyGFMModel(nn.Module):
    """Sparse, batch-size-one equivalent of the released zero-shot model."""

    def __init__(self, hidden_features: int = 400) -> None:
        super().__init__()
        self.gcn1 = AnomalyGFMGraphConv(8, hidden_features)
        self.gcn2 = AnomalyGFMGraphConv(hidden_features, hidden_features)
        self.classifier = nn.Linear(hidden_features, 1, bias=False)
        self.residual_classifier = nn.Linear(
            hidden_features, 1, bias=False
        )
        self.normal_prompt = nn.Linear(
            hidden_features, hidden_features, bias=False
        )
        self.anomaly_prompt = nn.Linear(
            hidden_features, hidden_features, bias=False
        )

    def forward(
        self,
        features: torch.Tensor,
        gcn_adjacency: torch.Tensor,
        neighbor_adjacency: torch.Tensor,
        normal_prompt_raw: torch.Tensor,
        anomaly_prompt_raw: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = self.gcn1(features, gcn_adjacency)
        embedding = self.gcn2(hidden, gcn_adjacency)
        normal_prompt = F.relu(self.normal_prompt(normal_prompt_raw))
        anomaly_prompt = F.relu(self.anomaly_prompt(anomaly_prompt_raw))
        neighbor_embedding = torch.sparse.mm(
            neighbor_adjacency, embedding
        )
        residual = embedding - neighbor_embedding
        logits = self.classifier(embedding).squeeze(1)
        residual_logits = self.residual_classifier(residual).squeeze(1)
        return (
            logits,
            residual_logits,
            embedding,
            residual,
            normal_prompt,
            anomaly_prompt,
        )


def anomalygfm_score_components(
    residual: torch.Tensor,
    normal_prompt: torch.Tensor,
    anomaly_prompt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = F.normalize(residual, p=2, dim=1)
    normal_prompt = F.normalize(normal_prompt, p=2, dim=0)
    anomaly_prompt = F.normalize(anomaly_prompt, p=2, dim=0)
    normal_similarity = torch.mv(residual, normal_prompt)
    anomaly_similarity = torch.mv(residual, anomaly_prompt)
    normal_component = torch.exp(-normal_similarity)
    anomaly_component = torch.exp(anomaly_similarity)
    return anomaly_component, normal_component
