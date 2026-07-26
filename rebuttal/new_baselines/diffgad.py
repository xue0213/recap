"""Leakage-free, large-graph-compatible DiffGAD implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN


@dataclass(frozen=True)
class DiffGADConfig:
    hidden_dim: int = 32
    diffusion_dim: int = 64
    autoencoder_dropout: float = 0.3
    autoencoder_lr: float = 0.01
    autoencoder_weight_decay: float = 0.01
    attribute_weight: float = 0.8
    autoencoder_epochs: int = 300
    diffusion_lr: float = 0.004
    diffusion_weight_decay: float = 0.0
    diffusion_epochs: int = 800
    diffusion_patience: int = 100
    prototype_alpha: float = 0.1
    guidance_weight: float = 1.0
    forward_timesteps: int = 500
    reverse_steps: int = 50
    inference_grid: tuple[int, ...] = (
        49,
        99,
        149,
        199,
        249,
        299,
        349,
        399,
        449,
        499,
    )


class DiffGADAutoencoder(nn.Module):
    """Released four-layer GAE without materializing a dense adjacency."""

    def __init__(self, in_dim: int, config: DiffGADConfig) -> None:
        super().__init__()
        hidden = config.hidden_dim
        self.encoder = GCN(
            in_channels=in_dim,
            hidden_channels=hidden,
            num_layers=2,
            out_channels=hidden,
            dropout=config.autoencoder_dropout,
            act=F.relu,
        )
        self.attribute_decoder = GCN(
            in_channels=hidden,
            hidden_channels=hidden,
            num_layers=2,
            out_channels=in_dim,
            dropout=config.autoencoder_dropout,
            act=F.relu,
        )
        self.structure_decoder = GCN(
            in_channels=hidden,
            hidden_channels=hidden,
            num_layers=1,
            out_channels=hidden,
            dropout=config.autoencoder_dropout,
            act=F.relu,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(
        self, embedding: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_hat = self.attribute_decoder(embedding, edge_index)
        structure_embedding = self.structure_decoder(embedding, edge_index)
        return x_hat, structure_embedding

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encode(x, edge_index)
        x_hat, structure_embedding = self.decode(embedding, edge_index)
        return x_hat, structure_embedding, embedding


def exact_structure_squared_error(
    structure_embedding: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Return exact rows of ``||A - ZZ^T||_F^2`` in O(Nd² + Ed)."""

    z = structure_embedding
    gram = z.T @ z
    quadratic = torch.sum((z @ gram) * z, dim=1)
    source, target = edge_index
    edge_dot = torch.sum(z[source] * z[target], dim=1)
    edge_sum = torch.zeros(
        z.shape[0], dtype=z.dtype, device=z.device
    ).index_add_(0, source, edge_dot)
    degree = torch.zeros(
        z.shape[0], dtype=z.dtype, device=z.device
    ).index_add_(0, source, torch.ones_like(edge_dot))
    return torch.clamp(quadratic - 2.0 * edge_sum + degree, min=0.0)


def dense_structure_squared_error(
    structure_embedding: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Dense reference used only by tests."""

    node_count = structure_embedding.shape[0]
    adjacency = torch.zeros(
        (node_count, node_count),
        dtype=structure_embedding.dtype,
        device=structure_embedding.device,
    )
    adjacency[edge_index[0], edge_index[1]] = 1.0
    reconstruction = structure_embedding @ structure_embedding.T
    return torch.sum((adjacency - reconstruction) ** 2, dim=1)


def joint_reconstruction_score(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    structure_embedding: torch.Tensor,
    edge_index: torch.Tensor,
    attribute_weight: float,
) -> torch.Tensor:
    attribute_error = torch.sqrt(
        torch.clamp(torch.sum((x - x_hat) ** 2, dim=1), min=0.0)
    )
    structure_error = torch.sqrt(
        exact_structure_squared_error(structure_embedding, edge_index)
    )
    return (
        attribute_weight * attribute_error
        + (1.0 - attribute_weight) * structure_error
    )


class SinusoidalNoiseEmbedding(nn.Module):
    def __init__(self, channels: int, max_positions: int = 10_000) -> None:
        super().__init__()
        self.channels = channels
        self.max_positions = max_positions

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        frequencies = torch.arange(
            self.channels // 2,
            device=noise.device,
            dtype=torch.float32,
        )
        frequencies = frequencies / (self.channels // 2)
        frequencies = (1.0 / self.max_positions) ** frequencies
        phase = noise.float().outer(frequencies)
        return torch.cat([phase.cos(), phase.sin()], dim=1)


class DiffusionMLP(nn.Module):
    def __init__(self, input_dim: int, time_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, time_dim)
        self.noise_embedding = SinusoidalNoiseEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.prototype_projection = nn.Linear(input_dim, time_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(time_dim, 2 * time_dim),
            nn.SiLU(),
            nn.Linear(2 * time_dim, 2 * time_dim),
            nn.SiLU(),
            nn.Linear(2 * time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_label: torch.Tensor,
        prototype: torch.Tensor | None = None,
        prototype_alpha: float | None = None,
    ) -> torch.Tensor:
        time = self.noise_embedding(noise_label)
        time = time.reshape(time.shape[0], 2, -1).flip(1).reshape_as(time)
        hidden = self.input_projection(x) + self.time_mlp(time)
        if prototype is not None:
            if prototype_alpha is None:
                raise ValueError("prototype_alpha is required with prototype")
            hidden = hidden + prototype_alpha * self.prototype_projection(
                prototype
            )
        return self.denoiser(hidden)


class PreconditionedDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        sigma_data: float = 0.5,
    ) -> None:
        super().__init__()
        self.network = DiffusionMLP(input_dim, time_dim)
        self.sigma_data = sigma_data
        self.sigma_min = 0.0
        self.sigma_max = float("inf")

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        prototype: torch.Tensor | None = None,
        prototype_alpha: float | None = None,
    ) -> torch.Tensor:
        sigma = sigma.float().reshape(-1, 1)
        sigma_data_sq = self.sigma_data**2
        c_skip = sigma_data_sq / (sigma**2 + sigma_data_sq)
        c_out = sigma * self.sigma_data / torch.sqrt(
            sigma**2 + sigma_data_sq
        )
        c_in = 1.0 / torch.sqrt(sigma_data_sq + sigma**2)
        c_noise = torch.log(sigma) / 4.0
        prediction = self.network(
            c_in * x.float(),
            c_noise.reshape(-1),
            prototype,
            prototype_alpha,
        )
        return c_skip * x + c_out * prediction.float()

    @staticmethod
    def round_sigma(sigma: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(sigma)


class DiffusionDetector(nn.Module):
    def __init__(self, input_dim: int, time_dim: int) -> None:
        super().__init__()
        self.denoiser = PreconditionedDenoiser(input_dim, time_dim)

    def training_loss(
        self,
        embeddings: torch.Tensor,
        *,
        prototype: torch.Tensor | None,
        prototype_alpha: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        random_normal = torch.randn(
            embeddings.shape[0], device=embeddings.device
        )
        sigma = (random_normal * 1.2 - 1.2).exp()
        weight = (sigma**2 + 0.5**2) / (sigma * 0.5) ** 2
        noisy = embeddings + torch.randn_like(embeddings) * sigma[:, None]
        reconstructed = self.denoiser(
            noisy, sigma, prototype, prototype_alpha
        )
        elementwise = weight[:, None] * (reconstructed - embeddings) ** 2
        return elementwise.mean(), reconstructed


def forward_noise_schedule(
    timesteps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta = torch.linspace(0.0001, 0.02, timesteps, device=device)
    alpha_product = torch.cumprod(1.0 - beta, dim=0)
    return torch.sqrt(alpha_product), torch.sqrt(1.0 - alpha_product)


def _reverse_schedule(
    denoiser: PreconditionedDenoiser,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    step = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min = max(0.002, denoiser.sigma_min)
    sigma_max = min(80.0, denoiser.sigma_max)
    schedule = (
        sigma_max ** (1.0 / 7.0)
        + step / (num_steps - 1)
        * (sigma_min ** (1.0 / 7.0) - sigma_max ** (1.0 / 7.0))
    ) ** 7.0
    return torch.cat([schedule, torch.zeros_like(schedule[:1])])


@torch.no_grad()
def guided_sample(
    conditional: PreconditionedDenoiser,
    unconditional: PreconditionedDenoiser,
    initial: torch.Tensor,
    *,
    num_steps: int,
    prototype: torch.Tensor,
    prototype_alpha: float,
    guidance_weight: float,
) -> torch.Tensor:
    schedule = _reverse_schedule(unconditional, num_steps, initial.device)
    current = initial.float() * schedule[0]
    for index, (sigma_current, sigma_next) in enumerate(
        zip(schedule[:-1], schedule[1:])
    ):
        gamma = min(1.0 / num_steps, math.sqrt(2.0) - 1.0)
        sigma_hat = sigma_current * (1.0 + gamma)
        perturbed = current + torch.sqrt(
            sigma_hat**2 - sigma_current**2
        ) * torch.randn_like(current)
        denoised_conditional = conditional(
            perturbed,
            sigma_hat,
            prototype,
            prototype_alpha,
        )
        denoised_unconditional = unconditional(perturbed, sigma_hat)
        derivative = (
            (1.0 + guidance_weight)
            * (perturbed - denoised_unconditional)
            / sigma_hat
            - guidance_weight
            * (perturbed - denoised_conditional)
            / sigma_hat
        )
        candidate = perturbed + (sigma_next - sigma_hat) * derivative
        if index < num_steps - 1:
            next_conditional = conditional(
                candidate,
                sigma_next,
                prototype,
                prototype_alpha,
            )
            next_unconditional = unconditional(candidate, sigma_next)
            derivative_next = (
                (1.0 + guidance_weight)
                * (candidate - next_unconditional)
                / sigma_next
                - guidance_weight
                * (candidate - next_conditional)
                / sigma_next
            )
            candidate = perturbed + (sigma_next - sigma_hat) * (
                0.5 * derivative + 0.5 * derivative_next
            )
        current = candidate
    return current


def update_prototype(
    prototype: torch.Tensor,
    reconstructed: torch.Tensor,
) -> torch.Tensor:
    similarity = F.cosine_similarity(
        prototype.reshape(1, -1), reconstructed, dim=1, eps=1e-6
    )
    weights = torch.softmax(similarity / 5.0, dim=0).reshape(1, -1)
    return (weights @ reconstructed).reshape(-1).detach()


__all__ = [
    "DiffGADAutoencoder",
    "DiffGADConfig",
    "DiffusionDetector",
    "dense_structure_squared_error",
    "exact_structure_squared_error",
    "forward_noise_schedule",
    "guided_sample",
    "joint_reconstruction_score",
    "update_prototype",
]
