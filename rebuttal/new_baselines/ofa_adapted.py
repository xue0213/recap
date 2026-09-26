"""Source-only OFA adaptations of GUIDE and DiffGAD.

The released methods are target-specific OFO detectors. This module supplies a
strict, explicitly adapted comparison: one model is trained on the four RECAP
source graphs and frozen before every target score is produced.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rebuttal.baselines.baseline_common import (
    LabelVault,
    atomic_json,
    atomic_npz,
    atomic_torch_save,
    environment_metadata,
    set_seed,
    sha256_array,
    sha256_file,
)
from rebuttal.baselines.baseline_protocol import DATASETS, SETTINGS
from rebuttal.new_baselines.common import (
    current_rss_mb,
    edge_index_from_adjacency,
    gpu_memory,
    load_raw_graph,
    normalized_adjacency_with_loops,
    reload_tolerance,
    score_metrics,
    utc_now,
)
from rebuttal.new_baselines.diffgad import (
    DiffGADAutoencoder,
    DiffGADConfig,
    DiffusionDetector,
    forward_noise_schedule,
    guided_sample,
    joint_reconstruction_score,
    update_prototype,
)
from rebuttal.new_baselines.guide import (
    GUIDEConfig,
    GUIDEModel,
    cached_guide_motifs,
    guide_score,
    minmax_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    PROJECT_ROOT / "rebuttal" / "OFA_ADAPTED_UNSUPERVISED_PROTOCOL.md"
)
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_VENDOR_ROOT = Path("/root/autodl-tmp/recap_three_baselines/vendor")
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "rebuttal" / "artifacts" / "ofa_adapted_unsupervised"
)
ALIGNMENT_VERSION = "robust_pca_post_zscore_v1"
ORCA_COMMIT = "e146a8b1a99a90f5e3096b7bcc2ab0ea246c3ca7"
METHODS = ("GUIDE-OFA-adapted", "DiffGAD-OFA-adapted")
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class AdaptedRunSpec:
    method: str
    setting: str
    seed: int
    source_graphs: tuple[str, ...]
    target_graphs: tuple[str, ...]

    @property
    def run_id(self) -> str:
        method = self.method.lower().replace("-", "_")
        return f"ofa_{self.setting.lower()}__{method}__seed{self.seed}"

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, **asdict(self)}


@dataclass
class GuideGraph:
    name: str
    attributes: torch.Tensor
    motifs: torch.Tensor
    adjacency: torch.Tensor
    node_count: int
    raw_sha256: str
    aligned_sha256: str
    motif_sha256: str
    motif_cache_path: str


@dataclass
class DiffGraph:
    name: str
    x: torch.Tensor
    edge_index: torch.Tensor
    node_count: int
    raw_sha256: str
    aligned_sha256: str


def build_manifest() -> list[AdaptedRunSpec]:
    runs = []
    for method in METHODS:
        for setting in ("A", "B", "C"):
            definition = SETTINGS[setting]
            for seed in SEEDS:
                runs.append(
                    AdaptedRunSpec(
                        method=method,
                        setting=setting,
                        seed=seed,
                        source_graphs=tuple(definition["sources"]),
                        target_graphs=tuple(definition["targets"]),
                    )
                )
    validate_manifest(runs)
    return runs


def validate_manifest(runs: list[AdaptedRunSpec]) -> None:
    if len(runs) != 18 or len({run.run_id for run in runs}) != 18:
        raise ValueError("adapted OFA manifest must contain 18 unique runs")
    for run in runs:
        if run.method not in METHODS or run.seed not in SEEDS:
            raise ValueError(f"invalid run: {run}")
        expected = SETTINGS[run.setting]
        if run.source_graphs != tuple(expected["sources"]):
            raise ValueError(f"{run.run_id}: source split drift")
        if run.target_graphs != tuple(expected["targets"]):
            raise ValueError(f"{run.run_id}: target split drift")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stable_inference_seed(spec: AdaptedRunSpec, target: str) -> int:
    payload = f"{spec.run_id}:{target}:frozen_inference_v1".encode()
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


def aligned_cache_path(dataset_dir: Path, name: str) -> Path:
    return dataset_dir / f"{name}_32.npz"


def load_aligned_features(
    dataset_dir: Path, name: str
) -> tuple[np.ndarray, str]:
    path = aligned_cache_path(dataset_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"missing frozen alignment cache: {path}")
    # The archive also contains an object-valued raw MAT payload. allow_pickle
    # remains false and only the numeric feature/version members are accessed.
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive["feat"], dtype=np.float32)
        version = str(archive["alignment_version"].item())
    if version != ALIGNMENT_VERSION:
        raise ValueError(f"{name}: alignment version {version!r}")
    if features.ndim != 2 or features.shape[1] != 32:
        raise ValueError(f"{name}: invalid aligned shape {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{name}: non-finite aligned features")
    return features, sha256_file(path)


def prepare_guide_graph(
    *,
    dataset_dir: Path,
    vendor_root: Path,
    name: str,
    device: torch.device,
) -> GuideGraph:
    graph = load_raw_graph(dataset_dir, name, undirected=True)
    aligned, aligned_hash = load_aligned_features(dataset_dir, name)
    if aligned.shape[0] != graph.node_count:
        raise ValueError(f"{name}: aligned feature node mismatch")
    motifs, motif_path, _ = cached_guide_motifs(
        graph.adjacency,
        orca_binary=vendor_root / "bin" / "orca",
        cache_dir=vendor_root / "guide_motif_cache",
        orca_commit=ORCA_COMMIT,
    )
    attributes_np = minmax_columns(aligned)
    motifs_np = minmax_columns(motifs)
    return GuideGraph(
        name=name,
        attributes=torch.from_numpy(attributes_np).to(device),
        motifs=torch.from_numpy(motifs_np).to(device),
        adjacency=normalized_adjacency_with_loops(graph.adjacency, device),
        node_count=graph.node_count,
        raw_sha256=graph.raw_sha256,
        aligned_sha256=aligned_hash,
        motif_sha256=sha256_array(motifs_np),
        motif_cache_path=str(motif_path),
    )


def prepare_diff_graph(
    *, dataset_dir: Path, name: str, device: torch.device
) -> DiffGraph:
    graph = load_raw_graph(dataset_dir, name, undirected=True)
    aligned, aligned_hash = load_aligned_features(dataset_dir, name)
    if aligned.shape[0] != graph.node_count:
        raise ValueError(f"{name}: aligned feature node mismatch")
    return DiffGraph(
        name=name,
        x=torch.from_numpy(aligned).to(device),
        edge_index=edge_index_from_adjacency(graph.adjacency, device),
        node_count=graph.node_count,
        raw_sha256=graph.raw_sha256,
        aligned_sha256=aligned_hash,
    )


def graph_provenance(graph: GuideGraph | DiffGraph) -> dict[str, Any]:
    output = {
        "nodes": graph.node_count,
        "raw_sha256": graph.raw_sha256,
        "aligned_sha256": graph.aligned_sha256,
        "alignment_version": ALIGNMENT_VERSION,
    }
    if isinstance(graph, GuideGraph):
        output.update(
            {
                "motif_sha256": graph.motif_sha256,
                "motif_cache_path": graph.motif_cache_path,
            }
        )
    return output


def train_guide_sources(
    sources: list[GuideGraph],
    *,
    config: GUIDEConfig,
    epochs: int,
    device: torch.device,
) -> tuple[GUIDEModel, list[dict[str, Any]]]:
    model = GUIDEModel(32, 6, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trace = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = []
        for graph in sources:
            attributes_hat, motifs_hat = model(
                graph.attributes, graph.motifs, graph.adjacency
            )
            score, _, _ = guide_score(
                graph.attributes,
                attributes_hat,
                graph.motifs,
                motifs_hat,
                config.attribute_weight,
            )
            source_loss = score.mean()
            (source_loss / len(sources)).backward()
            values.append(float(source_loss.detach().cpu()))
        optimizer.step()
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 20 == 0:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "source_macro_loss": float(np.mean(values)),
                    "source_losses": {
                        graph.name: value
                        for graph, value in zip(sources, values)
                    },
                }
            )
    return model, trace


@torch.no_grad()
def guide_target_scores(
    model: GUIDEModel, graph: GuideGraph, config: GUIDEConfig
) -> np.ndarray:
    model.eval()
    attributes_hat, motifs_hat = model(
        graph.attributes, graph.motifs, graph.adjacency
    )
    score, _, _ = guide_score(
        graph.attributes,
        attributes_hat,
        graph.motifs,
        motifs_hat,
        config.attribute_weight,
    )
    return score.detach().cpu().numpy().astype(np.float32)


def train_diffgad_autoencoder(
    sources: list[DiffGraph],
    *,
    config: DiffGADConfig,
    epochs: int,
    device: torch.device,
) -> tuple[DiffGADAutoencoder, list[dict[str, Any]]]:
    model = DiffGADAutoencoder(32, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.autoencoder_lr,
        weight_decay=config.autoencoder_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )
    trace = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = []
        for graph in sources:
            x_hat, structure_embedding, _ = model(graph.x, graph.edge_index)
            score = joint_reconstruction_score(
                graph.x,
                x_hat,
                structure_embedding,
                graph.edge_index,
                config.attribute_weight,
            )
            source_loss = score.mean()
            (source_loss / len(sources)).backward()
            values.append(float(source_loss.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 25 == 0:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "source_macro_loss": float(np.mean(values)),
                    "source_losses": {
                        graph.name: value
                        for graph, value in zip(sources, values)
                    },
                }
            )
    return model, trace


@torch.no_grad()
def source_embeddings(
    model: DiffGADAutoencoder, sources: list[DiffGraph]
) -> list[torch.Tensor]:
    model.eval()
    return [
        model.encode(graph.x, graph.edge_index).detach()
        for graph in sources
    ]


def train_multisource_diffusion(
    embeddings: list[torch.Tensor],
    source_names: list[str],
    *,
    config: DiffGADConfig,
    conditional: bool,
    fixed_prototype: torch.Tensor | None,
    epochs: int,
) -> tuple[DiffusionDetector, torch.Tensor | None, list[dict[str, Any]]]:
    device = embeddings[0].device
    detector = DiffusionDetector(config.hidden_dim, config.diffusion_dim).to(
        device
    )
    optimizer = torch.optim.Adam(
        detector.parameters(),
        lr=config.diffusion_lr,
        weight_decay=config.diffusion_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )
    prototypes = [embedding.mean(dim=0).detach() for embedding in embeddings]
    reconstructions: list[torch.Tensor | None] = [None] * len(embeddings)
    best_loss = float("inf")
    best_state = copy.deepcopy(detector.state_dict())
    best_prototypes = [prototype.clone() for prototype in prototypes]
    patience = 0
    trace = []
    for epoch in range(epochs):
        detector.train()
        if not conditional and epoch > 0:
            prototypes = [
                update_prototype(prototype, reconstructed)
                for prototype, reconstructed in zip(prototypes, reconstructions)
                if reconstructed is not None
            ]
            if len(prototypes) != len(embeddings):
                raise RuntimeError("missing source reconstruction prototype")
        optimizer.zero_grad(set_to_none=True)
        values = []
        next_reconstructions = []
        for embedding in embeddings:
            loss, reconstructed = detector.training_loss(
                embedding,
                prototype=fixed_prototype if conditional else None,
                prototype_alpha=(
                    config.prototype_alpha if conditional else None
                ),
            )
            (loss / len(embeddings)).backward()
            values.append(float(loss.detach().cpu()))
            next_reconstructions.append(reconstructed.detach())
        reconstructions = next_reconstructions
        torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        macro = float(np.mean(values))
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 25 == 0:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "source_macro_loss": macro,
                    "source_losses": dict(zip(source_names, values)),
                }
            )
        if macro < best_loss:
            best_loss = macro
            best_state = copy.deepcopy(detector.state_dict())
            best_prototypes = [prototype.clone() for prototype in prototypes]
            patience = 0
        else:
            patience += 1
            if patience >= config.diffusion_patience:
                trace.append(
                    {
                        "epoch": epoch + 1,
                        "event": "source_macro_training_loss_early_stop",
                        "best_loss": best_loss,
                    }
                )
                break
    detector.load_state_dict(best_state)
    if conditional:
        return detector, None, trace
    transferable = torch.stack(best_prototypes).mean(dim=0).detach()
    return detector, transferable, trace


@torch.no_grad()
def diffgad_target_scores(
    *,
    autoencoder: DiffGADAutoencoder,
    unconditional: DiffusionDetector,
    conditional: DiffusionDetector,
    prototype: torch.Tensor,
    graph: DiffGraph,
    config: DiffGADConfig,
    inference_seed: int,
    smoke: bool,
) -> np.ndarray:
    autoencoder.eval()
    unconditional.eval()
    conditional.eval()
    set_seed(inference_seed)
    embedding = autoencoder.encode(graph.x, graph.edge_index)
    base_noise = torch.randn_like(embedding)
    sqrt_alpha, sqrt_one_minus = forward_noise_schedule(
        config.forward_timesteps, graph.x.device
    )
    grid = config.inference_grid[:1] if smoke else config.inference_grid
    reverse_steps = min(3, config.reverse_steps) if smoke else config.reverse_steps
    scores = []
    for timestep in grid:
        noisy = (
            sqrt_alpha[timestep] * embedding
            + sqrt_one_minus[timestep] * base_noise
        )
        reconstructed = guided_sample(
            conditional.denoiser,
            unconditional.denoiser,
            noisy,
            num_steps=reverse_steps,
            prototype=prototype,
            prototype_alpha=config.prototype_alpha,
            guidance_weight=config.guidance_weight,
        )
        x_hat, structure_embedding = autoencoder.decode(
            reconstructed, graph.edge_index
        )
        score = joint_reconstruction_score(
            graph.x,
            x_hat,
            structure_embedding,
            graph.edge_index,
            config.attribute_weight,
        )
        scores.append(score.detach().cpu())
    return torch.stack(scores).mean(dim=0).numpy().astype(np.float32)


def save_and_freeze_score(
    *,
    directory: Path,
    target: str,
    scores: np.ndarray,
    vault: LabelVault,
) -> tuple[Path, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError(f"{target}: non-finite target scores")
    mask = np.ones(scores.shape[0], dtype=np.bool_)
    path = directory / "scores" / f"{target}.npz"
    atomic_npz(path, scores=scores, evaluation_mask=mask)
    vault.mark_score_frozen(
        target,
        score_path=path,
        score_sha256=sha256_array(scores),
        query_mask_sha256=sha256_array(mask),
    )
    return path, mask


def run_directory(output_root: Path, spec: AdaptedRunSpec) -> Path:
    return output_root / "runs" / spec.run_id


def run_guide(
    spec: AdaptedRunSpec,
    *,
    dataset_dir: Path,
    vendor_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_epochs: int | None,
    target_limit: int | None,
) -> dict[str, Any]:
    directory = run_directory(output_root, spec)
    complete_path = directory / "complete.json"
    if complete_path.exists() and smoke_epochs is None:
        return json.loads(complete_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    targets = (
        spec.target_graphs
        if target_limit is None
        else spec.target_graphs[:target_limit]
    )
    config = GUIDEConfig()
    epochs = smoke_epochs or config.epochs
    started = time.perf_counter()
    metadata = {
        "format": "recap_ofa_adapted_run_v1",
        "run": spec.to_dict(),
        "effective_targets": list(targets),
        "started_at": utc_now(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "environment": environment_metadata(device),
        "resolved_config": {
            **asdict(config),
            "effective_epochs": epochs,
            "alignment_version": ALIGNMENT_VERSION,
            "source_graph_weighting": "equal_graph_macro",
            "smoke": smoke_epochs is not None,
        },
    }
    atomic_json(directory / "run_start.json", metadata)
    vault = LabelVault(dataset_dir)
    preparation_started = time.perf_counter()
    sources = [
        prepare_guide_graph(
            dataset_dir=dataset_dir,
            vendor_root=vendor_root,
            name=name,
            device=device,
        )
        for name in spec.source_graphs
    ]
    source_provenance = {
        graph.name: graph_provenance(graph) for graph in sources
    }
    preparation_seconds = time.perf_counter() - preparation_started
    set_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    training_started = time.perf_counter()
    model, trace = train_guide_sources(
        sources, config=config, epochs=epochs, device=device
    )
    synchronize(device)
    training_seconds = time.perf_counter() - training_started
    atomic_json(directory / "loss_trace.json", trace)
    checkpoint = directory / "checkpoint.pt"
    atomic_torch_save(
        checkpoint,
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "spec": spec.to_dict(),
            "alignment_version": ALIGNMENT_VERSION,
        },
    )
    del sources
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    target_payload: dict[str, dict[str, Any]] = {}
    target_provenance = {}
    inference_started = time.perf_counter()
    for target in targets:
        graph = prepare_guide_graph(
            dataset_dir=dataset_dir,
            vendor_root=vendor_root,
            name=target,
            device=device,
        )
        target_provenance[target] = graph_provenance(graph)
        scores = guide_target_scores(model, graph, config)
        score_path, mask = save_and_freeze_score(
            directory=directory,
            target=target,
            scores=scores,
            vault=vault,
        )
        target_payload[target] = {
            "scores": scores,
            "mask": mask,
            "score_path": score_path,
            "score_sha256": sha256_array(scores),
        }
        del graph
    synchronize(device)
    model_inference_seconds = time.perf_counter() - inference_started

    # All targets are immutable before the first target label is read.
    metrics = {}
    for target in targets:
        labels = vault.load_target_for_evaluation(target)
        item = target_payload[target]
        metrics[target] = score_metrics(
            labels[item["mask"]], item["scores"][item["mask"]]
        )

    reloaded = GUIDEModel(32, 6, config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["model"])
    reload_differences = {}
    for target in targets:
        graph = prepare_guide_graph(
            dataset_dir=dataset_dir,
            vendor_root=vendor_root,
            name=target,
            device=device,
        )
        reload_scores = guide_target_scores(reloaded, graph, config)
        original = target_payload[target]["scores"]
        difference = float(np.max(np.abs(original - reload_scores)))
        tolerance = reload_tolerance(original)
        if difference > tolerance:
            raise ValueError(
                f"{spec.run_id}/{target}: reload {difference} > {tolerance}"
            )
        reload_differences[target] = {
            "max_abs_difference": difference,
            "tolerance": tolerance,
        }
        del graph
    result = {
        **metadata,
        "completed_at": utc_now(),
        "status": "complete",
        "source_provenance": source_provenance,
        "target_provenance": target_provenance,
        "metrics": metrics,
        "score_files": {
            target: {
                "path": str(item["score_path"]),
                "sha256": item["score_sha256"],
            }
            for target, item in target_payload.items()
        },
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "reload": reload_differences,
        "label_audit": vault.audit(),
        "timing_seconds": {
            "preprocessing_sources": preparation_seconds,
            "training": training_seconds,
            "target_inference_and_freeze": model_inference_seconds,
            "total": time.perf_counter() - started,
        },
        "resources": {**gpu_memory(device), "peak_rss_mb": current_rss_mb()},
        "smoke": smoke_epochs is not None,
    }
    atomic_json(complete_path, result)
    return result


def run_diffgad(
    spec: AdaptedRunSpec,
    *,
    dataset_dir: Path,
    vendor_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_epochs: int | None,
    target_limit: int | None,
) -> dict[str, Any]:
    del vendor_root
    directory = run_directory(output_root, spec)
    complete_path = directory / "complete.json"
    if complete_path.exists() and smoke_epochs is None:
        return json.loads(complete_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    targets = (
        spec.target_graphs
        if target_limit is None
        else spec.target_graphs[:target_limit]
    )
    config = DiffGADConfig()
    ae_epochs = smoke_epochs or config.autoencoder_epochs
    diffusion_epochs = smoke_epochs or config.diffusion_epochs
    started = time.perf_counter()
    metadata = {
        "format": "recap_ofa_adapted_run_v1",
        "run": spec.to_dict(),
        "effective_targets": list(targets),
        "started_at": utc_now(),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "environment": environment_metadata(device),
        "resolved_config": {
            **asdict(config),
            "effective_autoencoder_epochs": ae_epochs,
            "effective_diffusion_epochs": diffusion_epochs,
            "alignment_version": ALIGNMENT_VERSION,
            "source_graph_weighting": "equal_graph_macro",
            "prototype_aggregation": "mean_of_four_source_prototypes",
            "smoke": smoke_epochs is not None,
        },
    }
    atomic_json(directory / "run_start.json", metadata)
    vault = LabelVault(dataset_dir)
    preparation_started = time.perf_counter()
    sources = [
        prepare_diff_graph(dataset_dir=dataset_dir, name=name, device=device)
        for name in spec.source_graphs
    ]
    source_provenance = {
        graph.name: graph_provenance(graph) for graph in sources
    }
    preparation_seconds = time.perf_counter() - preparation_started
    set_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    training_started = time.perf_counter()
    autoencoder, ae_trace = train_diffgad_autoencoder(
        sources, config=config, epochs=ae_epochs, device=device
    )
    embeddings = source_embeddings(autoencoder, sources)
    names = [graph.name for graph in sources]
    unconditional, prototype, unconditional_trace = train_multisource_diffusion(
        embeddings,
        names,
        config=config,
        conditional=False,
        fixed_prototype=None,
        epochs=diffusion_epochs,
    )
    if prototype is None:
        raise RuntimeError("source-only transferable prototype was not produced")
    conditional, _, conditional_trace = train_multisource_diffusion(
        embeddings,
        names,
        config=config,
        conditional=True,
        fixed_prototype=prototype,
        epochs=diffusion_epochs,
    )
    synchronize(device)
    training_seconds = time.perf_counter() - training_started
    atomic_json(
        directory / "loss_trace.json",
        {
            "autoencoder": ae_trace,
            "unconditional_diffusion": unconditional_trace,
            "conditional_diffusion": conditional_trace,
        },
    )
    checkpoint = directory / "checkpoint.pt"
    atomic_torch_save(
        checkpoint,
        {
            "autoencoder": autoencoder.state_dict(),
            "unconditional": unconditional.state_dict(),
            "conditional": conditional.state_dict(),
            "prototype": prototype.detach().cpu(),
            "config": asdict(config),
            "spec": spec.to_dict(),
            "alignment_version": ALIGNMENT_VERSION,
        },
    )
    del sources, embeddings
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    target_payload: dict[str, dict[str, Any]] = {}
    target_provenance = {}
    inference_started = time.perf_counter()
    for target in targets:
        graph = prepare_diff_graph(
            dataset_dir=dataset_dir, name=target, device=device
        )
        target_provenance[target] = graph_provenance(graph)
        inference_seed = stable_inference_seed(spec, target)
        scores = diffgad_target_scores(
            autoencoder=autoencoder,
            unconditional=unconditional,
            conditional=conditional,
            prototype=prototype,
            graph=graph,
            config=config,
            inference_seed=inference_seed,
            smoke=smoke_epochs is not None,
        )
        score_path, mask = save_and_freeze_score(
            directory=directory,
            target=target,
            scores=scores,
            vault=vault,
        )
        target_payload[target] = {
            "scores": scores,
            "mask": mask,
            "score_path": score_path,
            "score_sha256": sha256_array(scores),
            "inference_seed": inference_seed,
        }
        del graph
    synchronize(device)
    model_inference_seconds = time.perf_counter() - inference_started

    metrics = {}
    for target in targets:
        labels = vault.load_target_for_evaluation(target)
        item = target_payload[target]
        metrics[target] = score_metrics(
            labels[item["mask"]], item["scores"][item["mask"]]
        )

    reload_autoencoder = DiffGADAutoencoder(32, config).to(device)
    reload_unconditional = DiffusionDetector(
        config.hidden_dim, config.diffusion_dim
    ).to(device)
    reload_conditional = DiffusionDetector(
        config.hidden_dim, config.diffusion_dim
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reload_autoencoder.load_state_dict(payload["autoencoder"])
    reload_unconditional.load_state_dict(payload["unconditional"])
    reload_conditional.load_state_dict(payload["conditional"])
    reload_prototype = payload["prototype"].to(device)
    reload_differences = {}
    for target in targets:
        graph = prepare_diff_graph(
            dataset_dir=dataset_dir, name=target, device=device
        )
        item = target_payload[target]
        reload_scores = diffgad_target_scores(
            autoencoder=reload_autoencoder,
            unconditional=reload_unconditional,
            conditional=reload_conditional,
            prototype=reload_prototype,
            graph=graph,
            config=config,
            inference_seed=item["inference_seed"],
            smoke=smoke_epochs is not None,
        )
        original = item["scores"]
        difference = float(np.max(np.abs(original - reload_scores)))
        tolerance = reload_tolerance(original)
        if difference > tolerance:
            raise ValueError(
                f"{spec.run_id}/{target}: reload {difference} > {tolerance}"
            )
        reload_differences[target] = {
            "max_abs_difference": difference,
            "tolerance": tolerance,
        }
        del graph
    result = {
        **metadata,
        "completed_at": utc_now(),
        "status": "complete",
        "source_provenance": source_provenance,
        "target_provenance": target_provenance,
        "metrics": metrics,
        "score_files": {
            target: {
                "path": str(item["score_path"]),
                "sha256": item["score_sha256"],
                "inference_seed": item["inference_seed"],
            }
            for target, item in target_payload.items()
        },
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "reload": reload_differences,
        "label_audit": vault.audit(),
        "timing_seconds": {
            "preprocessing_sources": preparation_seconds,
            "training": training_seconds,
            "target_inference_and_freeze": model_inference_seconds,
            "total": time.perf_counter() - started,
        },
        "resources": {**gpu_memory(device), "peak_rss_mb": current_rss_mb()},
        "smoke": smoke_epochs is not None,
    }
    atomic_json(complete_path, result)
    return result


def run_spec(
    spec: AdaptedRunSpec,
    *,
    dataset_dir: Path,
    vendor_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_epochs: int | None,
    target_limit: int | None,
) -> dict[str, Any]:
    arguments = dict(
        dataset_dir=dataset_dir,
        vendor_root=vendor_root,
        output_root=output_root,
        device=device,
        smoke_epochs=smoke_epochs,
        target_limit=target_limit,
    )
    if spec.method == "GUIDE-OFA-adapted":
        return run_guide(spec, **arguments)
    if spec.method == "DiffGAD-OFA-adapted":
        return run_diffgad(spec, **arguments)
    raise ValueError(spec.method)


def preflight(dataset_dir: Path, vendor_root: Path) -> dict[str, Any]:
    manifest = build_manifest()
    datasets = {}
    for name in DATASETS:
        graph = load_raw_graph(dataset_dir, name, undirected=True)
        features, aligned_hash = load_aligned_features(dataset_dir, name)
        if graph.node_count != features.shape[0]:
            raise ValueError(f"{name}: preflight node mismatch")
        datasets[name] = {
            "nodes": graph.node_count,
            "raw_feature_dim": graph.feature_count,
            "aligned_feature_dim": features.shape[1],
            "raw_sha256": graph.raw_sha256,
            "aligned_sha256": aligned_hash,
        }
    for required in (
        vendor_root / "bin" / "orca",
        vendor_root / "archives" / "guide.tar.gz",
        vendor_root / "archives" / "diffgad.tar.gz",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    return {
        "format": "recap_ofa_adapted_preflight_v1",
        "passed": True,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "manifest_runs": len(manifest),
        "evaluations": sum(len(run.target_graphs) for run in manifest),
        "datasets": datasets,
    }


def _fmt(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} +/- {100 * std:.2f}"


def analyze(
    *,
    dataset_dir: Path,
    output_root: Path,
    recap_macro_path: Path,
) -> dict[str, Any]:
    expected = build_manifest()
    records = []
    problems = []
    max_metric_difference = 0.0
    max_reload_difference = 0.0
    for spec in expected:
        path = run_directory(output_root, spec) / "complete.json"
        if not path.exists():
            problems.append(f"missing {spec.run_id}")
            continue
        result = json.loads(path.read_text())
        if result.get("status") != "complete" or result.get("smoke"):
            problems.append(f"invalid completion {spec.run_id}")
            continue
        audit = result.get("label_audit", {})
        events = audit.get("events", [])
        actions = [event.get("action") for event in events]
        expected_actions = (
            ["mark_target_score_frozen"] * len(spec.target_graphs)
            + ["load_target_labels_for_metric"] * len(spec.target_graphs)
        )
        if not audit.get("passed") or actions != expected_actions:
            problems.append(f"label audit {spec.run_id}: {actions}")
        for target in spec.target_graphs:
            score_info = result["score_files"][target]
            with np.load(score_info["path"], allow_pickle=False) as archive:
                scores = np.asarray(archive["scores"], dtype=np.float32)
                mask = np.asarray(archive["evaluation_mask"], dtype=np.bool_)
            if sha256_array(scores) != score_info["sha256"]:
                problems.append(f"score hash {spec.run_id}/{target}")
            raw = np.load  # keep labels outside model-facing preparation code
            del raw
            from rebuttal.ofo_baselines.common import load_labels

            labels = load_labels(dataset_dir, target)
            recomputed = score_metrics(labels[mask], scores[mask])
            recorded = result["metrics"][target]
            difference = max(
                abs(recomputed[key] - recorded[key]) for key in recomputed
            )
            max_metric_difference = max(max_metric_difference, difference)
            reload_value = result["reload"][target]["max_abs_difference"]
            max_reload_difference = max(max_reload_difference, reload_value)
            records.append(
                {
                    "method": spec.method,
                    "setting": spec.setting,
                    "seed": spec.seed,
                    "target": target,
                    **recomputed,
                }
            )
    if problems:
        raise RuntimeError("; ".join(problems))

    recap = {}
    with recap_macro_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["setting"] in {"A", "B", "C"} and row["aggregation"] == "dataset_macro":
                recap[row["setting"]] = {
                    "AUROC_mean": float(row["auroc_mean"]),
                    "AUROC_std": float(row["auroc_std"]),
                    "AUPRC_mean": float(row["auprc_mean"]),
                    "AUPRC_std": float(row["auprc_std"]),
                }

    macros = {}
    per_dataset = {}
    timing = {}
    for method in METHODS:
        macros[method] = {}
        per_dataset[method] = {}
        timing[method] = {}
        for setting in ("A", "B", "C"):
            subset = [
                row
                for row in records
                if row["method"] == method and row["setting"] == setting
            ]
            seed_values = {metric: [] for metric in ("AUROC", "AUPRC")}
            for seed in SEEDS:
                seed_rows = [row for row in subset if row["seed"] == seed]
                for metric in seed_values:
                    seed_values[metric].append(
                        float(np.mean([row[metric] for row in seed_rows]))
                    )
            macros[method][setting] = {
                f"{metric}_{stat}": float(
                    np.mean(values) if stat == "mean" else np.std(values, ddof=0)
                )
                for metric, values in seed_values.items()
                for stat in ("mean", "std")
            }
            per_dataset[method][setting] = {}
            for target in SETTINGS[setting]["targets"]:
                target_rows = [row for row in subset if row["target"] == target]
                per_dataset[method][setting][target] = {
                    f"{metric}_{stat}": float(
                        np.mean([row[metric] for row in target_rows])
                        if stat == "mean"
                        else np.std(
                            [row[metric] for row in target_rows], ddof=0
                        )
                    )
                    for metric in ("AUROC", "AUPRC")
                    for stat in ("mean", "std")
                }
            run_results = [
                json.loads((run_directory(output_root, spec) / "complete.json").read_text())
                for spec in expected
                if spec.method == method and spec.setting == setting
            ]
            timing[method][setting] = {
                "training_mean_seconds": float(
                    np.mean([item["timing_seconds"]["training"] for item in run_results])
                ),
                "inference_mean_seconds": float(
                    np.mean(
                        [
                            item["timing_seconds"]["target_inference_and_freeze"]
                            for item in run_results
                        ]
                    )
                ),
                "max_gpu_allocated_mb": float(
                    max(item["resources"]["allocated_mb"] for item in run_results)
                ),
            }

    report = {
        "format": "recap_ofa_adapted_analysis_v1",
        "passed": True,
        "runs": len(expected),
        "evaluations": len(records),
        "max_metric_recomputation_difference": max_metric_difference,
        "max_reload_score_difference": max_reload_difference,
        "macros": macros,
        "recap_macros": recap,
        "per_dataset": per_dataset,
        "timing": timing,
    }
    analysis_dir = output_root / "analysis"
    atomic_json(analysis_dir / "summary.json", report)
    lines = [
        "# GUIDE/DiffGAD Source-Only OFA Adaptation Report",
        "",
        "All numbers are dataset-macro AUROC/AUPRC in percent, mean +/- population standard deviation over seeds 0/1/2.",
        "",
        "| Method | Setting A | Setting B | Setting C |",
        "|---|---:|---:|---:|",
    ]
    for method in ("RECAP-OFA", *METHODS):
        cells = []
        for setting in ("A", "B", "C"):
            value = recap[setting] if method == "RECAP-OFA" else macros[method][setting]
            cells.append(
                f"{_fmt(value['AUROC_mean'], value['AUROC_std'])} / "
                f"{_fmt(value['AUPRC_mean'], value['AUPRC_std'])}"
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Runs: {len(expected)}/18",
            f"- Evaluations: {len(records)}/108",
            f"- Maximum metric recomputation difference: {max_metric_difference:.12g}",
            f"- Maximum checkpoint reload score difference: {max_reload_difference:.12g}",
            "",
        ]
    )
    for setting in ("A", "B", "C"):
        targets = SETTINGS[setting]["targets"]
        lines.extend(
            [
                f"## Setting {setting} per-target AUROC/AUPRC (%)",
                "",
                "| Method | " + " | ".join(targets) + " |",
                "|---|" + "---:|" * len(targets),
            ]
        )
        for method in METHODS:
            cells = []
            for target in targets:
                value = per_dataset[method][setting][target]
                cells.append(
                    f"{_fmt(value['AUROC_mean'], value['AUROC_std'])} / "
                    f"{_fmt(value['AUPRC_mean'], value['AUPRC_std'])}"
                )
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines.append("")
    report_path = analysis_dir / "REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run", "analyze"))
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--settings", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--recap-macro-path",
        type=Path,
        default=PROJECT_ROOT / "rebuttal" / "artifacts" / "phase1" / "analysis" / "metric_macros.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        report = preflight(args.dataset_dir, args.vendor_root)
        atomic_json(args.output_root / "preflight.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.command == "analyze":
        report = analyze(
            dataset_dir=args.dataset_dir,
            output_root=args.output_root,
            recap_macro_path=args.recap_macro_path,
        )
        print(json.dumps(report["macros"], indent=2, sort_keys=True))
        return
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selected = [
        spec
        for spec in build_manifest()
        if spec.method in args.methods
        and spec.setting in args.settings
        and spec.seed in args.seeds
    ]
    failures = []
    for index, spec in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] START {spec.run_id}", flush=True)
        try:
            result = run_spec(
                spec,
                dataset_dir=args.dataset_dir,
                vendor_root=args.vendor_root,
                output_root=args.output_root,
                device=device,
                smoke_epochs=args.smoke_epochs,
                target_limit=args.target_limit,
            )
            print(
                f"[{index}/{len(selected)}] COMPLETE {spec.run_id} "
                f"{result['metrics']}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "run_id": spec.run_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "failed_at": utc_now(),
            }
            failures.append(failure)
            directory = run_directory(args.output_root, spec)
            atomic_json(directory / "failure.json", failure)
            print(f"FAILED {spec.run_id}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    atomic_json(
        args.output_root / "progress.json",
        {
            "selected": len(selected),
            "failures": failures,
            "updated_at": utc_now(),
        },
    )
    if failures:
        raise RuntimeError(f"{len(failures)} adapted OFA runs failed")


if __name__ == "__main__":
    main()
