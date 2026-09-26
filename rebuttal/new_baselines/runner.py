"""Resumable runner for DiffGAD, GUIDE, and OWLEYE."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from rebuttal.new_baselines.common import (  # type: ignore
        DEFAULT_DATASET_DIR,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_VENDOR_ROOT,
        LabelVault,
        OFOLabelVault,
        PROTOCOL_PATH,
        UPSTREAM_MANIFEST_PATH,
        atomic_json,
        atomic_torch_save,
        base_metadata,
        current_rss_mb,
        dense_features,
        edge_index_from_adjacency,
        gpu_memory,
        load_raw_graph,
        normalized_adjacency_with_loops,
        reload_tolerance,
        save_target_scores,
        save_unsupervised_scores,
        score_metrics,
        set_seed,
        sha256_array,
        sha256_file,
        utc_now,
    )
    from rebuttal.new_baselines.diffgad import (  # type: ignore
        DiffGADAutoencoder,
        DiffGADConfig,
        DiffusionDetector,
        forward_noise_schedule,
        guided_sample,
        joint_reconstruction_score,
        update_prototype,
    )
    from rebuttal.new_baselines.guide import (  # type: ignore
        GUIDEConfig,
        GUIDEModel,
        cached_guide_motifs,
        guide_score,
        minmax_columns,
    )
    from rebuttal.new_baselines.owleye import (  # type: ignore
        OWLEYEConfig,
        OWLEYEModel,
        prepare_owleye_graph,
        sample_normal_indices,
        sample_target_indices,
    )
    from rebuttal.new_baselines.protocol import (  # type: ignore
        ExtensionRunSpec,
        build_manifest,
        expected_evaluations,
        validate_manifest,
    )
else:
    from .common import (
        DEFAULT_DATASET_DIR,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_VENDOR_ROOT,
        LabelVault,
        OFOLabelVault,
        PROTOCOL_PATH,
        UPSTREAM_MANIFEST_PATH,
        atomic_json,
        atomic_torch_save,
        base_metadata,
        current_rss_mb,
        dense_features,
        edge_index_from_adjacency,
        gpu_memory,
        load_raw_graph,
        normalized_adjacency_with_loops,
        reload_tolerance,
        save_target_scores,
        save_unsupervised_scores,
        score_metrics,
        set_seed,
        sha256_array,
        sha256_file,
        utc_now,
    )
    from .diffgad import (
        DiffGADAutoencoder,
        DiffGADConfig,
        DiffusionDetector,
        forward_noise_schedule,
        guided_sample,
        joint_reconstruction_score,
        update_prototype,
    )
    from .guide import (
        GUIDEConfig,
        GUIDEModel,
        cached_guide_motifs,
        guide_score,
        minmax_columns,
    )
    from .owleye import (
        OWLEYEConfig,
        OWLEYEModel,
        prepare_owleye_graph,
        sample_normal_indices,
        sample_target_indices,
    )
    from .protocol import (
        ExtensionRunSpec,
        build_manifest,
        expected_evaluations,
        validate_manifest,
    )


ARCHIVES = {
    "DiffGAD": {
        "commit": "197d60fe341f3a2080b242f04d4f81025e5f96b6",
        "url": "https://github.com/fortunato-all/DiffGAD/archive/197d60fe341f3a2080b242f04d4f81025e5f96b6.tar.gz",
        "sha256": "6f6f8dbc22f7b486a127f3c24fa19a342753468384fbb2d3dc3dd780ff940ce8",
    },
    "GUIDE": {
        "commit": "482103ef137ef88fb4d01c847dc266944e88e6fa",
        "url": "https://github.com/yushuowiki/GUIDE_pytorch/archive/482103ef137ef88fb4d01c847dc266944e88e6fa.tar.gz",
        "sha256": "b0cf88353dca694240fae9f9e98c00e3067f2ac2e320dca43a29d7116a19f8a4",
    },
    "OWLEYE": {
        "commit": "370435cd0442671e33b2874efb4cd15d30177225",
        "url": "https://github.com/zhenglecheng/ICLR-2026-OWLEYE/archive/370435cd0442671e33b2874efb4cd15d30177225.tar.gz",
        "sha256": "4e0358ce61e37ffbd05a4111a2acc44201da7160248e3cf499a3e8552e03fe52",
    },
    "ORCA": {
        "commit": "e146a8b1a99a90f5e3096b7bcc2ab0ea246c3ca7",
        "url": "https://github.com/thocevar/orca/archive/e146a8b1a99a90f5e3096b7bcc2ab0ea246c3ca7.tar.gz",
        "sha256": "efa21bbae4de2089f7cc6d3bc222ab7fe7bef05efc3153b32f8cde41dea6b08b",
    },
}


def run_directory(output_root: Path, spec: ExtensionRunSpec) -> Path:
    return output_root / "runs" / spec.run_id


def save_manifest(output_root: Path) -> Path:
    specs = build_manifest()
    validate_manifest(specs)
    path = output_root / "manifest.json"
    atomic_json(
        path,
        {
            "format": "recap_three_baseline_manifest_v1",
            "created_at": utc_now(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "upstream_manifest_sha256": sha256_file(UPSTREAM_MANIFEST_PATH),
            "training_runs": len(specs),
            "final_evaluations": expected_evaluations(),
            "runs": [spec.to_dict() for spec in specs],
        },
    )
    return path


def load_manifest(output_root: Path) -> list[ExtensionRunSpec]:
    path = output_root / "manifest.json"
    if not path.exists():
        save_manifest(output_root)
    payload = json.loads(path.read_text())
    specs = [ExtensionRunSpec.from_dict(value) for value in payload["runs"]]
    validate_manifest(specs)
    return specs


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    temporary.replace(path)


def prepare_vendor(vendor_root: Path, dataset_dir: Path) -> dict[str, Any]:
    vendor_root.mkdir(parents=True, exist_ok=True)
    archive_root = vendor_root / "archives"
    source_root = vendor_root / "sources"
    archive_root.mkdir(parents=True, exist_ok=True)
    source_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "format": "recap_three_baseline_vendor_v1",
        "prepared_at": utc_now(),
        "archives": {},
    }
    for name, metadata in ARCHIVES.items():
        archive = archive_root / f"{name.lower()}.tar.gz"
        if not archive.exists():
            _download(metadata["url"], archive)
        observed = sha256_file(archive)
        if observed != metadata["sha256"]:
            raise ValueError(f"{name}: archive hash mismatch {observed}")
        destination = source_root / name
        marker = destination / ".complete.json"
        if not marker.exists():
            if destination.exists():
                shutil.rmtree(destination)
            destination.mkdir(parents=True)
            with tarfile.open(archive, "r:gz") as package:
                members = package.getmembers()
                prefix = members[0].name.split("/", 1)[0] + "/"
                for member in members:
                    if not member.name.startswith(prefix):
                        continue
                    relative = member.name[len(prefix) :]
                    if not relative:
                        continue
                    member.name = relative
                    package.extract(member, destination, filter="data")
            atomic_json(
                marker,
                {
                    "commit": metadata["commit"],
                    "archive_sha256": observed,
                },
            )
        report["archives"][name] = {
            "path": str(archive),
            "sha256": observed,
            "commit": metadata["commit"],
        }

    orca_source = source_root / "ORCA" / "orca.cpp"
    if sha256_file(orca_source) != (
        "01c580febfd3653a632a04d6b466f7c6f2bf2f9baf3a75a2e553be30ac7ee778"
    ):
        raise ValueError("ORCA source hash mismatch")
    binary = vendor_root / "bin" / "orca"
    binary.parent.mkdir(parents=True, exist_ok=True)
    if not binary.exists():
        import subprocess

        subprocess.run(
            [
                "g++",
                "-O3",
                "-std=c++11",
                str(orca_source),
                "-o",
                str(binary),
            ],
            check=True,
        )
    report["orca_binary"] = str(binary)

    owleye_source = source_root / "OWLEYE"
    owleye_dataset = vendor_root / "owleye" / "dataset"
    if not (owleye_dataset / "cora_64.npz").exists():
        owleye_dataset.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(owleye_source / "dataset.zip") as package:
            for member in package.infolist():
                if not member.filename.startswith("dataset/"):
                    continue
                package.extract(member, owleye_dataset.parent)

    raw_hashes: dict[str, str] = {}
    with zipfile.ZipFile(owleye_source / "raw_data.zip") as package:
        for member in package.infolist():
            if not member.filename.endswith(".mat"):
                continue
            name = Path(member.filename).name
            digest = hashlib.sha256()
            with package.open(member) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            raw_hashes[name] = digest.hexdigest()
            current = dataset_dir / name
            if not current.exists() or sha256_file(current) != raw_hashes[name]:
                raise ValueError(f"OWLEYE raw-data mismatch: {name}")
    if len(raw_hashes) != 12:
        raise ValueError(f"Expected 12 OWLEYE raw MAT files, found {len(raw_hashes)}")
    report["owleye_raw_hashes"] = raw_hashes
    report["owleye_dataset_dir"] = str(owleye_dataset)
    atomic_json(vendor_root / "vendor_report.json", report)
    return report


def _train_diffusion(
    embeddings: torch.Tensor,
    *,
    config: DiffGADConfig,
    conditional: bool,
    prototype: torch.Tensor | None,
    epochs: int,
) -> tuple[DiffusionDetector, torch.Tensor | None, list[dict[str, Any]]]:
    detector = DiffusionDetector(
        config.hidden_dim, config.diffusion_dim
    ).to(embeddings.device)
    optimizer = torch.optim.Adam(
        detector.parameters(),
        lr=config.diffusion_lr,
        weight_decay=config.diffusion_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )
    current_prototype = (
        prototype.clone()
        if prototype is not None
        else embeddings.mean(dim=0).detach()
    )
    reconstructed: torch.Tensor | None = None
    best_loss = float("inf")
    best_state = copy.deepcopy(detector.state_dict())
    best_prototype = current_prototype.clone()
    patience = 0
    trace: list[dict[str, Any]] = []
    for epoch in range(epochs):
        detector.train()
        if not conditional and epoch > 0:
            assert reconstructed is not None
            current_prototype = update_prototype(
                current_prototype, reconstructed
            )
        loss, reconstructed = detector.training_loss(
            embeddings,
            prototype=current_prototype if conditional else None,
            prototype_alpha=config.prototype_alpha if conditional else None,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(detector.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_value = float(loss.detach().cpu())
        if (
            epoch == 0
            or epoch == epochs - 1
            or (epoch + 1) % 25 == 0
        ):
            trace.append({"epoch": epoch + 1, "loss": loss_value})
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = copy.deepcopy(detector.state_dict())
            best_prototype = current_prototype.clone()
            patience = 0
        else:
            patience += 1
            if patience >= config.diffusion_patience:
                trace.append(
                    {
                        "epoch": epoch + 1,
                        "event": "training_loss_early_stop",
                        "best_loss": best_loss,
                    }
                )
                break
    detector.load_state_dict(best_state)
    return (
        detector,
        best_prototype if not conditional else prototype,
        trace,
    )


@torch.no_grad()
def _diffgad_scores(
    *,
    autoencoder: DiffGADAutoencoder,
    unconditional: DiffusionDetector,
    conditional: DiffusionDetector,
    prototype: torch.Tensor,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    config: DiffGADConfig,
    inference_seed: int,
    smoke: bool,
) -> np.ndarray:
    autoencoder.eval()
    unconditional.eval()
    conditional.eval()
    set_seed(inference_seed)
    embeddings = autoencoder.encode(x, edge_index)
    base_noise = torch.randn_like(embeddings)
    sqrt_alpha, sqrt_one_minus = forward_noise_schedule(
        config.forward_timesteps, x.device
    )
    grid = config.inference_grid[:1] if smoke else config.inference_grid
    reverse_steps = min(3, config.reverse_steps) if smoke else config.reverse_steps
    scores = []
    for timestep in grid:
        noisy = (
            sqrt_alpha[timestep] * embeddings
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
            reconstructed, edge_index
        )
        score = joint_reconstruction_score(
            x,
            x_hat,
            structure_embedding,
            edge_index,
            config.attribute_weight,
        )
        scores.append(score.detach().cpu())
    return torch.stack(scores).mean(dim=0).numpy().astype(np.float32)


def run_diffgad(
    spec: ExtensionRunSpec,
    *,
    dataset_dir: Path,
    vendor_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    if spec.dataset is None:
        raise ValueError("DiffGAD requires OFO dataset")
    directory = run_directory(output_root, spec)
    complete_path = directory / "complete.json"
    if complete_path.exists() and smoke_epochs is None:
        return json.loads(complete_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    config = DiffGADConfig()
    metadata = base_metadata(
        run=spec.to_dict(),
        dataset_dir=dataset_dir,
        vendor_root=vendor_root,
        device=device,
    )
    epochs_ae = smoke_epochs or config.autoencoder_epochs
    epochs_diffusion = smoke_epochs or config.diffusion_epochs
    metadata["resolved_config"] = {
        **config.__dict__,
        "effective_autoencoder_epochs": epochs_ae,
        "effective_diffusion_epochs": epochs_diffusion,
        "smoke": smoke_epochs is not None,
    }
    atomic_json(directory / "run_start.json", metadata)
    started = time.perf_counter()
    vault = OFOLabelVault(dataset_dir, spec.dataset, False, spec.seed)

    preparation_started = time.perf_counter()
    graph = load_raw_graph(dataset_dir, spec.dataset, undirected=True)
    x = dense_features(
        graph.features, device, row_normalize=True
    )
    edge_index = edge_index_from_adjacency(graph.adjacency, device)
    preparation_seconds = time.perf_counter() - preparation_started
    set_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    autoencoder = DiffGADAutoencoder(graph.feature_count, config).to(device)
    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=config.autoencoder_lr,
        weight_decay=config.autoencoder_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )
    training_started = time.perf_counter()
    ae_trace: list[dict[str, Any]] = []
    for epoch in range(epochs_ae):
        autoencoder.train()
        x_hat, structure_embedding, _ = autoencoder(x, edge_index)
        score = joint_reconstruction_score(
            x,
            x_hat,
            structure_embedding,
            edge_index,
            config.attribute_weight,
        )
        loss = score.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if (
            epoch == 0
            or epoch == epochs_ae - 1
            or (epoch + 1) % 25 == 0
        ):
            ae_trace.append(
                {"epoch": epoch + 1, "loss": float(loss.detach().cpu())}
            )
    autoencoder.eval()
    with torch.no_grad():
        embeddings = autoencoder.encode(x, edge_index).detach()
    unconditional, prototype, unconditional_trace = _train_diffusion(
        embeddings,
        config=config,
        conditional=False,
        prototype=None,
        epochs=epochs_diffusion,
    )
    assert prototype is not None
    conditional, _, conditional_trace = _train_diffusion(
        embeddings,
        config=config,
        conditional=True,
        prototype=prototype,
        epochs=epochs_diffusion,
    )
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
            "config": config.__dict__,
            "spec": spec.to_dict(),
        },
    )

    inference_started = time.perf_counter()
    inference_seed = 700_001 + spec.seed
    scores = _diffgad_scores(
        autoencoder=autoencoder,
        unconditional=unconditional,
        conditional=conditional,
        prototype=prototype,
        x=x,
        edge_index=edge_index,
        config=config,
        inference_seed=inference_seed,
        smoke=smoke_epochs is not None,
    )
    score_path, mask = save_unsupervised_scores(
        directory=directory, scores=scores, vault=vault
    )
    labels = vault.evaluation_labels()
    metrics = score_metrics(labels[mask], scores[mask])
    inference_seconds = time.perf_counter() - inference_started

    reload_autoencoder = DiffGADAutoencoder(graph.feature_count, config).to(device)
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
    reload_scores = _diffgad_scores(
        autoencoder=reload_autoencoder,
        unconditional=reload_unconditional,
        conditional=reload_conditional,
        prototype=reload_prototype,
        x=x,
        edge_index=edge_index,
        config=config,
        inference_seed=inference_seed,
        smoke=smoke_epochs is not None,
    )
    reload_difference = float(np.max(np.abs(scores - reload_scores)))
    tolerance = reload_tolerance(scores)
    if reload_difference > tolerance:
        raise ValueError(
            f"{spec.run_id}: reload difference {reload_difference} > {tolerance}"
        )
    result = {
        **metadata,
        "completed_at": utc_now(),
        "status": "complete",
        "dataset": spec.dataset,
        "nodes": graph.node_count,
        "features": graph.feature_count,
        "raw_sha256": graph.raw_sha256,
        "metrics": metrics,
        "score_path": str(score_path),
        "score_sha256": sha256_array(scores),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "label_audit": vault.audit(),
        "reload_max_abs_difference": reload_difference,
        "reload_tolerance": tolerance,
        "timing_seconds": {
            "preprocessing": preparation_seconds,
            "training": training_seconds,
            "inference": inference_seconds,
            "total": time.perf_counter() - started,
        },
        "resources": {
            **gpu_memory(device),
            "peak_rss_mb": current_rss_mb(),
        },
        "smoke": smoke_epochs is not None,
    }
    atomic_json(complete_path, result)
    return result


def _guide_inputs(
    *,
    dataset_dir: Path,
    dataset: str,
    vendor_root: Path,
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    graph = load_raw_graph(dataset_dir, dataset, undirected=True)
    attributes_np = minmax_columns(
        np.asarray(graph.features.toarray(), dtype=np.float32)
    )
    motifs_np, cache_path, cache_state = cached_guide_motifs(
        graph.adjacency,
        orca_binary=vendor_root / "bin" / "orca",
        cache_dir=vendor_root / "guide_motif_cache",
        orca_commit=ARCHIVES["ORCA"]["commit"],
    )
    motifs_np = minmax_columns(motifs_np)
    attributes = torch.from_numpy(attributes_np).to(device)
    motifs = torch.from_numpy(motifs_np).to(device)
    adjacency = normalized_adjacency_with_loops(graph.adjacency, device)
    provenance = {
        "motif_cache_path": str(cache_path),
        "motif_cache_state": cache_state,
        "motif_sha256": sha256_array(motifs_np),
    }
    return graph, attributes, motifs, adjacency, provenance


@torch.no_grad()
def _guide_scores(
    model: GUIDEModel,
    attributes: torch.Tensor,
    motifs: torch.Tensor,
    adjacency: torch.Tensor,
    config: GUIDEConfig,
) -> np.ndarray:
    model.eval()
    attributes_hat, motifs_hat = model(attributes, motifs, adjacency)
    score, _, _ = guide_score(
        attributes,
        attributes_hat,
        motifs,
        motifs_hat,
        config.attribute_weight,
    )
    return score.detach().cpu().numpy().astype(np.float32)


def run_guide(
    spec: ExtensionRunSpec,
    *,
    dataset_dir: Path,
    vendor_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    if spec.dataset is None:
        raise ValueError("GUIDE requires OFO dataset")
    directory = run_directory(output_root, spec)
    complete_path = directory / "complete.json"
    if complete_path.exists() and smoke_epochs is None:
        return json.loads(complete_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    config = GUIDEConfig()
    epochs = smoke_epochs or config.epochs
    metadata = base_metadata(
        run=spec.to_dict(),
        dataset_dir=dataset_dir,
        vendor_root=vendor_root,
        device=device,
    )
    metadata["resolved_config"] = {
        **config.__dict__,
        "effective_epochs": epochs,
        "smoke": smoke_epochs is not None,
    }
    atomic_json(directory / "run_start.json", metadata)
    started = time.perf_counter()
    vault = OFOLabelVault(dataset_dir, spec.dataset, False, spec.seed)
    preparation_started = time.perf_counter()
    graph, attributes, motifs, adjacency, motif_provenance = _guide_inputs(
        dataset_dir=dataset_dir,
        dataset=spec.dataset,
        vendor_root=vendor_root,
        device=device,
    )
    preparation_seconds = time.perf_counter() - preparation_started
    set_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = GUIDEModel(graph.feature_count, motifs.shape[1], config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    trace = []
    training_started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        attributes_hat, motifs_hat = model(attributes, motifs, adjacency)
        score, attribute_loss, structure_loss = guide_score(
            attributes,
            attributes_hat,
            motifs,
            motifs_hat,
            config.attribute_weight,
        )
        loss = score.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 20 == 0:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.detach().cpu()),
                    "attribute_error": float(attribute_loss.detach().cpu()),
                    "structure_error": float(structure_loss.detach().cpu()),
                }
            )
    training_seconds = time.perf_counter() - training_started
    atomic_json(directory / "loss_trace.json", trace)
    checkpoint = directory / "checkpoint.pt"
    atomic_torch_save(
        checkpoint,
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "spec": spec.to_dict(),
        },
    )
    inference_started = time.perf_counter()
    scores = _guide_scores(model, attributes, motifs, adjacency, config)
    score_path, mask = save_unsupervised_scores(
        directory=directory, scores=scores, vault=vault
    )
    labels = vault.evaluation_labels()
    metrics = score_metrics(labels[mask], scores[mask])
    inference_seconds = time.perf_counter() - inference_started
    reloaded = GUIDEModel(graph.feature_count, motifs.shape[1], config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["model"])
    reload_scores = _guide_scores(
        reloaded, attributes, motifs, adjacency, config
    )
    reload_difference = float(np.max(np.abs(scores - reload_scores)))
    tolerance = reload_tolerance(scores)
    if reload_difference > tolerance:
        raise ValueError(
            f"{spec.run_id}: reload difference {reload_difference} > {tolerance}"
        )
    result = {
        **metadata,
        "completed_at": utc_now(),
        "status": "complete",
        "dataset": spec.dataset,
        "nodes": graph.node_count,
        "features": graph.feature_count,
        "raw_sha256": graph.raw_sha256,
        "motif_provenance": motif_provenance,
        "metrics": metrics,
        "score_path": str(score_path),
        "score_sha256": sha256_array(scores),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "label_audit": vault.audit(),
        "reload_max_abs_difference": reload_difference,
        "reload_tolerance": tolerance,
        "timing_seconds": {
            "preprocessing": preparation_seconds,
            "training": training_seconds,
            "inference": inference_seconds,
            "total": time.perf_counter() - started,
        },
        "resources": {
            **gpu_memory(device),
            "peak_rss_mb": current_rss_mb(),
        },
        "smoke": smoke_epochs is not None,
    }
    atomic_json(complete_path, result)
    return result


def _owleye_patterns(
    *,
    model: OWLEYEModel,
    source_graphs: list[Any],
    source_labels: list[torch.Tensor],
    support_count: int,
    fixed_indices: list[list[int]] | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[list[int]]]:
    feature_patterns = []
    structure_patterns = []
    indices_output: list[list[int]] = []
    with torch.no_grad():
        for index, graph in enumerate(source_graphs):
            feature_embedding, structure_embedding = model.embeddings(graph)
            indices = (
                fixed_indices[index]
                if fixed_indices is not None
                else sample_normal_indices(source_labels[index], support_count)
            )
            feature_patterns.append(feature_embedding[indices])
            structure_patterns.append(structure_embedding[indices])
            indices_output.append(list(indices))
    return feature_patterns, structure_patterns, indices_output


@torch.no_grad()
def _owleye_target_scores(
    *,
    model: OWLEYEModel,
    graph: Any,
    source_feature_patterns: list[torch.Tensor],
    source_structure_patterns: list[torch.Tensor],
    target_indices: list[int],
    chunk_size: int,
) -> np.ndarray:
    feature_embedding, structure_embedding = model.embeddings(graph)
    feature_patterns = list(source_feature_patterns) + [
        feature_embedding[target_indices]
    ]
    structure_patterns = list(source_structure_patterns) + [
        structure_embedding[target_indices]
    ]
    scores, _ = model.anomaly_scores(
        feature_embedding,
        structure_embedding,
        feature_patterns,
        structure_patterns,
        chunk_size=chunk_size,
    )
    return scores.detach().cpu().numpy().astype(np.float32)


def run_owleye(
    spec: ExtensionRunSpec,
    *,
    dataset_dir: Path,
    vendor_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    if spec.setting is None:
        raise ValueError("OWLEYE requires OFA setting")
    directory = run_directory(output_root, spec)
    complete_path = directory / "complete.json"
    if complete_path.exists() and smoke_epochs is None:
        return json.loads(complete_path.read_text())
    directory.mkdir(parents=True, exist_ok=True)
    config = OWLEYEConfig()
    epochs = smoke_epochs or config.epochs
    metadata = base_metadata(
        run=spec.to_dict(),
        dataset_dir=dataset_dir,
        vendor_root=vendor_root,
        device=device,
    )
    metadata["resolved_config"] = {
        **config.__dict__,
        "effective_epochs": epochs,
        "smoke": smoke_epochs is not None,
    }
    atomic_json(directory / "run_start.json", metadata)
    started = time.perf_counter()
    vault = LabelVault(dataset_dir)
    preparation_started = time.perf_counter()
    all_names = tuple(dict.fromkeys(spec.source_graphs + spec.target_graphs))
    raw_graphs = {
        name: load_raw_graph(dataset_dir, name, undirected=False)
        for name in all_names
    }
    cache_dir = vendor_root / "owleye" / "dataset"
    graphs = {
        name: prepare_owleye_graph(
            raw_graphs[name],
            cache_path=cache_dir / f"{name}_64.npz",
            config=config,
            device=device,
        )
        for name in all_names
    }
    source_labels = [
        torch.from_numpy(vault.load_source(name)).to(device)
        for name in spec.source_graphs
    ]
    preparation_seconds = time.perf_counter() - preparation_started
    set_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = OWLEYEModel(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    feature_pattern_map: dict[int, torch.Tensor] = {}
    structure_pattern_map: dict[int, torch.Tensor] = {}
    trace = []
    training_started = time.perf_counter()
    for epoch in range(epochs):
        epoch_losses = []
        for source_index, name in enumerate(spec.source_graphs):
            graph = graphs[name]
            model.train()
            feature_embedding, structure_embedding = model.embeddings(graph)
            pattern_indices = sample_normal_indices(
                source_labels[source_index], config.support_count
            )
            feature_pattern_map[source_index] = feature_embedding[
                pattern_indices
            ].detach()
            structure_pattern_map[source_index] = structure_embedding[
                pattern_indices
            ].detach()
            active = sorted(feature_pattern_map)
            loss = model.training_loss(
                feature_embedding,
                structure_embedding,
                source_labels[source_index],
                [feature_pattern_map[index] for index in active],
                [structure_pattern_map[index] for index in active],
                support_count=config.support_count,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            model.train()
            with torch.no_grad():
                feature_embedding, structure_embedding = model.embeddings(graph)
                pattern_indices = sample_normal_indices(
                    source_labels[source_index], config.support_count
                )
                feature_pattern_map[source_index] = feature_embedding[
                    pattern_indices
                ].detach()
                structure_pattern_map[source_index] = structure_embedding[
                    pattern_indices
                ].detach()
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": float(np.mean(epoch_losses)),
                    "source_losses": epoch_losses,
                }
            )
    training_seconds = time.perf_counter() - training_started
    atomic_json(directory / "loss_trace.json", trace)
    checkpoint = directory / "checkpoint.pt"
    atomic_torch_save(
        checkpoint,
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "spec": spec.to_dict(),
        },
    )
    model.eval()
    source_graph_list = [graphs[name] for name in spec.source_graphs]
    source_feature_patterns, source_structure_patterns, source_indices = (
        _owleye_patterns(
            model=model,
            source_graphs=source_graph_list,
            source_labels=source_labels,
            support_count=config.support_count,
        )
    )
    target_indices: dict[str, list[int]] = {}
    target_results: dict[str, dict[str, Any]] = {}
    reference_scores: dict[str, np.ndarray] = {}
    inference_started = time.perf_counter()
    for name in spec.target_graphs:
        indices = sample_target_indices(
            graphs[name].node_count, config.target_pattern_count
        )
        target_indices[name] = indices
        scores = _owleye_target_scores(
            model=model,
            graph=graphs[name],
            source_feature_patterns=source_feature_patterns,
            source_structure_patterns=source_structure_patterns,
            target_indices=indices,
            chunk_size=config.query_chunk_size,
        )
        _, mask = save_target_scores(
            directory=directory,
            name=name,
            scores=scores,
            pattern_indices=np.asarray(indices, dtype=np.int64),
            vault=vault,
        )
        labels = vault.load_target_for_evaluation(name)
        target_results[name] = {
            **score_metrics(labels[mask], scores[mask]),
            "nodes": int(labels.shape[0]),
            "query_nodes": int(mask.sum()),
            "target_pattern_nodes": len(indices),
            "raw_sha256": raw_graphs[name].raw_sha256,
            "score_sha256": sha256_array(scores),
        }
        reference_scores[name] = scores
    inference_seconds = time.perf_counter() - inference_started
    atomic_json(
        directory / "sampled_indices.json",
        {
            "source_normal_pattern_indices": {
                name: source_indices[index]
                for index, name in enumerate(spec.source_graphs)
            },
            "target_unlabeled_pattern_indices": target_indices,
        },
    )

    reloaded = OWLEYEModel(config).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["model"])
    reloaded.eval()
    reload_feature_patterns, reload_structure_patterns, _ = _owleye_patterns(
        model=reloaded,
        source_graphs=source_graph_list,
        source_labels=source_labels,
        support_count=config.support_count,
        fixed_indices=source_indices,
    )
    reload_differences = {}
    for name in spec.target_graphs:
        reload_scores = _owleye_target_scores(
            model=reloaded,
            graph=graphs[name],
            source_feature_patterns=reload_feature_patterns,
            source_structure_patterns=reload_structure_patterns,
            target_indices=target_indices[name],
            chunk_size=config.query_chunk_size,
        )
        difference = float(
            np.max(np.abs(reference_scores[name] - reload_scores))
        )
        tolerance = reload_tolerance(reference_scores[name])
        if difference > tolerance:
            raise ValueError(
                f"{spec.run_id}/{name}: reload {difference} > {tolerance}"
            )
        reload_differences[name] = {
            "max_abs_difference": difference,
            "tolerance": tolerance,
        }
    result = {
        **metadata,
        "completed_at": utc_now(),
        "status": "complete",
        "setting": spec.setting,
        "source_raw_sha256": {
            name: raw_graphs[name].raw_sha256 for name in spec.source_graphs
        },
        "target_results": target_results,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "label_audit": vault.audit(),
        "reload": reload_differences,
        "timing_seconds": {
            "preprocessing": preparation_seconds,
            "training": training_seconds,
            "inference": inference_seconds,
            "total": time.perf_counter() - started,
        },
        "resources": {
            **gpu_memory(device),
            "peak_rss_mb": current_rss_mb(),
        },
        "smoke": smoke_epochs is not None,
    }
    atomic_json(complete_path, result)
    return result


def select_specs(
    specs: list[ExtensionRunSpec],
    *,
    methods: set[str] | None,
    datasets: set[str] | None,
    settings: set[str] | None,
    seeds: set[int] | None,
) -> list[ExtensionRunSpec]:
    selected = []
    for spec in specs:
        if methods and spec.method not in methods:
            continue
        if datasets and spec.dataset not in datasets:
            continue
        if settings and spec.setting not in settings:
            continue
        if seeds and spec.seed not in seeds:
            continue
        selected.append(spec)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("manifest", "prepare-vendor", "run", "status")
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--settings", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "manifest":
        print(save_manifest(args.output_root))
        return
    if args.command == "prepare-vendor":
        print(
            json.dumps(
                prepare_vendor(args.vendor_root, args.dataset_dir),
                indent=2,
                sort_keys=True,
            )
        )
        return
    specs = select_specs(
        load_manifest(args.output_root),
        methods=set(args.methods) if args.methods else None,
        datasets=set(args.datasets) if args.datasets else None,
        settings=set(args.settings) if args.settings else None,
        seeds=set(args.seeds) if args.seeds else None,
    )
    if args.command == "status":
        completed = sum(
            (run_directory(args.output_root, spec) / "complete.json").exists()
            for spec in specs
        )
        print(
            json.dumps(
                {
                    "selected": len(specs),
                    "completed": completed,
                    "pending": len(specs) - completed,
                },
                indent=2,
            )
        )
        return
    prepare_vendor(args.vendor_root, args.dataset_dir)
    device = torch.device(args.device)
    for index, spec in enumerate(specs, start=1):
        print(
            f"[{index}/{len(specs)}] START {spec.run_id}",
            flush=True,
        )
        try:
            if spec.method == "DiffGAD":
                result = run_diffgad(
                    spec,
                    dataset_dir=args.dataset_dir,
                    vendor_root=args.vendor_root,
                    output_root=args.output_root,
                    device=device,
                    smoke_epochs=args.smoke_epochs,
                )
            elif spec.method == "GUIDE":
                result = run_guide(
                    spec,
                    dataset_dir=args.dataset_dir,
                    vendor_root=args.vendor_root,
                    output_root=args.output_root,
                    device=device,
                    smoke_epochs=args.smoke_epochs,
                )
            elif spec.method == "OWLEYE":
                result = run_owleye(
                    spec,
                    dataset_dir=args.dataset_dir,
                    vendor_root=args.vendor_root,
                    output_root=args.output_root,
                    device=device,
                    smoke_epochs=args.smoke_epochs,
                )
            else:
                raise ValueError(f"unknown method {spec.method}")
            summary = result.get("metrics", result.get("target_results"))
            print(
                f"[{index}/{len(specs)}] COMPLETE {spec.run_id} {summary}",
                flush=True,
            )
        except Exception as error:
            failure_dir = run_directory(args.output_root, spec)
            failure_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(
                failure_dir / f"failure_{int(time.time())}.json",
                {
                    "run": spec.to_dict(),
                    "failed_at": utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
