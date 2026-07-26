"""Resumable, label-isolated runner for the locked RECAP Phase 1 protocol."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REBUTTAL_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(REBUTTAL_ROOT))

from config import ModelConfig, TrainConfig  # noqa: E402
from model import recap  # noqa: E402
from rebuttal.phase1_protocol import (  # noqa: E402
    DATASETS,
    DIAGNOSTIC_EPOCHS,
    RunSpec,
    build_manifest,
    dataset_domain,
    display_name,
)
from utils import Dataset, set_seed  # noqa: E402


BASE_COMMIT = "c94c4d7985d2cb1438c430173ad868d68d0c1efe"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "params" / "recap_auprc_best.json"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = REBUTTAL_ROOT / "artifacts" / "phase1"
EXPECTED_EPOCHS = 100
CHECKPOINT_EPOCHS = (25, 50, 75, 100)

RAW_FIELDS = (
    "method",
    "paradigm",
    "setting",
    "seed",
    "source_graphs",
    "target_graph",
    "dataset_domain",
    "auroc",
    "auprc",
    "data_prepare_seconds",
    "train_seconds",
    "diagnostic_seconds",
    "inference_seconds",
    "cache_state",
    "peak_gpu_memory_mb",
    "config_path",
    "checkpoint_path",
    "community_output_path",
    "run_id",
    "resumed",
)


@dataclass
class PreparedDataset:
    name: str
    graph: Data
    labels: np.ndarray
    node_count: int
    feature_count: int
    anomaly_count: int
    raw_sha256: str
    feature_cache_state: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict], fieldnames: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_command_text(args: list[str]) -> str:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return (result.stdout or result.stderr).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def hardware_metadata(device: str) -> dict:
    gpu_query = run_command_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    cpu_text = run_command_text(["lscpu"])
    cpu_model = ""
    for line in cpu_text.splitlines():
        if line.startswith("Model name:"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    return {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_geometric": _package_version("torch_geometric"),
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "device": device,
        "cpu_model": cpu_model,
        "gpu_query": gpu_query,
        "base_commit": BASE_COMMIT,
    }


def raw_dataset_statistics(path: Path) -> dict:
    raw = sio.loadmat(path)
    adjacency = raw["Network"] if "Network" in raw else raw["A"]
    features = raw["Attributes"]
    labels = raw["Label"] if "Label" in raw else raw["gnd"]
    adjacency = sp.csr_matrix(adjacency)
    undirected = adjacency.maximum(adjacency.T).tocsr()
    undirected.setdiag(0)
    undirected.eliminate_zeros()
    unique_undirected_edges = int(sp.triu(undirected, k=1).nnz)
    labels = np.asarray(labels).reshape(-1)
    return {
        "nodes": int(adjacency.shape[0]),
        "raw_adjacency_nnz": int(adjacency.nnz),
        "unique_undirected_edges": unique_undirected_edges,
        "raw_feature_count": int(features.shape[1]),
        "anomalies": int(labels.sum()),
    }


def run_preflight(
    dataset_dir: Path,
    output_root: Path,
    config_path: Path,
    device: str,
    data_manifest_path: Path,
) -> dict:
    model_config = load_model_config(
        config_path,
        output_root / "cache" / "exact_knn",
    )
    prepared, data_prepare_seconds = prepare_graphs(
        names=list(DATASETS),
        dataset_dir=dataset_dir,
        dims=model_config.dims,
        num_hops=model_config.num_hops,
        device=device,
    )
    assert_training_graphs_label_free(prepared)
    rows = {}
    for name, definition in DATASETS.items():
        raw_path = dataset_dir / str(definition["file"])
        statistics = raw_dataset_statistics(raw_path)
        item = prepared[name]
        if statistics["nodes"] != item.node_count:
            raise ValueError(f"{name}: raw/prepared node count mismatch")
        if statistics["anomalies"] != item.anomaly_count:
            raise ValueError(f"{name}: raw/prepared anomaly count mismatch")
        rows[name] = {
            "display_name": display_name(name),
            "domain": dataset_domain(name),
            "file": str(definition["file"]),
            "sha256": item.raw_sha256,
            "aligned_feature_count": item.feature_count,
            "feature_alignment_cache_state": item.feature_cache_state,
            **statistics,
        }
    if rows["Amazon"]["nodes"] != 10224:
        raise ValueError(
            f"Amazon actual-data decision violated: {rows['Amazon']['nodes']} nodes"
        )
    if rows["questions"]["nodes"] != 48921:
        raise ValueError(
            f"Questions dataset mismatch: {rows['questions']['nodes']} nodes"
        )
    if (PROJECT_ROOT / "large_graph_knn.py").exists():
        raise ValueError("ANN module must not exist in the clean Phase 1 worktree")
    metadata = {
        "format": "recap_phase1_data_manifest_v1",
        "locked_at": utc_now(),
        "base_commit": BASE_COMMIT,
        "dataset_dir": str(dataset_dir.resolve()),
        "paper_config_sha256": sha256_file(config_path),
        "data_prepare_seconds": data_prepare_seconds,
        "datasets": rows,
        "amazon_count_decision": (
            "Use actual 10,224-node file; manuscript 10,244 is a documented likely typo."
        ),
    }
    atomic_json(data_manifest_path, metadata)
    hardware = hardware_metadata(device)
    atomic_json(output_root / "preflight" / "hardware.json", hardware)
    atomic_json(output_root / "preflight" / "data_manifest.json", metadata)
    del prepared
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "passed": True,
        "dataset_count": len(rows),
        "data_prepare_seconds": data_prepare_seconds,
        "data_manifest_path": str(data_manifest_path.resolve()),
        "hardware_path": str((output_root / "preflight" / "hardware.json").resolve()),
    }


def _package_version(package: str) -> str:
    try:
        module = __import__(package)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def load_model_config(config_path: Path, cache_dir: Path) -> ModelConfig:
    model_config = ModelConfig.from_json(str(config_path))
    if model_config is None:
        raise FileNotFoundError(f"Missing locked model config: {config_path}")
    model_config.knn_cache_enabled = True
    model_config.knn_cache_dir = str(cache_dir)

    locked = {
        "dims": 32,
        "num_hops": 4,
        "num_clusters": 36,
        "knn_k": 64,
        "lr": 5e-5,
        "weight_decay": 5e-5,
        "tau_s": 0.3,
        "tau_c": 0.3,
        "tau_e": 1.0,
        "lambda_H": 0.1,
        "lambda_usage_entropy": 0.1,
        "lambda_E": 0.0,
        "beta": 0.02,
    }
    mismatches = {
        key: {"expected": expected, "actual": getattr(model_config, key)}
        for key, expected in locked.items()
        if getattr(model_config, key) != expected
    }
    if mismatches:
        raise ValueError(f"Locked paper config mismatch: {mismatches}")
    return model_config


def build_optimizer(model: recap, model_config: ModelConfig) -> torch.optim.Optimizer:
    lr = model_config.lr
    weight_decay = model_config.weight_decay
    multiplier = float(model_config.cluster_lr_multiplier)
    if multiplier == 1.0:
        parameters: Any = model.parameters()
    else:
        cluster_ids = {id(param) for param in model.ego_clusters.parameters()}
        base_parameters = [
            param for param in model.parameters() if id(param) not in cluster_ids
        ]
        parameters = [
            {
                "params": base_parameters,
                "lr": lr,
                "weight_decay": weight_decay,
            },
            {
                "params": list(model.ego_clusters.parameters()),
                "lr": lr * multiplier,
                "weight_decay": 0.0,
            },
        ]
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)


def label_free_graph(original: Data, raw_sha256: str) -> Data:
    """Return the only graph object allowed to cross the training boundary."""
    graph = Data(
        x_list=original.x_list,
        dataset_name=str(original.dataset_name),
        feature_alignment_version=str(original.feature_alignment_version),
        feature_dims=int(original.feature_dims),
        adjacency_version=(
            f"{original.adjacency_version}|raw_sha256={raw_sha256[:20]}"
        ),
    )
    if "ano_labels" in graph:
        raise AssertionError("Label isolation failure: sanitized graph contains labels")
    return graph


def prepare_graphs(
    names: list[str],
    dataset_dir: Path,
    dims: int,
    num_hops: int,
    device: str,
) -> tuple[dict[str, PreparedDataset], float]:
    unique_names = list(dict.fromkeys(names))
    synchronize(device)
    started = time.perf_counter()
    output: dict[str, PreparedDataset] = {}
    for name in unique_names:
        if name not in DATASETS:
            raise KeyError(f"Unknown locked dataset: {name}")
        raw_path = dataset_dir / str(DATASETS[name]["file"])
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing dataset file: {raw_path}")
        raw_hash = sha256_file(raw_path)
        feature_cache = dataset_dir / f"{name}_{dims}.npz"
        feature_state = "reused" if feature_cache.exists() else "new"
        loaded = Dataset(dims, name, prefix=f"{dataset_dir}{os.sep}")
        labels = np.asarray(loaded.label).reshape(-1).astype(np.int64, copy=True)
        loaded.propagated(num_hops, device=device)
        graph = label_free_graph(loaded.graph, raw_hash)
        node_count = int(graph.x_list[0].shape[0])
        feature_count = int(graph.x_list[0].shape[1])
        if labels.shape[0] != node_count:
            raise ValueError(
                f"{name}: label count {labels.shape[0]} != node count {node_count}"
            )
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError(f"{name}: labels are not binary with both classes")
        output[name] = PreparedDataset(
            name=name,
            graph=graph,
            labels=labels,
            node_count=node_count,
            feature_count=feature_count,
            anomaly_count=int(labels.sum()),
            raw_sha256=raw_hash,
            feature_cache_state=feature_state,
        )
        del loaded
    synchronize(device)
    return output, time.perf_counter() - started


def assert_training_graphs_label_free(graphs: dict[str, PreparedDataset]) -> None:
    offenders = [name for name, item in graphs.items() if "ano_labels" in item.graph]
    if offenders:
        raise AssertionError(f"Training boundary contains labels: {offenders}")


def knn_cache_probe(model: recap) -> dict:
    cluster = model.ego_clusters[0]
    residual = model._view_embeds_init[0]
    cache_key = model._view_cache_keys[0]
    search_dtype = cluster._resolve_knn_search_dtype(residual.device)
    full_key = (
        "recap_knn_candidates_v1",
        cache_key,
        residual.shape[0],
        residual.shape[1],
        cluster.knn_k,
        str(search_dtype).replace("torch.", ""),
    )
    path = cluster._disk_cache_path(full_key)
    return {
        "path": path,
        "state_before": "reused" if path and os.path.exists(path) else "new",
        "key_sha256": hashlib.sha256(repr(full_key).encode("utf-8")).hexdigest(),
    }


@torch.no_grad()
def collect_diagnostics(
    model: recap,
    graph: Data,
    optimizer_step_loss: float,
    epoch: int,
    spec: RunSpec,
    dataset_name: str,
) -> dict:
    if "ano_labels" in graph:
        raise AssertionError("Diagnostics received labels")
    model.eval()
    model(graph)
    cluster = model.ego_clusters[0]
    residual = model._view_embeds[0]
    residual_initial = model._view_embeds_init[0]
    cache_key = model._view_cache_keys[0]
    edge_index, edge_weight = cluster.build_ego_graph(
        residual,
        E_init=residual_initial,
        cache_key=cache_key,
    )
    assignments = cluster.cluster(residual)
    l_con = cluster._compute_con_loss(assignments, edge_index, edge_weight)

    eps = cluster.eps
    community_count = assignments.shape[1]
    log_c = torch.log(assignments.new_tensor(max(community_count, 2)))
    node_entropy = -(assignments * torch.log(assignments + eps)).sum(dim=1)
    assignment_entropy_norm = node_entropy.mean() / log_c.clamp(min=eps)
    l_assign = assignments.new_tensor(0.0)
    if cluster.assignment_entropy_upper is None and cluster.assignment_entropy_lower is None:
        l_assign = node_entropy.mean()
    else:
        if cluster.assignment_entropy_upper is not None:
            l_assign = l_assign + torch.relu(
                assignment_entropy_norm - float(cluster.assignment_entropy_upper)
            )
        if cluster.assignment_entropy_lower is not None:
            l_assign = l_assign + torch.relu(
                float(cluster.assignment_entropy_lower) - assignment_entropy_norm
            )
        l_assign = l_assign * log_c

    usage = assignments.mean(dim=0)
    usage_entropy = -(usage * torch.log(usage + eps)).sum()
    usage_entropy_norm = usage_entropy / log_c.clamp(min=eps)
    l_usage = assignments.new_tensor(0.0)
    if cluster.usage_entropy_upper is not None:
        l_usage = l_usage + torch.relu(
            usage_entropy_norm - float(cluster.usage_entropy_upper)
        )
    if cluster.usage_entropy_lower is not None:
        l_usage = l_usage + torch.relu(
            float(cluster.usage_entropy_lower) - usage_entropy_norm
        )
    l_usage = l_usage * log_c
    l_balance = (usage * torch.log(usage + eps)).sum()
    l_regularization = (
        l_assign
        + cluster.lambda_bal * l_balance
        + cluster.lambda_usage_entropy * l_usage
    )
    residual_std = residual.std(dim=0, unbiased=False)
    l_var = (
        torch.relu(cluster.gamma - residual_std).mean()
        if cluster.lambda_E != 0
        else residual.new_tensor(0.0)
    )
    total_loss = (
        l_con + cluster.lambda_H * l_regularization + cluster.lambda_E * l_var
    )
    active_threshold = float(cluster.gamma)

    return {
        "epoch": epoch,
        "seed": spec.seed,
        "paradigm": spec.paradigm,
        "setting": spec.setting,
        "dataset": dataset_name,
        "row_type": "source",
        "optimizer_step_loss": float(optimizer_step_loss),
        "total_loss": float(total_loss.item()),
        "l_con": float(l_con.item()),
        "assignment_regularization_loss": float(l_assign.item()),
        "community_balance_loss": float(l_balance.item()),
        "community_usage_regularization_loss": float(l_usage.item()),
        "combined_regularization_loss": float(l_regularization.item()),
        "assignment_entropy": float(node_entropy.mean().item()),
        "assignment_entropy_normalized": float(assignment_entropy_norm.item()),
        "community_usage_entropy": float(usage_entropy.item()),
        "community_usage_entropy_normalized": float(usage_entropy_norm.item()),
        "effective_communities": float(torch.exp(usage_entropy).item()),
        "usage_min": float(usage.min().item()),
        "usage_max": float(usage.max().item()),
        "usage_std": float(usage.std(unbiased=False).item()),
        "residual_mean_std": float(residual_std.mean().item()),
        "active_usage_threshold": active_threshold,
        "active_communities": int((usage > active_threshold).sum().item()),
    }


def macro_diagnostic(rows: list[dict], spec: RunSpec, epoch: int) -> dict:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float))
        and key not in {"epoch", "seed", "active_communities"}
    ]
    output: dict[str, Any] = {
        "epoch": epoch,
        "seed": spec.seed,
        "paradigm": spec.paradigm,
        "setting": spec.setting,
        "dataset": "__source_macro__",
        "row_type": "macro",
    }
    for key in numeric_keys:
        output[key] = float(np.mean([float(row[key]) for row in rows]))
    output["active_communities"] = float(
        np.mean([float(row["active_communities"]) for row in rows])
    )
    return output


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temp)
    os.replace(temp, path)


def checkpoint_payload(
    model: recap,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    spec: RunSpec,
    epoch: int,
    loss_history: list[float],
    diagnostics: list[dict],
    train_seconds: float,
    diagnostic_seconds: float,
    train_cache: dict[str, dict],
    config_hash: str,
) -> dict:
    return {
        "format": "recap_phase1_checkpoint_v1",
        "base_commit": BASE_COMMIT,
        "run_spec": spec.to_dict(),
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "loss_history": loss_history,
        "diagnostics": diagnostics,
        "train_seconds": train_seconds,
        "diagnostic_seconds": diagnostic_seconds,
        "train_cache": train_cache,
        "config_hash": config_hash,
        "rng_state": rng_state(),
        "saved_at": utc_now(),
    }


def latest_resume_checkpoint(run_dir: Path) -> Path | None:
    candidates = list((run_dir / "checkpoints").glob("resume_epoch_*.pt"))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )


def train_model(
    spec: RunSpec,
    prepared: dict[str, PreparedDataset],
    model_config: ModelConfig,
    train_config: TrainConfig,
    run_dir: Path,
    config_hash: str,
    resume: bool,
) -> tuple[recap, dict]:
    assert_training_graphs_label_free(prepared)
    device = train_config.device
    set_seed(spec.seed)
    model = recap(**model_config.to_dict()).to(device)
    optimizer = build_optimizer(model, model_config)
    start_epoch = 1
    loss_history: list[float] = []
    diagnostics: list[dict] = []
    train_seconds = 0.0
    diagnostic_seconds = 0.0
    train_cache: dict[str, dict] = {}
    resumed = False

    resume_path = latest_resume_checkpoint(run_dir) if resume else None
    if resume_path is not None:
        # Checkpoints are atomically created by this runner in the same trusted
        # run directory and include optimizer/RNG Python objects, so the
        # PyTorch 2.6+ weights-only default is intentionally disabled.
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("config_hash") != config_hash:
            raise ValueError(f"Resume config hash mismatch: {resume_path}")
        if checkpoint.get("run_spec") != spec.to_dict():
            raise ValueError(f"Resume run specification mismatch: {resume_path}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_history = list(checkpoint["loss_history"])
        diagnostics = list(checkpoint["diagnostics"])
        train_seconds = float(checkpoint["train_seconds"])
        diagnostic_seconds = float(checkpoint["diagnostic_seconds"])
        train_cache = dict(checkpoint.get("train_cache", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        restore_rng_state(checkpoint["rng_state"])
        resumed = int(checkpoint["epoch"]) < train_config.epochs

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    for epoch in range(start_epoch, train_config.epochs + 1):
        model.train()
        per_source_step_loss: dict[str, float] = {}
        synchronize(device)
        epoch_started = time.perf_counter()
        for source_name in spec.source_graphs:
            graph = prepared[source_name].graph
            if "ano_labels" in graph:
                raise AssertionError(f"Training graph {source_name} contains labels")
            model(graph)
            if source_name not in train_cache:
                train_cache[source_name] = knn_cache_probe(model)
            loss = model.get_cluster_loss()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{spec.run_id} epoch {epoch} {source_name}: non-finite loss"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            per_source_step_loss[source_name] = float(loss.detach().item())
        synchronize(device)
        train_seconds += time.perf_counter() - epoch_started
        mean_loss = float(np.mean(list(per_source_step_loss.values())))
        loss_history.append(mean_loss)

        if epoch in DIAGNOSTIC_EPOCHS:
            synchronize(device)
            diagnostic_started = time.perf_counter()
            epoch_rows = [
                collect_diagnostics(
                    model=model,
                    graph=prepared[source_name].graph,
                    optimizer_step_loss=per_source_step_loss[source_name],
                    epoch=epoch,
                    spec=spec,
                    dataset_name=source_name,
                )
                for source_name in spec.source_graphs
            ]
            diagnostics.extend(epoch_rows)
            if len(epoch_rows) > 1:
                diagnostics.append(macro_diagnostic(epoch_rows, spec, epoch))
            synchronize(device)
            diagnostic_seconds += time.perf_counter() - diagnostic_started

        if epoch in CHECKPOINT_EPOCHS:
            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                train_config=train_config,
                spec=spec,
                epoch=epoch,
                loss_history=loss_history,
                diagnostics=diagnostics,
                train_seconds=train_seconds,
                diagnostic_seconds=diagnostic_seconds,
                train_cache=train_cache,
                config_hash=config_hash,
            )
            checkpoint_path = run_dir / "checkpoints" / f"resume_epoch_{epoch}.pt"
            atomic_torch_save(checkpoint_path, payload)

    if len(loss_history) != train_config.epochs:
        raise AssertionError(
            f"{spec.run_id}: expected {train_config.epochs} losses, got {len(loss_history)}"
        )
    expected_rows_per_epoch = len(spec.source_graphs) + (
        1 if len(spec.source_graphs) > 1 else 0
    )
    expected_diagnostic_rows = len(DIAGNOSTIC_EPOCHS) * expected_rows_per_epoch
    if len(diagnostics) != expected_diagnostic_rows:
        raise AssertionError(
            f"{spec.run_id}: expected {expected_diagnostic_rows} diagnostic rows, "
            f"got {len(diagnostics)}"
        )

    final_payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        train_config=train_config,
        spec=spec,
        epoch=train_config.epochs,
        loss_history=loss_history,
        diagnostics=diagnostics,
        train_seconds=train_seconds,
        diagnostic_seconds=diagnostic_seconds,
        train_cache=train_cache,
        config_hash=config_hash,
    )
    final_path = run_dir / "checkpoints" / "final.pt"
    atomic_torch_save(final_path, final_payload)
    diagnostics_fields = sorted({key for row in diagnostics for key in row})
    atomic_csv(run_dir / "training_diagnostics.csv", diagnostics, diagnostics_fields)
    atomic_json(
        run_dir / "training_history.json",
        {
            "losses": loss_history,
            "epochs": train_config.epochs,
            "train_seconds": train_seconds,
            "diagnostic_seconds": diagnostic_seconds,
            "knn_cache": train_cache,
            "resumed": resumed,
        },
    )
    return model, {
        "checkpoint_path": str(final_path.resolve()),
        "train_seconds": train_seconds,
        "diagnostic_seconds": diagnostic_seconds,
        "knn_cache": train_cache,
        "resumed": resumed,
        "loss_history": loss_history,
    }


def community_path(output_root: Path, spec: RunSpec, target: str) -> Path:
    if spec.setting == "OFO":
        return (
            output_root
            / "community_stability"
            / "one_for_one"
            / display_name(target)
            / f"seed_{spec.seed}.npz"
        )
    return (
        output_root
        / "community_stability"
        / "one_for_all"
        / f"setting_{spec.setting}"
        / display_name(target)
        / f"seed_{spec.seed}.npz"
    )


@torch.no_grad()
def infer_target(
    model: recap,
    prepared: PreparedDataset,
    spec: RunSpec,
    output_root: Path,
    device: str,
) -> tuple[dict, dict]:
    graph = prepared.graph
    if "ano_labels" in graph:
        raise AssertionError(f"Inference graph {prepared.name} contains labels")
    model.eval()
    synchronize(device)
    started = time.perf_counter()
    model(graph)
    cache_probe = knn_cache_probe(model)
    components = model.get_ego_score_components()
    assignments = model.ego_clusters[0].cluster(model._view_embeds[0])
    synchronize(device)
    inference_seconds = time.perf_counter() - started

    # Scores and communities are finalized before labels are accessed.
    arrays = {
        "H": assignments.detach().float().cpu().numpy(),
        "adhesion_raw": components["adhesion_raw"].detach().float().cpu().numpy(),
        "adhesion_standardized": components["adhesion"].detach().float().cpu().numpy(),
        "context_raw": components["scale_raw"].detach().float().cpu().numpy(),
        "context_standardized": components["scale"].detach().float().cpu().numpy(),
        "final_scores": components["total"].detach().float().cpu().numpy(),
    }
    usage = arrays["H"].mean(axis=0, dtype=np.float64).astype(np.float32)
    eps = 1e-8
    assignment_entropy = -np.sum(
        arrays["H"] * np.log(arrays["H"] + eps),
        axis=1,
    ).astype(np.float32)
    effective = float(np.exp(-np.sum(usage * np.log(usage + eps))))
    arrays.update(
        {
            "hard_assignments": arrays["H"].argmax(axis=1).astype(np.int64),
            "usage": usage,
            "assignment_entropy": assignment_entropy,
            "node_indices": np.arange(prepared.node_count, dtype=np.int64),
            "effective_communities": np.asarray(effective, dtype=np.float32),
            "num_communities": np.asarray(arrays["H"].shape[1], dtype=np.int64),
        }
    )
    if any(not np.all(np.isfinite(value)) for value in arrays.values() if value.dtype.kind == "f"):
        raise FloatingPointError(f"{spec.run_id}/{prepared.name}: non-finite output array")

    output_path = community_path(output_root, spec, prepared.name)
    atomic_npz(output_path, **arrays)

    # This is the first line in this function that accesses labels.
    labels = prepared.labels
    auroc = float(roc_auc_score(labels, arrays["final_scores"]))
    auprc = float(average_precision_score(labels, arrays["final_scores"]))
    return {
        "target_graph": prepared.name,
        "dataset_domain": dataset_domain(prepared.name),
        "auroc": auroc,
        "auprc": auprc,
        "inference_seconds": inference_seconds,
        "community_output_path": str(output_path.resolve()),
        "effective_communities": effective,
    }, cache_probe


@torch.no_grad()
def verify_checkpoint_reload(
    checkpoint_path: Path,
    model_config: ModelConfig,
    prepared: PreparedDataset,
    reference_scores_path: Path,
    device: str,
) -> dict:
    # This is a trusted, self-generated Phase 1 checkpoint containing more than
    # tensor weights (resolved configs, optimizer and RNG state).
    payload = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    reloaded = recap(**model_config.to_dict()).to(device)
    reloaded.load_state_dict(payload["model_state_dict"], strict=True)
    reloaded.eval()
    reloaded(prepared.graph)
    scores = reloaded.get_ego_scores().detach().float().cpu().numpy()
    with np.load(reference_scores_path, allow_pickle=False) as saved:
        reference = saved["final_scores"]
    max_abs = float(np.max(np.abs(scores - reference)))
    if max_abs > 1e-6:
        raise AssertionError(f"Checkpoint reload mismatch: max_abs={max_abs}")
    return {"passed": True, "max_abs_score_difference": max_abs}


def run_one(
    spec: RunSpec,
    dataset_dir: Path,
    output_root: Path,
    config_path: Path,
    device: str,
    resume: bool,
) -> list[dict]:
    run_dir = output_root / "runs" / spec.run_id
    complete_path = run_dir / "complete.json"
    if complete_path.exists():
        for log_name in ("stdout.log", "stderr.log"):
            log_path = run_dir / log_name
            if not log_path.exists():
                log_path.write_text(
                    "Formal seed-0 gate was invoked directly before subprocess "
                    "stream capture was enabled. Structured status, history, "
                    "diagnostics, configs and checkpoint audits are complete.\n",
                    encoding="utf-8",
                )
        with (run_dir / "result_records.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    run_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(
        run_dir / "events.jsonl",
        {"event": "run_started", "run_id": spec.run_id, "at": utc_now()},
    )
    atomic_json(
        run_dir / "status.json",
        {"status": "running", "run_id": spec.run_id, "started_at": utc_now()},
    )
    try:
        cache_dir = output_root / "cache" / "exact_knn"
        model_config = load_model_config(config_path, cache_dir)
        train_config = TrainConfig(
            device=device,
            epochs=EXPECTED_EPOCHS,
            trials=1,
            seed=spec.seed,
            output_dir=str(run_dir),
            save_checkpoint=True,
            early_stop=False,
            log_diagnostics=False,
        )
        resolved = {
            "base_commit": BASE_COMMIT,
            "run_spec": spec.to_dict(),
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
            "diagnostic_epochs": list(DIAGNOSTIC_EPOCHS),
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "standard_deviation_ddof": 0,
            "label_isolation": "sanitized Data objects; labels accessed after scores",
        }
        config_hash = stable_hash(resolved)
        resolved["config_hash"] = config_hash
        resolved_path = run_dir / "resolved_config.json"
        atomic_json(resolved_path, resolved)

        needed = list(dict.fromkeys((*spec.source_graphs, *spec.target_graphs)))
        prepared, data_prepare_seconds = prepare_graphs(
            names=needed,
            dataset_dir=dataset_dir,
            dims=model_config.dims,
            num_hops=model_config.num_hops,
            device=device,
        )
        assert_training_graphs_label_free(prepared)
        data_metadata = {
            name: {
                "nodes": item.node_count,
                "features": item.feature_count,
                "anomalies": item.anomaly_count,
                "raw_sha256": item.raw_sha256,
                "feature_cache_state": item.feature_cache_state,
            }
            for name, item in prepared.items()
        }
        atomic_json(run_dir / "data_metadata.json", data_metadata)
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "event": "data_prepared",
                "run_id": spec.run_id,
                "seconds": data_prepare_seconds,
                "at": utc_now(),
            },
        )

        model, train_info = train_model(
            spec=spec,
            prepared=prepared,
            model_config=model_config,
            train_config=train_config,
            run_dir=run_dir,
            config_hash=config_hash,
            resume=resume,
        )
        append_jsonl(
            run_dir / "events.jsonl",
            {
                "event": "training_complete",
                "run_id": spec.run_id,
                "train_seconds": train_info["train_seconds"],
                "diagnostic_seconds": train_info["diagnostic_seconds"],
                "at": utc_now(),
            },
        )

        inference_rows = []
        inference_cache = {}
        for target_name in spec.target_graphs:
            row, cache_probe = infer_target(
                model=model,
                prepared=prepared[target_name],
                spec=spec,
                output_root=output_root,
                device=device,
            )
            inference_rows.append(row)
            inference_cache[target_name] = cache_probe
            append_jsonl(
                run_dir / "events.jsonl",
                {
                    "event": "target_inference_complete",
                    "run_id": spec.run_id,
                    "target": target_name,
                    "inference_seconds": row["inference_seconds"],
                    "at": utc_now(),
                },
            )

        checkpoint_path = Path(train_info["checkpoint_path"])
        first_target = spec.target_graphs[0]
        reload_audit = verify_checkpoint_reload(
            checkpoint_path=checkpoint_path,
            model_config=model_config,
            prepared=prepared[first_target],
            reference_scores_path=Path(inference_rows[0]["community_output_path"]),
            device=device,
        )
        atomic_json(run_dir / "checkpoint_reload_audit.json", reload_audit)

        peak_mb = (
            float(torch.cuda.max_memory_allocated(torch.device(device)) / (1024**2))
            if torch.cuda.is_available()
            else 0.0
        )
        records = []
        for inference_row in inference_rows:
            target = inference_row["target_graph"]
            cache_state = {
                "feature": prepared[target].feature_cache_state,
                "train_knn": train_info["knn_cache"],
                "inference_knn": inference_cache[target],
            }
            record = {
                "method": spec.method,
                "paradigm": spec.paradigm,
                "setting": spec.setting,
                "seed": spec.seed,
                "source_graphs": "|".join(spec.source_graphs),
                "target_graph": target,
                "dataset_domain": inference_row["dataset_domain"],
                "auroc": inference_row["auroc"],
                "auprc": inference_row["auprc"],
                "data_prepare_seconds": data_prepare_seconds,
                "train_seconds": train_info["train_seconds"],
                "diagnostic_seconds": train_info["diagnostic_seconds"],
                "inference_seconds": inference_row["inference_seconds"],
                "cache_state": json.dumps(cache_state, sort_keys=True),
                "peak_gpu_memory_mb": peak_mb,
                "config_path": str(resolved_path.resolve()),
                "checkpoint_path": str(checkpoint_path.resolve()),
                "community_output_path": inference_row["community_output_path"],
                "run_id": spec.run_id,
                "resumed": train_info["resumed"],
            }
            records.append(record)

        atomic_json(run_dir / "result_records.json", records)
        atomic_csv(run_dir / "result_records.csv", records, RAW_FIELDS)
        atomic_json(
            complete_path,
            {
                "status": "complete",
                "run_id": spec.run_id,
                "completed_at": utc_now(),
                "record_count": len(records),
                "config_hash": config_hash,
                "checkpoint_reload_audit": reload_audit,
            },
        )
        atomic_json(
            run_dir / "status.json",
            {"status": "complete", "run_id": spec.run_id, "completed_at": utc_now()},
        )
        append_jsonl(
            run_dir / "events.jsonl",
            {"event": "run_complete", "run_id": spec.run_id, "at": utc_now()},
        )
        return records
    except Exception as exc:
        failure = {
            "status": "failed",
            "run_id": spec.run_id,
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(run_dir / "status.json", failure)
        append_jsonl(output_root / "failure_log.jsonl", failure)
        append_jsonl(run_dir / "events.jsonl", {"event": "run_failed", **failure})
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_manifest(path: Path) -> list[dict]:
    specs = build_manifest()
    payload = {
        "format": "recap_phase1_manifest_v1",
        "created_at": utc_now(),
        "base_commit": BASE_COMMIT,
        "locked_protocol": "rebuttal/PHASE1_EXECUTION_LOCK.md",
        "training_run_count": len(specs),
        "final_evaluation_count": sum(len(spec.target_graphs) for spec in specs),
        "runs": [spec.to_dict() for spec in specs],
    }
    atomic_json(path, payload)
    return payload["runs"]


def load_manifest(path: Path) -> list[RunSpec]:
    if not path.exists():
        write_manifest(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("base_commit") != BASE_COMMIT:
        raise ValueError("Manifest base commit mismatch")
    # JSON has no tuple type. Normalize the declarative Python manifest through
    # a JSON round trip before comparing it with the persisted JSON payload.
    expected = json.loads(
        json.dumps([spec.to_dict() for spec in build_manifest()])
    )
    if payload.get("runs") != expected:
        raise ValueError("Manifest differs from locked protocol constants")
    return [RunSpec.from_dict(value) for value in payload["runs"]]


def collect_records(output_root: Path, manifest: list[RunSpec]) -> list[dict]:
    records = []
    for spec in manifest:
        path = output_root / "runs" / spec.run_id / "result_records.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                records.extend(json.load(handle))
    records.sort(key=lambda row: (row["setting"], row["target_graph"], row["seed"]))
    return records


def write_progress(output_root: Path, manifest: list[RunSpec]) -> dict:
    complete = []
    failed = []
    running = []
    pending = []
    for spec in manifest:
        status_path = output_root / "runs" / spec.run_id / "status.json"
        if not status_path.exists():
            pending.append(spec.run_id)
            continue
        with status_path.open("r", encoding="utf-8") as handle:
            status = json.load(handle)
        state = status.get("status")
        if state == "complete":
            complete.append(spec.run_id)
        elif state == "failed":
            failed.append(spec.run_id)
        elif state == "running":
            running.append(spec.run_id)
        else:
            pending.append(spec.run_id)
    records = collect_records(output_root, manifest)
    progress = {
        "captured_at": utc_now(),
        "training_runs_total": len(manifest),
        "training_runs_complete": len(complete),
        "training_runs_failed": len(failed),
        "training_runs_running": len(running),
        "training_runs_pending": len(pending),
        "final_evaluations_complete": len(records),
        "final_evaluations_total": 87,
        "complete_run_ids": complete,
        "failed_run_ids": failed,
        "running_run_ids": running,
        "next_pending_run_ids": pending[:10],
    }
    atomic_json(output_root / "progress.json", progress)
    if records:
        atomic_json(output_root / "raw_results.json", records)
        atomic_csv(output_root / "raw_results.csv", records, RAW_FIELDS)
    return progress


def execute_manifest(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    specs = load_manifest(manifest_path)
    if args.run_ids:
        selected = set(args.run_ids)
        missing = selected - {spec.run_id for spec in specs}
        if missing:
            raise KeyError(f"Unknown run IDs: {sorted(missing)}")
        specs = [spec for spec in specs if spec.run_id in selected]
    elif args.gates_only:
        specs = specs[:2]

    for index, spec in enumerate(specs, start=1):
        complete = output_root / "runs" / spec.run_id / "complete.json"
        if complete.exists():
            print(f"[{index}/{len(specs)}] SKIP complete {spec.run_id}", flush=True)
        else:
            print(f"[{index}/{len(specs)}] START {spec.run_id}", flush=True)
            run_dir = output_root / "runs" / spec.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output-root",
                str(output_root),
                "--dataset-dir",
                str(Path(args.dataset_dir)),
                "--config",
                str(Path(args.config)),
                "--manifest",
                str(manifest_path),
                "--device",
                args.device,
                "run-one",
                spec.run_id,
            ]
            if args.no_resume:
                command.append("--no-resume")
            with (run_dir / "stdout.log").open("a", encoding="utf-8") as stdout_handle:
                with (run_dir / "stderr.log").open("a", encoding="utf-8") as stderr_handle:
                    completed = subprocess.run(
                        command,
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        check=False,
                    )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{spec.run_id} failed with exit code {completed.returncode}; "
                    f"see {run_dir / 'stderr.log'}"
                )
            print(f"[{index}/{len(specs)}] COMPLETE {spec.run_id}", flush=True)
        progress = write_progress(output_root, load_manifest(manifest_path))
        print(
            "PROGRESS "
            + json.dumps(
                {
                    "complete": progress["training_runs_complete"],
                    "failed": progress["training_runs_failed"],
                    "evaluations": progress["final_evaluations_complete"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Phase 1 artifact root",
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Directory containing the 12 MAT datasets and versioned caches",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Locked paper model configuration",
    )
    parser.add_argument(
        "--manifest",
        default=str(REBUTTAL_ROOT / "phase1_manifest.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("manifest")
    subparsers.add_parser("preflight")

    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("run_id")
    run_parser.add_argument("--no-resume", action="store_true")

    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--run-ids", nargs="*")
    execute_parser.add_argument("--gates-only", action="store_true")
    execute_parser.add_argument("--no-resume", action="store_true")

    subparsers.add_parser("status")
    subparsers.add_parser("hardware")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)
    if args.command == "manifest":
        write_manifest(manifest_path)
        print(manifest_path)
    elif args.command == "preflight":
        result = run_preflight(
            dataset_dir=Path(args.dataset_dir),
            output_root=output_root,
            config_path=Path(args.config),
            device=args.device,
            data_manifest_path=REBUTTAL_ROOT / "data_manifest.json",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "hardware":
        metadata = hardware_metadata(args.device)
        atomic_json(output_root / "hardware.json", metadata)
        print(json.dumps(metadata, indent=2, sort_keys=True))
    elif args.command == "status":
        progress = write_progress(output_root, load_manifest(manifest_path))
        print(json.dumps(progress, indent=2, sort_keys=True))
    elif args.command == "run-one":
        specs = {spec.run_id: spec for spec in load_manifest(manifest_path)}
        if args.run_id not in specs:
            raise KeyError(f"Unknown run ID: {args.run_id}")
        run_one(
            spec=specs[args.run_id],
            dataset_dir=Path(args.dataset_dir),
            output_root=output_root,
            config_path=Path(args.config),
            device=args.device,
            resume=not args.no_resume,
        )
        write_progress(output_root, list(specs.values()))
    elif args.command == "execute":
        execute_manifest(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
