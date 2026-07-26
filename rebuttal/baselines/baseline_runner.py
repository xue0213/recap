"""Resumable Phase 2 runner for supervised OFA baselines.

Formal scope and method settings are locked in
``rebuttal/BASELINE_OFA_REPROTOCOL.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from rebuttal.baselines.baseline_common import (  # type: ignore
        LabelVault,
        atomic_json,
        atomic_npz,
        atomic_torch_save,
        environment_metadata,
        prepare_arc_graph,
        set_seed,
        sha256_array,
        sha256_file,
        utc_now,
    )
    from rebuttal.baselines.baseline_models import (  # type: ignore
        ArcModel,
        SparseAffinityEncoder,
        ia_forward,
        ia_invariant_score,
        ia_training_loss,
        load_ia_vendor_model,
        minmax_anomaly_from_affinity,
        sparse_affinity_message,
    )
    from rebuttal.baselines.baseline_protocol import (  # type: ignore
        BaselineRunSpec,
        build_manifest,
        expected_evaluations,
        validate_manifest,
    )
else:
    from .baseline_common import (
        LabelVault,
        atomic_json,
        atomic_npz,
        atomic_torch_save,
        environment_metadata,
        prepare_arc_graph,
        set_seed,
        sha256_array,
        sha256_file,
        utc_now,
    )
    from .baseline_models import (
        ArcModel,
        SparseAffinityEncoder,
        ia_forward,
        ia_invariant_score,
        ia_training_loss,
        load_ia_vendor_model,
        minmax_anomaly_from_affinity,
        sparse_affinity_message,
    )
    from .baseline_protocol import (
        BaselineRunSpec,
        build_manifest,
        expected_evaluations,
        validate_manifest,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = Path("/root/autodl-tmp/recap/dataset")
DEFAULT_VENDOR_ROOT = Path("/root/autodl-tmp/recap_baselines/vendor")
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "rebuttal" / "artifacts" / "phase2_baselines"
)
PROTOCOL_PATH = PROJECT_ROOT / "rebuttal" / "BASELINE_OFA_REPROTOCOL.md"
UPSTREAM_MANIFEST_PATH = (
    PROJECT_ROOT / "rebuttal" / "baselines" / "upstream_manifest.json"
)


def score_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
    }


def run_dir(output_root: Path, spec: BaselineRunSpec) -> Path:
    return output_root / "runs" / spec.run_id


def calibration_path(
    output_root: Path, method: str, setting: str
) -> Path:
    slug = method.lower().replace("-", "_")
    return output_root / "calibration" / f"{slug}__setting_{setting}.json"


def save_manifest(output_root: Path) -> Path:
    specs = build_manifest()
    validate_manifest(specs)
    path = output_root / "manifest.json"
    atomic_json(
        path,
        {
            "format": "recap_phase2_baseline_manifest_v1",
            "created_at": utc_now(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "upstream_manifest_sha256": sha256_file(UPSTREAM_MANIFEST_PATH),
            "training_runs": len(specs),
            "final_evaluations": expected_evaluations(),
            "runs": [spec.to_dict() for spec in specs],
        },
    )
    return path


def load_manifest(output_root: Path) -> list[BaselineRunSpec]:
    path = output_root / "manifest.json"
    if not path.exists():
        save_manifest(output_root)
    payload = json.loads(path.read_text())
    specs = [BaselineRunSpec.from_dict(value) for value in payload["runs"]]
    validate_manifest(specs)
    return specs


def _save_target_scores(
    *,
    directory: Path,
    name: str,
    scores: np.ndarray,
    query_mask: np.ndarray,
    context_indices: np.ndarray,
    components: dict[str, np.ndarray] | None,
    vault: LabelVault,
) -> Path:
    path = directory / "scores" / f"{name}.npz"
    arrays: dict[str, Any] = {
        "scores": np.asarray(scores, dtype=np.float32),
        "query_mask": np.asarray(query_mask, dtype=np.bool_),
        "context_indices": np.asarray(context_indices, dtype=np.int64),
    }
    for key, value in (components or {}).items():
        arrays[key] = np.asarray(value, dtype=np.float32)
    atomic_npz(path, **arrays)
    vault.mark_score_frozen(
        name,
        score_path=path,
        score_sha256=sha256_array(arrays["scores"]),
        query_mask_sha256=sha256_array(arrays["query_mask"]),
    )
    return path


def _base_run_metadata(
    spec: BaselineRunSpec,
    dataset_dir: Path,
    vendor_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "format": "recap_phase2_baseline_run_v1",
        "run": spec.to_dict(),
        "started_at": utc_now(),
        "dataset_dir": str(dataset_dir.resolve()),
        "vendor_root": str(vendor_root.resolve()),
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "upstream_manifest_path": str(UPSTREAM_MANIFEST_PATH),
        "upstream_manifest_sha256": sha256_file(UPSTREAM_MANIFEST_PATH),
        "environment": environment_metadata(device),
    }


def _prepare_graphs(
    spec: BaselineRunSpec,
    dataset_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    names = tuple(dict.fromkeys(spec.source_graphs + spec.target_graphs))
    graphs = {
        name: prepare_arc_graph(dataset_dir, name, device, num_hops=2)
        for name in names
    }
    return graphs, time.perf_counter() - started


def run_arc(
    spec: BaselineRunSpec,
    *,
    dataset_dir: Path,
    vendor_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    directory = run_dir(output_root, spec)
    if (directory / "complete.json").exists() and smoke_epochs is None:
        return json.loads((directory / "complete.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    metadata = _base_run_metadata(spec, dataset_dir, vendor_root, device)
    metadata["resolved_config"] = {
        "method": "ARC",
        "feature_dims": 64,
        "num_hops": 2,
        "hidden_feats": 1024,
        "num_layers": 4,
        "num_prompt": 10,
        "target_normal_context": 10,
        "epochs": smoke_epochs or 40,
        "formal_epochs": 40,
        "lr": 1e-5,
        "weight_decay": 5e-5,
    }
    atomic_json(directory / "run_start.json", metadata)

    vault = LabelVault(dataset_dir)
    graphs, preparation_seconds = _prepare_graphs(spec, dataset_dir, device)
    source_labels = {
        name: torch.from_numpy(vault.load_source(name)).to(device)
        for name in spec.source_graphs
    }

    # Official order: set the seed, sample all target contexts, then initialize
    # and train the detector. Target labels are permitted only for ARC context.
    set_seed(spec.seed)
    contexts: dict[str, torch.Tensor] = {}
    query_masks: dict[str, torch.Tensor] = {}
    for name in spec.target_graphs:
        labels = vault.load_arc_context(name)
        normal = np.flatnonzero(labels == 0).tolist()
        random.shuffle(normal)
        context = torch.tensor(normal[:10], device=device, dtype=torch.long)
        mask = torch.ones(labels.shape[0], device=device, dtype=torch.bool)
        mask[context] = False
        contexts[name] = context
        query_masks[name] = mask

    model = ArcModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-5, weight_decay=5e-5
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    loss_log: list[dict[str, Any]] = []
    epochs = smoke_epochs or 40
    for epoch in range(epochs):
        epoch_losses = []
        for name in spec.source_graphs:
            model.train()
            residual = model(graphs[name].propagated)
            loss = model.cross_attention.training_loss(
                residual, source_labels[name], 10
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            loss_log.append(
                {
                    "epoch": epoch + 1,
                    "source_loss_mean": float(np.mean(epoch_losses)),
                    "source_losses": epoch_losses,
                }
            )
    training_seconds = time.perf_counter() - training_started
    checkpoint_path = directory / "checkpoint.pt"
    atomic_torch_save(
        checkpoint_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "spec": spec.to_dict(),
            "epochs": epochs,
        },
    )

    evaluation_started = time.perf_counter()
    target_results: dict[str, dict[str, Any]] = {}
    reference_scores: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for name in spec.target_graphs:
            residual = model(graphs[name].propagated)
            scores_tensor = model.cross_attention.target_score(
                residual, contexts[name], query_masks[name]
            )
            scores = scores_tensor.detach().cpu().numpy().astype(np.float32)
            query_mask = query_masks[name].detach().cpu().numpy()
            context = contexts[name].detach().cpu().numpy()
            _save_target_scores(
                directory=directory,
                name=name,
                scores=scores,
                query_mask=query_mask,
                context_indices=context,
                components=None,
                vault=vault,
            )
            labels = vault.load_target_for_evaluation(name)
            metrics = score_metrics(labels[query_mask], scores)
            target_results[name] = {
                **metrics,
                "nodes": int(labels.shape[0]),
                "query_nodes": int(query_mask.sum()),
                "context_nodes": int((~query_mask).sum()),
            }
            reference_scores[name] = scores
    evaluation_seconds = time.perf_counter() - evaluation_started

    # Checkpoint-reload gate on every target to catch state or context drift.
    reloaded = ArcModel().to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    reloaded.load_state_dict(checkpoint["model"])
    reloaded.eval()
    reload_diffs = {}
    with torch.no_grad():
        for name in spec.target_graphs:
            residual = reloaded(graphs[name].propagated)
            score = reloaded.cross_attention.target_score(
                residual, contexts[name], query_masks[name]
            )
            actual = score.detach().cpu().numpy()
            reload_diffs[name] = float(
                np.max(np.abs(actual - reference_scores[name]))
            )
    reload_max_diff = max(reload_diffs.values(), default=0.0)
    reload_passed = all(
        np.allclose(
            reference_scores[name],
            (
                reloaded.cross_attention.target_score(
                    reloaded(graphs[name].propagated),
                    contexts[name],
                    query_masks[name],
                )
                .detach()
                .cpu()
                .numpy()
            ),
            atol=1e-5,
            rtol=1e-5,
        )
        for name in spec.target_graphs
    )
    if not reload_passed:
        raise RuntimeError(f"{spec.run_id}: checkpoint reload audit failed")

    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if torch.cuda.is_available()
        else 0
    )
    result = {
        **metadata,
        "completed_at": utc_now(),
        "status": "smoke_complete" if smoke_epochs is not None else "complete",
        "preparation_seconds": preparation_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "peak_gpu_memory_bytes": peak_bytes,
        "loss_log": loss_log,
        "target_results": target_results,
        "dataset_hashes": {
            name: graphs[name].raw_sha256 for name in graphs
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "reload_passed": reload_passed,
            "reload_max_abs_diff": reload_max_diff,
            "reload_target_diffs": reload_diffs,
        },
        "label_audit": vault.audit(),
    }
    atomic_json(directory / "result.json", result)
    if smoke_epochs is None:
        atomic_json(directory / "complete.json", result)
    return result


def _sample_unlabeled_references(
    node_count: int, count: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = list(range(node_count))
    random.shuffle(indices)
    reference = torch.tensor(indices[:count], dtype=torch.long)
    mask = torch.ones(node_count, dtype=torch.bool)
    mask[reference] = False
    return reference, mask


def _ia_source_calibration(
    *,
    spec: BaselineRunSpec,
    graphs: dict[str, Any],
    source_labels: dict[str, torch.Tensor],
    invariant_model: torch.nn.Module,
    affinity_model: SparseAffinityEncoder,
    final_codebook: torch.Tensor,
    grid: tuple[float, ...],
) -> tuple[float, list[dict[str, Any]]]:
    rng_state = random.getstate()
    random.seed(2026 + ord(spec.setting))
    records: list[dict[str, Any]] = []
    try:
        components: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        invariant_model.eval()
        affinity_model.eval()
        with torch.no_grad():
            for name in spec.source_graphs:
                graph = graphs[name]
                reference_cpu, query_cpu = _sample_unlabeled_references(
                    graph.node_count
                )
                reference_mask = ~query_cpu
                residual, _, _, _ = ia_forward(
                    invariant_model, graph.propagated, name
                )
                invariant = ia_invariant_score(
                    invariant_model,
                    residual,
                    final_codebook,
                    reference_mask.to(residual.device),
                )
                affinity_embedding = affinity_model(
                    graph.affinity_adj_norm, graph.x
                )
                _, message = sparse_affinity_message(
                    affinity_embedding,
                    graph.affinity_edge_index,
                    graph.node_count,
                )
                affinity = minmax_anomaly_from_affinity(message)[
                    query_cpu.to(message.device)
                ]
                labels = source_labels[name][query_cpu.to(source_labels[name].device)]
                components[name] = (
                    invariant.detach().cpu().numpy(),
                    affinity.detach().cpu().numpy(),
                    labels.detach().cpu().numpy(),
                )
        for weight in grid:
            per_dataset = []
            for name, (invariant, affinity, labels) in components.items():
                fused = (1.0 - weight) * invariant + weight * affinity
                per_dataset.append(score_metrics(labels, fused))
            records.append(
                {
                    "weight": weight,
                    "source_macro_AUROC": float(
                        np.mean([value["AUROC"] for value in per_dataset])
                    ),
                    "source_macro_AUPRC": float(
                        np.mean([value["AUPRC"] for value in per_dataset])
                    ),
                }
            )
    finally:
        random.setstate(rng_state)
    best = max(
        records,
        key=lambda value: (
            value["source_macro_AUROC"],
            value["source_macro_AUPRC"],
            -value["weight"],
        ),
    )
    return float(best["weight"]), records


def run_ia_ggad(
    spec: BaselineRunSpec,
    *,
    dataset_dir: Path,
    vendor_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    directory = run_dir(output_root, spec)
    if (directory / "complete.json").exists() and smoke_epochs is None:
        return json.loads((directory / "complete.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    metadata = _base_run_metadata(spec, dataset_dir, vendor_root, device)
    metadata["resolved_config"] = {
        "method": "IA-GGAD",
        "feature_dims": 64,
        "num_hops": 2,
        "hidden_feats": 1024,
        "codebook_size": 2048,
        "affinity_hidden_feats": 128,
        "num_prompt": 10,
        "unlabeled_internal_target_references": 10,
        "epochs": smoke_epochs or 40,
        "formal_epochs": 40,
        "lr": 1e-5,
        "weight_decay": 5e-5,
        "fusion_calibration": "setting seed-0 source macro only",
    }
    atomic_json(directory / "run_start.json", metadata)

    vault = LabelVault(dataset_dir)
    graphs, preparation_seconds = _prepare_graphs(spec, dataset_dir, device)
    source_labels = {
        name: torch.from_numpy(vault.load_source(name)).to(device)
        for name in spec.source_graphs
    }
    set_seed(spec.seed)
    target_references: dict[str, torch.Tensor] = {}
    target_query_masks: dict[str, torch.Tensor] = {}
    for name in spec.target_graphs:
        reference, query = _sample_unlabeled_references(
            graphs[name].node_count
        )
        target_references[name] = reference.to(device)
        target_query_masks[name] = query.to(device)

    invariant_model = load_ia_vendor_model(vendor_root).to(device)
    affinity_model = SparseAffinityEncoder().to(device)
    invariant_optimizer = torch.optim.AdamW(
        invariant_model.parameters(), lr=1e-5, weight_decay=5e-5
    )
    affinity_optimizer = torch.optim.AdamW(
        affinity_model.parameters(), lr=1e-5, weight_decay=5e-5
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    loss_log: list[dict[str, Any]] = []
    final_source_codebooks: dict[str, torch.Tensor] = {}
    epochs = smoke_epochs or 40
    for epoch in range(epochs):
        invariant_losses = []
        affinity_losses = []
        for name in spec.source_graphs:
            graph = graphs[name]
            invariant_model.train()
            affinity_model.train()
            residual, code_loss, quantized, codebook = ia_forward(
                invariant_model, graph.propagated, name
            )
            invariant_loss = ia_training_loss(
                invariant_model,
                residual,
                quantized,
                codebook,
                source_labels[name],
                num_prompt=10,
            ) + code_loss.squeeze()
            affinity_embedding = affinity_model(
                graph.affinity_adj_norm, graph.x
            )
            affinity_loss, _ = sparse_affinity_message(
                affinity_embedding,
                graph.affinity_edge_index,
                graph.node_count,
            )
            invariant_optimizer.zero_grad(set_to_none=True)
            affinity_optimizer.zero_grad(set_to_none=True)
            invariant_loss.backward()
            affinity_loss.backward()
            invariant_optimizer.step()
            affinity_optimizer.step()
            invariant_losses.append(float(invariant_loss.detach().cpu()))
            affinity_losses.append(float(affinity_loss.detach().cpu()))
            if epoch == epochs - 1:
                final_source_codebooks[name] = codebook.detach().clone()
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 10 == 0:
            loss_log.append(
                {
                    "epoch": epoch + 1,
                    "invariant_loss_mean": float(np.mean(invariant_losses)),
                    "affinity_loss_mean": float(np.mean(affinity_losses)),
                    "invariant_losses": invariant_losses,
                    "affinity_losses": affinity_losses,
                }
            )
    training_seconds = time.perf_counter() - training_started
    final_codebook = torch.cat(
        [final_source_codebooks[name] for name in spec.source_graphs], dim=0
    ).to(device)

    grid = (
        0.0,
        0.01,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
        1.0,
    )
    locked_calibration_path = calibration_path(
        output_root, "IA-GGAD", spec.setting
    )
    if smoke_epochs is not None:
        fusion_weight, calibration_records = _ia_source_calibration(
            spec=spec,
            graphs=graphs,
            source_labels=source_labels,
            invariant_model=invariant_model,
            affinity_model=affinity_model,
            final_codebook=final_codebook,
            grid=grid,
        )
        calibration_status = "smoke_only"
    elif spec.seed == 0:
        fusion_weight, calibration_records = _ia_source_calibration(
            spec=spec,
            graphs=graphs,
            source_labels=source_labels,
            invariant_model=invariant_model,
            affinity_model=affinity_model,
            final_codebook=final_codebook,
            grid=grid,
        )
        atomic_json(
            locked_calibration_path,
            {
                "format": "recap_phase2_source_only_calibration_v1",
                "method": "IA-GGAD",
                "setting": spec.setting,
                "calibration_seed": 0,
                "selected_weight": fusion_weight,
                "grid_records": calibration_records,
                "target_labels_used": False,
                "created_at": utc_now(),
            },
        )
        calibration_status = "created_and_locked"
    else:
        if not locked_calibration_path.exists():
            raise FileNotFoundError(
                f"Seed-0 calibration must run first: {locked_calibration_path}"
            )
        calibration = json.loads(locked_calibration_path.read_text())
        fusion_weight = float(calibration["selected_weight"])
        calibration_records = calibration["grid_records"]
        calibration_status = "reused_seed0_lock"

    checkpoint_path = directory / "checkpoint.pt"
    atomic_torch_save(
        checkpoint_path,
        {
            "invariant_model": invariant_model.state_dict(),
            "affinity_model": affinity_model.state_dict(),
            "invariant_optimizer": invariant_optimizer.state_dict(),
            "affinity_optimizer": affinity_optimizer.state_dict(),
            "source_codebook": final_codebook.detach().cpu(),
            "fusion_weight": fusion_weight,
            "spec": spec.to_dict(),
            "epochs": epochs,
        },
    )

    evaluation_started = time.perf_counter()
    target_results: dict[str, dict[str, Any]] = {}
    reference_scores: dict[str, np.ndarray] = {}
    invariant_model.eval()
    affinity_model.eval()
    with torch.no_grad():
        for name in spec.target_graphs:
            graph = graphs[name]
            residual, _, _, _ = ia_forward(
                invariant_model, graph.propagated, name
            )
            reference_mask = ~target_query_masks[name]
            invariant_score = ia_invariant_score(
                invariant_model,
                residual,
                final_codebook,
                reference_mask,
            )
            affinity_embedding = affinity_model(
                graph.affinity_adj_norm, graph.x
            )
            _, affinity_message = sparse_affinity_message(
                affinity_embedding,
                graph.affinity_edge_index,
                graph.node_count,
            )
            affinity_full = minmax_anomaly_from_affinity(affinity_message)
            affinity_score = affinity_full[target_query_masks[name]]
            fused = (
                (1.0 - fusion_weight) * invariant_score
                + fusion_weight * affinity_score
            )
            scores = fused.detach().cpu().numpy().astype(np.float32)
            invariant_values = (
                invariant_score.detach().cpu().numpy().astype(np.float32)
            )
            affinity_values = (
                affinity_score.detach().cpu().numpy().astype(np.float32)
            )
            query_mask = target_query_masks[name].detach().cpu().numpy()
            context = target_references[name].detach().cpu().numpy()
            _save_target_scores(
                directory=directory,
                name=name,
                scores=scores,
                query_mask=query_mask,
                context_indices=context,
                components={
                    "invariant_score": invariant_values,
                    "affinity_score": affinity_values,
                },
                vault=vault,
            )
            labels = vault.load_target_for_evaluation(name)
            metrics = score_metrics(labels[query_mask], scores)
            target_results[name] = {
                **metrics,
                "nodes": int(labels.shape[0]),
                "query_nodes": int(query_mask.sum()),
                "internal_reference_nodes": int((~query_mask).sum()),
                "fusion_weight": fusion_weight,
            }
            reference_scores[name] = scores
    evaluation_seconds = time.perf_counter() - evaluation_started

    # Full two-branch checkpoint reload audit.
    reloaded_invariant = load_ia_vendor_model(vendor_root).to(device)
    reloaded_affinity = SparseAffinityEncoder().to(device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    reloaded_invariant.load_state_dict(checkpoint["invariant_model"])
    reloaded_affinity.load_state_dict(checkpoint["affinity_model"])
    reloaded_codebook = checkpoint["source_codebook"].to(device)
    reloaded_invariant.eval()
    reloaded_affinity.eval()
    reload_diffs: dict[str, float] = {}
    with torch.no_grad():
        for name in spec.target_graphs:
            graph = graphs[name]
            residual, _, _, _ = ia_forward(
                reloaded_invariant, graph.propagated, name
            )
            invariant_score = ia_invariant_score(
                reloaded_invariant,
                residual,
                reloaded_codebook,
                ~target_query_masks[name],
            )
            affinity_embedding = reloaded_affinity(
                graph.affinity_adj_norm, graph.x
            )
            _, message = sparse_affinity_message(
                affinity_embedding,
                graph.affinity_edge_index,
                graph.node_count,
            )
            affinity_score = minmax_anomaly_from_affinity(message)[
                target_query_masks[name]
            ]
            fused = (
                (1.0 - fusion_weight) * invariant_score
                + fusion_weight * affinity_score
            )
            actual = fused.detach().cpu().numpy()
            reload_diffs[name] = float(
                np.max(np.abs(actual - reference_scores[name]))
            )
    reload_max_diff = max(reload_diffs.values(), default=0.0)
    reload_passed = reload_max_diff <= 1e-5
    if not reload_passed:
        raise RuntimeError(
            f"{spec.run_id}: IA-GGAD reload max diff {reload_max_diff}"
        )

    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if torch.cuda.is_available()
        else 0
    )
    result = {
        **metadata,
        "completed_at": utc_now(),
        "status": "smoke_complete" if smoke_epochs is not None else "complete",
        "preparation_seconds": preparation_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "peak_gpu_memory_bytes": peak_bytes,
        "loss_log": loss_log,
        "fusion_calibration": {
            "status": calibration_status,
            "path": str(locked_calibration_path),
            "selected_weight": fusion_weight,
            "grid_records": calibration_records,
            "target_labels_used": False,
        },
        "target_results": target_results,
        "dataset_hashes": {
            name: graphs[name].raw_sha256 for name in graphs
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "reload_passed": reload_passed,
            "reload_max_abs_diff": reload_max_diff,
            "reload_target_diffs": reload_diffs,
        },
        "label_audit": vault.audit(),
    }
    atomic_json(directory / "result.json", result)
    if smoke_epochs is None:
        atomic_json(directory / "complete.json", result)
    return result


def execute_spec(
    spec: BaselineRunSpec,
    *,
    dataset_dir: Path,
    vendor_root: Path,
    output_root: Path,
    device: torch.device,
    smoke_epochs: int | None = None,
) -> dict[str, Any]:
    if spec.method == "ARC":
        return run_arc(
            spec,
            dataset_dir=dataset_dir,
            vendor_root=vendor_root,
            output_root=output_root,
            device=device,
            smoke_epochs=smoke_epochs,
        )
    if spec.method == "IA-GGAD":
        return run_ia_ggad(
            spec,
            dataset_dir=dataset_dir,
            vendor_root=vendor_root,
            output_root=output_root,
            device=device,
            smoke_epochs=smoke_epochs,
        )
    raise NotImplementedError(f"Adapter pending for {spec.method}")


def status(output_root: Path) -> dict[str, Any]:
    specs = load_manifest(output_root)
    complete = []
    partial = []
    pending = []
    evaluations = 0
    for spec in specs:
        directory = run_dir(output_root, spec)
        if (directory / "complete.json").exists():
            complete.append(spec.run_id)
            payload = json.loads((directory / "complete.json").read_text())
            evaluations += len(payload.get("target_results", {}))
        elif directory.exists():
            partial.append(spec.run_id)
        else:
            pending.append(spec.run_id)
    return {
        "training_complete": len(complete),
        "training_expected": len(specs),
        "evaluations_complete": evaluations,
        "evaluations_expected": expected_evaluations(),
        "complete": complete,
        "partial": partial,
        "pending": pending,
    }


def select_spec(
    specs: list[BaselineRunSpec], run_id_value: str
) -> BaselineRunSpec:
    matches = [spec for spec in specs if spec.run_id == run_id_value]
    if len(matches) != 1:
        raise KeyError(f"Unknown run ID: {run_id_value}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR
    )
    parser.add_argument(
        "--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    parser.add_argument("--device", default="cuda:0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("manifest")
    subparsers.add_parser("status")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--smoke-epochs", type=int)
    pending_parser = subparsers.add_parser("run-pending")
    pending_parser.add_argument("--method")
    pending_parser.add_argument("--setting")
    pending_parser.add_argument("--max-runs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "manifest":
        print(save_manifest(output_root))
        return
    if args.command == "status":
        print(json.dumps(status(output_root), indent=2))
        return

    specs = load_manifest(output_root)
    device = torch.device(args.device)
    if args.command == "run":
        spec = select_spec(specs, args.run_id)
        result = execute_spec(
            spec,
            dataset_dir=args.dataset_dir.resolve(),
            vendor_root=args.vendor_root.resolve(),
            output_root=output_root,
            device=device,
            smoke_epochs=args.smoke_epochs,
        )
        print(json.dumps({"run_id": spec.run_id, "status": result["status"]}))
        return

    selected = [
        spec
        for spec in specs
        if (args.method is None or spec.method == args.method)
        and (args.setting is None or spec.setting == args.setting)
        and not (run_dir(output_root, spec) / "complete.json").exists()
    ]
    if args.max_runs is not None:
        selected = selected[: args.max_runs]
    for index, spec in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {spec.run_id}", flush=True)
        execute_spec(
            spec,
            dataset_dir=args.dataset_dir.resolve(),
            vendor_root=args.vendor_root.resolve(),
            output_root=output_root,
            device=device,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps(status(output_root), indent=2))


if __name__ == "__main__":
    main()

