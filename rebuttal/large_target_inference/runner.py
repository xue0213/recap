"""Preflight, gates, and resumable formal large-target inference runner."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig  # noqa: E402
from model import EgoCluster, recap  # noqa: E402
from rebuttal.large_target_inference.ann import (  # noqa: E402
    build_exact_candidates,
    build_faiss_candidates,
    candidate_recall,
    exact_neighbors_for_queries,
    validate_candidates,
)
from rebuttal.large_target_inference.common import (  # noqa: E402
    LabelVault,
    append_jsonl,
    atomic_csv,
    atomic_json,
    atomic_npy,
    measured_phase,
    sha256_file,
    stable_hash,
    synchronize,
    utc_now,
)
from rebuttal.large_target_inference.data import (  # noqa: E402
    FEATURE_ALIGNMENT_VERSION,
    canonical_paths,
    initial_residual,
    load_and_propagate,
)
from rebuttal.large_target_inference.protocol import (  # noqa: E402
    ANN_CONFIG,
    CHECKPOINTS,
    DATASETS,
    MODEL_LOCK,
    SEEDS,
    TARGETS,
    InferenceSpec,
    build_manifest,
)
from rebuttal.large_target_inference.scoring import (  # noqa: E402
    compute_score_components_chunked,
)


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "rebuttal" / "artifacts" / "large_target_inference"
)
DEFAULT_PROTOCOL_PATH = (
    PROJECT_ROOT / "rebuttal" / "LARGE_TARGET_INFERENCE_PROTOCOL.md"
)
SCORE_BATCH_SIZE = 1024
FIDELITY_QUERY_COUNT = 512
FIDELITY_QUERY_SEED = 20260726


def _command_text(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return (result.stdout or result.stderr).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def environment_record(device: str) -> dict:
    try:
        import faiss

        faiss_version = str(faiss.__version__)
    except Exception:
        faiss_version = "unavailable"
    return {
        "captured_at": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "faiss": faiss_version,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": device,
        "gpu": _command_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "cpu": _command_text(["lscpu"]),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL_PATH),
    }


def checkpoint_path(checkpoint_root: Path, seed: int) -> Path:
    return checkpoint_root / CHECKPOINTS[seed]["relative_path"]


def validate_checkpoint(
    path: Path, seed: int, device: str = "cpu"
) -> tuple[dict, ModelConfig]:
    expected_hash = CHECKPOINTS[seed]["sha256"]
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise ValueError(
            f"Checkpoint hash mismatch for seed {seed}: {observed_hash}"
        )
    payload = torch.load(
        path, map_location=device, weights_only=False
    )
    if payload.get("format") != "recap_phase1_checkpoint_v1":
        raise ValueError(f"Unexpected checkpoint format: {path}")
    run_spec = payload.get("run_spec", {})
    if run_spec.get("setting") != "A" or int(run_spec.get("seed", -1)) != seed:
        raise ValueError(f"Checkpoint is not accepted Setting-A seed {seed}")
    expected_sources = ("pubmed", "Flickr", "questions", "YelpChi")
    if tuple(run_spec.get("source_graphs", ())) != expected_sources:
        raise ValueError(f"Unexpected Setting-A sources: {run_spec}")
    if int(payload.get("epoch", -1)) != 100:
        raise ValueError(f"Checkpoint is not epoch 100: {path}")
    model_config = ModelConfig.from_dict(payload["model_config"])
    mismatches = {
        key: {"expected": expected, "actual": getattr(model_config, key)}
        for key, expected in MODEL_LOCK.items()
        if getattr(model_config, key) != expected
    }
    if mismatches:
        raise ValueError(f"Locked model mismatch: {mismatches}")
    return payload, model_config


def _finite_memmap(path: Path, chunk_size: int = 500_000) -> bool:
    array = np.load(path, mmap_mode="r")
    for start in range(0, len(array), chunk_size):
        if not np.all(np.isfinite(array[start : start + chunk_size])):
            return False
    return True


def run_preflight(
    *,
    dataset_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    device: str,
) -> dict:
    output = {
        "format": "recap_large_target_preflight_v1",
        "created_at": utc_now(),
        "manifest": [item.to_dict() for item in build_manifest()],
        "environment": environment_record(device),
        "datasets": {},
        "checkpoints": {},
    }
    for name in TARGETS:
        expected = DATASETS[name]
        paths = canonical_paths(dataset_root, name, MODEL_LOCK["dims"])
        required_keys = [
            "metadata",
            "adjacency",
            "features",
            "aligned_features",
            "labels",
        ]
        missing = [str(paths[key]) for key in required_keys if not paths[key].exists()]
        if missing:
            raise FileNotFoundError(f"{name}: missing {missing}")
        adjacency = sp.load_npz(paths["adjacency"]).tocsr()
        raw_features = np.load(paths["features"], mmap_mode="r")
        aligned = np.load(paths["aligned_features"], mmap_mode="r")
        labels = np.asarray(
            np.load(paths["labels"], mmap_mode="r"), dtype=np.int64
        ).reshape(-1)
        mask = (
            np.asarray(
                np.load(paths["evaluation_mask"], mmap_mode="r"),
                dtype=np.bool_,
            ).reshape(-1)
            if paths["evaluation_mask"].exists()
            else np.ones(expected["nodes"], dtype=np.bool_)
        )
        if adjacency.shape != (expected["nodes"], expected["nodes"]):
            raise ValueError(f"{name}: invalid adjacency shape")
        if int(adjacency.nnz) != expected["adjacency_nnz"]:
            raise ValueError(f"{name}: invalid adjacency nnz")
        if raw_features.shape != (expected["nodes"], expected["raw_features"]):
            raise ValueError(f"{name}: invalid raw features")
        if aligned.shape != (expected["nodes"], MODEL_LOCK["dims"]):
            raise ValueError(f"{name}: invalid aligned features")
        if labels.shape != (expected["nodes"],):
            raise ValueError(f"{name}: invalid label shape")
        if mask.shape != (expected["nodes"],):
            raise ValueError(f"{name}: invalid evaluation mask")
        query_labels = labels[mask]
        if set(np.unique(query_labels).tolist()) != {0, 1}:
            raise ValueError(f"{name}: evaluation labels are not binary")
        if int((query_labels == 1).sum()) != expected["anomalies"]:
            raise ValueError(f"{name}: anomaly count mismatch")
        if int(mask.sum()) != expected["evaluation_nodes"]:
            raise ValueError(f"{name}: evaluation population mismatch")
        if not _finite_memmap(paths["aligned_features"]):
            raise FloatingPointError(f"{name}: non-finite aligned features")
        component_hashes = {
            key: sha256_file(paths[key])
            for key in required_keys
        }
        if paths["evaluation_mask"].exists():
            component_hashes["evaluation_mask"] = sha256_file(
                paths["evaluation_mask"]
            )
        output["datasets"][name] = {
            "nodes": expected["nodes"],
            "adjacency_nnz": int(adjacency.nnz),
            "raw_features": int(raw_features.shape[1]),
            "aligned_features": int(aligned.shape[1]),
            "evaluation_nodes": int(mask.sum()),
            "anomalies": int((query_labels == 1).sum()),
            "feature_alignment_version": FEATURE_ALIGNMENT_VERSION,
            "primary_knn": expected["primary_knn"],
            "component_hashes": component_hashes,
        }
        del adjacency, raw_features, aligned, labels, mask, query_labels
        gc.collect()
    for seed in SEEDS:
        path = checkpoint_path(checkpoint_root, seed)
        _, model_config = validate_checkpoint(path, seed)
        output["checkpoints"][str(seed)] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "model_config": model_config.to_dict(),
        }
    output["passed"] = True
    output["preflight_hash"] = stable_hash(output)
    path = output_root / "preflight" / "preflight.json"
    atomic_json(path, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def _ann_kwargs() -> dict:
    return {
        "nlist": ANN_CONFIG["nlist"],
        "nprobe": ANN_CONFIG["nprobe"],
        "pq_m": ANN_CONFIG["pq_m"],
        "train_size": ANN_CONFIG["train_size"],
        "query_batch_size": ANN_CONFIG["query_batch_size"],
        "add_batch_size": ANN_CONFIG["add_batch_size"],
        "rerank_factor": ANN_CONFIG["rerank_factor"],
        "max_rerank_candidates": ANN_CONFIG["max_rerank_candidates"],
        "seed": ANN_CONFIG["seed"],
    }


def run_numerical_gates(output_root: Path, device: str) -> dict:
    torch.manual_seed(314159)
    resolved = torch.device(device)
    residual = torch.randn(127, 19, device=resolved)
    cluster = EgoCluster(
        embed_dim=19,
        num_clusters=7,
        knn_k=8,
        tau_s=0.3,
        tau_c=0.3,
        tau_e=1.0,
        beta=0.02,
        sim_block_size=31,
        knn_cache_enabled=False,
    ).to(resolved)
    candidates = (
        cluster._select_knn_candidates(residual)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int32)
    )
    native = cluster.compute_score_components(
        residual, E_init=residual, cache_key=None
    )
    chunked = compute_score_components_chunked(
        residual=residual,
        cluster=cluster,
        candidates=candidates,
        score_batch_size=23,
    )
    component_differences = {
        "total": float(
            np.max(
                np.abs(
                    native["total"].detach().cpu().numpy()
                    - chunked["total"]
                )
            )
        ),
        "adhesion_raw": float(
            np.max(
                np.abs(
                    native["adhesion_raw"].detach().cpu().numpy()
                    - chunked["adhesion_raw"]
                )
            )
        ),
        "context_raw": float(
            np.max(
                np.abs(
                    native["scale_raw"].detach().cpu().numpy()
                    - chunked["context_raw"]
                )
            )
        ),
    }
    if component_differences["total"] > 1e-5:
        raise AssertionError(
            f"Chunked score equivalence failed: {component_differences}"
        )

    gate_dir = output_root / "gates" / "ann_determinism"
    embeddings = torch.randn(20_000, 128, device=resolved)
    first = gate_dir / "first.npy"
    second = gate_dir / "second.npy"
    first_meta = build_faiss_candidates(
        embeddings, k=64, output_path=first, **_ann_kwargs()
    )
    second_meta = build_faiss_candidates(
        embeddings, k=64, output_path=second, **_ann_kwargs()
    )
    deterministic = bool(
        np.array_equal(
            np.load(first, mmap_mode="r"),
            np.load(second, mmap_mode="r"),
        )
    )
    if not deterministic:
        raise AssertionError("FAISS fixed-seed candidate gate is not deterministic")
    report = {
        "format": "recap_large_target_numerical_gates_v1",
        "passed": True,
        "manifest_training_runs": 0,
        "manifest_primary_evaluations": len(build_manifest()),
        "chunked_native_component_max_abs_difference": component_differences,
        "ann_fixed_seed_deterministic": deterministic,
        "ann_first": first_meta,
        "ann_second": second_meta,
        "created_at": utc_now(),
    }
    atomic_json(output_root / "gates" / "numerical_gates.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def _candidate_paths(shared_dir: Path, target: str) -> dict[str, Path]:
    return {
        "exact": shared_dir / f"{target}_exact_top64.npy",
        "faiss_ivfpq": shared_dir / f"{target}_ann_top64.npy",
    }


def _candidate_setup(
    *,
    target: str,
    residual_initial: torch.Tensor,
    shared_dir: Path,
    device: str,
) -> tuple[dict[str, dict], dict]:
    paths = _candidate_paths(shared_dir, target)
    node_count = int(residual_initial.shape[0])
    k = int(MODEL_LOCK["knn_k"])
    candidate_metadata: dict[str, dict] = {}
    phase_records: list[dict] = []

    required_routes = (
        ("exact", "faiss_ivfpq")
        if target == "tfinance"
        else ("faiss_ivfpq",)
    )
    for route in required_routes:
        path = paths[route]
        with measured_phase(f"candidate_build_{route}", device) as phase:
            if path.exists():
                metadata = validate_candidates(path, node_count, k)
                metadata["route"] = route
                metadata["state"] = "reused"
            elif route == "exact":
                metadata = build_exact_candidates(
                    residual_initial,
                    k=k,
                    output_path=path,
                    query_batch_size=256,
                )
                metadata["state"] = "new"
            else:
                metadata = build_faiss_candidates(
                    residual_initial,
                    k=k,
                    output_path=path,
                    **_ann_kwargs(),
                )
                metadata["state"] = "new"
            phase["candidate_path"] = str(path.resolve())
            phase["candidate_sha256"] = metadata["sha256"]
        phase_records.append(dict(phase))
        candidate_metadata[route] = metadata

    with measured_phase("ann_fidelity_queries", device) as phase:
        ann = np.load(paths["faiss_ivfpq"], mmap_mode="r")
        if target == "tfinance":
            exact = np.load(paths["exact"], mmap_mode="r")
            fidelity = candidate_recall(exact, ann)
            fidelity["query_selection"] = "all_nodes"
        else:
            rng = np.random.default_rng(FIDELITY_QUERY_SEED)
            query_indices = np.sort(
                rng.choice(
                    node_count,
                    size=min(FIDELITY_QUERY_COUNT, node_count),
                    replace=False,
                ).astype(np.int64)
            )
            exact_sample = exact_neighbors_for_queries(
                residual_initial,
                query_indices,
                k=k,
            )
            approximate_sample = np.asarray(ann[query_indices])
            fidelity = candidate_recall(exact_sample, approximate_sample)
            fidelity.update(
                {
                    "query_selection": "fixed_label_blind_uniform_sample",
                    "query_seed": FIDELITY_QUERY_SEED,
                    "query_indices_sha256": stable_hash(
                        query_indices.tolist()
                    ),
                }
            )
            atomic_npy(
                shared_dir / f"{target}_fidelity_query_indices.npy",
                query_indices,
            )
            atomic_npy(
                shared_dir / f"{target}_fidelity_exact_neighbors.npy",
                exact_sample,
            )
        phase["mean_recall"] = fidelity["mean_recall"]
    phase_records.append(dict(phase))
    return candidate_metadata, {
        "record": fidelity,
        "phase_records": phase_records,
    }


def _metric(labels: np.ndarray, mask: np.ndarray, scores: np.ndarray) -> dict:
    query_labels = labels[mask]
    query_scores = scores[mask]
    return {
        "auroc": float(roc_auc_score(query_labels, query_scores)),
        "auprc": float(average_precision_score(query_labels, query_scores)),
        "evaluation_nodes": int(mask.sum()),
        "anomalies": int((query_labels == 1).sum()),
        "anomaly_prevalence": float(query_labels.mean()),
    }


def _write_query_mask(
    context, run_dir: Path
) -> tuple[Path, str]:
    if context.evaluation_mask_path is None:
        mask = np.ones(context.node_count, dtype=np.bool_)
    else:
        mask = np.asarray(
            np.load(context.evaluation_mask_path, mmap_mode="r"),
            dtype=np.bool_,
        )
    path = run_dir / "query_mask.npy"
    atomic_npy(path, mask)
    return path, sha256_file(path)


def _run_checkpoint(
    *,
    spec: InferenceSpec,
    context,
    candidate_paths: dict[str, Path],
    checkpoint_root: Path,
    output_root: Path,
    device: str,
) -> dict:
    run_dir = output_root / "runs" / spec.run_id
    complete_path = run_dir / "complete.json"
    if complete_path.exists():
        with (run_dir / "result.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)
    run_dir.mkdir(parents=True, exist_ok=True)
    events_path = run_dir / "label_audit.jsonl"
    append_jsonl(
        run_dir / "events.jsonl",
        {"event": "run_started", "run_id": spec.run_id, "at": utc_now()},
    )
    atomic_json(
        run_dir / "status.json",
        {"status": "running", "run_id": spec.run_id, "at": utc_now()},
    )
    try:
        checkpoint = checkpoint_root / spec.checkpoint_relative_path
        phase_records: list[dict] = []
        with measured_phase("checkpoint_load", device) as phase:
            payload, model_config = validate_checkpoint(
                checkpoint, spec.seed, device="cpu"
            )
            model = recap(**model_config.to_dict())
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model = model.to(device)
            model.eval()
            phase["checkpoint_sha256"] = spec.checkpoint_sha256
        phase_records.append(dict(phase))

        with torch.inference_mode():
            with measured_phase("model_residual_forward", device) as phase:
                model(context.graph)
                residual = model._view_embeds[0]
                if not torch.isfinite(residual).all():
                    raise FloatingPointError("Non-finite model residual")
                # Candidates were frozen from the target-only initial residual.
                model._view_embeds_init = []
                phase["residual_shape"] = list(residual.shape)
            phase_records.append(dict(phase))

            routes = [spec.primary_knn]
            if spec.target == "tfinance":
                routes.append("faiss_ivfpq")
            route_outputs: dict[str, dict] = {}
            for route in routes:
                candidates = np.load(
                    candidate_paths[route], mmap_mode="r"
                )
                with measured_phase(f"score_{route}", device) as phase:
                    components = compute_score_components_chunked(
                        residual=residual,
                        cluster=model.ego_clusters[0],
                        candidates=candidates,
                        score_batch_size=SCORE_BATCH_SIZE,
                    )
                    phase["nodes"] = context.node_count
                    phase["score_batch_size"] = SCORE_BATCH_SIZE
                phase_records.append(dict(phase))
                route_dir = run_dir / route
                with measured_phase(f"serialize_{route}", device) as phase:
                    score_path = route_dir / "scores.npy"
                    atomic_npy(score_path, components["total"])
                    score_hash = sha256_file(score_path)
                    if route == spec.primary_knn:
                        atomic_npy(
                            run_dir / "hard_assignments.npy",
                            components["hard_assignments"],
                        )
                        atomic_npy(
                            run_dir / "community_usage.npy",
                            components["usage"],
                        )
                        atomic_json(
                            run_dir / "score_summary.json",
                            {
                                "effective_communities": (
                                    components["effective_communities"]
                                ),
                                "score_mean": float(
                                    np.mean(components["total"])
                                ),
                                "score_std": float(
                                    np.std(components["total"], ddof=0)
                                ),
                                "adhesion_raw_mean": float(
                                    np.mean(components["adhesion_raw"])
                                ),
                                "context_raw_mean": float(
                                    np.mean(components["context_raw"])
                                ),
                            },
                        )
                    phase["score_path"] = str(score_path.resolve())
                    phase["score_sha256"] = score_hash
                phase_records.append(dict(phase))
                route_outputs[route] = {
                    "score_path": score_path,
                    "score_sha256": score_hash,
                    "scores": components["total"],
                    "effective_communities": (
                        components["effective_communities"]
                    ),
                }
                del components, candidates

        mask_path, mask_hash = _write_query_mask(context, run_dir)
        vault = LabelVault(
            labels_path=context.labels_path,
            evaluation_mask_path=context.evaluation_mask_path,
            node_count=context.node_count,
            events_path=events_path,
        )
        for route, output in route_outputs.items():
            vault.freeze(
                route=route,
                scores_path=output["score_path"],
                scores_sha256=output["score_sha256"],
                mask_path=mask_path,
                mask_sha256=mask_hash,
            )

        with measured_phase("label_unlock_and_metrics", device) as phase:
            labels, mask = vault.unlock(tuple(routes))
            metrics = {
                route: _metric(labels, mask, output["scores"])
                for route, output in route_outputs.items()
            }
        phase_records.append(dict(phase))
        fidelity = None
        if spec.target == "tfinance":
            exact_scores = route_outputs["exact"]["scores"]
            ann_scores = route_outputs["faiss_ivfpq"]["scores"]
            fidelity = {
                "score_spearman": float(
                    spearmanr(exact_scores, ann_scores).statistic
                ),
                "auroc_difference_ann_minus_exact": (
                    metrics["faiss_ivfpq"]["auroc"]
                    - metrics["exact"]["auroc"]
                ),
                "auprc_difference_ann_minus_exact": (
                    metrics["faiss_ivfpq"]["auprc"]
                    - metrics["exact"]["auprc"]
                ),
            }

        primary = route_outputs[spec.primary_knn]
        result = {
            "format": "recap_large_target_result_v1",
            "run_id": spec.run_id,
            "seed": spec.seed,
            "target": spec.target,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": spec.checkpoint_sha256,
            "primary_knn": spec.primary_knn,
            "primary_score_path": str(primary["score_path"].resolve()),
            "primary_score_sha256": primary["score_sha256"],
            "query_mask_path": str(mask_path.resolve()),
            "query_mask_sha256": mask_hash,
            "node_count": context.node_count,
            "adjacency_nnz": context.adjacency_nnz,
            "primary_metrics": metrics[spec.primary_knn],
            "all_route_metrics": metrics,
            "tfinance_ann_fidelity": fidelity,
            "effective_communities": primary["effective_communities"],
            "phase_records": phase_records,
            "label_audit_path": str(events_path.resolve()),
            "completed_at": utc_now(),
        }
        atomic_json(run_dir / "result.json", result)
        atomic_json(
            complete_path,
            {
                "status": "complete",
                "run_id": spec.run_id,
                "primary_score_sha256": primary["score_sha256"],
                "completed_at": utc_now(),
            },
        )
        atomic_json(
            run_dir / "status.json",
            {"status": "complete", "run_id": spec.run_id, "at": utc_now()},
        )
        append_jsonl(
            run_dir / "events.jsonl",
            {"event": "run_complete", "run_id": spec.run_id, "at": utc_now()},
        )
        del (
            model,
            payload,
            residual,
            labels,
            mask,
            route_outputs,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"PASS {spec.run_id}: "
            f"AUROC={result['primary_metrics']['auroc']:.6f} "
            f"AUPRC={result['primary_metrics']['auprc']:.6f}",
            flush=True,
        )
        return result
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
        raise


def run_target(
    *,
    target: str,
    dataset_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    device: str,
    seeds: tuple[int, ...] = SEEDS,
) -> list[dict]:
    if target not in TARGETS:
        raise KeyError(f"Unknown locked target: {target}")
    preflight_path = output_root / "preflight" / "preflight.json"
    if not preflight_path.exists():
        raise FileNotFoundError("Run preflight before target inference")
    target_dir = output_root / "shared" / target
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading and propagating {target}", flush=True)
    context = load_and_propagate(
        dataset_root=dataset_root,
        name=target,
        dims=MODEL_LOCK["dims"],
        num_hops=MODEL_LOCK["num_hops"],
        device=device,
    )
    with measured_phase("initial_residual_construction", device) as phase:
        residual_initial = initial_residual(context.graph.x_list)
        if not torch.isfinite(residual_initial).all():
            raise FloatingPointError(f"{target}: non-finite initial residual")
        phase["shape"] = list(residual_initial.shape)
    context.phase_records.append(dict(phase))
    candidate_metadata, fidelity = _candidate_setup(
        target=target,
        residual_initial=residual_initial,
        shared_dir=target_dir,
        device=device,
    )
    context.phase_records.extend(fidelity["phase_records"])
    del residual_initial
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    setup = {
        "format": "recap_large_target_shared_setup_v1",
        "target": target,
        "node_count": context.node_count,
        "adjacency_nnz": context.adjacency_nnz,
        "phase_records": context.phase_records,
        "candidate_metadata": candidate_metadata,
        "ann_fidelity": fidelity["record"],
        "created_at": utc_now(),
    }
    setup_path = target_dir / "shared_setup.json"
    if setup_path.exists():
        append_jsonl(target_dir / "setup_reinvocations.jsonl", setup)
    else:
        atomic_json(setup_path, setup)

    paths = _candidate_paths(target_dir, target)
    manifest = [
        item
        for item in build_manifest()
        if item.target == target and item.seed in seeds
    ]
    results = [
        _run_checkpoint(
            spec=spec,
            context=context,
            candidate_paths=paths,
            checkpoint_root=checkpoint_root,
            output_root=output_root,
            device=device,
        )
        for spec in manifest
    ]
    expected = len(seeds)
    if len(results) != expected:
        raise AssertionError(
            f"{target}: expected {expected} results, got {len(results)}"
        )
    atomic_json(
        target_dir / "complete.json",
        {
            "status": "complete",
            "target": target,
            "seeds": list(seeds),
            "result_count": len(results),
            "completed_at": utc_now(),
        },
    )
    return results


def status(output_root: Path) -> dict:
    manifest = build_manifest()
    completed = []
    failed = []
    for spec in manifest:
        run_dir = output_root / "runs" / spec.run_id
        if (run_dir / "complete.json").exists():
            completed.append(spec.run_id)
        elif (run_dir / "status.json").exists():
            with (run_dir / "status.json").open("r", encoding="utf-8") as handle:
                state = json.load(handle)
            if state.get("status") == "failed":
                failed.append(
                    {"run_id": spec.run_id, "error": state.get("error")}
                )
    output = {
        "expected_primary_evaluations": len(manifest),
        "completed_primary_evaluations": len(completed),
        "pending_primary_evaluations": len(manifest) - len(completed),
        "completed": completed,
        "failed": failed,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("preflight", "gates", "run-target", "status"),
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = parser.parse_args()
    if args.command == "preflight":
        if args.dataset_root is None or args.checkpoint_root is None:
            parser.error("preflight requires --dataset-root and --checkpoint-root")
        run_preflight(
            dataset_root=args.dataset_root,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            device=args.device,
        )
    elif args.command == "gates":
        run_numerical_gates(args.output_root, args.device)
    elif args.command == "run-target":
        if (
            args.dataset_root is None
            or args.checkpoint_root is None
            or args.target is None
        ):
            parser.error(
                "run-target requires --dataset-root, --checkpoint-root, --target"
            )
        run_target(
            target=args.target,
            dataset_root=args.dataset_root,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            device=args.device,
            seeds=tuple(args.seeds),
        )
    else:
        status(args.output_root)


if __name__ == "__main__":
    main()
