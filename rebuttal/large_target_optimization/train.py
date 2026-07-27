"""Memory-bounded label-free target adaptation for RECAP large graphs."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig  # noqa: E402
from model import recap  # noqa: E402
from rebuttal.large_target_inference.ann import (  # noqa: E402
    build_faiss_candidates,
    validate_candidates,
)
from rebuttal.large_target_inference.common import (  # noqa: E402
    atomic_json,
    atomic_npy,
    sha256_file,
    utc_now,
)
from rebuttal.large_target_inference.data import (  # noqa: E402
    initial_residual,
    load_and_propagate,
    load_and_propagate_cpu_csr,
)
from rebuttal.large_target_inference.protocol import (  # noqa: E402
    ANN_CONFIG,
    DATASETS,
    MODEL_LOCK,
)
from rebuttal.large_target_inference.scoring import (  # noqa: E402
    compute_score_components_chunked,
)


PROTOCOL_PATH = (
    PROJECT_ROOT / "rebuttal" / "LARGE_TARGET_OPTIMIZATION_PROTOCOL.md"
)
SEEDS = (0, 1, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def locked_model_config(cache_dir: Path) -> ModelConfig:
    path = PROJECT_ROOT / "params" / "recap_auprc_best.json"
    config = ModelConfig.from_json(str(path))
    if config is None:
        raise FileNotFoundError(path)
    config.knn_cache_enabled = False
    config.knn_cache_dir = str(cache_dir)
    mismatches = {
        key: (expected, getattr(config, key))
        for key, expected in MODEL_LOCK.items()
        if getattr(config, key) != expected
    }
    if mismatches:
        raise ValueError(f"Model lock mismatch: {mismatches}")
    return config


def build_optimizer(
    model: recap, config: ModelConfig
) -> torch.optim.Optimizer:
    cluster_ids = {id(value) for value in model.ego_clusters.parameters()}
    base = [value for value in model.parameters() if id(value) not in cluster_ids]
    return torch.optim.Adam(
        [
            {
                "params": base,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            },
            {
                "params": list(model.ego_clusters.parameters()),
                "lr": config.lr * float(config.cluster_lr_multiplier),
                "weight_decay": 0.0,
            },
        ],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


def deterministic_sample(node_count: int, sample_size: int) -> np.ndarray:
    if sample_size >= node_count:
        return np.arange(node_count, dtype=np.int64)
    rng = np.random.default_rng(20260727)
    return np.sort(
        rng.choice(node_count, size=sample_size, replace=False).astype(np.int64)
    )


def _edge_terms(
    residual: torch.Tensor,
    assignments: torch.Tensor,
    candidates: np.ndarray,
    start: int,
    stop: int,
    tau_s: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = residual.device
    columns = torch.from_numpy(
        np.asarray(candidates[start:stop], dtype=np.int64)
    ).to(device=device)
    query = residual[start:stop]
    query_norm = F.normalize(query, p=2, dim=1)
    candidate = residual[columns]
    candidate_norm = F.normalize(candidate, p=2, dim=2)
    weights = F.softmax(
        (candidate_norm * query_norm[:, None, :]).sum(dim=2)
        / max(tau_s, eps),
        dim=1,
    )
    left = assignments[start:stop, None, :]
    right = assignments[columns]
    numerator = (
        0.5 * weights * ((left - right) ** 2).sum(dim=2)
    ).sum()
    denominator = (
        0.5
        * weights
        * ((left * left).sum(dim=2) + (right * right).sum(dim=2))
    ).sum()
    return numerator, denominator


def chunked_loss_value_and_backward(
    *,
    residual: torch.Tensor,
    cluster,
    candidates: np.ndarray,
    batch_size: int,
) -> dict:
    """Backpropagate an algebraically exact fixed-candidate RECAP loss.

    A no-grad first pass obtains the global numerator and denominator. The
    second pass backpropagates the exact quotient derivative one edge chunk at
    a time, avoiding materializing N x K x embed_dim.
    """
    assignments = cluster.cluster(residual)
    node_count = int(residual.shape[0])
    with torch.no_grad():
        numerator_value = residual.new_tensor(0.0)
        denominator_value = residual.new_tensor(0.0)
        for start in range(0, node_count, batch_size):
            stop = min(start + batch_size, node_count)
            numerator, denominator = _edge_terms(
                residual,
                assignments,
                candidates,
                start,
                stop,
                cluster.tau_s,
                cluster.eps,
            )
            numerator_value += numerator
            denominator_value += denominator
    denominator_value = denominator_value.clamp(min=cluster.eps)
    ratio_value = numerator_value / denominator_value

    l_h, l_sharp, l_bal = cluster._compute_H_loss(assignments)
    l_var = (
        cluster._compute_var_loss(residual)
        if cluster.lambda_E != 0
        else residual.new_tensor(0.0)
    )
    regularization = cluster.lambda_H * l_h + cluster.lambda_E * l_var
    regularization.backward(retain_graph=True)

    starts = list(range(0, node_count, batch_size))
    for index, start in enumerate(starts):
        stop = min(start + batch_size, node_count)
        numerator, denominator = _edge_terms(
            residual,
            assignments,
            candidates,
            start,
            stop,
            cluster.tau_s,
            cluster.eps,
        )
        local_surrogate = (
            numerator / denominator_value
            - numerator_value
            * denominator
            / (denominator_value * denominator_value)
        )
        local_surrogate.backward(retain_graph=index < len(starts) - 1)
        del numerator, denominator, local_surrogate

    return {
        "total": float((ratio_value + regularization.detach()).item()),
        "l_con": float(ratio_value.item()),
        "l_h": float(l_h.detach().item()),
        "l_sharp": float(l_sharp.detach().item()),
        "l_bal": float(l_bal.detach().item()),
        "l_var": float(l_var.detach().item()),
    }


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sample_graph(context, sample: np.ndarray, device: str) -> Data:
    index = torch.from_numpy(sample).to(device=context.graph.x_list[0].device)
    graph = Data(
        x_list=[values[index] for values in context.graph.x_list],
        dataset_name=f"{context.name}__target_adapt_sample_{len(sample)}",
        feature_alignment_version=context.graph.feature_alignment_version,
        feature_dims=context.graph.feature_dims,
        adjacency_version=(
            f"{context.graph.adjacency_version}|full_graph_propagation_context"
        ),
    )
    if "ano_labels" in graph or "evaluation_mask" in graph:
        raise AssertionError("Sample graph contains labels or evaluation mask")
    return graph


def _ann_kwargs(sample_size: int) -> dict:
    nlist = min(
        ANN_CONFIG["nlist"],
        max(64, 2 ** int(math.floor(math.log2(max(sample_size // 64, 64))))),
    )
    return {
        "nlist": nlist,
        "nprobe": min(32, nlist),
        "pq_m": ANN_CONFIG["pq_m"],
        "train_size": min(ANN_CONFIG["train_size"], sample_size),
        "query_batch_size": ANN_CONFIG["query_batch_size"],
        "add_batch_size": ANN_CONFIG["add_batch_size"],
        "rerank_factor": ANN_CONFIG["rerank_factor"],
        "max_rerank_candidates": ANN_CONFIG["max_rerank_candidates"],
        "seed": ANN_CONFIG["seed"],
    }


def prepare_training_candidates(
    graph: Data,
    path: Path,
    device: str,
    full_candidates_path: Path | None,
) -> dict:
    node_count = int(graph.x_list[0].shape[0])
    if path.exists():
        metadata = validate_candidates(path, node_count, MODEL_LOCK["knn_k"])
        metadata["state"] = "reused"
        return metadata
    if full_candidates_path is not None and node_count == DATASETS["tfinance"]["nodes"]:
        source = np.load(full_candidates_path, mmap_mode="r")
        if source.shape != (node_count, MODEL_LOCK["knn_k"]):
            raise ValueError("T-Finance full candidate shape mismatch")
        atomic_npy(path, np.asarray(source, dtype=np.int32))
        metadata = validate_candidates(path, node_count, MODEL_LOCK["knn_k"])
        metadata["state"] = "copied_from_locked_exact_full_target"
        return metadata
    residual_initial = initial_residual(graph.x_list)
    metadata = build_faiss_candidates(
        residual_initial,
        k=MODEL_LOCK["knn_k"],
        output_path=path,
        **_ann_kwargs(node_count),
    )
    metadata["state"] = "new_label_blind_sample_ann"
    return metadata


def train_seed(
    *,
    seed: int,
    graph: Data,
    candidates: np.ndarray,
    config: ModelConfig,
    output_root: Path,
    epochs: int,
    batch_size: int,
    device: str,
    sample_sha256: str,
    candidate_sha256: str,
) -> tuple[recap, dict]:
    run_dir = output_root / "runs" / f"seed{seed}"
    final_path = run_dir / "checkpoints" / "final.pt"
    if final_path.exists():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        if (
            payload.get("protocol_sha256") == sha256_file(PROTOCOL_PATH)
            and payload.get("sample_sha256") == sample_sha256
            and payload.get("candidate_sha256") == candidate_sha256
            and int(payload.get("epoch", -1)) == epochs
        ):
            model = recap(**config.to_dict())
            model.load_state_dict(payload["model_state_dict"], strict=True)
            return model.to(device).eval(), {
                "state": "reused",
                "checkpoint_path": str(final_path.resolve()),
                "checkpoint_sha256": sha256_file(final_path),
                "history": payload["history"],
            }

    set_seed(seed)
    model = recap(**config.to_dict()).to(device)
    optimizer = build_optimizer(model, config)
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        model(graph)
        values = chunked_loss_value_and_backward(
            residual=model._view_embeds[0],
            cluster=model.ego_clusters[0],
            candidates=candidates,
            batch_size=batch_size,
        )
        optimizer.step()
        values["epoch"] = epoch
        values["elapsed_seconds"] = time.perf_counter() - started
        history.append(values)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"TRAIN seed={seed} epoch={epoch}/{epochs} "
                f"loss={values['total']:.6f} "
                f"time={values['elapsed_seconds']:.1f}s",
                flush=True,
            )
        if epoch in {25, 50, 75, 100, epochs}:
            payload = {
                "format": "recap_large_target_adapt_checkpoint_v1",
                "seed": seed,
                "epoch": epoch,
                "model_config": config.to_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "sample_sha256": sample_sha256,
                "candidate_sha256": candidate_sha256,
                "protocol_sha256": sha256_file(PROTOCOL_PATH),
                "labels_accessed": False,
                "saved_at": utc_now(),
            }
            checkpoint = (
                final_path
                if epoch == epochs
                else run_dir / "checkpoints" / f"epoch_{epoch}.pt"
            )
            _atomic_checkpoint(checkpoint, payload)
    return model.eval(), {
        "state": "new",
        "checkpoint_path": str(final_path.resolve()),
        "checkpoint_sha256": sha256_file(final_path),
        "history": history,
    }


def _score_and_freeze(
    *,
    model: recap,
    context,
    candidates: np.ndarray,
    run_dir: Path,
    checkpoint_record: dict,
    score_batch_size: int,
) -> dict:
    score_dir = run_dir / "scores"
    paths = {
        "full": score_dir / "full.npy",
        "adhesion_only": score_dir / "adhesion_only.npy",
        "context_only": score_dir / "context_only.npy",
    }
    frozen_path = run_dir / "scores_frozen.json"
    if frozen_path.exists() and all(path.exists() for path in paths.values()):
        with frozen_path.open("r", encoding="utf-8") as handle:
            frozen = json.load(handle)
        if all(
            frozen["score_sha256"][name] == sha256_file(path)
            for name, path in paths.items()
        ):
            return frozen
    model.eval()
    with torch.inference_mode():
        model(context.graph)
        components = compute_score_components_chunked(
            residual=model._view_embeds[0],
            cluster=model.ego_clusters[0],
            candidates=candidates,
            score_batch_size=score_batch_size,
        )
    values = {
        "full": np.asarray(components["total"], dtype=np.float32),
        "adhesion_only": np.asarray(
            components["adhesion"], dtype=np.float32
        ),
        "context_only": np.asarray(components["context"], dtype=np.float32),
    }
    for name, score in values.items():
        atomic_npy(paths[name], score)
    frozen = {
        "format": "recap_large_target_adapt_scores_frozen_v1",
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "score_paths": {name: str(path.resolve()) for name, path in paths.items()},
        "score_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "labels_accessed": False,
        "frozen_at": utc_now(),
    }
    atomic_json(frozen_path, frozen)
    del components, values
    return frozen


def run(args: argparse.Namespace) -> None:
    if args.target not in DATASETS:
        raise KeyError(args.target)
    output_root = Path(args.output_root).resolve() / args.target
    output_root.mkdir(parents=True, exist_ok=True)
    config = locked_model_config(output_root / "knn_cache")
    loader_kwargs = {
        "dataset_root": Path(args.dataset_root).resolve(),
        "name": args.target,
        "dims": config.dims,
        "num_hops": config.num_hops,
    }
    context = (
        load_and_propagate_cpu_csr(**loader_kwargs)
        if args.device == "cpu"
        else load_and_propagate(**loader_kwargs, device=args.device)
    )
    sample = deterministic_sample(context.node_count, args.sample_size)
    sample_path = output_root / "sample_indices.npy"
    atomic_npy(sample_path, sample)
    sample_sha256 = sha256_file(sample_path)
    graph = _sample_graph(context, sample, args.device)
    training_candidate_path = output_root / "training_candidates.npy"
    metadata = prepare_training_candidates(
        graph,
        training_candidate_path,
        args.device,
        Path(args.full_candidates).resolve() if args.full_candidates else None,
    )
    candidates = np.load(training_candidate_path, mmap_mode="r")
    candidate_sha256 = sha256_file(training_candidate_path)
    atomic_json(
        output_root / "training_manifest.json",
        {
            "format": "recap_large_target_adapt_manifest_v1",
            "target": args.target,
            "full_nodes": context.node_count,
            "sample_nodes": len(sample),
            "sample_sha256": sample_sha256,
            "candidate_metadata": metadata,
            "candidate_sha256": candidate_sha256,
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "score_batch_size": args.score_batch_size,
            "seeds": list(SEEDS),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "labels_accessed": False,
            "created_at": utc_now(),
        },
    )

    full_candidates = np.load(Path(args.inference_candidates), mmap_mode="r")
    if full_candidates.shape != (context.node_count, config.knn_k):
        raise ValueError(f"Full inference candidate mismatch: {full_candidates.shape}")
    frozen = []
    training_records = []
    for seed in SEEDS:
        model, training = train_seed(
            seed=seed,
            graph=graph,
            candidates=candidates,
            config=config,
            output_root=output_root,
            epochs=args.epochs,
            batch_size=args.train_batch_size,
            device=args.device,
            sample_sha256=sample_sha256,
            candidate_sha256=candidate_sha256,
        )
        record = _score_and_freeze(
            model=model,
            context=context,
            candidates=full_candidates,
            run_dir=output_root / "runs" / f"seed{seed}",
            checkpoint_record=training,
            score_batch_size=args.score_batch_size,
        )
        frozen.append(record)
        training_records.append(training)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_json(
        output_root / "all_scores_frozen.json",
        {
            "format": "recap_large_target_adapt_all_scores_frozen_v1",
            "target": args.target,
            "seeds": list(SEEDS),
            "records": frozen,
            "labels_accessed_before_global_freeze": False,
            "frozen_at": utc_now(),
        },
    )

    # First label/mask access in this process.
    dataset_dir = Path(args.dataset_root).resolve() / args.target
    labels = np.asarray(np.load(dataset_dir / "labels.npy"), dtype=np.int64)
    mask_path = dataset_dir / "evaluation_mask.npy"
    mask = (
        np.asarray(np.load(mask_path), dtype=np.bool_)
        if mask_path.exists()
        else np.ones(context.node_count, dtype=np.bool_)
    )
    rows = []
    for seed, record in zip(SEEDS, frozen, strict=True):
        for route, score_path in record["score_paths"].items():
            scores = np.load(score_path, mmap_mode="r")
            rows.append(
                {
                    "target": args.target,
                    "seed": seed,
                    "route": route,
                    "auroc": float(roc_auc_score(labels[mask], scores[mask])),
                    "auprc": float(
                        average_precision_score(labels[mask], scores[mask])
                    ),
                    "evaluation_nodes": int(mask.sum()),
                    "anomalies": int(labels[mask].sum()),
                    "score_sha256": record["score_sha256"][route],
                    "checkpoint_sha256": training_records[seed][
                        "checkpoint_sha256"
                    ],
                }
            )
    csv_path = output_root / "results.csv"
    fields = sorted({key for row in rows for key in row})
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    summary = []
    for route in ("full", "adhesion_only", "context_only"):
        current = [row for row in rows if row["route"] == route]
        auroc = np.asarray([row["auroc"] for row in current])
        auprc = np.asarray([row["auprc"] for row in current])
        summary.append(
            {
                "target": args.target,
                "route": route,
                "auroc_mean": float(auroc.mean()),
                "auroc_std": float(auroc.std(ddof=0)),
                "auprc_mean": float(auprc.mean()),
                "auprc_std": float(auprc.std(ddof=0)),
                "seeds": 3,
            }
        )
    atomic_json(
        output_root / "summary.json",
        {
            "format": "recap_large_target_adapt_summary_v1",
            "summary": summary,
            "labels_loaded_after_global_freeze": True,
            "results_sha256": sha256_file(csv_path),
            "created_at": utc_now(),
        },
    )
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--target", required=True, choices=tuple(DATASETS))
    value.add_argument("--dataset-root", required=True)
    value.add_argument("--inference-candidates", required=True)
    value.add_argument("--full-candidates")
    value.add_argument("--output-root", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--sample-size", type=int, default=131_072)
    value.add_argument("--epochs", type=int, default=100)
    value.add_argument("--train-batch-size", type=int, default=256)
    value.add_argument("--score-batch-size", type=int, default=1024)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
