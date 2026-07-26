"""Resumable one-run executor for the locked 12-dataset OFO baselines."""

from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from rebuttal.baselines.baseline_common import (
    sha256_file,
    symmetric_normalize,
)
from rebuttal.baselines.baseline_protocol import DATASETS
from rebuttal.ofo_baselines.common import (
    PREPROCESS_VERSION,
    SPLIT_VERSION,
    OFOLabelVault,
    adjacency_tensors,
    atomic_json,
    atomic_npz,
    atomic_torch_save,
    environment_metadata,
    prepare_graph,
    sample_rowwise_nonedges,
    set_seed,
    torch_sparse,
    utc_now,
)
from rebuttal.ofo_baselines.models import (
    ADAGADAutoencoder,
    AnomalyDAEModel,
    BWGNNClassifier,
    CoLAModel,
    DOMINANTModel,
    GATClassifier,
    GCNClassifier,
    average_states,
    dominant_scores,
    label_free_candidate_score,
    sampled_sigmoid_structure_error,
    weighted_attribute_error,
)
from rebuttal.ofo_baselines.protocol import (
    OFOBaselineRun,
    build_manifest,
    validate_manifest,
)


FORMAL_CONFIGS: dict[str, dict[str, Any]] = {
    "GCN": {
        "epochs": 200,
        "patience": 50,
        "lr": 0.01,
        "weight_decay": 0.0,
        "hidden_dim": 32,
        "dropout": 0.0,
        "selection_metric": "validation_auprc",
    },
    "GAT": {
        "epochs": 200,
        "patience": 50,
        "lr": 0.01,
        "weight_decay": 0.0,
        "hidden_dim": 32,
        "heads": 2,
        "dropout": 0.0,
        "selection_metric": "validation_auprc",
    },
    "BWGNN": {
        "epochs": 200,
        "patience": 50,
        "lr": 0.01,
        "weight_decay": 0.0,
        "hidden_dim": 32,
        "order": 2,
        "dropout": 0.0,
        "selection_metric": "validation_auprc",
    },
    "XGBGraph": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.3,
        "tree_method": "hist",
        "num_layers": 2,
        "aggregation": "mean",
        "init_eps": -1.0,
        "n_jobs": 1,
    },
    "DOMINANT": {
        "epochs": 100,
        "lr": 0.005,
        "weight_decay": 0.0,
        "hidden_dim": 64,
        "dropout": 0.3,
        "alpha": 0.8,
    },
    "AnomalyDAE": {
        "epochs": 100,
        "lr": 0.001,
        "weight_decay": 0.0,
        "embedding_dim": 64,
        "hidden_dim": 64,
        "alpha": 0.7,
        "eta": 5.0,
        "theta": 40.0,
        "negative_ratio": 1.0,
        "inference_rounds": 4,
    },
    "CoLA": {
        "epochs": 100,
        "lr": 0.001,
        "weight_decay": 0.0,
        "hidden_dim": 64,
        "num_layers": 4,
        "negative_ratio": 1,
        "inference_rounds": 64,
        "context_sampler": "pygod_random_neighbor_full_graph",
    },
    "ADA-GAD": {
        "pretrain_epochs_per_view": 20,
        "detector_epochs": 20,
        "lr": 0.001,
        "weight_decay": 0.0002,
        "hidden_dim": 32,
        "view_drop_rates": [0.0, 0.05, 0.10],
        "alpha": 0.5,
        "negative_ratio": 1.0,
        "inference_rounds": 4,
    },
}


def resolved_config(method: str, smoke: bool) -> dict[str, Any]:
    config = copy.deepcopy(FORMAL_CONFIGS[method])
    if smoke:
        if method in {"GCN", "GAT", "BWGNN"}:
            config["epochs"] = 2
            config["patience"] = 1
        elif method == "XGBGraph":
            config["n_estimators"] = 3
            config["max_depth"] = 2
        elif method in {"DOMINANT", "AnomalyDAE", "CoLA"}:
            config["epochs"] = 2
            if "inference_rounds" in config:
                config["inference_rounds"] = 2
        elif method == "ADA-GAD":
            config["pretrain_epochs_per_view"] = 1
            config["detector_epochs"] = 1
            config["inference_rounds"] = 2
    config["smoke"] = smoke
    config["preprocess_version"] = PREPROCESS_VERSION
    config["split_version"] = SPLIT_VERSION
    return config


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_features(
    sparse_features: sp.csr_matrix, device: torch.device
) -> torch.Tensor:
    dense = sparse_features.toarray().astype(np.float32, copy=False)
    values = torch.from_numpy(dense).to(device)
    return values


def validation_auprc(
    labels: np.ndarray, mask: np.ndarray, logits: torch.Tensor
) -> float:
    probability = logits.softmax(dim=1)[:, 1].detach().cpu().numpy()
    return float(average_precision_score(labels[mask], probability[mask]))


def build_supervised_model(
    method: str, in_features: int, config: dict[str, Any]
) -> torch.nn.Module:
    if method == "GCN":
        return GCNClassifier(in_features, config["hidden_dim"])
    if method == "GAT":
        return GATClassifier(
            in_features, config["hidden_dim"], config["heads"]
        )
    if method == "BWGNN":
        return BWGNNClassifier(
            in_features, config["hidden_dim"], config["order"]
        )
    raise ValueError(method)


def supervised_forward(
    method: str,
    model: torch.nn.Module,
    features: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    if method == "GCN":
        return model(features, tensors["adj_norm_loop"])
    if method == "GAT":
        return model(features, tensors["edge_index_no_loop"])
    if method == "BWGNN":
        return model(features, tensors["adj_norm_no_loop"])
    raise ValueError(method)


def train_torch_supervised(
    *,
    method: str,
    graph,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    features = tensor_features(graph.features, device)
    tensors = adjacency_tensors(graph.adjacency, device)
    labels_tensor = torch.from_numpy(labels).long().to(device)
    train_mask = torch.from_numpy(masks["train"]).to(device)

    model = build_supervised_model(
        method, graph.feature_count, config
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    train_labels = labels[masks["train"]]
    anomaly_count = int(np.sum(train_labels == 1))
    normal_count = int(np.sum(train_labels == 0))
    class_weight = torch.tensor(
        [1.0, normal_count / anomaly_count],
        dtype=torch.float32,
        device=device,
    )
    best_value = -math.inf
    best_epoch = -1
    best_state = None
    wait = 0
    history = []

    synchronize(device)
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        logits = supervised_forward(method, model, features, tensors)
        loss = F.cross_entropy(
            logits[train_mask], labels_tensor[train_mask], weight=class_weight
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = supervised_forward(method, model, features, tensors)
            value = validation_auprc(labels, masks["validation"], logits)
        history.append(
            {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "validation_auprc": value,
            }
        )
        if value > best_value:
            best_value = value
            best_epoch = epoch
            best_state = cpu_state(model)
            wait = 0
        else:
            wait += 1
            if wait > config["patience"]:
                break
    synchronize(device)
    train_seconds = time.perf_counter() - started

    if best_state is None:
        raise RuntimeError("No supervised checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    synchronize(device)
    infer_started = time.perf_counter()
    with torch.no_grad():
        logits = supervised_forward(method, model, features, tensors)
        scores = logits.softmax(dim=1)[:, 1].cpu().numpy().astype(np.float32)
    synchronize(device)
    inference_seconds = time.perf_counter() - infer_started

    reloaded = build_supervised_model(
        method, graph.feature_count, config
    ).to(device)
    reloaded.load_state_dict(best_state)
    reloaded.eval()
    with torch.no_grad():
        reload_scores = (
            supervised_forward(method, reloaded, features, tensors)
            .softmax(dim=1)[:, 1]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return {
        "scores": scores,
        "checkpoint": {
            "format": "recap_ofo12_torch_checkpoint_v1",
            "method": method,
            "in_features": graph.feature_count,
            "config": config,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_auprc": best_value,
        },
        "history": history,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "reload_max_abs_diff": float(np.max(np.abs(scores - reload_scores))),
        "best_epoch": best_epoch,
        "best_validation_auprc": best_value,
    }


def xgbgraph_features(
    adjacency: sp.csr_matrix, features: sp.csr_matrix, layers: int
) -> sp.csr_matrix:
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(degree, dtype=np.float32)
    inverse[degree > 0] = 1.0 / degree[degree > 0]
    propagation = sp.diags(inverse).dot(adjacency).tocsr()
    values = features
    outputs = [features]
    for _ in range(layers):
        values = propagation.dot(values).tocsr()
        outputs.append(values)
    return sp.hstack(outputs, format="csr", dtype=np.float32)


def train_xgbgraph(
    *,
    graph,
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    config: dict[str, Any],
    seed: int,
    model_path: Path,
) -> dict[str, Any]:
    features = xgbgraph_features(
        graph.adjacency, graph.features, config["num_layers"]
    )
    train_labels = labels[masks["train"]]
    anomaly_count = int(np.sum(train_labels == 1))
    normal_count = int(np.sum(train_labels == 0))
    weights = np.where(
        train_labels == 1, normal_count / anomaly_count, 1.0
    ).astype(np.float32)
    model = xgb.XGBClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        tree_method=config["tree_method"],
        n_jobs=config["n_jobs"],
        random_state=seed,
        objective="binary:logistic",
        eval_metric="aucpr",
    )
    started = time.perf_counter()
    model.fit(features[masks["train"]], train_labels, sample_weight=weights)
    train_seconds = time.perf_counter() - started
    infer_started = time.perf_counter()
    scores = model.predict_proba(features)[:, 1].astype(np.float32)
    inference_seconds = time.perf_counter() - infer_started
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)

    reloaded = xgb.XGBClassifier()
    reloaded.load_model(model_path)
    reload_scores = reloaded.predict_proba(features)[:, 1].astype(np.float32)
    return {
        "scores": scores,
        "checkpoint": None,
        "history": [],
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "reload_max_abs_diff": float(np.max(np.abs(scores - reload_scores))),
        "best_epoch": None,
        "best_validation_auprc": float(
            average_precision_score(
                labels[masks["validation"]],
                scores[masks["validation"]],
            )
        ),
        "xgb_model_path": str(model_path),
        "derived_feature_shape": list(features.shape),
        "derived_feature_nnz": int(features.nnz),
    }


def train_dominant(
    *, graph, config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    features = tensor_features(graph.features, device)
    tensors = adjacency_tensors(graph.adjacency, device)
    model = DOMINANTModel(
        graph.feature_count, config["hidden_dim"], config["dropout"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    history = []
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        reconstructed, latent = model(features, tensors["adj_norm_loop"])
        score = dominant_scores(
            features,
            reconstructed,
            latent,
            tensors["edge_index_with_loop"],
            tensors["edge_weight_with_loop"],
            config["alpha"],
        )
        loss = score.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
    synchronize(device)
    train_seconds = time.perf_counter() - started

    model.eval()
    synchronize(device)
    infer_started = time.perf_counter()
    with torch.no_grad():
        reconstructed, latent = model(features, tensors["adj_norm_loop"])
        scores_tensor = dominant_scores(
            features,
            reconstructed,
            latent,
            tensors["edge_index_with_loop"],
            tensors["edge_weight_with_loop"],
            config["alpha"],
        )
        scores = scores_tensor.cpu().numpy().astype(np.float32)
    synchronize(device)
    inference_seconds = time.perf_counter() - infer_started
    state = cpu_state(model)

    reloaded = DOMINANTModel(
        graph.feature_count, config["hidden_dim"], config["dropout"]
    ).to(device)
    reloaded.load_state_dict(state)
    reloaded.eval()
    with torch.no_grad():
        reconstructed, latent = reloaded(
            features, tensors["adj_norm_loop"]
        )
        reload_scores = (
            dominant_scores(
                features,
                reconstructed,
                latent,
                tensors["edge_index_with_loop"],
                tensors["edge_weight_with_loop"],
                config["alpha"],
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return {
        "scores": scores,
        "checkpoint": {
            "format": "recap_ofo12_torch_checkpoint_v1",
            "method": "DOMINANT",
            "in_features": graph.feature_count,
            "config": config,
            "state_dict": state,
        },
        "history": history,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "reload_max_abs_diff": float(np.max(np.abs(scores - reload_scores))),
    }


def sampled_edges(
    adjacency: sp.csr_matrix,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with_loop = adjacency + sp.eye(
        adjacency.shape[0], dtype=np.float32, format="csr"
    )
    with_loop.data[:] = 1.0
    positive = sp.coo_matrix(with_loop)
    rows, cols, row_weight = sample_rowwise_nonedges(
        with_loop,
        positive.row.astype(np.int64, copy=False),
        np.random.default_rng(seed),
    )
    positive_edge = torch.from_numpy(
        np.vstack([positive.row, positive.col]).astype(np.int64)
    ).to(device)
    negative_edge = torch.from_numpy(np.vstack([rows, cols])).to(device)
    weights = torch.from_numpy(row_weight).to(device)
    return positive_edge, negative_edge, weights


def anomalydae_score(
    *,
    model: AnomalyDAEModel,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    graph,
    config: dict[str, Any],
    seed_base: int,
    device: torch.device,
) -> torch.Tensor:
    reconstructed, latent = model(features, edge_index)
    attribute = weighted_attribute_error(
        features, reconstructed, config["eta"]
    )
    structures = []
    for round_index in range(config["inference_rounds"]):
        positive, negative, weights = sampled_edges(
            graph.adjacency,
            seed=seed_base + 10_000 + round_index,
            device=device,
        )
        structures.append(
            sampled_sigmoid_structure_error(
                latent,
                positive,
                negative,
                weights,
                config["theta"],
            )
        )
    structure = torch.stack(structures).mean(dim=0)
    return config["alpha"] * attribute + (1.0 - config["alpha"]) * structure


def train_anomalydae(
    *,
    graph,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    features = tensor_features(graph.features, device)
    tensors = adjacency_tensors(graph.adjacency, device)
    edge_index = tensors["edge_index_no_loop"]
    model = AnomalyDAEModel(
        graph.feature_count,
        graph.node_count,
        config["embedding_dim"],
        config["hidden_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    history = []
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        reconstructed, latent = model(features, edge_index)
        attribute = weighted_attribute_error(
            features, reconstructed, config["eta"]
        )
        positive, negative, weights = sampled_edges(
            graph.adjacency,
            seed=seed * 1_000_003 + epoch,
            device=device,
        )
        structure = sampled_sigmoid_structure_error(
            latent,
            positive,
            negative,
            weights,
            config["theta"],
        )
        score = (
            config["alpha"] * attribute
            + (1.0 - config["alpha"]) * structure
        )
        loss = score.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
    synchronize(device)
    train_seconds = time.perf_counter() - started

    model.eval()
    synchronize(device)
    infer_started = time.perf_counter()
    with torch.no_grad():
        scores = (
            anomalydae_score(
                model=model,
                features=features,
                edge_index=edge_index,
                graph=graph,
                config=config,
                seed_base=seed * 1_000_003,
                device=device,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    synchronize(device)
    inference_seconds = time.perf_counter() - infer_started
    state = cpu_state(model)

    reloaded = AnomalyDAEModel(
        graph.feature_count,
        graph.node_count,
        config["embedding_dim"],
        config["hidden_dim"],
    ).to(device)
    reloaded.load_state_dict(state)
    reloaded.eval()
    with torch.no_grad():
        reload_scores = (
            anomalydae_score(
                model=reloaded,
                features=features,
                edge_index=edge_index,
                graph=graph,
                config=config,
                seed_base=seed * 1_000_003,
                device=device,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return {
        "scores": scores,
        "checkpoint": {
            "format": "recap_ofo12_torch_checkpoint_v1",
            "method": "AnomalyDAE",
            "in_features": graph.feature_count,
            "node_count": graph.node_count,
            "config": config,
            "state_dict": state,
        },
        "history": history,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "reload_max_abs_diff": float(np.max(np.abs(scores - reload_scores))),
    }


def cola_scores(
    *,
    model: CoLAModel,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    rounds: int,
    seed: int,
) -> torch.Tensor:
    embedding = model.encoder(features, edge_index)
    positive = model.discriminator(features, embedding).squeeze(1)
    generator = torch.Generator(device=features.device).manual_seed(seed)
    values = []
    for _ in range(rounds):
        permutation = torch.randperm(
            features.shape[0], device=features.device, generator=generator
        )
        negative = model.discriminator(
            features[permutation], embedding
        ).squeeze(1)
        values.append(negative - positive)
    return torch.stack(values).mean(dim=0)


def train_cola(
    *,
    graph,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    features = tensor_features(graph.features, device)
    edge_index = adjacency_tensors(
        graph.adjacency, device
    )["edge_index_no_loop"]
    model = CoLAModel(
        graph.feature_count, config["hidden_dim"], config["num_layers"]
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    history = []
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        permutation = torch.randperm(
            graph.node_count, device=device, generator=generator
        )
        positive, negative = model.logits(
            features, edge_index, permutation
        )
        logits = torch.cat([positive, negative])
        target = torch.cat(
            [torch.ones_like(positive), torch.zeros_like(negative)]
        )
        loss = F.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
    synchronize(device)
    train_seconds = time.perf_counter() - started

    model.eval()
    synchronize(device)
    infer_started = time.perf_counter()
    with torch.no_grad():
        scores = (
            cola_scores(
                model=model,
                features=features,
                edge_index=edge_index,
                rounds=config["inference_rounds"],
                seed=seed * 1_000_003 + 99_991,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    synchronize(device)
    inference_seconds = time.perf_counter() - infer_started
    state = cpu_state(model)

    reloaded = CoLAModel(
        graph.feature_count, config["hidden_dim"], config["num_layers"]
    ).to(device)
    reloaded.load_state_dict(state)
    reloaded.eval()
    with torch.no_grad():
        reload_scores = (
            cola_scores(
                model=reloaded,
                features=features,
                edge_index=edge_index,
                rounds=config["inference_rounds"],
                seed=seed * 1_000_003 + 99_991,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return {
        "scores": scores,
        "checkpoint": {
            "format": "recap_ofo12_torch_checkpoint_v1",
            "method": "CoLA",
            "in_features": graph.feature_count,
            "config": config,
            "state_dict": state,
        },
        "history": history,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "reload_max_abs_diff": float(np.max(np.abs(scores - reload_scores))),
    }


def make_deniosed_adjacency(
    adjacency: sp.csr_matrix,
    candidate_score: np.ndarray,
    drop_rate: float,
) -> sp.csr_matrix:
    if drop_rate <= 0:
        return adjacency.copy()
    node_count = adjacency.shape[0]
    drop_count = int(math.floor(node_count * drop_rate))
    selected = np.argsort(candidate_score, kind="stable")[-drop_count:]
    keep = np.ones(node_count, dtype=np.float32)
    keep[selected] = 0.0
    selector = sp.diags(keep)
    value = selector.dot(adjacency).dot(selector).tocsr()
    value.eliminate_zeros()
    return value


def adag_score(
    *,
    model: ADAGADAutoencoder,
    features: torch.Tensor,
    adjacency: torch.Tensor,
    graph,
    config: dict[str, Any],
    seed_base: int,
    device: torch.device,
) -> torch.Tensor:
    reconstructed, latent = model(features, adjacency)
    attribute = torch.linalg.vector_norm(reconstructed - features, dim=1)
    structures = []
    for round_index in range(config["inference_rounds"]):
        positive, negative, weights = sampled_edges(
            graph.adjacency,
            seed=seed_base + 50_000 + round_index,
            device=device,
        )
        structures.append(
            sampled_sigmoid_structure_error(
                latent, positive, negative, weights, 1.0
            )
        )
    structure = torch.stack(structures).mean(dim=0)
    return config["alpha"] * attribute + (1.0 - config["alpha"]) * structure


def train_adagad(
    *,
    graph,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    features = tensor_features(graph.features, device)
    base_tensors = adjacency_tensors(graph.adjacency, device)
    with torch.no_grad():
        candidate = (
            label_free_candidate_score(
                features, base_tensors["adj_norm_no_loop"]
            )
            .cpu()
            .numpy()
        )
    encoder_states = []
    history = []
    synchronize(device)
    started = time.perf_counter()
    for view_index, drop_rate in enumerate(config["view_drop_rates"]):
        view_adjacency = make_deniosed_adjacency(
            graph.adjacency, candidate, drop_rate
        )
        normalized = symmetric_normalize(
            view_adjacency
            + sp.eye(graph.node_count, dtype=np.float32, format="csr")
        )
        normalized_tensor = torch_sparse(normalized, device)
        view_graph = copy.copy(graph)
        view_graph.adjacency = view_adjacency
        model = ADAGADAutoencoder(
            graph.feature_count, config["hidden_dim"]
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        for epoch in range(config["pretrain_epochs_per_view"]):
            model.train()
            reconstructed, latent = model(features, normalized_tensor)
            attribute = torch.linalg.vector_norm(
                reconstructed - features, dim=1
            )
            positive, negative, weights = sampled_edges(
                view_adjacency,
                seed=(
                    seed * 1_000_003
                    + view_index * 100_003
                    + epoch
                ),
                device=device,
            )
            structure = sampled_sigmoid_structure_error(
                latent, positive, negative, weights, 1.0
            )
            loss = (0.5 * attribute + 0.5 * structure).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(
                {
                    "stage": f"pretrain_view_{view_index}",
                    "drop_rate": drop_rate,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                }
            )
        encoder_states.append(model.encoder_state())
        del model

    averaged = average_states(encoder_states)
    detector = ADAGADAutoencoder(
        graph.feature_count, config["hidden_dim"]
    ).to(device)
    detector.load_encoder_state(averaged, freeze=True)
    optimizer = torch.optim.Adam(
        [parameter for parameter in detector.parameters() if parameter.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    for epoch in range(config["detector_epochs"]):
        detector.train()
        reconstructed, latent = detector(
            features, base_tensors["adj_norm_loop"]
        )
        attribute = torch.linalg.vector_norm(
            reconstructed - features, dim=1
        )
        positive, negative, weights = sampled_edges(
            graph.adjacency,
            seed=seed * 1_000_003 + 500_009 + epoch,
            device=device,
        )
        structure = sampled_sigmoid_structure_error(
            latent, positive, negative, weights, 1.0
        )
        loss = (
            config["alpha"] * attribute
            + (1.0 - config["alpha"]) * structure
        ).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(
            {
                "stage": "detector",
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
            }
        )
    synchronize(device)
    train_seconds = time.perf_counter() - started

    detector.eval()
    synchronize(device)
    infer_started = time.perf_counter()
    with torch.no_grad():
        scores = (
            adag_score(
                model=detector,
                features=features,
                adjacency=base_tensors["adj_norm_loop"],
                graph=graph,
                config=config,
                seed_base=seed * 1_000_003,
                device=device,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    synchronize(device)
    inference_seconds = time.perf_counter() - infer_started
    state = cpu_state(detector)

    reloaded = ADAGADAutoencoder(
        graph.feature_count, config["hidden_dim"]
    ).to(device)
    reloaded.load_state_dict(state)
    reloaded.load_encoder_state(averaged, freeze=True)
    reloaded.eval()
    with torch.no_grad():
        reload_scores = (
            adag_score(
                model=reloaded,
                features=features,
                adjacency=base_tensors["adj_norm_loop"],
                graph=graph,
                config=config,
                seed_base=seed * 1_000_003,
                device=device,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return {
        "scores": scores,
        "checkpoint": {
            "format": "recap_ofo12_torch_checkpoint_v1",
            "method": "ADA-GAD",
            "in_features": graph.feature_count,
            "config": config,
            "state_dict": state,
            "averaged_encoder_state": {
                key: value.cpu() for key, value in averaged.items()
            },
            "candidate_score": candidate.astype(np.float32),
        },
        "history": history,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "reload_max_abs_diff": float(np.max(np.abs(scores - reload_scores))),
    }


def execute_training(
    *,
    spec: OFOBaselineRun,
    graph,
    labels: np.ndarray | None,
    masks: dict[str, np.ndarray] | None,
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    if spec.method in {"GCN", "GAT", "BWGNN"}:
        assert labels is not None and masks is not None
        return train_torch_supervised(
            method=spec.method,
            graph=graph,
            labels=labels,
            masks=masks,
            config=config,
            device=device,
        )
    if spec.method == "XGBGraph":
        assert labels is not None and masks is not None
        return train_xgbgraph(
            graph=graph,
            labels=labels,
            masks=masks,
            config=config,
            seed=spec.seed,
            model_path=run_dir / "model.json",
        )
    if spec.method == "DOMINANT":
        return train_dominant(graph=graph, config=config, device=device)
    if spec.method == "AnomalyDAE":
        return train_anomalydae(
            graph=graph, config=config, seed=spec.seed, device=device
        )
    if spec.method == "CoLA":
        return train_cola(
            graph=graph, config=config, seed=spec.seed, device=device
        )
    if spec.method == "ADA-GAD":
        return train_adagad(
            graph=graph, config=config, seed=spec.seed, device=device
        )
    raise ValueError(spec.method)


def run_one(
    *,
    spec: OFOBaselineRun,
    dataset_dir: Path,
    output_root: Path,
    device: torch.device,
    smoke: bool,
    force: bool,
) -> dict[str, Any]:
    run_dir = output_root / spec.run_id
    success_path = run_dir / "_SUCCESS.json"
    if success_path.exists() and not force:
        return json.loads(success_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        run_dir / "status.json",
        {
            "status": "running",
            "started_at": utc_now(),
            "spec": spec.to_dict(),
            "smoke": smoke,
        },
    )
    set_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    config = resolved_config(spec.method, smoke)
    total_started = time.perf_counter()

    try:
        graph = prepare_graph(dataset_dir, spec.dataset)
        vault = OFOLabelVault(
            dataset_dir, spec.dataset, spec.supervised, spec.seed
        )
        labels = None
        masks = None
        if spec.supervised:
            labels, masks = vault.supervised_partitions()
            atomic_npz(run_dir / "split_masks.npz", **masks)

        training = execute_training(
            spec=spec,
            graph=graph,
            labels=labels,
            masks=masks,
            config=config,
            device=device,
            run_dir=run_dir,
        )
        scores = np.asarray(training.pop("scores"), dtype=np.float32)
        if scores.shape != (graph.node_count,):
            raise ValueError(
                f"{spec.run_id}: invalid score shape {scores.shape}"
            )
        if not np.isfinite(scores).all():
            raise ValueError(f"{spec.run_id}: non-finite scores")
        if training["reload_max_abs_diff"] > 1e-5:
            raise ValueError(
                f"{spec.run_id}: checkpoint reload drift "
                f"{training['reload_max_abs_diff']}"
            )

        if spec.supervised:
            assert masks is not None
            query_mask = masks["test"].copy()
        else:
            query_mask = np.ones(graph.node_count, dtype=np.bool_)
        score_path = run_dir / "scores.npz"
        atomic_npz(score_path, scores=scores, query_mask=query_mask)
        frozen_hashes = vault.freeze(
            score_path=score_path,
            scores=scores,
            query_mask=query_mask,
        )
        evaluation_labels = vault.evaluation_labels()
        auroc = float(
            roc_auc_score(evaluation_labels[query_mask], scores[query_mask])
        )
        auprc = float(
            average_precision_score(
                evaluation_labels[query_mask], scores[query_mask]
            )
        )
        if training.get("checkpoint") is not None:
            atomic_torch_save(
                run_dir / "checkpoint.pt", training.pop("checkpoint")
            )
        atomic_json(run_dir / "history.json", training.pop("history"))
        atomic_json(run_dir / "label_audit.json", vault.audit())
        atomic_json(run_dir / "resolved_config.json", config)

        total_seconds = time.perf_counter() - total_started
        peak_gpu_allocated = (
            int(torch.cuda.max_memory_allocated())
            if device.type == "cuda"
            else 0
        )
        peak_gpu_reserved = (
            int(torch.cuda.max_memory_reserved())
            if device.type == "cuda"
            else 0
        )
        max_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result = {
            "format": "recap_ofo12_run_result_v1",
            "status": "complete",
            "completed_at": utc_now(),
            "run_id": spec.run_id,
            "method": spec.method,
            "dataset": spec.dataset,
            "domain": DATASETS[spec.dataset]["domain"],
            "seed": spec.seed,
            "supervised": spec.supervised,
            "smoke": smoke,
            "auroc": auroc,
            "auprc": auprc,
            "query_nodes": int(query_mask.sum()),
            "node_count": graph.node_count,
            "feature_count": graph.feature_count,
            "directed_edges_after_binary_symmetrization": int(
                graph.adjacency.nnz
            ),
            "raw_sha256": graph.raw_sha256,
            **frozen_hashes,
            **training,
            "total_seconds": total_seconds,
            "peak_gpu_allocated_bytes": peak_gpu_allocated,
            "peak_gpu_reserved_bytes": peak_gpu_reserved,
            "peak_process_rss_kib": max_rss_kib,
            "process_rss_at_end_bytes": int(
                psutil.Process().memory_info().rss
            ),
            "environment": environment_metadata(device),
            "config": config,
        }
        atomic_json(run_dir / "result.json", result)
        success = {
            "status": "complete",
            "run_id": spec.run_id,
            "result_path": str(run_dir / "result.json"),
            "result_sha256": sha256_file(run_dir / "result.json"),
            "completed_at": result["completed_at"],
        }
        atomic_json(success_path, success)
        atomic_json(
            run_dir / "status.json",
            {
                "status": "complete",
                "completed_at": result["completed_at"],
                "spec": spec.to_dict(),
                "smoke": smoke,
            },
        )
        return success
    except Exception as exc:
        failure = {
            "status": "failed",
            "failed_at": utc_now(),
            "spec": spec.to_dict(),
            "smoke": smoke,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_json(run_dir / "failure.json", failure)
        atomic_json(run_dir / "status.json", failure)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/root/autodl-tmp/recap/dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "rebuttal/artifacts/ofo_12_baselines/formal/runs"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest()
    validate_manifest(manifest)
    by_id = {spec.run_id: spec for spec in manifest}
    if args.run_id not in by_id:
        raise KeyError(f"Unknown run ID: {args.run_id}")
    result = run_one(
        spec=by_id[args.run_id],
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        device=torch.device(args.device),
        smoke=args.smoke,
        force=args.force,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
