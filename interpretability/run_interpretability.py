#!/usr/bin/env python3
"""Generate interpretability artifacts for the RECAP community layer.

The script first trains RECAP on source graphs and saves the trained checkpoint,
then reloads the checkpoint and directly forwards target graphs for explanation.
Target labels are used only for post-hoc validation of the exported artifacts.

1. Community cards: post-hoc diagnostic profiles for learned communities.
2. Node explanations: language-style explanations for top-ranked anomalies.
3. Faithfulness proxies: component AP/AUROC, top-risk lift, context-mismatch lift.
4. Residual-community visualizations: pre-training residual embeddings, trained
   residual-community embeddings, community-risk maps, and score maps.

These artifacts are designed to support the claim that the community layer is
an interpretable prototype/context layer, even when it is not the dominant
driver of average anomaly-ranking performance.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, roc_auc_score


try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import patheffects
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime.
    plt = None
    patheffects = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ModelConfig, TrainConfig, create_default_configs  # noqa: E402
from detector import recapDetector  # noqa: E402
from model_checkpoint import ModelCheckpoint  # noqa: E402
from utils import prepare_datasets, set_seed  # noqa: E402


DEFAULT_TRAIN_DATASETS = ["pubmed", "Flickr", "questions", "YelpChi"]
DEFAULT_ANALYSIS_DATASETS = ["weibo"]


@dataclass
class TrialArtifacts:
    trial: int
    dataset: str
    scores: np.ndarray
    labels: np.ndarray
    hard_community: np.ndarray
    top_node_ids: set[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RECAP interpretability reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dims", type=int, default=32)
    parser.add_argument("--model", default="recap_auprc_best")
    parser.add_argument("--json-dir", default="params")
    parser.add_argument("--output-dir", default="interpretability/results")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory used to save source-trained checkpoints; defaults to <output-dir>/checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-paths",
        nargs="+",
        default=None,
        help="Load existing checkpoint(s) and skip source training. One path is treated as one trial.",
    )
    parser.add_argument(
        "--no-save-trained-checkpoints",
        action="store_true",
        help="Do not save/reload source-trained checkpoints before target inference.",
    )
    parser.add_argument(
        "--no-pretrain-embedding",
        action="store_true",
        help="Skip the pre-training residual embedding visualization.",
    )
    parser.add_argument("--train-datasets", nargs="+", default=None)
    parser.add_argument("--analysis-datasets", nargs="+", default=None)
    parser.add_argument("--top-nodes", type=int, default=20)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--community-top-nodes", type=int, default=5)
    parser.add_argument("--top-communities", type=int, default=3)
    parser.add_argument("--viz-max-nodes", type=int, default=4000)
    parser.add_argument("--figure-format", default="pdf", choices=["pdf", "png"])
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--diagnostics-interval", type=int, default=10)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--num-clusters", type=int, default=None)
    parser.add_argument("--knn-k", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run: one seed, <=2 epochs, pubmed -> Facebook, small reports",
    )
    return parser.parse_args()


def apply_quick_mode(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.trials = 1
    args.epochs = min(args.epochs, 2)
    args.top_nodes = min(args.top_nodes, 5)
    args.viz_max_nodes = min(args.viz_max_nodes, 1000)
    if args.train_datasets is None:
        args.train_datasets = ["pubmed"]
    if args.analysis_datasets is None:
        args.analysis_datasets = ["Facebook"]
    if args.checkpoint_paths is None:
        args.checkpoint_dir = args.checkpoint_dir or "/tmp/recap_interpretability_quick_checkpoints"


def ensure_device_available(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device={device}, but CUDA is unavailable. Use --device cpu."
        )


def apply_model_overrides(model_config, args: argparse.Namespace) -> None:
    if args.num_clusters is not None:
        model_config.num_clusters = args.num_clusters
    if args.knn_k is not None:
        model_config.knn_k = args.knn_k
    if args.beta is not None:
        model_config.beta = args.beta


def to_float(value: Any) -> float:
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def maybe_metric(labels: np.ndarray, scores: np.ndarray, metric: str) -> float:
    if len(np.unique(labels)) < 2:
        return math.nan
    if metric == "auroc":
        return float(roc_auc_score(labels, scores))
    if metric == "auprc":
        return float(average_precision_score(labels, scores))
    raise ValueError(metric)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size <= 1:
        return np.ones_like(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks / float(values.size - 1)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return math.nan
    return float(np.sum(values * weights) / total)


def top_pairs_text(values: np.ndarray, top_k: int, prefix: str = "C") -> str:
    if values.size == 0:
        return ""
    idx = np.argsort(values)[::-1][:top_k]
    return "; ".join(f"{prefix}{int(i)}:{values[i]:.3f}" for i in idx)


def node_list_text(node_ids: np.ndarray | list[int]) -> str:
    return ";".join(str(int(i)) for i in node_ids)


def normalized_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-12)
    active = probs > 0
    if probs.size <= 1:
        return 0.0
    entropy = -float(np.sum(probs[active] * np.log(probs[active])))
    return max(0.0, entropy / math.log(max(2, probs.size)))


def community_profile_vectors(artifact: dict[str, Any]) -> dict[str, np.ndarray | float]:
    H = artifact["H"]
    labels = artifact["labels"]
    hard = H.argmax(axis=1)
    n, c_count = H.shape
    global_rate = float(labels.mean()) if n else math.nan

    soft_mass = H.sum(axis=0).astype(np.float64)
    soft_rate = np.divide(
        (H * labels[:, None]).sum(axis=0),
        np.maximum(soft_mass, 1e-12),
    )
    lift = soft_rate / global_rate if global_rate > 0 else np.full(c_count, np.nan)
    hard_size = np.bincount(hard, minlength=c_count).astype(np.float64)
    hard_share = hard_size / max(1, n)

    return {
        "global_rate": global_rate,
        "soft_mass": soft_mass,
        "soft_share": soft_mass / max(1, n),
        "soft_rate": soft_rate,
        "lift": lift,
        "hard_size": hard_size,
        "hard_share": hard_share,
    }


def risk_tier(lift: float, soft_share: float) -> str:
    if math.isnan(lift):
        return "unknown"
    if lift >= 2.0 and soft_share >= 0.005:
        return "high-risk"
    if lift >= 1.2:
        return "elevated-risk"
    if lift <= 0.8:
        return "low-risk"
    return "baseline-risk"


def label_note(label: float) -> str:
    return "labeled anomaly" if int(label) == 1 else "label=0"


def node_diagnosis_type(
    *,
    adhesion_pct: float,
    context_pct: float,
    dominant_prob: float,
    neighbor_dominant_prob: float,
    community_lift: float,
    closest: int,
    dominant: int,
) -> str:
    if closest != dominant and context_pct >= 0.95:
        return "prototype-assignment tension"
    if dominant_prob < 0.75:
        return "mixed-community boundary anomaly"
    if neighbor_dominant_prob < 0.25 and context_pct >= 0.98:
        return "neighborhood context break"
    if adhesion_pct >= 0.995 and neighbor_dominant_prob >= 0.65:
        return "locally coherent prototype outlier"
    if community_lift >= 1.5 and adhesion_pct >= 0.95:
        return "high-risk community prototype outlier"
    if context_pct > adhesion_pct:
        return "context-driven anomaly"
    return "prototype-distance anomaly"


def node_explanation_text(
    *,
    node_id: int,
    rank: int,
    label: float,
    diagnosis: str,
    dominant: int,
    dominant_prob: float,
    community_rate: float,
    community_lift: float,
    community_share: float,
    distance_pct: float,
    distance_to_dominant: float,
    adhesion_pct: float,
    context_pct: float,
    context_contribution: float,
    neighbor_dominant_prob: float,
    neighbor_top: str,
    assignment_entropy_norm: float,
    closest: int,
    closest_distance: float,
) -> str:
    community_clause = (
        f"C{dominant} is {risk_tier(community_lift, community_share)} "
        f"(soft share {community_share:.1%}, anomaly rate {community_rate:.1%}, "
        f"lift {community_lift:.2f}x)."
    )

    if neighbor_dominant_prob >= 0.7:
        context_clause = (
            f"Its neighbors mostly agree with C{dominant} "
            f"({neighbor_dominant_prob:.1%} neighbor mass), so the alert is mainly a "
            "within-community residual outlier."
        )
    elif neighbor_dominant_prob >= 0.35:
        context_clause = (
            f"Its neighbors only partially support C{dominant} "
            f"({neighbor_dominant_prob:.1%} neighbor mass; {neighbor_top}), adding a "
            "moderate context-mismatch signal."
        )
    else:
        context_clause = (
            f"Its neighborhood distribution is different from its own assignment "
            f"({neighbor_dominant_prob:.1%} neighbor mass on C{dominant}; {neighbor_top}), "
            "which is a strong context-break signal."
        )

    if closest != dominant:
        prototype_clause = (
            f"The closest prototype is C{closest} (distance {closest_distance:.2f}), "
            f"but the soft assignment favors C{dominant}; this assignment/prototype "
            "tension makes the case more diagnostic."
        )
    else:
        prototype_clause = (
            f"Its distance to the assigned prototype is {distance_to_dominant:.2f}, "
            f"at the {distance_pct:.1%} percentile among nodes."
        )

    signal_clause = (
        f"Adhesion percentile={adhesion_pct:.1%}; context percentile={context_pct:.1%}; "
        f"beta-weighted context contribution={context_contribution:.3f}; "
        f"assignment entropy={assignment_entropy_norm:.3f}."
    )

    return (
        f"#{rank} node {node_id} ({label_note(label)}) is a {diagnosis}. "
        f"It assigns to C{dominant} with p={dominant_prob:.3f}. {community_clause} "
        f"{prototype_clause} {context_clause} {signal_clause}"
    )


@torch.no_grad()
def collect_dataset_artifacts(detector: recapDetector, dataset) -> dict[str, Any]:
    model = detector.model
    model.eval()
    graph = dataset.graph.to(detector.train_config["device"])
    model(graph)

    cluster = model.ego_clusters[0]
    E = model._view_embeds[0]
    E_init = model._view_embeds_init[0]
    cache_key = model._view_cache_keys[0]
    components = cluster.compute_score_components(E, E_init=E_init, cache_key=cache_key)
    H = cluster.cluster(E)

    denom = H.sum(dim=0).clamp(min=cluster.eps)
    prototypes = torch.mm(H.t(), E) / denom.unsqueeze(1)
    diff = E.unsqueeze(1) - prototypes.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=2)

    topk_idx = cluster._get_knn_candidates(E_init, cache_key=cache_key)

    return {
        "E": E.detach().cpu().numpy().astype(np.float32),
        "E_init": E_init.detach().cpu().numpy().astype(np.float32),
        "H": H.detach().cpu().numpy().astype(np.float32),
        "prototypes": prototypes.detach().cpu().numpy().astype(np.float32),
        "dist_sq": dist_sq.detach().cpu().numpy().astype(np.float32),
        "knn_idx": topk_idx.detach().cpu().numpy().astype(np.int64),
        "labels": graph.ano_labels.detach().cpu().numpy().astype(np.float32),
        "total": components["total"].detach().cpu().numpy().astype(np.float32),
        "adhesion": components["adhesion"].detach().cpu().numpy().astype(np.float32),
        "context": components["scale"].detach().cpu().numpy().astype(np.float32),
        "adhesion_raw": components["adhesion_raw"].detach().cpu().numpy().astype(np.float32),
        "context_raw": components["scale_raw"].detach().cpu().numpy().astype(np.float32),
        "neighbor_H": components["neighbor_context"].detach().cpu().numpy().astype(np.float32),
        "beta": float(cluster.beta),
    }


@torch.no_grad()
def collect_residual_embedding(detector: recapDetector, dataset) -> np.ndarray:
    """Collect residual embeddings from the detector's current model state."""
    model = detector.model
    model.eval()
    graph = dataset.graph.to(detector.train_config["device"])
    model(graph)
    return model._view_embeds[0].detach().cpu().numpy().astype(np.float32)


def detector_from_loaded_checkpoint(checkpoint_path: str, device: str):
    """Load a saved RECAP checkpoint and wrap it with the fields used here."""
    model, train_config, model_config, checkpoint_info = ModelCheckpoint().load_checkpoint(
        checkpoint_path,
        device=device,
    )
    train_config.device = device
    model.eval()
    return SimpleNamespace(model=model, train_config=train_config.to_dict()), model_config, checkpoint_info


def build_community_cards(
    artifact: dict[str, Any],
    dataset_name: str,
    trial: int,
    top_nodes: int,
) -> list[dict[str, Any]]:
    H = artifact["H"]
    labels = artifact["labels"]
    scores = artifact["total"]
    adhesion = artifact["adhesion"]
    context = artifact["context"]
    context_raw = artifact["context_raw"]
    hard = H.argmax(axis=1)
    prototypes = artifact["prototypes"]

    n, c_count = H.shape
    profile = community_profile_vectors(artifact)
    global_anomaly_rate = float(profile["global_rate"])
    rows: list[dict[str, Any]] = []

    for c in range(c_count):
        hard_mask = hard == c
        weights = H[:, c]
        soft_mass = float(weights.sum())
        hard_size = int(hard_mask.sum())
        if hard_size:
            hard_labels = labels[hard_mask]
            hard_scores = scores[hard_mask]
            local_indices = np.where(hard_mask)[0]
            top_local = local_indices[np.argsort(hard_scores)[::-1][:top_nodes]]
            hard_anomaly_rate = float(hard_labels.mean())
            mean_assignment = float(H[hard_mask, c].mean())
            mean_total = float(scores[hard_mask].mean())
            mean_adhesion = float(adhesion[hard_mask].mean())
            mean_context = float(context[hard_mask].mean())
            mean_context_raw = float(context_raw[hard_mask].mean())
        else:
            top_local = np.array([], dtype=np.int64)
            hard_anomaly_rate = math.nan
            mean_assignment = math.nan
            mean_total = math.nan
            mean_adhesion = math.nan
            mean_context = math.nan
            mean_context_raw = math.nan

        soft_anomaly_rate = float(profile["soft_rate"][c])
        anomaly_lift = (
            soft_anomaly_rate / global_anomaly_rate
            if global_anomaly_rate and not math.isnan(soft_anomaly_rate)
            else math.nan
        )

        rows.append(
            {
                "trial": trial,
                "dataset": dataset_name,
                "community": c,
                "hard_size": hard_size,
                "hard_share": hard_size / max(1, n),
                "soft_mass": soft_mass,
                "soft_share": soft_mass / max(1, n),
                "hard_anomaly_rate": hard_anomaly_rate,
                "soft_anomaly_rate": soft_anomaly_rate,
                "anomaly_lift": anomaly_lift,
                "risk_tier": risk_tier(anomaly_lift, soft_mass / max(1, n)),
                "mean_total_score": mean_total,
                "mean_adhesion_score": mean_adhesion,
                "mean_context_score": mean_context,
                "mean_context_raw": mean_context_raw,
                "mean_assignment_prob_hard": mean_assignment,
                "prototype_norm": float(np.linalg.norm(prototypes[c])),
                "top_nodes_by_score": node_list_text(top_local),
            }
        )

    rows.sort(
        key=lambda row: (
            -1.0 if math.isnan(row["anomaly_lift"]) else -row["anomaly_lift"],
            -row["soft_share"],
        )
    )
    return rows


def build_node_explanations(
    artifact: dict[str, Any],
    dataset_name: str,
    trial: int,
    top_nodes: int,
    top_communities: int,
) -> list[dict[str, Any]]:
    H = artifact["H"]
    labels = artifact["labels"]
    total = artifact["total"]
    adhesion = artifact["adhesion"]
    context = artifact["context"]
    adhesion_raw = artifact["adhesion_raw"]
    context_raw = artifact["context_raw"]
    neighbor_H = artifact["neighbor_H"]
    dist_sq = artifact["dist_sq"]
    beta = artifact["beta"]
    profile = community_profile_vectors(artifact)

    total_pct = percentile_ranks(total)
    adhesion_pct = percentile_ranks(adhesion)
    context_pct = percentile_ranks(context)
    hard = H.argmax(axis=1)
    dominant_distances = np.sqrt(np.maximum(dist_sq[np.arange(H.shape[0]), hard], 0.0))
    dominant_distance_pct = percentile_ranks(dominant_distances)
    node_ids = np.argsort(total)[::-1][:top_nodes]

    rows: list[dict[str, Any]] = []
    for rank, node_id in enumerate(node_ids, start=1):
        h_i = H[node_id]
        neighbor_i = neighbor_H[node_id]
        dominant = int(np.argmax(h_i))
        closest = int(np.argmin(dist_sq[node_id]))
        context_contribution = float(beta * context[node_id])
        neighbor_dominant_prob = float(neighbor_i[dominant])
        assignment_entropy_norm = normalized_entropy(h_i)
        community_rate = float(profile["soft_rate"][dominant])
        community_lift = float(profile["lift"][dominant])
        community_share = float(profile["soft_share"][dominant])
        distance_to_dominant = float(math.sqrt(max(dist_sq[node_id, dominant], 0.0)))
        distance_to_closest = float(math.sqrt(max(dist_sq[node_id, closest], 0.0)))
        diagnosis = node_diagnosis_type(
            adhesion_pct=float(adhesion_pct[node_id]),
            context_pct=float(context_pct[node_id]),
            dominant_prob=float(h_i[dominant]),
            neighbor_dominant_prob=neighbor_dominant_prob,
            community_lift=community_lift,
            closest=closest,
            dominant=dominant,
        )
        neighbor_top = top_pairs_text(neighbor_i, top_communities)
        explanation = node_explanation_text(
            node_id=int(node_id),
            rank=rank,
            label=float(labels[node_id]),
            diagnosis=diagnosis,
            dominant=dominant,
            dominant_prob=float(h_i[dominant]),
            community_rate=community_rate,
            community_lift=community_lift,
            community_share=community_share,
            distance_pct=float(dominant_distance_pct[node_id]),
            distance_to_dominant=distance_to_dominant,
            adhesion_pct=float(adhesion_pct[node_id]),
            context_pct=float(context_pct[node_id]),
            context_contribution=context_contribution,
            neighbor_dominant_prob=neighbor_dominant_prob,
            neighbor_top=neighbor_top,
            assignment_entropy_norm=assignment_entropy_norm,
            closest=closest,
            closest_distance=distance_to_closest,
        )

        rows.append(
            {
                "trial": trial,
                "dataset": dataset_name,
                "rank": rank,
                "node_id": int(node_id),
                "label": int(labels[node_id]),
                "diagnosis_type": diagnosis,
                "total_score": float(total[node_id]),
                "total_percentile": float(total_pct[node_id]),
                "adhesion_score": float(adhesion[node_id]),
                "adhesion_percentile": float(adhesion_pct[node_id]),
                "context_score": float(context[node_id]),
                "context_percentile": float(context_pct[node_id]),
                "context_contribution": context_contribution,
                "adhesion_raw": float(adhesion_raw[node_id]),
                "context_raw_js": float(context_raw[node_id]),
                "dominant_community": dominant,
                "dominant_prob": float(h_i[dominant]),
                "dominant_community_soft_share": community_share,
                "dominant_community_anomaly_rate": community_rate,
                "dominant_community_lift": community_lift,
                "neighbor_dominant_prob": neighbor_dominant_prob,
                "assignment_entropy_norm": assignment_entropy_norm,
                "closest_prototype": closest,
                "distance_to_dominant_prototype": distance_to_dominant,
                "distance_to_dominant_percentile": float(dominant_distance_pct[node_id]),
                "distance_to_closest_prototype": distance_to_closest,
                "top_communities": top_pairs_text(h_i, top_communities),
                "neighbor_top_communities": neighbor_top,
                "explanation": explanation,
            }
        )
    return rows


def build_explanation_metrics(
    artifact: dict[str, Any],
    dataset_name: str,
    trial: int,
    top_fraction: float,
) -> dict[str, Any]:
    labels = artifact["labels"]
    total = artifact["total"]
    adhesion = artifact["adhesion"]
    context = artifact["context"]
    adhesion_raw = artifact["adhesion_raw"]
    context_raw = artifact["context_raw"]
    hard = artifact["H"].argmax(axis=1)

    n = labels.size
    top_count = max(1, int(math.ceil(n * top_fraction)))
    top_idx = np.argsort(total)[::-1][:top_count]
    rest_mask = np.ones(n, dtype=bool)
    rest_mask[top_idx] = False

    top_labels = labels[top_idx]
    overall_anomaly_rate = float(labels.mean())
    top_anomaly_rate = float(top_labels.mean())
    top_lift = top_anomaly_rate / overall_anomaly_rate if overall_anomaly_rate > 0 else math.nan

    top_counts = np.bincount(hard[top_idx], minlength=artifact["H"].shape[1]).astype(np.float64)
    top_dist = top_counts / max(1.0, top_counts.sum())
    entropy = -np.sum(top_dist[top_dist > 0] * np.log(top_dist[top_dist > 0]))
    norm_entropy = entropy / math.log(max(2, len(top_dist)))

    def lift(values: np.ndarray) -> float:
        rest_values = values[rest_mask]
        denom = float(rest_values.mean()) if rest_values.size else math.nan
        if denom == 0 or math.isnan(denom):
            return math.nan
        return float(values[top_idx].mean() / denom)

    return {
        "trial": trial,
        "dataset": dataset_name,
        "n_nodes": int(n),
        "anomaly_rate": overall_anomaly_rate,
        "top_fraction": float(top_fraction),
        "top_count": int(top_count),
        "top_anomaly_rate": top_anomaly_rate,
        "top_anomaly_lift": top_lift,
        "top_community_entropy_norm": float(norm_entropy),
        "top_dominant_community": int(np.argmax(top_counts)),
        "total_AUROC": maybe_metric(labels, total, "auroc"),
        "total_AUPRC": maybe_metric(labels, total, "auprc"),
        "adhesion_AUROC": maybe_metric(labels, adhesion, "auroc"),
        "adhesion_AUPRC": maybe_metric(labels, adhesion, "auprc"),
        "context_AUROC": maybe_metric(labels, context, "auroc"),
        "context_AUPRC": maybe_metric(labels, context, "auprc"),
        "adhesion_raw_lift_top_vs_rest": lift(adhesion_raw),
        "context_raw_lift_top_vs_rest": lift(context_raw),
    }


def rank_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size != b.size or a.size < 2:
        return math.nan
    ra = percentile_ranks(a)
    rb = percentile_ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def build_stability_rows(
    trial_artifacts: list[TrialArtifacts],
    top_fraction: float,
) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[TrialArtifacts]] = defaultdict(list)
    for artifact in trial_artifacts:
        by_dataset[artifact.dataset].append(artifact)

    rows = []
    for dataset_name, items in sorted(by_dataset.items()):
        if len(items) < 2:
            continue
        jaccards = []
        spearmans = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i]
                b = items[j]
                inter = len(a.top_node_ids & b.top_node_ids)
                union = len(a.top_node_ids | b.top_node_ids)
                jaccards.append(inter / union if union else math.nan)
                spearmans.append(rank_spearman(a.scores, b.scores))
        rows.append(
            {
                "dataset": dataset_name,
                "num_trials": len(items),
                "top_fraction": top_fraction,
                "top_node_jaccard_mean": float(np.nanmean(jaccards)),
                "top_node_jaccard_std": float(np.nanstd(jaccards)),
                "score_rank_spearman_mean": float(np.nanmean(spearmans)),
                "score_rank_spearman_std": float(np.nanstd(spearmans)),
            }
        )
    return rows


def choose_plot_indices(labels: np.ndarray, scores: np.ndarray, max_nodes: int, seed: int) -> np.ndarray:
    n = labels.size
    if n <= max_nodes:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    top_count = min(max_nodes // 5, n)
    top_idx = np.argsort(scores)[::-1][:top_count]
    anomaly_idx = np.where(labels > 0)[0]
    keep = set(int(i) for i in top_idx)
    if anomaly_idx.size:
        anomaly_sample = rng.choice(
            anomaly_idx,
            size=min(max_nodes // 4, anomaly_idx.size),
            replace=False,
        )
        keep.update(int(i) for i in anomaly_sample)
    remaining = np.array([i for i in range(n) if i not in keep], dtype=np.int64)
    sample_size = max(0, max_nodes - len(keep))
    if remaining.size and sample_size:
        keep.update(int(i) for i in rng.choice(remaining, size=min(sample_size, remaining.size), replace=False))
    return np.array(sorted(keep), dtype=np.int64)


def choose_embedding_plot_indices(labels: np.ndarray, max_nodes: int, seed: int) -> np.ndarray:
    """Sample nodes for embedding visualizations without relying on anomaly scores."""
    n = labels.size
    if n <= max_nodes:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    anomaly_idx = np.where(labels > 0)[0]
    keep = set()
    if anomaly_idx.size:
        keep.update(
            int(i)
            for i in rng.choice(
                anomaly_idx,
                size=min(max_nodes // 4, anomaly_idx.size),
                replace=False,
            )
        )
    remaining = np.array([i for i in range(n) if i not in keep], dtype=np.int64)
    sample_size = max(0, max_nodes - len(keep))
    if remaining.size and sample_size:
        keep.update(int(i) for i in rng.choice(remaining, size=min(sample_size, remaining.size), replace=False))
    return np.array(sorted(keep), dtype=np.int64)


def community_palette(num_communities: int) -> np.ndarray:
    cmap = plt.get_cmap("gist_ncar", max(num_communities, 2))
    return np.asarray([cmap(i) for i in range(max(num_communities, 2))])


def choose_paper_plot_indices(
    labels: np.ndarray,
    scores: np.ndarray,
    H: np.ndarray,
    max_nodes: int,
    seed: int,
    min_confidence: float = 0.6,
) -> np.ndarray:
    """Balanced sample for paper figures so one giant hard community does not dominate."""
    n, c_count = H.shape
    if n <= max_nodes:
        return np.arange(n)

    rng = np.random.default_rng(seed)
    hard = H.argmax(axis=1)
    confidence = H.max(axis=1)
    keep: set[int] = set()

    # Always retain the highest-ranked examples because node annotations refer to them.
    top_count = min(max(50, max_nodes // 10), n)
    keep.update(int(i) for i in np.argsort(scores)[::-1][:top_count])

    # Keep a balanced set of confident members from every community.
    per_community = max(30, max_nodes // max(2 * c_count, 1))
    for c in range(c_count):
        candidates = np.where((hard == c) & (confidence >= min_confidence))[0]
        if candidates.size == 0:
            candidates = np.where(hard == c)[0]
        if candidates.size == 0:
            continue
        order = np.lexsort((-scores[candidates], -confidence[candidates]))
        chosen = candidates[order[:per_community]]
        keep.update(int(i) for i in chosen)

    # Add labeled anomalies for post-hoc visual validation.
    anomaly_idx = np.where(labels > 0)[0]
    if anomaly_idx.size:
        keep.update(
            int(i)
            for i in rng.choice(
                anomaly_idx,
                size=min(max_nodes // 8, anomaly_idx.size),
                replace=False,
            )
        )

    if len(keep) > max_nodes:
        arr = np.array(list(keep), dtype=np.int64)
        priority = scores[arr] + 0.02 * confidence[arr] + 0.01 * labels[arr]
        arr = arr[np.argsort(priority)[::-1][:max_nodes]]
        return np.array(sorted(arr), dtype=np.int64)

    remaining = np.array([i for i in range(n) if i not in keep], dtype=np.int64)
    sample_size = max_nodes - len(keep)
    if remaining.size and sample_size > 0:
        keep.update(int(i) for i in rng.choice(remaining, size=min(sample_size, remaining.size), replace=False))
    return np.array(sorted(keep), dtype=np.int64)


def compute_tsne_layout(
    E: np.ndarray,
    prototypes: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute a t-SNE layout after PCA pre-reduction for readable paper plots."""
    if prototypes is None:
        values = E
        proto_count = 0
    else:
        proto_count = len(prototypes)
        values = np.vstack([E, prototypes])

    if len(values) <= 3:
        coords = PCA(n_components=2, random_state=seed).fit_transform(values)
    else:
        pre_dim = min(30, values.shape[1], len(values) - 1)
        if pre_dim >= 2 and values.shape[1] > pre_dim:
            values_2d_input = PCA(n_components=pre_dim, random_state=seed).fit_transform(values)
        else:
            values_2d_input = values
        perplexity = min(35.0, max(5.0, (len(values_2d_input) - 1) / 4.0))
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=seed,
        ).fit_transform(values_2d_input)

    node_coords = coords[: len(E)]
    proto_coords = coords[len(E):] if proto_count else None
    return node_coords, proto_coords


def select_paper_communities(
    lift: np.ndarray,
    soft_share: np.ndarray,
    max_count: int,
) -> list[int]:
    selected: list[int] = []
    for c in np.argsort(np.nan_to_num(lift, nan=-1.0))[::-1]:
        if soft_share[c] >= 0.005:
            selected.append(int(c))
        if len(selected) >= max_count:
            return selected
    for c in np.argsort(np.nan_to_num(soft_share, nan=0.0))[::-1]:
        if int(c) not in selected and soft_share[c] > 0:
            selected.append(int(c))
        if len(selected) >= max_count:
            return selected
    return selected


def setup_plot_style() -> None:
    if plt is None:
        return
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def annotate_top_nodes(
    ax,
    coords: np.ndarray,
    plotted_idx: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    count: int = 10,
) -> None:
    plotted_pos = {int(node_id): pos for pos, node_id in enumerate(plotted_idx)}
    effects = [patheffects.withStroke(linewidth=2.5, foreground="white")] if patheffects else None
    for node_id in np.argsort(scores)[::-1][:count]:
        pos = plotted_pos.get(int(node_id))
        if pos is None:
            continue
        color = "#8b1e1e" if int(labels[node_id]) == 1 else "#252525"
        ax.annotate(
            str(int(node_id)),
            xy=(coords[pos, 0], coords[pos, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color=color,
            path_effects=effects,
        )


def make_figures(
    artifact: dict[str, Any],
    dataset_name: str,
    trial: int,
    figure_dir: Path,
    max_nodes: int,
    figure_format: str,
    pretrain_E: np.ndarray | None = None,
) -> list[str]:
    if plt is None:
        print("matplotlib is unavailable; skipping figures.")
        return []

    setup_plot_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    E = artifact["E"]
    labels = artifact["labels"]
    scores = artifact["total"]
    H = artifact["H"]
    hard = H.argmax(axis=1)
    adhesion = artifact["adhesion"]
    context = artifact["context"]
    prototypes = artifact["prototypes"]
    profile = community_profile_vectors(artifact)
    lift = np.asarray(profile["lift"], dtype=np.float64)
    soft_share = np.asarray(profile["soft_share"], dtype=np.float64)
    soft_rate = np.asarray(profile["soft_rate"], dtype=np.float64)

    idx = choose_plot_indices(labels, scores, max_nodes=max_nodes, seed=trial)
    pca_input = np.vstack([E[idx], prototypes])
    pca_coords = PCA(n_components=2, random_state=0).fit_transform(pca_input)
    coords = pca_coords[: len(idx)]
    prototype_coords = pca_coords[len(idx) :]
    label_idx = labels[idx] > 0
    score_pct = percentile_ranks(scores)
    adhesion_pct = percentile_ranks(adhesion)
    context_pct = percentile_ranks(context)
    top_nodes = np.argsort(scores)[::-1][:12]

    paths: list[str] = []

    def save_current(name: str) -> None:
        path = figure_dir / f"{dataset_name}_trial{trial}_{name}.{figure_format}"
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        paths.append(str(path))

    # Paper-oriented figures: emphasize community risk, soft responses, and
    # confident community assignments instead of forcing every soft community
    # into a single hard-color PCA view.
    paper_idx = choose_paper_plot_indices(
        labels=labels,
        scores=scores,
        H=H,
        max_nodes=max_nodes,
        seed=trial + 2027,
    )
    paper_coords, paper_proto_coords = compute_tsne_layout(
        E[paper_idx],
        prototypes,
        seed=trial + 11,
    )
    paper_conf = H[paper_idx].max(axis=1)
    paper_hard = hard[paper_idx]
    c_count = H.shape[1]
    palette = community_palette(c_count)
    paper_communities = select_paper_communities(lift, soft_share, max_count=10)
    paper_community_set = set(paper_communities)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    low_conf = paper_conf < 0.7
    other_comm = np.array([int(c) not in paper_community_set for c in paper_hard])
    background = low_conf | other_comm
    if background.any():
        ax.scatter(
            paper_coords[background, 0],
            paper_coords[background, 1],
            s=7,
            c="#d1d5db",
            alpha=0.32,
            linewidths=0,
            label="low-confidence / other communities",
        )
    for c in paper_communities:
        mask = (paper_hard == c) & (paper_conf >= 0.7)
        if not mask.any():
            continue
        ax.scatter(
            paper_coords[mask, 0],
            paper_coords[mask, 1],
            s=11,
            c=[palette[c]],
            alpha=0.76,
            linewidths=0,
            label=f"C{c}: lift={lift[c]:.1f}x, share={soft_share[c]:.1%}",
        )
    anomaly_mask = labels[paper_idx] > 0
    if anomaly_mask.any():
        ax.scatter(
            paper_coords[anomaly_mask, 0],
            paper_coords[anomaly_mask, 1],
            facecolors="none",
            edgecolors="#111827",
            s=34,
            linewidths=0.65,
            label="labeled anomaly",
            zorder=4,
        )
    if paper_proto_coords is not None:
        for c in paper_communities[:8]:
            ax.scatter(
                paper_proto_coords[c, 0],
                paper_proto_coords[c, 1],
                marker="*",
                s=170,
                facecolor=palette[c],
                edgecolor="#111827",
                linewidth=0.8,
                zorder=5,
            )
            ax.text(
                paper_proto_coords[c, 0],
                paper_proto_coords[c, 1],
                f"C{c}",
                fontsize=7,
                color="#111827",
                path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")] if patheffects else None,
            )
    annotate_top_nodes(ax, paper_coords, paper_idx, scores, labels, count=10)
    ax.set_title(f"{dataset_name} trial {trial}: high-confidence residual communities")
    ax.set_xlabel("t-SNE-1 of trained residual embeddings")
    ax.set_ylabel("t-SNE-2 of trained residual embeddings")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    save_current("paper_confident_communities_tsne")

    response_communities = select_paper_communities(lift, soft_share, max_count=4)
    if response_communities:
        ncols = 2
        nrows = int(math.ceil(len(response_communities) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.8, 3.7 * nrows), squeeze=False)
        for ax, c in zip(axes.ravel(), response_communities):
            sc = ax.scatter(
                paper_coords[:, 0],
                paper_coords[:, 1],
                c=H[paper_idx, c],
                s=8,
                cmap="YlOrRd",
                vmin=0.0,
                vmax=max(1e-6, float(np.max(H[paper_idx, c]))),
                alpha=0.78,
                linewidths=0,
            )
            if anomaly_mask.any():
                ax.scatter(
                    paper_coords[anomaly_mask, 0],
                    paper_coords[anomaly_mask, 1],
                    facecolors="none",
                    edgecolors="#0f172a",
                    s=24,
                    linewidths=0.55,
                )
            if paper_proto_coords is not None:
                ax.scatter(
                    paper_proto_coords[c, 0],
                    paper_proto_coords[c, 1],
                    marker="*",
                    s=150,
                    facecolor="#111827",
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )
            ax.set_title(
                f"C{c}: response, lift={lift[c]:.1f}x, share={soft_share[c]:.1%}",
                fontsize=10,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.01)
        for ax in axes.ravel()[len(response_communities):]:
            ax.axis("off")
        fig.suptitle(
            f"{dataset_name} trial {trial}: soft responses of risk-relevant communities",
            y=1.01,
            fontsize=13,
        )
        save_current("paper_high_risk_response_maps")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    valid = soft_share > 0
    bubble_sizes = 3500 * np.sqrt(np.maximum(soft_share[valid], 1e-8))
    bubble_colors = np.clip(np.nan_to_num(lift[valid], nan=0.0), 0.0, 3.0)
    sc = ax.scatter(
        soft_share[valid],
        lift[valid],
        s=bubble_sizes,
        c=bubble_colors,
        cmap="YlOrRd",
        alpha=0.72,
        edgecolors="#475569",
        linewidths=0.5,
    )
    ax.axhline(1.0, color="#334155", linestyle="--", linewidth=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("community soft share (log scale)")
    ax.set_ylabel("anomaly lift vs graph average")
    ax.set_title(f"{dataset_name} trial {trial}: community mass-risk profile")
    valid_ids = np.where(valid)[0]
    for c in select_paper_communities(lift, soft_share, max_count=8):
        ax.text(
            soft_share[c] * 1.03,
            lift[c] + 0.025,
            f"C{c}",
            fontsize=8,
            color="#111827",
            path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")] if patheffects else None,
        )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.84)
    cbar.set_label("anomaly lift (clipped at 3x)")
    save_current("paper_community_mass_lift")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    bar_order = select_paper_communities(lift, soft_share, max_count=10)
    y = np.arange(len(bar_order))
    bar_values = lift[bar_order]
    colors = [
        "#b91c1c" if lift[c] >= 2.0 and soft_share[c] >= 0.005
        else "#f97316" if lift[c] >= 1.0
        else "#94a3b8"
        for c in bar_order
    ]
    ax.barh(y, bar_values, color=colors, edgecolor="#7f1d1d", linewidth=0.4)
    ax.axvline(1.0, color="#334155", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"C{c} ({soft_share[c]:.1%})" for c in bar_order])
    ax.invert_yaxis()
    ax.set_xlabel("anomaly lift vs graph average")
    ax.set_title(f"{dataset_name} trial {trial}: risk-relevant residual communities")
    for yi, c in zip(y, bar_order):
        ax.text(
            bar_values[yi] + 0.03,
            yi,
            f"rate={soft_rate[c]:.1%}",
            va="center",
            fontsize=8,
            color="#111827",
        )
    save_current("paper_community_lift_bars")

    embed_idx = choose_embedding_plot_indices(labels, max_nodes=max_nodes, seed=trial + 97)
    trained_coords_for_communities = coords
    prototype_coords_for_communities = prototype_coords
    community_idx = idx
    has_pretrain_plot = pretrain_E is not None and pretrain_E.shape == E.shape
    if has_pretrain_plot:
        pca_input_embed = np.vstack([pretrain_E[embed_idx], E[embed_idx], prototypes])
        pca_embed = PCA(n_components=2, random_state=0).fit_transform(pca_input_embed)
        pretrain_coords = pca_embed[: len(embed_idx)]
        trained_coords_for_communities = pca_embed[len(embed_idx): 2 * len(embed_idx)]
        prototype_coords_for_communities = pca_embed[2 * len(embed_idx):]
        community_idx = embed_idx

        fig, ax = plt.subplots(figsize=(7.8, 6.2))
        normal = labels[embed_idx] <= 0
        anomaly = labels[embed_idx] > 0
        ax.scatter(
            pretrain_coords[normal, 0],
            pretrain_coords[normal, 1],
            s=8,
            c="#cbd5e1",
            alpha=0.42,
            linewidths=0,
            label="label=0",
        )
        if anomaly.any():
            ax.scatter(
                pretrain_coords[anomaly, 0],
                pretrain_coords[anomaly, 1],
                s=15,
                c="#ef4444",
                alpha=0.68,
                linewidths=0,
                label="labeled anomaly",
            )
        ax.set_title(f"{dataset_name} trial {trial}: residual embedding before source training")
        ax.set_xlabel("Joint residual PCA-1")
        ax.set_ylabel("Joint residual PCA-2")
        ax.legend(loc="upper right", frameon=True)
        save_current("pretrain_residual_embedding")

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    c_count = H.shape[1]
    hard_counts = np.bincount(hard, minlength=c_count)
    palette = community_palette(c_count)
    point_colors = palette[hard[community_idx]]
    ax.scatter(
        trained_coords_for_communities[:, 0],
        trained_coords_for_communities[:, 1],
        c=point_colors,
        s=8,
        alpha=0.58,
        linewidths=0,
    )
    anomaly_for_community = labels[community_idx] > 0
    if anomaly_for_community.any():
        ax.scatter(
            trained_coords_for_communities[anomaly_for_community, 0],
            trained_coords_for_communities[anomaly_for_community, 1],
            facecolors="none",
            edgecolors="#111827",
            s=32,
            linewidths=0.65,
            label="labeled anomaly",
            zorder=3,
        )
    active_for_legend = [
        int(c)
        for c in np.argsort(hard_counts)[::-1]
        if hard_counts[c] > 0
    ][:10]
    for c in active_for_legend:
        ax.scatter(
            [],
            [],
            s=28,
            c=[palette[c]],
            label=f"C{c} n={int(hard_counts[c])}",
        )
    high_mass = [
        int(c)
        for c in np.argsort(np.nan_to_num(soft_share, nan=0.0))[::-1]
        if soft_share[c] >= 0.005
    ][:12]
    for c in high_mass:
        ax.scatter(
            prototype_coords_for_communities[c, 0],
            prototype_coords_for_communities[c, 1],
            marker="*",
            s=145,
            facecolor=palette[c],
            edgecolor="#111827",
            linewidth=0.8,
            zorder=4,
        )
        ax.text(
            prototype_coords_for_communities[c, 0],
            prototype_coords_for_communities[c, 1],
            f"C{c}",
            fontsize=7,
            color="#111827",
            path_effects=[patheffects.withStroke(linewidth=2.4, foreground="white")] if patheffects else None,
        )
    annotate_top_nodes(
        ax,
        trained_coords_for_communities,
        community_idx,
        scores,
        labels,
        count=10,
    )
    ax.set_title(f"{dataset_name} trial {trial}: trained residual communities")
    ax.set_xlabel("Joint residual PCA-1" if has_pretrain_plot else "Residual PCA-1")
    ax.set_ylabel("Joint residual PCA-2" if has_pretrain_plot else "Residual PCA-2")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    save_current("trained_residual_communities")

    high_risk = [
        int(c)
        for c in np.argsort(np.nan_to_num(lift, nan=-1.0))[::-1]
        if soft_share[c] >= 0.005
    ][:8]

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    risk_values = np.clip(np.nan_to_num(lift[hard[idx]], nan=0.0), 0.0, 3.0)
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=risk_values,
        s=8,
        cmap="YlOrRd",
        alpha=0.62,
        linewidths=0,
    )
    if label_idx.any():
        ax.scatter(
            coords[label_idx, 0],
            coords[label_idx, 1],
            facecolors="none",
            edgecolors="#0f172a",
            s=38,
            linewidths=0.7,
            label="labeled anomaly",
        )
    for c in high_risk:
        ax.scatter(
            prototype_coords[c, 0],
            prototype_coords[c, 1],
            marker="*",
            s=185,
            facecolor="#1f2937",
            edgecolor="white",
            linewidth=0.9,
            zorder=4,
        )
        ax.text(
            prototype_coords[c, 0],
            prototype_coords[c, 1],
            f" C{c}\n {lift[c]:.1f}x",
            fontsize=7,
            color="#111827",
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")] if patheffects else None,
        )
    annotate_top_nodes(ax, coords, idx, scores, labels, count=10)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.86)
    cbar.set_label("community anomaly lift (clipped at 3x)")
    ax.set_title(f"{dataset_name} trial {trial}: residual community risk map")
    ax.set_xlabel("Residual PCA-1")
    ax.set_ylabel("Residual PCA-2")
    ax.legend(loc="upper right", frameon=True)
    save_current("community_risk_map")

    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=score_pct[idx],
        s=8,
        cmap="magma",
        alpha=0.72,
        linewidths=0,
    )
    if label_idx.any():
        ax.scatter(
            coords[label_idx, 0],
            coords[label_idx, 1],
            facecolors="none",
            edgecolors="#22d3ee",
            s=40,
            linewidths=0.8,
            label="labeled anomaly",
        )
    top_mask = np.isin(idx, top_nodes)
    if top_mask.any():
        ax.scatter(
            coords[top_mask, 0],
            coords[top_mask, 1],
            s=58,
            facecolors="none",
            edgecolors="white",
            linewidths=1.1,
            label="top-ranked nodes",
        )
    annotate_top_nodes(ax, coords, idx, scores, labels, count=12)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.86)
    cbar.set_label("score percentile")
    ax.set_title(f"{dataset_name} trial {trial}: anomaly score landscape")
    ax.set_xlabel("Residual PCA-1")
    ax.set_ylabel("Residual PCA-2")
    ax.legend(loc="upper right", frameon=True)
    save_current("anomaly_score_map")

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    normal_idx = labels[idx] <= 0
    anomaly_idx = labels[idx] > 0
    ax.scatter(
        adhesion_pct[idx][normal_idx],
        context_pct[idx][normal_idx],
        s=10,
        c="#cbd5e1",
        alpha=0.42,
        linewidths=0,
        label="label=0",
    )
    if anomaly_idx.any():
        ax.scatter(
            adhesion_pct[idx][anomaly_idx],
            context_pct[idx][anomaly_idx],
            s=18,
            c="#ef4444",
            alpha=0.65,
            linewidths=0,
            label="labeled anomaly",
        )
    plotted_pos = {int(node_id): pos for pos, node_id in enumerate(idx)}
    for node_id in top_nodes:
        pos = plotted_pos.get(int(node_id))
        if pos is None:
            continue
        ax.scatter(
            adhesion_pct[node_id],
            context_pct[node_id],
            s=70,
            facecolors="none",
            edgecolors="#111827",
            linewidths=1.0,
            zorder=4,
        )
        ax.annotate(
            str(int(node_id)),
            xy=(adhesion_pct[node_id], context_pct[node_id]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")] if patheffects else None,
        )
    ax.axvline(0.9, color="#475569", linestyle="--", linewidth=0.9)
    ax.axhline(0.9, color="#475569", linestyle="--", linewidth=0.9)
    ax.text(0.905, 0.03, "high adhesion", fontsize=8, color="#475569")
    ax.text(0.03, 0.905, "high context mismatch", fontsize=8, color="#475569")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("adhesion score percentile")
    ax.set_ylabel("context mismatch percentile")
    ax.set_title(f"{dataset_name} trial {trial}: decomposed anomaly evidence")
    ax.legend(loc="lower right", frameon=True)
    save_current("score_component_percentiles")

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    bar_order = [
        int(c)
        for c in np.argsort(np.nan_to_num(lift, nan=-1.0))[::-1]
        if soft_share[c] >= 0.002
    ][:12]
    y = np.arange(len(bar_order))
    bar_values = lift[bar_order]
    colors = plt.cm.YlOrRd(np.clip(bar_values / max(3.0, float(np.nanmax(bar_values))), 0.1, 1.0))
    ax.barh(y, bar_values, color=colors, edgecolor="#7f1d1d", linewidth=0.4)
    ax.axvline(1.0, color="#334155", linestyle="--", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"C{c}" for c in bar_order])
    ax.invert_yaxis()
    ax.set_xlabel("anomaly lift vs graph average")
    ax.set_title(f"{dataset_name} trial {trial}: highest-risk residual communities")
    for yi, c in zip(y, bar_order):
        ax.text(
            bar_values[yi] + 0.03,
            yi,
            f"rate={soft_rate[c]:.1%}, share={soft_share[c]:.1%}",
            va="center",
            fontsize=8,
            color="#111827",
        )
    save_current("community_lift_bars")

    return paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_text(value: Any, digits: int = 4) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_dataset_report(
    path: Path,
    dataset_name: str,
    trial: int,
    community_rows: list[dict[str, Any]],
    node_rows: list[dict[str, Any]],
    metric_row: dict[str, Any],
    figure_paths: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# RECAP Interpretability Report: {dataset_name} trial {trial}",
        "",
        "## Explanation Metrics",
        "",
        f"- Total AUPRC: `{metric_text(metric_row['total_AUPRC'])}`",
        f"- Adhesion-only AUPRC: `{metric_text(metric_row['adhesion_AUPRC'])}`",
        f"- Context-only AUPRC: `{metric_text(metric_row['context_AUPRC'])}`",
        f"- Top anomaly lift: `{metric_text(metric_row['top_anomaly_lift'])}`",
        f"- Context mismatch lift among top nodes: `{metric_text(metric_row['context_raw_lift_top_vs_rest'])}`",
        "",
        "## Highest-Risk Communities",
        "",
        "| Community | Tier | Soft Share | Soft Anomaly Rate | Lift | Mean Score | Top Nodes |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in community_rows[:10]:
        lines.append(
            f"| C{row['community']} | {row.get('risk_tier', '')} | "
            f"{metric_text(row['soft_share'])} | "
            f"{metric_text(row['soft_anomaly_rate'])} | {metric_text(row['anomaly_lift'])} | "
            f"{metric_text(row['mean_total_score'])} | {row['top_nodes_by_score']} |"
        )

    lines.extend(
        [
            "",
            "## Top Node Explanations",
            "",
            "| Rank | Node | Label | Diagnosis | Community | Adhesion % | Context % | Explanation |",
            "|---:|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in node_rows[:10]:
        lines.append(
            f"| {row['rank']} | {row['node_id']} | {row['label']} | "
            f"{row.get('diagnosis_type', '')} | C{row['dominant_community']} | "
            f"{metric_text(row['adhesion_percentile'])} | "
            f"{metric_text(row['context_percentile'])} | {row['explanation']} |"
        )

    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for figure_path in figure_paths:
            rel = os.path.relpath(figure_path, path.parent)
            lines.append(f"- `{rel}`")

    path.write_text("\n".join(lines) + "\n")


def write_summary_report(
    path: Path,
    args: argparse.Namespace,
    metric_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# RECAP Interpretability Summary",
        "",
        "This directory contains community-level and node-level interpretability artifacts.",
        "",
        "## Configuration",
        "",
        f"- Model: `{args.model}`",
        f"- Epochs: `{args.epochs}`",
        f"- Trials: `{args.trials}`",
        f"- Source-train datasets: `{args.train_datasets}`",
        f"- Analysis datasets: `{args.analysis_datasets}`",
        f"- Checkpoint dir: `{args.checkpoint_dir}`",
        "- Target usage: source-trained checkpoints are reloaded and directly forwarded on analysis graphs; target labels are post-hoc only.",
        "",
        "## Dataset Metrics",
        "",
        "| Dataset | Trial | Total AUPRC | Adhesion AUPRC | Context AUPRC | Top Lift | Context Lift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['dataset']} | {row['trial']} | {metric_text(row['total_AUPRC'])} | "
            f"{metric_text(row['adhesion_AUPRC'])} | {metric_text(row['context_AUPRC'])} | "
            f"{metric_text(row['top_anomaly_lift'])} | "
            f"{metric_text(row['context_raw_lift_top_vs_rest'])} |"
        )

    if training_rows:
        lines.extend(
            [
                "",
                "## Training / Inference Manifest",
                "",
                "| Trial | Mode | Seed | Epochs | Checkpoint |",
                "|---:|---|---:|---:|---|",
            ]
        )
        for row in training_rows:
            lines.append(
                f"| {row['trial']} | {row['mode']} | {row['seed']} | "
                f"{row['epochs']} | `{row['checkpoint_path']}` |"
            )

    if stability_rows:
        lines.extend(
            [
                "",
                "## Stability Across Seeds",
                "",
                "| Dataset | Trials | Top-Node Jaccard | Score Spearman |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in stability_rows:
            lines.append(
                f"| {row['dataset']} | {row['num_trials']} | "
                f"{metric_text(row['top_node_jaccard_mean'])} +/- "
                f"{metric_text(row['top_node_jaccard_std'])} | "
                f"{metric_text(row['score_rank_spearman_mean'])} +/- "
                f"{metric_text(row['score_rank_spearman_std'])} |"
            )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `community_cards.csv`: community-level diagnostic cards.",
            "- `node_explanations.csv`: top-node diagnosis types and node-level explanations.",
            "- `explanation_metrics.csv`: faithfulness and concentration metrics.",
            "- `stability_summary.csv`: seed-level explanation stability, when multiple trials are used.",
            "- `training_manifest.csv`: source-training checkpoints and target-inference provenance.",
            "- `reports/`: per-dataset Markdown reports.",
            "- `figures/`: paper-ready community lift/mass/response plots, t-SNE confident-community maps, pre-training embeddings, trained community embeddings, community risk maps, score maps, component-percentile plots, and lift bars.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def run_one_trial(
    args: argparse.Namespace,
    trial: int,
    base_train_config,
    base_model_config,
    data_train: list,
    data_analysis: list,
    analysis_names: list[str],
    output_dir: Path,
) -> tuple[list[dict], list[dict], list[dict], list[TrialArtifacts], list[dict]]:
    set_seed(trial)
    train_config = copy.deepcopy(base_train_config)
    model_config = copy.deepcopy(base_model_config)
    train_config.device = args.device
    train_config.epochs = args.epochs
    train_config.trials = args.trials
    train_config.save_checkpoint = not args.no_save_trained_checkpoints
    train_config.output_dir = str(output_dir)
    train_config.log_diagnostics = not args.no_diagnostics
    train_config.diagnostics_interval = args.diagnostics_interval

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    training_rows: list[dict] = []
    pretrain_embeddings: dict[str, np.ndarray] = {}

    if args.checkpoint_paths:
        checkpoint_path = args.checkpoint_paths[trial]
        detector, loaded_model_config, checkpoint_info = detector_from_loaded_checkpoint(
            checkpoint_path,
            device=args.device,
        )
        model_config = loaded_model_config
        source_mode = "loaded_checkpoint"
        if not args.no_pretrain_embedding:
            pretrain_seed = checkpoint_info.get("seed")
            set_seed(int(pretrain_seed) if pretrain_seed is not None else trial)
            pretrain_detector = recapDetector(
                train_config,
                model_config,
                {"train": [], "test": data_analysis},
            )
            for dataset, dataset_name in zip(data_analysis, analysis_names):
                pretrain_embeddings[dataset_name] = collect_residual_embedding(pretrain_detector, dataset)
        training_rows.append(
            {
                "trial": trial,
                "mode": source_mode,
                "seed": checkpoint_info.get("seed", trial),
                "epochs": checkpoint_info.get("epoch", args.epochs),
                "checkpoint_path": checkpoint_path,
                "train_datasets": node_list_text([]),
                "analysis_datasets": ";".join(analysis_names),
                "model": model_config.model,
                "num_clusters": model_config.num_clusters,
                "knn_k": model_config.knn_k,
                "beta": model_config.beta,
            }
        )
    else:
        detector = recapDetector(train_config, model_config, {"train": data_train, "test": data_analysis})
        if not args.no_pretrain_embedding:
            print("  Capturing pre-training residual embeddings on analysis graphs")
            for dataset, dataset_name in zip(data_analysis, analysis_names):
                pretrain_embeddings[dataset_name] = collect_residual_embedding(detector, dataset)

        print("  Source training RECAP before target inference")
        train_history = detector.train(verbose=True)

        checkpoint_path = ""
        if not args.no_save_trained_checkpoints:
            checkpoint_manager = ModelCheckpoint(str(checkpoint_dir))
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model=detector.get_model(),
                train_config=train_config,
                model_config=model_config,
                epoch=train_config.epochs,
                trial=trial,
                metrics=None,
                seed=trial,
                history=train_history,
            )
            print("  Reloading saved checkpoint for target-only inference")
            detector, loaded_model_config, checkpoint_info = detector_from_loaded_checkpoint(
                checkpoint_path,
                device=args.device,
            )
            model_config = loaded_model_config
            completed_epochs = checkpoint_info.get("epoch", train_config.epochs)
        else:
            completed_epochs = train_history.get("epochs", train_config.epochs)

        training_rows.append(
            {
                "trial": trial,
                "mode": "source_train_then_reload" if checkpoint_path else "source_train_in_memory",
                "seed": trial,
                "epochs": completed_epochs,
                "checkpoint_path": checkpoint_path,
                "train_datasets": ";".join(args.train_datasets),
                "analysis_datasets": ";".join(analysis_names),
                "model": model_config.model,
                "num_clusters": model_config.num_clusters,
                "knn_k": model_config.knn_k,
                "beta": model_config.beta,
            }
        )

    community_rows: list[dict] = []
    node_rows: list[dict] = []
    metric_rows: list[dict] = []
    trial_artifacts: list[TrialArtifacts] = []

    for dataset, dataset_name in zip(data_analysis, analysis_names):
        print(f"  Explaining {dataset_name} (trial {trial})")
        artifact = collect_dataset_artifacts(detector, dataset)
        community = build_community_cards(
            artifact,
            dataset_name=dataset_name,
            trial=trial,
            top_nodes=args.community_top_nodes,
        )
        nodes = build_node_explanations(
            artifact,
            dataset_name=dataset_name,
            trial=trial,
            top_nodes=args.top_nodes,
            top_communities=args.top_communities,
        )
        metrics = build_explanation_metrics(
            artifact,
            dataset_name=dataset_name,
            trial=trial,
            top_fraction=args.top_fraction,
        )
        figures = []
        if not args.no_figures:
            figures = make_figures(
                artifact,
                dataset_name=dataset_name,
                trial=trial,
                figure_dir=output_dir / "figures",
                max_nodes=args.viz_max_nodes,
                figure_format=args.figure_format,
                pretrain_E=pretrain_embeddings.get(dataset_name),
            )

        write_dataset_report(
            output_dir / "reports" / f"{dataset_name}_trial{trial}.md",
            dataset_name=dataset_name,
            trial=trial,
            community_rows=community,
            node_rows=nodes,
            metric_row=metrics,
            figure_paths=figures,
        )

        community_rows.extend(community)
        node_rows.extend(nodes)
        metric_rows.append(metrics)

        top_count = max(1, int(math.ceil(len(artifact["labels"]) * args.top_fraction)))
        top_node_ids = set(int(i) for i in np.argsort(artifact["total"])[::-1][:top_count])
        trial_artifacts.append(
            TrialArtifacts(
                trial=trial,
                dataset=dataset_name,
                scores=artifact["total"],
                labels=artifact["labels"],
                hard_community=artifact["H"].argmax(axis=1),
                top_node_ids=top_node_ids,
            )
        )

    return community_rows, node_rows, metric_rows, trial_artifacts, training_rows


def main() -> None:
    args = parse_args()
    apply_quick_mode(args)
    ensure_device_available(args.device)
    os.chdir(PROJECT_ROOT)

    train_datasets = args.train_datasets or DEFAULT_TRAIN_DATASETS
    analysis_datasets = args.analysis_datasets or DEFAULT_ANALYSIS_DATASETS
    if args.checkpoint_paths:
        args.trials = len(args.checkpoint_paths)
    args.train_datasets = train_datasets
    args.analysis_datasets = analysis_datasets

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = args.checkpoint_dir or str(output_dir / "checkpoints")

    if args.checkpoint_paths:
        checkpoint = torch.load(args.checkpoint_paths[0], map_location="cpu")
        train_config = TrainConfig.from_dict(checkpoint["train_config"])
        model_config = ModelConfig.from_dict(checkpoint["model_config"])
        args.model = model_config.model
        args.dims = model_config.dims
        args.epochs = train_config.epochs
    else:
        train_config, model_config = create_default_configs(
            model_name=args.model,
            json_dir=args.json_dir,
            dims=args.dims,
        )
        apply_model_overrides(model_config, args)

    print("\n" + "=" * 70)
    print("RECAP Interpretability Experiments")
    print("=" * 70)
    print(f"Project root      : {PROJECT_ROOT}")
    print(f"Train datasets    : {train_datasets}")
    print(f"Analysis datasets : {analysis_datasets}")
    print(f"Trials            : {args.trials}")
    print(f"Epochs            : {args.epochs}")
    print(f"Device            : {args.device}")
    print(f"Output dir        : {output_dir}")
    print(f"Checkpoint dir    : {args.checkpoint_dir}")
    if args.checkpoint_paths:
        print(f"Checkpoint mode   : loading {len(args.checkpoint_paths)} checkpoint(s); no source training")

    data_train, data_analysis = prepare_datasets(
        dims=args.dims,
        train_datasets=[] if args.checkpoint_paths else train_datasets,
        test_datasets=analysis_datasets,
        num_hops=model_config.num_hops,
    )

    all_community_rows: list[dict] = []
    all_node_rows: list[dict] = []
    all_metric_rows: list[dict] = []
    all_trial_artifacts: list[TrialArtifacts] = []
    all_training_rows: list[dict] = []

    start = time.perf_counter()
    for trial in range(args.trials):
        print("\n" + "=" * 70)
        print(f"Trial {trial}")
        print("=" * 70)
        try:
            community_rows, node_rows, metric_rows, trial_artifacts, training_rows = run_one_trial(
                args=args,
                trial=trial,
                base_train_config=train_config,
                base_model_config=model_config,
                data_train=data_train,
                data_analysis=data_analysis,
                analysis_names=analysis_datasets,
                output_dir=output_dir,
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"Trial {trial} failed: {exc}")
            continue

        all_community_rows.extend(community_rows)
        all_node_rows.extend(node_rows)
        all_metric_rows.extend(metric_rows)
        all_trial_artifacts.extend(trial_artifacts)
        all_training_rows.extend(training_rows)

    stability_rows = build_stability_rows(all_trial_artifacts, top_fraction=args.top_fraction)

    write_csv(output_dir / "community_cards.csv", all_community_rows)
    write_csv(output_dir / "node_explanations.csv", all_node_rows)
    write_csv(output_dir / "explanation_metrics.csv", all_metric_rows)
    write_csv(output_dir / "stability_summary.csv", stability_rows)
    write_csv(output_dir / "training_manifest.csv", all_training_rows)

    payload = {
        "config": {
            **vars(args),
            "project_root": str(PROJECT_ROOT),
            "model_config": model_config.to_dict(),
        },
        "community_cards": all_community_rows,
        "node_explanations": all_node_rows,
        "explanation_metrics": all_metric_rows,
        "stability": stability_rows,
        "training_manifest": all_training_rows,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "interpretability_raw.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )
    write_summary_report(
        output_dir / "interpretability_summary.md",
        args,
        all_metric_rows,
        stability_rows,
        all_training_rows,
    )

    print("\n" + "=" * 70)
    print("Saved interpretability artifacts")
    print("=" * 70)
    print(f"  Community cards : {output_dir / 'community_cards.csv'}")
    print(f"  Node reports    : {output_dir / 'node_explanations.csv'}")
    print(f"  Metrics         : {output_dir / 'explanation_metrics.csv'}")
    print(f"  Stability       : {output_dir / 'stability_summary.csv'}")
    print(f"  Training manifest: {output_dir / 'training_manifest.csv'}")
    print(f"  Summary         : {output_dir / 'interpretability_summary.md'}")


if __name__ == "__main__":
    main()
