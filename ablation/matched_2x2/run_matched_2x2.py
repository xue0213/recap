#!/usr/bin/env python3
"""Run residual/non-residual + simple detector ablations against RECAP.

The simple-combine baselines intentionally do not train a transferable mapping.
They use the same aligned features and multi-hop propagation as RECAP, construct
either the raw residual embedding

    E0 = [X^(1) - X^(0) || ... || X^(L) - X^(0)],

or the matched non-residual embedding

    F0 = [X^(1) || ... || X^(L)],

and then score target-graph nodes with a classical unsupervised detector.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import create_default_configs  # noqa: E402
from detector import recapDetector  # noqa: E402
from utils import prepare_datasets, set_seed, test_eval  # noqa: E402


SIMPLE_METHODS = (
    "Non-residual + KMeans",
    "Residual + KMeans",
    "Residual + Spectral Clustering",
    "Residual + GMM",
    "Residual + LOF",
)

DEFAULT_TRAIN_DATASETS = ["pubmed", "Flickr", "questions", "YelpChi"]
DEFAULT_TEST_DATASETS = [
    "Facebook",
    "cora",
    "citeseer",
    "ACM",
    "BlogCatalog",
    "weibo",
    "Reddit",
    "Amazon",
]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


@dataclass
class MethodResult:
    method: str
    trial: int
    dataset: str
    AUROC: float
    AUPRC: float
    seconds: float
    status: str = "ok"
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual simple-combine ablation for RECAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--device", default="cpu", help="Device for RECAP training/evaluation")
    parser.add_argument("--trials", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--epochs", type=int, default=100, help="RECAP training epochs")
    parser.add_argument("--dims", type=int, default=32, help="Aligned feature dimension")
    parser.add_argument("--model", default="recap_auprc_best", help="Model config name")
    parser.add_argument("--json-dir", default="params", help="Model JSON config directory")
    parser.add_argument("--output-dir", default="ablation/simple_combine/results")
    parser.add_argument("--train-datasets", nargs="+", default=None)
    parser.add_argument("--test-datasets", nargs="+", default=None)

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=SIMPLE_METHODS,
        default=list(SIMPLE_METHODS),
        help="Simple residual baselines to run",
    )
    parser.add_argument("--no-recap", action="store_true", help="Skip full RECAP training/eval")
    parser.add_argument(
        "--save-recap-checkpoints",
        action="store_true",
        help="Save RECAP checkpoints for each trial",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Disable RECAP training diagnostics logs",
    )
    parser.add_argument("--diagnostics-interval", type=int, default=10)

    parser.add_argument("--num-clusters", type=int, default=None, help="Override C for baselines/RECAP")
    parser.add_argument("--knn-k", type=int, default=None, help="Override RECAP KNN k")
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--gmm-covariance", default="diag", choices=["full", "tied", "diag", "spherical"])
    parser.add_argument("--gmm-max-iter", type=int, default=300)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    parser.add_argument(
        "--gmm-reg-covar-retries",
        type=parse_float_list,
        default=parse_float_list("1e-6,1e-5,1e-4,1e-3,1e-2"),
        help="Comma-separated reg_covar retry schedule for robust GMM fitting",
    )
    parser.add_argument(
        "--gmm-min-std",
        type=float,
        default=1e-8,
        help="Drop near-constant residual dimensions before fitting GMM",
    )
    parser.add_argument(
        "--gmm-no-component-reduction",
        action="store_true",
        help="Disable GMM fallback retries with fewer mixture components",
    )
    parser.add_argument("--spectral-neighbors", type=int, default=None)
    parser.add_argument("--spectral-n-init", type=int, default=5)
    parser.add_argument("--spectral-max-nodes", type=int, default=3000)
    parser.add_argument(
        "--spectral-large-mode",
        default="sample",
        choices=["sample", "error"],
        help=(
            "For graphs above --spectral-max-nodes, either run spectral clustering "
            "on a sampled residual subgraph and score all nodes by sampled centroids, "
            "or record an error when --continue-on-error is set."
        ),
    )
    parser.add_argument("--lof-neighbors", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed method/dataset runs as NaN instead of stopping",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny smoke run: 1 trial, <=2 RECAP epochs, and two small target datasets",
    )

    return parser.parse_args()


def apply_quick_mode(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.trials = 1
    args.epochs = min(args.epochs, 2)
    if args.train_datasets is None:
        args.train_datasets = ["pubmed"]
    if args.test_datasets is None:
        args.test_datasets = ["Facebook", "cora"]
    args.spectral_max_nodes = min(args.spectral_max_nodes, 5000)


def ensure_device_available(device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested device={device}, but torch.cuda.is_available() is False. "
            "Use --device cpu on this machine."
        )


def to_serializable_config(args: argparse.Namespace, model_config) -> dict:
    config = vars(args).copy()
    config["project_root"] = str(PROJECT_ROOT)
    config["model_config"] = model_config.to_dict()
    return config


def residual_embedding_np(dataset) -> np.ndarray:
    """Return raw residual embedding from propagated hop features."""
    x_list = dataset.graph.x_list
    if x_list is None:
        raise RuntimeError(f"{dataset.name} has no propagated x_list")
    if isinstance(x_list[0], list):
        raise ValueError("simple_combine currently supports single-view datasets only")

    x0 = x_list[0]
    residuals = [x_i - x0 for x_i in x_list[1:]]
    embedding = torch.hstack(residuals).detach().cpu().numpy().astype(np.float32)
    return np.nan_to_num(embedding, copy=False)


def non_residual_embedding_np(dataset) -> np.ndarray:
    """Return the propagated-hop concatenation used by RECAP w/o Residual."""
    x_list = dataset.graph.x_list
    if x_list is None:
        raise RuntimeError(f"{dataset.name} has no propagated x_list")
    if isinstance(x_list[0], list):
        raise ValueError("simple_combine currently supports single-view datasets only")

    embedding = torch.hstack(x_list[1:]).detach().cpu().numpy().astype(np.float32)
    return np.nan_to_num(embedding, copy=False)


def labels_np(dataset) -> np.ndarray:
    return dataset.graph.ano_labels.detach().cpu().numpy()


def standardized_residual(dataset) -> np.ndarray:
    embedding = residual_embedding_np(dataset)
    embedding = StandardScaler().fit_transform(embedding)
    return np.nan_to_num(embedding.astype(np.float32), copy=False)


def standardized_non_residual(dataset) -> np.ndarray:
    embedding = non_residual_embedding_np(dataset)
    embedding = StandardScaler().fit_transform(embedding)
    return np.nan_to_num(embedding.astype(np.float32), copy=False)


def cluster_centroid_scores(embedding: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    centers = np.zeros((n_clusters, embedding.shape[1]), dtype=np.float32)
    global_center = embedding.mean(axis=0, keepdims=True)
    for c in range(n_clusters):
        mask = labels == c
        centers[c] = embedding[mask].mean(axis=0) if np.any(mask) else global_center
    diff = embedding - centers[labels]
    return np.linalg.norm(diff, axis=1)


def score_kmeans(embedding: np.ndarray, n_clusters: int, seed: int, args: argparse.Namespace) -> np.ndarray:
    n_clusters = min(n_clusters, embedding.shape[0])
    model = KMeans(
        n_clusters=n_clusters,
        n_init=args.kmeans_n_init,
        random_state=seed,
        algorithm="lloyd",
    )
    labels = model.fit_predict(embedding)
    return np.min(model.transform(embedding), axis=1)


def score_gmm(embedding: np.ndarray, n_clusters: int, seed: int, args: argparse.Namespace) -> np.ndarray:
    n_clusters = min(n_clusters, embedding.shape[0])
    gmm_embedding = embedding.astype(np.float64, copy=False)

    std = gmm_embedding.std(axis=0)
    keep = std > args.gmm_min_std
    if np.any(keep) and not np.all(keep):
        gmm_embedding = gmm_embedding[:, keep]

    covariances = [args.gmm_covariance]
    for fallback in ("diag", "spherical"):
        if fallback not in covariances:
            covariances.append(fallback)

    reg_values = [args.gmm_reg_covar, *args.gmm_reg_covar_retries]
    reg_values = sorted({float(v) for v in reg_values if float(v) > 0.0})

    component_values = [n_clusters]
    if not args.gmm_no_component_reduction:
        component_values.extend(
            [
                int(round(n_clusters * 0.75)),
                int(round(n_clusters * 0.50)),
                int(round(n_clusters * 0.33)),
            ]
        )
    component_values = sorted(
        {min(max(2, int(v)), gmm_embedding.shape[0]) for v in component_values},
        reverse=True,
    )

    last_error: Exception | None = None
    first_attempt = True
    for n_components in component_values:
        for covariance_type in covariances:
            for reg_covar in reg_values:
                try:
                    model = GaussianMixture(
                        n_components=n_components,
                        covariance_type=covariance_type,
                        reg_covar=reg_covar,
                        max_iter=args.gmm_max_iter,
                        random_state=seed,
                        n_init=3,
                    )
                    model.fit(gmm_embedding)
                    if not first_attempt:
                        print(
                            "    GMM fallback succeeded: "
                            f"components={n_components}, covariance={covariance_type}, "
                            f"reg_covar={reg_covar:g}"
                        )
                    return -model.score_samples(gmm_embedding)
                except Exception as exc:
                    last_error = exc
                    first_attempt = False

    raise RuntimeError(f"GMM failed after robust retries; last error: {last_error}")


def score_lof(embedding: np.ndarray, n_clusters: int, seed: int, args: argparse.Namespace) -> np.ndarray:
    del n_clusters, seed
    n_neighbors = min(max(2, args.lof_neighbors), embedding.shape[0] - 1)
    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        metric="euclidean",
        contamination="auto",
        n_jobs=args.n_jobs,
    )
    model.fit_predict(embedding)
    return -model.negative_outlier_factor_


def score_spectral(embedding: np.ndarray, n_clusters: int, seed: int, args: argparse.Namespace) -> np.ndarray:
    n = embedding.shape[0]
    n_clusters = min(n_clusters, n)
    n_neighbors = args.spectral_neighbors
    if n_neighbors is None:
        n_neighbors = max(10, args.knn_k or 10)
    n_neighbors = min(max(n_clusters, n_neighbors), n - 1)

    graph_embedding = normalize(embedding, norm="l2", axis=1)
    fit_embedding = graph_embedding
    sample_idx = None
    if args.spectral_max_nodes and n > args.spectral_max_nodes:
        if args.spectral_large_mode == "error":
            raise RuntimeError(
                f"Spectral clustering skipped for n={n} > --spectral-max-nodes={args.spectral_max_nodes}. "
                "Increase the limit or use --spectral-large-mode sample."
            )
        rng = np.random.default_rng(seed)
        sample_size = min(args.spectral_max_nodes, n)
        sample_idx = np.sort(rng.choice(n, size=sample_size, replace=False))
        fit_embedding = graph_embedding[sample_idx]
        n_neighbors = min(n_neighbors, sample_size - 1)

    distance_graph = kneighbors_graph(
        fit_embedding,
        n_neighbors=n_neighbors,
        mode="distance",
        include_self=False,
        n_jobs=args.n_jobs,
    )
    distances = distance_graph.data
    positive = distances[distances > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-6)
    affinity = distance_graph.tocsr(copy=True)
    affinity.data = np.exp(-(affinity.data ** 2) / (2.0 * sigma * sigma))
    affinity = 0.5 * (affinity + affinity.T)
    affinity = sparse.csr_matrix(affinity)

    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=args.spectral_n_init,
        eigen_solver="arpack",
    ).fit_predict(affinity)

    if sample_idx is None:
        return cluster_centroid_scores(graph_embedding.astype(np.float32), labels, n_clusters)

    centers = np.zeros((n_clusters, graph_embedding.shape[1]), dtype=np.float32)
    global_center = fit_embedding.mean(axis=0, keepdims=True)
    for c in range(n_clusters):
        mask = labels == c
        centers[c] = fit_embedding[mask].mean(axis=0) if np.any(mask) else global_center
    assigned = pairwise_distances_argmin(graph_embedding, centers)
    diff = graph_embedding - centers[assigned]
    return np.linalg.norm(diff, axis=1)


SCORE_FNS: dict[str, Callable[[np.ndarray, int, int, argparse.Namespace], np.ndarray]] = {
    "Non-residual + KMeans": score_kmeans,
    "Residual + KMeans": score_kmeans,
    "Residual + Spectral Clustering": score_spectral,
    "Residual + GMM": score_gmm,
    "Residual + LOF": score_lof,
}


def evaluate_scores(dataset, scores: np.ndarray) -> tuple[float, float]:
    scores = np.nan_to_num(np.asarray(scores, dtype=np.float64), copy=False)
    metrics = test_eval(labels_np(dataset), scores)
    return float(metrics["AUROC"]), float(metrics["AUPRC"])


def run_simple_methods(
    data_test: list,
    dataset_names: list[str],
    methods: list[str],
    n_clusters: int,
    args: argparse.Namespace,
) -> list[MethodResult]:
    results: list[MethodResult] = []

    print("\n" + "=" * 70)
    print("Simple residual-combine baselines")
    print("=" * 70)
    for trial in range(args.trials):
        set_seed(trial)
        print(f"\n--- Simple baselines trial {trial} ---")
        for dataset, dataset_name in zip(data_test, dataset_names):
            embeddings = {}
            for method in methods:
                embedding_kind = (
                    "non_residual"
                    if method == "Non-residual + KMeans"
                    else "residual"
                )
                if embedding_kind not in embeddings:
                    embeddings[embedding_kind] = (
                        standardized_non_residual(dataset)
                        if embedding_kind == "non_residual"
                        else standardized_residual(dataset)
                    )
                embedding = embeddings[embedding_kind]
                start = time.perf_counter()
                try:
                    scores = SCORE_FNS[method](embedding, n_clusters, trial, args)
                    auroc, auprc = evaluate_scores(dataset, scores)
                    elapsed = time.perf_counter() - start
                    results.append(MethodResult(method, trial, dataset_name, auroc, auprc, elapsed))
                    print(
                        f"  {method:32s} {dataset_name:12s} "
                        f"AUROC={auroc:.4f} AUPRC={auprc:.4f} ({elapsed:.1f}s)"
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - start
                    if not args.continue_on_error:
                        raise
                    results.append(
                        MethodResult(
                            method=method,
                            trial=trial,
                            dataset=dataset_name,
                            AUROC=math.nan,
                            AUPRC=math.nan,
                            seconds=elapsed,
                            status="failed",
                            error=str(exc),
                        )
                    )
                    print(f"  {method:32s} {dataset_name:12s} FAILED: {exc}")
    return results


def run_recap(
    data_train: list,
    data_test: list,
    dataset_names: list[str],
    train_config,
    model_config,
    args: argparse.Namespace,
) -> list[MethodResult]:
    results: list[MethodResult] = []

    print("\n" + "=" * 70)
    print("Full RECAP")
    print("=" * 70)
    for trial in range(args.trials):
        set_seed(trial)
        print(f"\n--- RECAP trial {trial} ---")
        trial_train_config = copy.deepcopy(train_config)
        trial_model_config = copy.deepcopy(model_config)

        trial_train_config.device = args.device
        trial_train_config.epochs = args.epochs
        trial_train_config.trials = args.trials
        trial_train_config.save_checkpoint = args.save_recap_checkpoints
        trial_train_config.output_dir = args.output_dir
        trial_train_config.log_diagnostics = not args.no_diagnostics
        trial_train_config.diagnostics_interval = args.diagnostics_interval

        data = {"train": data_train, "test": data_test}
        start = time.perf_counter()
        detector = recapDetector(trial_train_config, trial_model_config, data)
        detector.train(verbose=True)
        train_seconds = time.perf_counter() - start

        eval_start = time.perf_counter()
        scores = detector.evaluate(
            data_list=data_test,
            dataset_names=dataset_names,
            verbose=True,
        )
        eval_seconds = time.perf_counter() - eval_start
        per_dataset_seconds = eval_seconds / max(1, len(dataset_names))

        for dataset_name in dataset_names:
            metric = scores[dataset_name]
            results.append(
                MethodResult(
                    method="RECAP",
                    trial=trial,
                    dataset=dataset_name,
                    AUROC=float(metric["AUROC"]),
                    AUPRC=float(metric["AUPRC"]),
                    seconds=per_dataset_seconds,
                )
            )
        print(f"  RECAP trial {trial} train_seconds={train_seconds:.1f} eval_seconds={eval_seconds:.1f}")

    return results


def summarize(records: list[MethodResult]) -> list[dict]:
    grouped: dict[tuple[str, str], list[MethodResult]] = defaultdict(list)
    for record in records:
        grouped[(record.method, record.dataset)].append(record)

    rows = []
    for (method, dataset), items in sorted(grouped.items()):
        aurocs = np.asarray([r.AUROC for r in items], dtype=np.float64)
        auprcs = np.asarray([r.AUPRC for r in items], dtype=np.float64)
        seconds = np.asarray([r.seconds for r in items], dtype=np.float64)
        valid_aurocs = aurocs[~np.isnan(aurocs)]
        valid_auprcs = auprcs[~np.isnan(auprcs)]
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "n": int(len(valid_aurocs)),
                "AUROC_mean": float(np.mean(valid_aurocs)) if len(valid_aurocs) else math.nan,
                "AUROC_std": float(np.std(valid_aurocs)) if len(valid_aurocs) else math.nan,
                "AUPRC_mean": float(np.mean(valid_auprcs)) if len(valid_auprcs) else math.nan,
                "AUPRC_std": float(np.std(valid_auprcs)) if len(valid_auprcs) else math.nan,
                "seconds_mean": float(np.nanmean(seconds)),
            }
        )
    return rows


def summarize_average(records: list[MethodResult]) -> list[dict]:
    per_trial: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: {"AUROC": [], "AUPRC": []})
    for record in records:
        if not math.isnan(record.AUROC):
            per_trial[(record.method, record.trial)]["AUROC"].append(record.AUROC)
        if not math.isnan(record.AUPRC):
            per_trial[(record.method, record.trial)]["AUPRC"].append(record.AUPRC)

    by_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"AUROC": [], "AUPRC": []})
    for (method, _trial), values in per_trial.items():
        if values["AUROC"]:
            by_method[method]["AUROC"].append(float(np.mean(values["AUROC"])))
        if values["AUPRC"]:
            by_method[method]["AUPRC"].append(float(np.mean(values["AUPRC"])))

    rows = []
    for method, values in sorted(by_method.items()):
        aurocs = np.asarray(values["AUROC"], dtype=np.float64)
        auprcs = np.asarray(values["AUPRC"], dtype=np.float64)
        rows.append(
            {
                "method": method,
                "dataset": "Average",
                "n": int(len(aurocs)),
                "AUROC_mean": float(np.nanmean(aurocs)),
                "AUROC_std": float(np.nanstd(aurocs)),
                "AUPRC_mean": float(np.nanmean(auprcs)),
                "AUPRC_std": float(np.nanstd(auprcs)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_cell(row: dict, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}"


def write_markdown(path: Path, average_rows: list[dict], summary_rows: list[dict]) -> None:
    lines = []
    lines.append("# Simple Combine Ablation Results")
    lines.append("")
    lines.append("## Average Across Target Graphs")
    lines.append("")
    lines.append("| Method | AUROC | AUPRC | Seeds |")
    lines.append("|---|---:|---:|---:|")
    for row in average_rows:
        lines.append(
            f"| {row['method']} | {metric_cell(row, 'AUROC')} | "
            f"{metric_cell(row, 'AUPRC')} | {row['n']} |"
        )

    lines.append("")
    lines.append("## Per Dataset")
    lines.append("")
    lines.append("| Method | Dataset | AUROC | AUPRC | Seeds |")
    lines.append("|---|---|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['dataset']} | {metric_cell(row, 'AUROC')} | "
            f"{metric_cell(row, 'AUPRC')} | {row['n']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    model_config,
    records: list[MethodResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "simple_combine_raw.json"
    summary_path = output_dir / "simple_combine_summary.csv"
    average_path = output_dir / "simple_combine_average.csv"
    markdown_path = output_dir / "simple_combine_table.md"

    summary_rows = summarize(records)
    average_rows = summarize_average(records)

    payload = {
        "config": to_serializable_config(args, model_config),
        "records": [asdict(record) for record in records],
        "summary": summary_rows,
        "average": average_rows,
    }
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_csv(summary_path, summary_rows)
    write_csv(average_path, average_rows)
    write_markdown(markdown_path, average_rows, summary_rows)

    print("\n" + "=" * 70)
    print("Saved outputs")
    print("=" * 70)
    print(f"  Raw JSON : {raw_path}")
    print(f"  Summary  : {summary_path}")
    print(f"  Average  : {average_path}")
    print(f"  Markdown : {markdown_path}")


def main() -> None:
    args = parse_args()
    apply_quick_mode(args)
    ensure_device_available(args.device)

    os.chdir(PROJECT_ROOT)
    train_datasets = args.train_datasets or DEFAULT_TRAIN_DATASETS
    test_datasets = args.test_datasets or DEFAULT_TEST_DATASETS

    train_config, model_config = create_default_configs(
        model_name=args.model,
        json_dir=args.json_dir,
        dims=args.dims,
    )
    if args.num_clusters is not None:
        model_config.num_clusters = args.num_clusters
    if args.knn_k is not None:
        model_config.knn_k = args.knn_k
    args.knn_k = model_config.knn_k

    print("\n" + "=" * 70)
    print("Simple Combine Ablation")
    print("=" * 70)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Train graphs : {train_datasets}")
    print(f"Test graphs  : {test_datasets}")
    print(f"Trials       : {args.trials}")
    print(f"RECAP epochs : {args.epochs}")
    print(f"C / KNN k    : {model_config.num_clusters} / {model_config.knn_k}")

    # Classical target-specific baselines do not consume source graphs. Avoid
    # loading them in --no-recap mode so the matched 2x2 CPU run remains valid
    # under memory-constrained instances.
    prepared_train_datasets = [] if args.no_recap else train_datasets
    data_train, data_test = prepare_datasets(
        dims=args.dims,
        train_datasets=prepared_train_datasets,
        test_datasets=test_datasets,
        num_hops=model_config.num_hops,
    )

    records: list[MethodResult] = []
    if args.methods:
        records.extend(
            run_simple_methods(
                data_test=data_test,
                dataset_names=test_datasets,
                methods=args.methods,
                n_clusters=model_config.num_clusters,
                args=args,
            )
        )
    if not args.no_recap:
        records.extend(
            run_recap(
                data_train=data_train,
                data_test=data_test,
                dataset_names=test_datasets,
                train_config=train_config,
                model_config=model_config,
                args=args,
            )
        )

    write_outputs(Path(args.output_dir), args, model_config, records)


if __name__ == "__main__":
    main()
