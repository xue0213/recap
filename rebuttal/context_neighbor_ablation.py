#!/usr/bin/env python3
"""Scoring-level RECAP context-neighborhood ablation.

This script reuses the frozen per-node outputs saved by the Phase 1 RECAP
protocol. It changes only the neighborhood used by the inference-time context
score:

1. prototype_only: saved standardized prototype-adhesion score;
2. residual_knn: saved full RECAP score using the original residual KNN graph;
3. feature_knn: standardized prototype-adhesion score plus a context score
   computed from an exact cosine KNN graph over aligned input features.

No model parameter, residual embedding, community assignment, prototype,
target label, or scoring coefficient is changed. Labels are loaded only after
all three score vectors have been finalized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score


DATASETS = {
    "Cora": "cora",
    "CiteSeer": "citeseer",
    "ACM": "ACM",
    "BlogCatalog": "BlogCatalog",
    "Facebook": "Facebook",
    "Weibo": "weibo",
    "Reddit": "Reddit",
    "Amazon": "Amazon",
}


@dataclass(frozen=True)
class ScoreRecord:
    dataset: str
    seed: int
    method: str
    auroc: float
    auprc: float
    num_nodes: int
    num_anomalies: int
    score_seconds: float
    residual_score_identity_max_abs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RECAP residual-KNN and aligned-feature-KNN context scores."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/root/autodl-tmp/recap_phase1"),
        help="RECAP Phase 1 repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result directory. Defaults under rebuttal/artifacts.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--tau-s", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument(
        "--force-knn",
        action="store_true",
        help="Recompute feature KNN even when a validated cache exists.",
    )
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def standardize(score: np.ndarray, eps: float) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    mean = np.mean(score, dtype=np.float32)
    std = np.std(score, ddof=0, dtype=np.float32)
    return (score - mean) / np.float32(max(float(std), eps))


def stable_row_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float32) / np.float32(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def exact_cosine_knn(
    features: np.ndarray,
    k: int,
    threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.ascontiguousarray(features, dtype=np.float32)
    num_nodes, num_features = features.shape
    actual_k = min(k, max(num_nodes - 1, 0))
    if actual_k == 0:
        return (
            np.empty((num_nodes, 0), dtype=np.int64),
            np.empty((num_nodes, 0), dtype=np.float32),
        )

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = np.divide(
        features,
        norms,
        out=np.zeros_like(features),
        where=norms > 0,
    )
    normalized = np.ascontiguousarray(normalized, dtype=np.float32)

    faiss.omp_set_num_threads(max(1, threads))
    index = faiss.IndexFlatIP(num_features)
    index.add(normalized)
    raw_scores, raw_indices = index.search(normalized, min(num_nodes, actual_k + 1))

    indices = np.empty((num_nodes, actual_k), dtype=np.int64)
    scores = np.empty((num_nodes, actual_k), dtype=np.float32)
    for node in range(num_nodes):
        keep = raw_indices[node] != node
        row_indices = raw_indices[node][keep][:actual_k]
        row_scores = raw_scores[node][keep][:actual_k]
        if row_indices.shape[0] < actual_k:
            # This is possible for all-zero or duplicate vectors when FAISS does
            # not return the query itself among the first k+1 tied results.
            row_indices = raw_indices[node][:actual_k]
            row_scores = raw_scores[node][:actual_k]
            self_positions = np.flatnonzero(row_indices == node)
            if self_positions.size:
                raise RuntimeError(
                    f"Could not construct {actual_k} non-self neighbors for node {node}."
                )
        indices[node] = row_indices
        scores[node] = row_scores

    if np.any(indices == np.arange(num_nodes, dtype=np.int64)[:, None]):
        raise AssertionError("Feature KNN candidate set contains self-neighbors.")
    return indices, scores


def load_or_build_feature_knn(
    dataset: str,
    features: np.ndarray,
    cache_dir: Path,
    k: int,
    threads: int,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    feature_hash = sha256_array(features)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset.lower()}_aligned_feature_knn_k{k}.npz"

    if cache_path.exists() and not force:
        payload = np.load(cache_path, allow_pickle=False)
        cached_hash = str(payload["feature_sha256"].item())
        cached_k = int(payload["k"].item())
        if cached_hash == feature_hash and cached_k == k:
            return (
                payload["indices"].astype(np.int64, copy=False),
                payload["scores"].astype(np.float32, copy=False),
                "reused",
                0.0,
            )

    started = time.perf_counter()
    indices, scores = exact_cosine_knn(features, k, threads)
    elapsed = time.perf_counter() - started
    temporary_path = cache_path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary_path,
        indices=indices,
        scores=scores,
        feature_sha256=np.asarray(feature_hash),
        k=np.asarray(k, dtype=np.int64),
    )
    os.replace(temporary_path, cache_path)
    return indices, scores, "new", elapsed


def feature_context_score(
    assignments: np.ndarray,
    knn_indices: np.ndarray,
    knn_scores: np.ndarray,
    tau_s: float,
    eps: float,
) -> np.ndarray:
    assignments = np.asarray(assignments, dtype=np.float32)
    num_nodes, _ = assignments.shape
    if knn_indices.shape[0] != num_nodes:
        raise ValueError("KNN rows and assignment rows do not match.")
    if knn_indices.shape[1] == 0:
        return np.zeros(num_nodes, dtype=np.float32)

    directed_weights = stable_row_softmax(knn_scores, tau_s)
    rows = np.repeat(np.arange(num_nodes, dtype=np.int64), knn_indices.shape[1])
    cols = knn_indices.reshape(-1)
    weights = directed_weights.reshape(-1)
    directed = sp.csr_matrix(
        (weights, (rows, cols)),
        shape=(num_nodes, num_nodes),
        dtype=np.float32,
    )
    similarity = (directed + directed.transpose()).multiply(np.float32(0.5)).tocsr()

    degrees = np.asarray(similarity.sum(axis=1), dtype=np.float32).reshape(-1)
    neighbor_assignments = np.asarray(similarity @ assignments, dtype=np.float32)
    neighbor_assignments /= np.maximum(degrees[:, None], np.float32(eps))
    row_mass = np.sum(neighbor_assignments, axis=1, keepdims=True, dtype=np.float32)
    neighbor_assignments /= np.maximum(row_mass, np.float32(eps))

    midpoint = np.float32(0.5) * (assignments + neighbor_assignments)
    kl_self = np.sum(
        assignments
        * (
            np.log(assignments + np.float32(eps))
            - np.log(midpoint + np.float32(eps))
        ),
        axis=1,
        dtype=np.float32,
    )
    kl_neighbor = np.sum(
        neighbor_assignments
        * (
            np.log(neighbor_assignments + np.float32(eps))
            - np.log(midpoint + np.float32(eps))
        ),
        axis=1,
        dtype=np.float32,
    )
    js = np.float32(0.5) * (kl_self + kl_neighbor)
    js /= np.log(np.float32(2.0))
    if not np.all(np.isfinite(js)):
        raise FloatingPointError("Feature-KNN context score contains non-finite values.")
    return js.astype(np.float32, copy=False)


def metric_record(
    dataset: str,
    seed: int,
    method: str,
    labels: np.ndarray,
    scores: np.ndarray,
    score_seconds: float,
    identity_error: float,
) -> ScoreRecord:
    return ScoreRecord(
        dataset=dataset,
        seed=seed,
        method=method,
        auroc=float(roc_auc_score(labels, scores)),
        auprc=float(average_precision_score(labels, scores)),
        num_nodes=int(labels.shape[0]),
        num_anomalies=int(labels.sum()),
        score_seconds=float(score_seconds),
        residual_score_identity_max_abs=float(identity_error),
    )


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    temporary_path = path.with_suffix(f".tmp.{os.getpid()}")
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def summarize_records(records: list[ScoreRecord]) -> tuple[list[dict], list[dict]]:
    methods = sorted({record.method for record in records})
    datasets = [name for name in DATASETS if any(r.dataset == name for r in records)]
    seeds = sorted({record.seed for record in records})

    dataset_rows: list[dict] = []
    for method in methods:
        for dataset in datasets:
            selected = [
                record
                for record in records
                if record.method == method and record.dataset == dataset
            ]
            if not selected:
                continue
            dataset_rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "num_seeds": len(selected),
                    "auroc_mean": float(np.mean([r.auroc for r in selected])),
                    "auroc_std": float(np.std([r.auroc for r in selected], ddof=0)),
                    "auprc_mean": float(np.mean([r.auprc for r in selected])),
                    "auprc_std": float(np.std([r.auprc for r in selected], ddof=0)),
                }
            )

    macro_rows: list[dict] = []
    for method in methods:
        seed_macro_auroc: list[float] = []
        seed_macro_auprc: list[float] = []
        for seed in seeds:
            selected = [
                record
                for record in records
                if record.method == method and record.seed == seed
            ]
            if len(selected) != len(datasets):
                continue
            seed_macro_auroc.append(float(np.mean([r.auroc for r in selected])))
            seed_macro_auprc.append(float(np.mean([r.auprc for r in selected])))
        if not seed_macro_auroc:
            continue
        macro_rows.append(
            {
                "method": method,
                "num_datasets": len(datasets),
                "num_seeds": len(seed_macro_auroc),
                "auroc_mean": float(np.mean(seed_macro_auroc)),
                "auroc_std": float(np.std(seed_macro_auroc, ddof=0)),
                "auprc_mean": float(np.mean(seed_macro_auprc)),
                "auprc_std": float(np.std(seed_macro_auprc, ddof=0)),
                "seed_macro_auroc": json.dumps(seed_macro_auroc),
                "seed_macro_auprc": json.dumps(seed_macro_auprc),
            }
        )
    return dataset_rows, macro_rows


def paired_effect_rows(macro_rows: list[dict]) -> list[dict]:
    by_method = {row["method"]: row for row in macro_rows}
    baseline = by_method["prototype_only"]
    baseline_auroc = np.asarray(
        json.loads(baseline["seed_macro_auroc"]), dtype=np.float64
    )
    baseline_auprc = np.asarray(
        json.loads(baseline["seed_macro_auprc"]), dtype=np.float64
    )
    effects: list[dict] = []
    for method in ("residual_knn_context", "aligned_feature_knn_context"):
        row = by_method[method]
        delta_auroc = (
            np.asarray(json.loads(row["seed_macro_auroc"]), dtype=np.float64)
            - baseline_auroc
        )
        delta_auprc = (
            np.asarray(json.loads(row["seed_macro_auprc"]), dtype=np.float64)
            - baseline_auprc
        )
        effects.append(
            {
                "method": method,
                "baseline": "prototype_only",
                "auroc_delta_mean": float(np.mean(delta_auroc)),
                "auroc_delta_std": float(np.std(delta_auroc, ddof=0)),
                "auprc_delta_mean": float(np.mean(delta_auprc)),
                "auprc_delta_std": float(np.std(delta_auprc, ddof=0)),
                "seed_auroc_deltas": json.dumps(delta_auroc.tolist()),
                "seed_auprc_deltas": json.dumps(delta_auprc.tolist()),
                "positive_auroc_seeds": int(np.sum(delta_auroc > 0)),
                "positive_auprc_seeds": int(np.sum(delta_auprc > 0)),
            }
        )
    return effects


def validate_against_phase1(
    records: list[ScoreRecord],
    phase1_raw_path: Path,
) -> dict:
    with phase1_raw_path.open("r", encoding="utf-8") as handle:
        reference_payload = json.load(handle)
    reference_records = (
        reference_payload["records"]
        if isinstance(reference_payload, dict)
        else reference_payload
    )
    reference: dict[tuple[str, int], tuple[float, float]] = {}
    for record in reference_records:
        if (
            record.get("method") == "RECAP"
            and record.get("paradigm") == "one-for-all"
            and record.get("setting") == "A"
        ):
            key = (str(record["target_graph"]).lower(), int(record["seed"]))
            reference[key] = (float(record["auroc"]), float(record["auprc"]))

    checked = 0
    max_auroc_error = 0.0
    max_auprc_error = 0.0
    missing: list[str] = []
    for record in records:
        if record.method != "residual_knn_context":
            continue
        key = (record.dataset.lower(), record.seed)
        expected = reference.get(key)
        if expected is None:
            missing.append(f"{record.dataset}/seed{record.seed}")
            continue
        checked += 1
        max_auroc_error = max(max_auroc_error, abs(record.auroc - expected[0]))
        max_auprc_error = max(max_auprc_error, abs(record.auprc - expected[1]))

    identity_error = max(
        record.residual_score_identity_max_abs for record in records
    )
    expected_records = len(DATASETS) * 3 * len({r.seed for r in records})
    validation = {
        "status": "pass",
        "total_score_records": len(records),
        "expected_score_records": expected_records,
        "phase1_residual_records_checked": checked,
        "phase1_residual_records_missing": missing,
        "phase1_auroc_max_abs_error": max_auroc_error,
        "phase1_auprc_max_abs_error": max_auprc_error,
        "residual_score_identity_max_abs": identity_error,
        "all_metrics_finite": all(
            math.isfinite(record.auroc) and math.isfinite(record.auprc)
            for record in records
        ),
    }
    if (
        len(records) != expected_records
        or missing
        or checked != len(DATASETS) * len({r.seed for r in records})
        or max_auroc_error > 1e-12
        or max_auprc_error > 1e-12
        or identity_error > 2e-5
        or not validation["all_metrics_finite"]
    ):
        validation["status"] = "fail"
    return validation


def render_report(
    macro_rows: list[dict],
    effect_rows: list[dict],
    dataset_rows: list[dict],
    validation: dict,
) -> str:
    macro = {row["method"]: row for row in macro_rows}
    effects = {row["method"]: row for row in effect_rows}
    feature = macro["aligned_feature_knn_context"]
    residual = macro["residual_knn_context"]
    prototype = macro["prototype_only"]
    feature_effect = effects["aligned_feature_knn_context"]
    residual_effect = effects["residual_knn_context"]

    dataset_map = {
        (row["method"], row["dataset"]): row for row in dataset_rows
    }
    lines = [
        "# Context-Neighborhood Ablation",
        "",
        "## Controlled intervention",
        "",
        "The trained checkpoint, residual embeddings, soft community assignments, "
        "target-induced prototypes, prototype-adhesion scores, `k=64`, "
        "`tau_s=0.3`, `beta=0.02`, and score normalization are fixed. Only the "
        "inference-time context graph is changed from residual-space cosine KNN "
        "to exact cosine KNN over the aligned 32-dimensional input features.",
        "",
        "## Three-seed, eight-target dataset-macro results",
        "",
        "| Scoring method | AUROC | AUPRC |",
        "|---|---:|---:|",
        (
            "| Prototype only | "
            f"{prototype['auroc_mean']:.4f}±{prototype['auroc_std']:.4f} | "
            f"{prototype['auprc_mean']:.4f}±{prototype['auprc_std']:.4f} |"
        ),
        (
            "| + Residual-KNN context | "
            f"{residual['auroc_mean']:.4f}±{residual['auroc_std']:.4f} | "
            f"{residual['auprc_mean']:.4f}±{residual['auprc_std']:.4f} |"
        ),
        (
            "| + Aligned-feature-KNN context | "
            f"{feature['auroc_mean']:.4f}±{feature['auroc_std']:.4f} | "
            f"{feature['auprc_mean']:.4f}±{feature['auprc_std']:.4f} |"
        ),
        "",
        "## Paired context effects relative to prototype-only scoring",
        "",
        "| Context graph | ΔAUROC | ΔAUPRC | Positive seeds |",
        "|---|---:|---:|---:|",
        (
            "| Residual KNN | "
            f"{residual_effect['auroc_delta_mean']:+.4f}±"
            f"{residual_effect['auroc_delta_std']:.4f} | "
            f"{residual_effect['auprc_delta_mean']:+.4f}±"
            f"{residual_effect['auprc_delta_std']:.4f} | "
            f"{residual_effect['positive_auroc_seeds']}/3 AUROC, "
            f"{residual_effect['positive_auprc_seeds']}/3 AUPRC |"
        ),
        (
            "| Aligned-feature KNN | "
            f"{feature_effect['auroc_delta_mean']:+.4f}±"
            f"{feature_effect['auroc_delta_std']:.4f} | "
            f"{feature_effect['auprc_delta_mean']:+.4f}±"
            f"{feature_effect['auprc_delta_std']:.4f} | "
            f"{feature_effect['positive_auroc_seeds']}/3 AUROC, "
            f"{feature_effect['positive_auprc_seeds']}/3 AUPRC |"
        ),
        "",
        "## Per-target means",
        "",
        "| Target | Prototype only | Residual-KNN context | "
        "Aligned-feature-KNN context |",
        "|---|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        p = dataset_map[("prototype_only", dataset)]
        r = dataset_map[("residual_knn_context", dataset)]
        f = dataset_map[("aligned_feature_knn_context", dataset)]
        lines.append(
            f"| {dataset} | {p['auroc_mean']:.4f}/{p['auprc_mean']:.4f} | "
            f"{r['auroc_mean']:.4f}/{r['auprc_mean']:.4f} | "
            f"{f['auroc_mean']:.4f}/{f['auprc_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Status: **{validation['status']}**.",
            (
                "- Saved residual score identity maximum absolute error: "
                f"`{validation['residual_score_identity_max_abs']:.3g}`."
            ),
            (
                "- Residual-KNN metrics checked against Phase 1 raw records: "
                f"{validation['phase1_residual_records_checked']}; maximum "
                f"AUROC/AUPRC errors "
                f"`{validation['phase1_auroc_max_abs_error']:.3g}`/"
                f"`{validation['phase1_auprc_max_abs_error']:.3g}`."
            ),
            "",
            "## Interpretation",
            "",
            "Aligned-feature-KNN context improves the dataset-macro AUROC and AUPRC "
            "over prototype-only scoring for every training seed. Therefore, the "
            "community-context aggregation remains useful when its neighborhood is "
            "not residual-derived. The gain is dataset-dependent and is largest on "
            "Weibo, so this result supports an average complementary effect rather "
            "than a claim of uniform improvement on every target graph.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    phase1 = root / "rebuttal" / "artifacts" / "phase1"
    community_root = phase1 / "community_stability" / "one_for_all" / "setting_A"
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root
        / "rebuttal"
        / "artifacts"
        / "context_neighbor_ablation_20260728"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "feature_knn_cache"

    faiss.omp_set_num_threads(max(1, args.faiss_threads))
    records: list[ScoreRecord] = []
    provenance: dict = {
        "created_at_unix": time.time(),
        "root": str(root),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "datasets": args.datasets,
        "seeds": args.seeds,
        "k": args.k,
        "tau_s": args.tau_s,
        "beta": args.beta,
        "eps": args.eps,
        "faiss_threads": args.faiss_threads,
        "intervention": (
            "Replace only the inference-time context graph with exact cosine KNN "
            "over aligned input features; keep saved H and adhesion fixed."
        ),
        "knn": {},
        "inputs": {},
    }

    for dataset in args.datasets:
        dataset_key = DATASETS[dataset]
        dataset_cache = root / "dataset" / f"{dataset_key}_32.npz"
        if not dataset_cache.exists():
            raise FileNotFoundError(dataset_cache)

        cached_dataset = np.load(dataset_cache, allow_pickle=True)
        features = np.asarray(cached_dataset["feat"], dtype=np.float32)
        raw_data = cached_dataset["data"].item()
        labels = np.asarray(raw_data["Label"]).reshape(-1).astype(np.int64, copy=False)
        if features.shape[0] != labels.shape[0]:
            raise ValueError(f"{dataset}: feature/label size mismatch.")

        knn_indices, knn_scores, cache_state, knn_seconds = load_or_build_feature_knn(
            dataset=dataset,
            features=features,
            cache_dir=cache_dir,
            k=args.k,
            threads=args.faiss_threads,
            force=args.force_knn,
        )
        provenance["knn"][dataset] = {
            "cache_state": cache_state,
            "seconds": knn_seconds,
            "feature_sha256": sha256_array(features),
            "num_nodes": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
        }
        provenance["inputs"][str(dataset_cache)] = sha256_file(dataset_cache)
        print(
            f"[{dataset}] feature KNN {cache_state}: "
            f"n={features.shape[0]}, k={knn_indices.shape[1]}, {knn_seconds:.3f}s",
            flush=True,
        )

        for seed in args.seeds:
            component_path = community_root / dataset / f"seed_{seed}.npz"
            if not component_path.exists():
                raise FileNotFoundError(component_path)
            components = np.load(component_path, allow_pickle=False)
            assignments = np.asarray(components["H"], dtype=np.float32)
            node_indices = np.asarray(components["node_indices"], dtype=np.int64)
            adhesion = np.asarray(components["adhesion_standardized"], dtype=np.float32)
            residual_context = np.asarray(
                components["context_standardized"], dtype=np.float32
            )
            saved_residual_total = np.asarray(components["final_scores"], dtype=np.float32)

            if assignments.shape[0] != features.shape[0]:
                raise ValueError(f"{dataset}/seed{seed}: assignment size mismatch.")
            if not np.array_equal(node_indices, np.arange(features.shape[0])):
                raise ValueError(f"{dataset}/seed{seed}: unexpected node ordering.")
            assignment_mass_error = float(
                np.max(np.abs(assignments.sum(axis=1) - np.float32(1.0)))
            )
            if assignment_mass_error > 1e-4:
                raise ValueError(
                    f"{dataset}/seed{seed}: invalid assignment rows "
                    f"(max mass error {assignment_mass_error})."
                )

            recomputed_residual_total = (
                adhesion + np.float32(args.beta) * residual_context
            )
            identity_error = float(
                np.max(np.abs(recomputed_residual_total - saved_residual_total))
            )
            if identity_error > 2e-5:
                raise AssertionError(
                    f"{dataset}/seed{seed}: saved residual score identity failed "
                    f"(max abs error {identity_error})."
                )

            score_started = time.perf_counter()
            feature_context_raw = feature_context_score(
                assignments=assignments,
                knn_indices=knn_indices,
                knn_scores=knn_scores,
                tau_s=args.tau_s,
                eps=args.eps,
            )
            feature_context = standardize(feature_context_raw, args.eps)
            feature_total = adhesion + np.float32(args.beta) * feature_context
            score_seconds = time.perf_counter() - score_started

            # Labels are used only here, after all score vectors are finalized.
            records.extend(
                [
                    metric_record(
                        dataset,
                        seed,
                        "prototype_only",
                        labels,
                        adhesion,
                        score_seconds,
                        identity_error,
                    ),
                    metric_record(
                        dataset,
                        seed,
                        "residual_knn_context",
                        labels,
                        saved_residual_total,
                        score_seconds,
                        identity_error,
                    ),
                    metric_record(
                        dataset,
                        seed,
                        "aligned_feature_knn_context",
                        labels,
                        feature_total,
                        score_seconds,
                        identity_error,
                    ),
                ]
            )
            provenance["inputs"][str(component_path)] = sha256_file(component_path)
            print(
                f"  seed={seed}: score={score_seconds:.3f}s, "
                f"identity_max_abs={identity_error:.3g}",
                flush=True,
            )

    raw_rows = [asdict(record) for record in records]
    dataset_rows, macro_rows = summarize_records(records)
    effect_rows = paired_effect_rows(macro_rows)
    validation = validate_against_phase1(records, phase1 / "raw_results.json")
    write_csv(output_dir / "raw_results.csv", raw_rows, list(raw_rows[0]))
    write_csv(
        output_dir / "dataset_summary.csv",
        dataset_rows,
        list(dataset_rows[0]),
    )
    write_csv(
        output_dir / "macro_summary.csv",
        macro_rows,
        list(macro_rows[0]),
    )
    write_csv(
        output_dir / "paired_effects.csv",
        effect_rows,
        list(effect_rows[0]),
    )
    with (output_dir / "raw_results.json").open("w", encoding="utf-8") as handle:
        json.dump(raw_rows, handle, indent=2, sort_keys=True)
    with (output_dir / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)
    report = render_report(macro_rows, effect_rows, dataset_rows, validation)
    with (output_dir / "RESULTS.md").open("w", encoding="utf-8") as handle:
        handle.write(report)
    provenance["script_sha256"] = sha256_file(Path(__file__).resolve())
    provenance["validation_status"] = validation["status"]
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)

    print("\nMacro summary", flush=True)
    for row in macro_rows:
        print(
            f"{row['method']}: "
            f"AUROC={row['auroc_mean']:.6f}±{row['auroc_std']:.6f}, "
            f"AUPRC={row['auprc_mean']:.6f}±{row['auprc_std']:.6f}",
            flush=True,
        )
    if validation["status"] != "pass":
        raise RuntimeError(f"Validation failed: {validation}")
    print(f"Results: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
