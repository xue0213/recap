"""Independent audit and compact report for large-target inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from rebuttal.large_target_inference.common import (
    atomic_csv,
    atomic_json,
    sha256_file,
    stable_hash,
    utc_now,
)
from rebuttal.large_target_inference.data import canonical_paths
from rebuttal.large_target_inference.protocol import (
    CHECKPOINTS,
    DATASETS,
    MODEL_LOCK,
    TARGETS,
    build_manifest,
)


HASH_CHUNK = 8 * 1024 * 1024
ARRAY_CHUNK = 500_000
SOURCE_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = SOURCE_DIR.parent / "LARGE_TARGET_INFERENCE_PROTOCOL.md"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _finite(array: np.ndarray) -> bool:
    for start in range(0, len(array), ARRAY_CHUNK):
        if not np.all(np.isfinite(array[start : start + ARRAY_CHUNK])):
            return False
    return True


def _phase_seconds(records: list[dict], names: set[str] | None = None) -> float:
    return float(
        sum(
            float(item["seconds"])
            for item in records
            if names is None or item["phase"] in names
        )
    )


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024.0**3)


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _fmt_mean_std(mean: float, std: float, scale: float = 1.0) -> str:
    return f"{mean * scale:.4f} ± {std * scale:.4f}"


def _verify_component_hashes(
    *,
    preflight: dict,
    dataset_root: Path,
    checkpoint_root: Path,
) -> dict:
    records: dict[str, Any] = {"datasets": {}, "checkpoints": {}}
    for target in TARGETS:
        paths = canonical_paths(dataset_root, target, MODEL_LOCK["dims"])
        expected_hashes = preflight["datasets"][target]["component_hashes"]
        observed: dict[str, str] = {}
        for component, expected_hash in expected_hashes.items():
            path = paths[component]
            actual_hash = sha256_file(path, chunk_size=HASH_CHUNK)
            if actual_hash != expected_hash:
                raise AssertionError(
                    f"{target}/{component}: data hash mismatch"
                )
            observed[component] = actual_hash
        records["datasets"][target] = observed
    for seed, checkpoint in CHECKPOINTS.items():
        path = checkpoint_root / checkpoint["relative_path"]
        observed_hash = sha256_file(path, chunk_size=HASH_CHUNK)
        if observed_hash != checkpoint["sha256"]:
            raise AssertionError(f"seed {seed}: checkpoint hash mismatch")
        if (
            preflight["checkpoints"][str(seed)]["sha256"]
            != observed_hash
        ):
            raise AssertionError(f"seed {seed}: preflight hash mismatch")
        records["checkpoints"][str(seed)] = observed_hash
    return records


def _audit_label_events(
    *,
    path: Path,
    primary_route: str,
    result: dict,
) -> dict:
    events = _read_jsonl(path)
    expected_routes = (
        {"exact", "faiss_ivfpq"}
        if result["target"] == "tfinance"
        else {primary_route}
    )
    if not events or events[-1].get("event") != "labels_unlocked":
        raise AssertionError(f"{result['run_id']}: missing final label unlock")
    frozen = [event for event in events if event.get("event") == "scores_frozen"]
    unlocks = [event for event in events if event.get("event") == "labels_unlocked"]
    if len(unlocks) != 1 or {event["route"] for event in frozen} != expected_routes:
        raise AssertionError(f"{result['run_id']}: invalid label audit routes")
    if events.index(unlocks[0]) <= max(events.index(event) for event in frozen):
        raise AssertionError(f"{result['run_id']}: labels unlocked too early")
    for event in frozen:
        score_path = path.parent / event["route"] / "scores.npy"
        if sha256_file(score_path) != event["scores_sha256"]:
            raise AssertionError(f"{result['run_id']}: frozen score hash mismatch")
        if event["mask_sha256"] != result["query_mask_sha256"]:
            raise AssertionError(f"{result['run_id']}: frozen mask hash mismatch")
    if set(unlocks[0]["required_routes"]) != expected_routes:
        raise AssertionError(f"{result['run_id']}: unlock declaration mismatch")
    return {
        "event_count": len(events),
        "frozen_routes": sorted(expected_routes),
        "labels_unlocked_after_all_scores": True,
    }


def _audit_run(
    *,
    spec,
    output_root: Path,
    dataset_root: Path,
) -> tuple[dict, dict]:
    run_dir = output_root / "runs" / spec.run_id
    if not (run_dir / "complete.json").exists():
        raise AssertionError(f"{spec.run_id}: missing completion marker")
    result = _read_json(run_dir / "result.json")
    if result["run_id"] != spec.run_id:
        raise AssertionError(f"{spec.run_id}: result ID mismatch")
    if result["checkpoint_sha256"] != spec.checkpoint_sha256:
        raise AssertionError(f"{spec.run_id}: checkpoint declaration mismatch")
    if result["primary_knn"] != spec.primary_knn:
        raise AssertionError(f"{spec.run_id}: primary route mismatch")

    score_path = run_dir / spec.primary_knn / "scores.npy"
    score_hash = sha256_file(score_path, chunk_size=HASH_CHUNK)
    if score_hash != result["primary_score_sha256"]:
        raise AssertionError(f"{spec.run_id}: primary score hash mismatch")
    scores = np.load(score_path, mmap_mode="r")
    if scores.shape != (DATASETS[spec.target]["nodes"],) or not _finite(scores):
        raise AssertionError(f"{spec.run_id}: invalid primary score vector")

    mask_path = run_dir / "query_mask.npy"
    mask_hash = sha256_file(mask_path, chunk_size=HASH_CHUNK)
    if mask_hash != result["query_mask_sha256"]:
        raise AssertionError(f"{spec.run_id}: query mask hash mismatch")
    mask = np.load(mask_path, mmap_mode="r")
    if mask.shape != scores.shape or int(mask.sum()) != DATASETS[spec.target][
        "evaluation_nodes"
    ]:
        raise AssertionError(f"{spec.run_id}: invalid evaluation mask")

    label_audit = _audit_label_events(
        path=run_dir / "label_audit.jsonl",
        primary_route=spec.primary_knn,
        result=result,
    )
    labels_path = canonical_paths(
        dataset_root, spec.target, MODEL_LOCK["dims"]
    )["labels"]
    labels = np.load(labels_path, mmap_mode="r")
    query = np.asarray(mask, dtype=np.bool_)
    recomputed = {
        "auroc": float(roc_auc_score(labels[query], scores[query])),
        "auprc": float(average_precision_score(labels[query], scores[query])),
    }
    stored = result["primary_metrics"]
    differences = {
        key: abs(recomputed[key] - float(stored[key]))
        for key in ("auroc", "auprc")
    }
    if max(differences.values()) > 1e-12:
        raise AssertionError(f"{spec.run_id}: metric recomputation mismatch")

    phases = result["phase_records"]
    primary_names = {
        "checkpoint_load",
        "model_residual_forward",
        f"score_{spec.primary_knn}",
        f"serialize_{spec.primary_knn}",
    }
    primary_seconds = _phase_seconds(phases, primary_names)
    total_seconds = _phase_seconds(phases)
    peak_gpu = max(
        int(item["gpu_peak_allocated_bytes"]) for item in phases
    )
    peak_reserved = max(
        int(item["gpu_peak_reserved_bytes"]) for item in phases
    )
    peak_rss = max(int(item["rss_peak_bytes"]) for item in phases)
    row = {
        "run_id": spec.run_id,
        "target": spec.target,
        "seed": spec.seed,
        "primary_knn": spec.primary_knn,
        "nodes_scored": DATASETS[spec.target]["nodes"],
        "evaluation_nodes": int(stored["evaluation_nodes"]),
        "anomalies": int(stored["anomalies"]),
        "auroc": recomputed["auroc"],
        "auprc": recomputed["auprc"],
        "primary_inference_seconds": primary_seconds,
        "total_run_phase_seconds": total_seconds,
        "node_throughput_per_second": (
            DATASETS[spec.target]["nodes"] / primary_seconds
        ),
        "peak_gpu_allocated_gib": _bytes_to_gib(peak_gpu),
        "peak_gpu_reserved_gib": _bytes_to_gib(peak_reserved),
        "peak_rss_gib": _bytes_to_gib(peak_rss),
        "effective_communities": float(result["effective_communities"]),
        "score_sha256": score_hash,
        "mask_sha256": mask_hash,
    }
    audit = {
        "run_id": spec.run_id,
        "status": "pass",
        "score_shape": list(scores.shape),
        "score_finite": True,
        "score_sha256": score_hash,
        "mask_sha256": mask_hash,
        "label_audit": label_audit,
        "recomputed_metrics": recomputed,
        "stored_metric_absolute_differences": differences,
    }
    return row, audit


def _audit_shared(output_root: Path) -> tuple[dict, list[dict]]:
    summaries: dict[str, dict] = {}
    audits: list[dict] = []
    for target in TARGETS:
        shared_dir = output_root / "shared" / target
        if not (shared_dir / "complete.json").exists():
            raise AssertionError(f"{target}: missing shared completion marker")
        setup = _read_json(shared_dir / "shared_setup.json")
        cache_bytes = 0
        candidate_audit: dict[str, dict] = {}
        for route, metadata in setup["candidate_metadata"].items():
            path = Path(metadata["path"])
            observed = sha256_file(path, chunk_size=HASH_CHUNK)
            if observed != metadata["sha256"]:
                raise AssertionError(f"{target}/{route}: candidate hash mismatch")
            candidates = np.load(path, mmap_mode="r")
            expected_shape = (
                DATASETS[target]["nodes"],
                MODEL_LOCK["knn_k"],
            )
            if candidates.shape != expected_shape or candidates.dtype != np.int32:
                raise AssertionError(
                    f"{target}/{route}: candidate shape/dtype mismatch"
                )
            invalid_ids = 0
            self_neighbor_rows = 0
            duplicate_rows = 0
            for start in range(0, len(candidates), 50_000):
                stop = min(start + 50_000, len(candidates))
                block = np.asarray(candidates[start:stop])
                invalid_ids += int(
                    np.count_nonzero(
                        (block < 0) | (block >= DATASETS[target]["nodes"])
                    )
                )
                row_ids = np.arange(start, stop, dtype=np.int64)[:, None]
                self_neighbor_rows += int(
                    np.count_nonzero(np.any(block == row_ids, axis=1))
                )
                ordered = np.sort(block, axis=1)
                duplicate_rows += int(
                    np.count_nonzero(
                        np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)
                    )
                )
            if invalid_ids or self_neighbor_rows or duplicate_rows:
                raise AssertionError(f"{target}/{route}: invalid candidates")
            size = path.stat().st_size
            cache_bytes += size
            candidate_audit[route] = {
                "sha256": observed,
                "bytes": size,
                "shape": metadata["shape"],
                "invalid_ids": invalid_ids,
                "self_neighbor_rows": self_neighbor_rows,
                "duplicate_rows": duplicate_rows,
            }
        records = setup["phase_records"]
        setup_seconds = _phase_seconds(records)
        ann_metadata = setup["candidate_metadata"]["faiss_ivfpq"]
        summaries[target] = {
            "setup_seconds": setup_seconds,
            "setup_peak_gpu_allocated_gib": _bytes_to_gib(
                max(int(item["gpu_peak_allocated_bytes"]) for item in records)
            ),
            "setup_peak_gpu_reserved_gib": _bytes_to_gib(
                max(int(item["gpu_peak_reserved_bytes"]) for item in records)
            ),
            "setup_peak_rss_gib": _bytes_to_gib(
                max(int(item["rss_peak_bytes"]) for item in records)
            ),
            "candidate_cache_gib": _bytes_to_gib(cache_bytes),
            "ann_fidelity": setup["ann_fidelity"],
            "ann_nlist": int(ann_metadata["nlist"]),
            "ann_nprobe": int(ann_metadata["nprobe"]),
            "ann_search_k": int(ann_metadata["search_k"]),
        }
        audits.append(
            {
                "target": target,
                "status": "pass",
                "candidate_files": candidate_audit,
                "ann_fidelity": setup["ann_fidelity"],
            }
        )
    return summaries, audits


def _paired_tfinance_fidelity(output_root: Path) -> dict:
    records: list[dict] = []
    for seed in CHECKPOINTS:
        run_dir = output_root / "runs" / f"ofa_a__seed{seed}__tfinance"
        result = _read_json(run_dir / "result.json")
        exact = np.load(run_dir / "exact" / "scores.npy", mmap_mode="r")
        approximate = np.load(
            run_dir / "faiss_ivfpq" / "scores.npy", mmap_mode="r"
        )
        observed_spearman = float(spearmanr(exact, approximate).statistic)
        stored = result["tfinance_ann_fidelity"]
        if abs(observed_spearman - stored["score_spearman"]) > 1e-12:
            raise AssertionError(f"T-Finance seed {seed}: fidelity mismatch")
        records.append({"seed": seed, **stored})
    return {
        "per_seed": records,
        "score_spearman_mean": float(
            np.mean([item["score_spearman"] for item in records])
        ),
        "absolute_auroc_difference_mean": float(
            np.mean(
                [
                    abs(item["auroc_difference_ann_minus_exact"])
                    for item in records
                ]
            )
        ),
        "absolute_auprc_difference_mean": float(
            np.mean(
                [
                    abs(item["auprc_difference_ann_minus_exact"])
                    for item in records
                ]
            )
        ),
    }


def _aggregate(
    rows: list[dict], shared: dict[str, dict]
) -> list[dict]:
    output: list[dict] = []
    for target in TARGETS:
        subset = [row for row in rows if row["target"] == target]
        auroc_mean, auroc_std = _mean_std([row["auroc"] for row in subset])
        auprc_mean, auprc_std = _mean_std([row["auprc"] for row in subset])
        inference_mean, inference_std = _mean_std(
            [row["primary_inference_seconds"] for row in subset]
        )
        throughput_mean, throughput_std = _mean_std(
            [row["node_throughput_per_second"] for row in subset]
        )
        setup_seconds = shared[target]["setup_seconds"]
        output.append(
            {
                "target": target,
                "display": DATASETS[target]["display"],
                "nodes": DATASETS[target]["nodes"],
                "adjacency_nnz": DATASETS[target]["adjacency_nnz"],
                "evaluation_nodes": DATASETS[target]["evaluation_nodes"],
                "anomaly_prevalence": (
                    DATASETS[target]["anomalies"]
                    / DATASETS[target]["evaluation_nodes"]
                ),
                "primary_knn": DATASETS[target]["primary_knn"],
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
                "setup_seconds": setup_seconds,
                "primary_inference_seconds_mean": inference_mean,
                "primary_inference_seconds_std": inference_std,
                "cold_latency_seconds_mean": setup_seconds + inference_mean,
                "node_throughput_mean": throughput_mean,
                "node_throughput_std": throughput_std,
                "adjacency_nnz_throughput": (
                    DATASETS[target]["adjacency_nnz"] / setup_seconds
                ),
                "peak_gpu_allocated_gib": max(
                    shared[target]["setup_peak_gpu_allocated_gib"],
                    max(row["peak_gpu_allocated_gib"] for row in subset),
                ),
                "peak_gpu_reserved_gib": max(
                    shared[target]["setup_peak_gpu_reserved_gib"],
                    max(row["peak_gpu_reserved_gib"] for row in subset),
                ),
                "peak_rss_gib": max(
                    shared[target]["setup_peak_rss_gib"],
                    max(row["peak_rss_gib"] for row in subset),
                ),
                "candidate_cache_gib": shared[target]["candidate_cache_gib"],
                "ann_recall_at_64_mean": shared[target]["ann_fidelity"][
                    "mean_recall"
                ],
                "ann_nlist": shared[target]["ann_nlist"],
                "ann_nprobe": shared[target]["ann_nprobe"],
                "ann_search_k": shared[target]["ann_search_k"],
            }
        )
    return output


def _report_markdown(
    *,
    aggregate: list[dict],
    rows: list[dict],
    fidelity: dict,
    audit_hash: str,
    preflight_hash: str,
) -> str:
    lines = [
        "# RECAP Large-Target Full-Graph Inference Results",
        "",
        "Scope: **target-side inference scalability only**. These experiments "
        "reuse the accepted RECAP-OFA Setting-A checkpoints and perform no "
        "training or target-label tuning.",
        "",
        "## Primary results",
        "",
        "| Target | Full nodes | Eval nodes | Anomaly rate (%) | KNN | "
        "AUROC (%) | AUPRC (%) |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for item in aggregate:
        lines.append(
            f"| {item['display']} | {item['nodes']:,} | "
            f"{item['evaluation_nodes']:,} | "
            f"{item['anomaly_prevalence'] * 100:.4f} | "
            f"{item['primary_knn']} | "
            f"{_fmt_mean_std(item['auroc_mean'], item['auroc_std'], 100)} | "
            f"{_fmt_mean_std(item['auprc_mean'], item['auprc_std'], 100)} |"
        )
    lines.extend(
        [
            "",
            "Mean ± population standard deviation over the three immutable "
            "Setting-A checkpoints. Every target receives a score for every "
            "node; DGraph-Fin metrics use only its frozen 0/1 evaluation mask.",
            "AUROC 50% and the listed anomaly rate for AUPRC are the random-"
            "ranking references; computational completion is not treated as "
            "evidence of predictive effectiveness.",
            "",
            "## Scalability",
            "",
            "| Target | Shared cold setup (s) | Warm checkpoint inference (s) | "
            "Estimated cold latency (s) | Nodes/s | Adjacency nnz/s | "
            "Peak GPU alloc. (GiB) | Peak RSS (GiB) | KNN cache (GiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in aggregate:
        lines.append(
            f"| {item['display']} | {item['setup_seconds']:.2f} | "
            f"{item['primary_inference_seconds_mean']:.2f} ± "
            f"{item['primary_inference_seconds_std']:.2f} | "
            f"{item['cold_latency_seconds_mean']:.2f} | "
            f"{item['node_throughput_mean']:,.0f} | "
            f"{item['adjacency_nnz_throughput']:,.0f} | "
            f"{item['peak_gpu_allocated_gib']:.2f} | "
            f"{item['peak_rss_gib']:.2f} | "
            f"{item['candidate_cache_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Shared setup includes canonical loading/normalization, four-hop "
            "propagation, initial residual construction, KNN construction and "
            "the pre-registered ANN-fidelity query. Warm checkpoint inference "
            "includes checkpoint load, model residual forward, primary score "
            "construction and score serialization; it excludes metric "
            "calculation. Peak memory is the maximum observed in setup or any "
            "checkpoint run.",
            "",
            "## Approximation fidelity",
            "",
            "| Target | Fidelity population | IVF lists / probes | "
            "Retrieved | Top-64 recall |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in aggregate:
        population = (
            "all nodes"
            if item["target"] == "tfinance"
            else "512 fixed label-blind queries"
        )
        lines.append(
            f"| {item['display']} | {population} | "
            f"{item['ann_nlist']} / {item['ann_nprobe']} | "
            f"{item['ann_search_k']} | "
            f"{item['ann_recall_at_64_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "For T-Finance, the auxiliary ANN route has mean exact/ANN score "
            f"Spearman {fidelity['score_spearman_mean']:.6f}, mean absolute "
            "AUROC difference "
            f"{fidelity['absolute_auroc_difference_mean']:.6f}, and mean "
            "absolute AUPRC difference "
            f"{fidelity['absolute_auprc_difference_mean']:.6f}. Its primary "
            "results remain exact KNN.",
            "",
            "The million-node primary routes use the locked 4,096 IVF lists. "
            "T-Finance's descriptive auxiliary ANN index uses 614 lists "
            "(the adapter's predeclared 64 training vectors/list safety cap "
            "on a 39,357-node graph); this does not affect its exact primary "
            "result.",
            "",
            "## Per-checkpoint records",
            "",
            "| Target | Seed | AUROC | AUPRC | Primary inference (s) | "
            "Peak GPU alloc. (GiB) | Peak RSS (GiB) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {DATASETS[row['target']]['display']} | {row['seed']} | "
            f"{row['auroc']:.6f} | {row['auprc']:.6f} | "
            f"{row['primary_inference_seconds']:.2f} | "
            f"{row['peak_gpu_allocated_gib']:.2f} | "
            f"{row['peak_rss_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance and interpretation",
            "",
            "- 9/9 pre-registered checkpoint-target cells completed with a "
            "finite full-node score vector.",
            "- Every checkpoint, dataset component, candidate cache, score "
            "vector and query mask matched its recorded SHA-256.",
            "- Every label audit froze all declared score routes before the "
            "single label-unlock event.",
            "- Independent AUROC/AUPRC recomputation matched every stored "
            "metric within 1e-12.",
            "- This establishes computational applicability of target-side "
            "full-graph inference under the recorded hardware and adapter. It "
            "does not establish million-node training scalability, and "
            "predictive quality must be read from the primary table rather "
            "than inferred from successful completion.",
            "",
            f"Preflight hash: `{preflight_hash}`  ",
            f"Independent audit hash: `{audit_hash}`",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    *,
    dataset_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    report_path: Path,
) -> dict:
    preflight_path = output_root / "preflight" / "preflight.json"
    preflight = _read_json(preflight_path)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    if protocol_hash != preflight["environment"]["protocol_sha256"]:
        raise AssertionError("Protocol changed after formal preflight")
    expected_ids = [item.run_id for item in build_manifest()]
    observed_ids = sorted(
        path.name
        for path in (output_root / "runs").iterdir()
        if path.is_dir()
    )
    if sorted(expected_ids) != observed_ids:
        raise AssertionError(
            f"Manifest coverage mismatch: expected={expected_ids}, "
            f"observed={observed_ids}"
        )

    component_hashes = _verify_component_hashes(
        preflight=preflight,
        dataset_root=dataset_root,
        checkpoint_root=checkpoint_root,
    )
    shared, shared_audits = _audit_shared(output_root)
    rows: list[dict] = []
    run_audits: list[dict] = []
    for spec in build_manifest():
        row, audit = _audit_run(
            spec=spec,
            output_root=output_root,
            dataset_root=dataset_root,
        )
        rows.append(row)
        run_audits.append(audit)
    fidelity = _paired_tfinance_fidelity(output_root)
    aggregate = _aggregate(rows, shared)
    audit = {
        "format": "recap_large_target_independent_audit_v1",
        "created_at": utc_now(),
        "status": "pass",
        "manifest_expected": expected_ids,
        "manifest_observed": observed_ids,
        "manifest_hash": stable_hash(expected_ids),
        "protocol_sha256": protocol_hash,
        "adapter_source_hashes": {
            path.name: sha256_file(path)
            for path in sorted(SOURCE_DIR.glob("*.py"))
        },
        "component_hash_verification": component_hashes,
        "shared_target_audits": shared_audits,
        "run_audits": run_audits,
        "tfinance_paired_ann_fidelity": fidelity,
        "aggregate": aggregate,
    }
    analysis_dir = output_root / "analysis"
    audit_path = analysis_dir / "independent_audit.json"
    atomic_json(audit_path, audit)
    audit_hash = sha256_file(audit_path)
    summary = {
        "format": "recap_large_target_summary_v1",
        "created_at": utc_now(),
        "status": "pass",
        "completed_primary_cells": len(rows),
        "aggregate": aggregate,
        "tfinance_paired_ann_fidelity": fidelity,
        "preflight_sha256": sha256_file(preflight_path),
        "independent_audit_sha256": audit_hash,
    }
    atomic_json(analysis_dir / "summary.json", summary)
    fieldnames = list(rows[0].keys())
    atomic_csv(analysis_dir / "per_run_results.csv", rows, fieldnames)
    atomic_csv(
        analysis_dir / "aggregate_results.csv",
        aggregate,
        list(aggregate[0].keys()),
    )
    report = _report_markdown(
        aggregate=aggregate,
        rows=rows,
        fidelity=fidelity,
        audit_hash=audit_hash,
        preflight_hash=summary["preflight_sha256"],
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze(
        dataset_root=args.dataset_root.resolve(),
        checkpoint_root=args.checkpoint_root.resolve(),
        output_root=args.output_root.resolve(),
        report_path=args.report_path.resolve(),
    )
    print(
        f"PASS independent audit: "
        f"{summary['completed_primary_cells']}/9 primary cells"
    )


if __name__ == "__main__":
    main()
