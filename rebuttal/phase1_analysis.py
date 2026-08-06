"""Aggregate, validate, and analyze locked RECAP Phase 1 results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REBUTTAL_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(REBUTTAL_ROOT))

from rebuttal.phase1_protocol import (  # noqa: E402
    DIAGNOSTIC_EPOCHS,
    RunSpec,
    build_manifest,
    dataset_domain,
)
from rebuttal.phase1_runner import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    RAW_FIELDS,
    atomic_csv,
    atomic_json,
    collect_records,
    load_manifest,
)


PAIR_FIELDS = (
    "paradigm",
    "setting",
    "target_graph",
    "dataset_domain",
    "seed_a",
    "seed_b",
    "nmi",
    "ari",
    "soft_coassignment_similarity",
    "score_spearman",
    "effective_communities_a",
    "effective_communities_b",
    "community_path_a",
    "community_path_b",
)


def population_stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def exact_soft_coassignment_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Permutation-invariant exact similarity without constructing N x N arrays."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape[0] != second.shape[0]:
        raise ValueError(f"Node count mismatch: {first.shape} vs {second.shape}")
    numerator = float(np.square(first.T @ second).sum())
    first_norm = float(np.sqrt(np.square(first.T @ first).sum()))
    second_norm = float(np.sqrt(np.square(second.T @ second).sum()))
    denominator = first_norm * second_norm
    if denominator <= 0:
        raise ValueError("Degenerate soft co-assignment denominator")
    return numerator / denominator


def load_community(path: Path) -> dict[str, np.ndarray]:
    required = {
        "H",
        "hard_assignments",
        "usage",
        "assignment_entropy",
        "adhesion_raw",
        "adhesion_standardized",
        "context_raw",
        "context_standardized",
        "final_scores",
        "node_indices",
        "effective_communities",
        "num_communities",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path}: missing community arrays {sorted(missing)}")
        output = {key: archive[key] for key in required}
    node_count = output["H"].shape[0]
    vector_keys = (
        "hard_assignments",
        "assignment_entropy",
        "adhesion_raw",
        "adhesion_standardized",
        "context_raw",
        "context_standardized",
        "final_scores",
        "node_indices",
    )
    for key in vector_keys:
        if output[key].shape[0] != node_count:
            raise ValueError(f"{path}: {key} does not align with H")
    if output["H"].dtype != np.float32:
        raise ValueError(f"{path}: H must be float32, got {output['H'].dtype}")
    if not np.array_equal(output["node_indices"], np.arange(node_count)):
        raise ValueError(f"{path}: node indices are not deterministic arange order")
    for key, value in output.items():
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"{path}: non-finite values in {key}")
    return output


def stability_rows(records: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for record in records:
        key = (
            str(record["paradigm"]),
            str(record["setting"]),
            str(record["target_graph"]),
        )
        seed = int(record["seed"])
        if seed in grouped[key]:
            raise ValueError(f"Duplicate seed record for {key}, seed {seed}")
        grouped[key][seed] = record

    if len(grouped) != 29:
        raise ValueError(f"Expected 29 stability scopes, got {len(grouped)}")

    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for (paradigm, setting, target), seed_records in sorted(grouped.items()):
        if set(seed_records) != {0, 1, 2}:
            raise ValueError(
                f"{paradigm}/{setting}/{target}: expected seeds 0,1,2; "
                f"got {sorted(seed_records)}"
            )
        loaded = {
            seed: load_community(Path(record["community_output_path"]))
            for seed, record in seed_records.items()
        }
        for seed in (1, 2):
            if not np.array_equal(
                loaded[0]["node_indices"],
                loaded[seed]["node_indices"],
            ):
                raise ValueError(f"{setting}/{target}: node order differs across seeds")

        for seed_a, seed_b in ((0, 1), (0, 2), (1, 2)):
            first = loaded[seed_a]
            second = loaded[seed_b]
            spearman = float(
                spearmanr(first["final_scores"], second["final_scores"]).statistic
            )
            row = {
                "paradigm": paradigm,
                "setting": setting,
                "target_graph": target,
                "dataset_domain": dataset_domain(target),
                "seed_a": seed_a,
                "seed_b": seed_b,
                "nmi": float(
                    normalized_mutual_info_score(
                        first["hard_assignments"],
                        second["hard_assignments"],
                    )
                ),
                "ari": float(
                    adjusted_rand_score(
                        first["hard_assignments"],
                        second["hard_assignments"],
                    )
                ),
                "soft_coassignment_similarity": exact_soft_coassignment_similarity(
                    first["H"],
                    second["H"],
                ),
                "score_spearman": spearman,
                "effective_communities_a": float(first["effective_communities"]),
                "effective_communities_b": float(second["effective_communities"]),
                "community_path_a": seed_records[seed_a]["community_output_path"],
                "community_path_b": seed_records[seed_b]["community_output_path"],
            }
            if any(
                not math.isfinite(float(row[key]))
                for key in ("nmi", "ari", "soft_coassignment_similarity", "score_spearman")
            ):
                raise ValueError(f"Non-finite stability result: {row}")
            pair_rows.append(row)

        scope_pairs = pair_rows[-3:]
        effective_values = [
            float(loaded[seed]["effective_communities"]) for seed in (0, 1, 2)
        ]
        summary = {
            "paradigm": paradigm,
            "setting": setting,
            "target_graph": target,
            "dataset_domain": dataset_domain(target),
        }
        for metric in ("nmi", "ari", "soft_coassignment_similarity", "score_spearman"):
            mean, std = population_stats([float(row[metric]) for row in scope_pairs])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        effective_mean, effective_std = population_stats(effective_values)
        summary["effective_communities_mean"] = effective_mean
        summary["effective_communities_std"] = effective_std
        summary_rows.append(summary)

    if len(pair_rows) != 87 or len(summary_rows) != 29:
        raise AssertionError(
            f"Expected 87 pair rows and 29 summaries, got "
            f"{len(pair_rows)} and {len(summary_rows)}"
        )
    return pair_rows, summary_rows


def metric_summaries(records: list[dict]) -> tuple[list[dict], list[dict]]:
    dataset_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        dataset_groups[(str(record["setting"]), str(record["target_graph"]))].append(
            record
        )

    dataset_rows = []
    for (setting, target), rows in sorted(dataset_groups.items()):
        if sorted(int(row["seed"]) for row in rows) != [0, 1, 2]:
            raise ValueError(f"{setting}/{target}: missing or duplicate seeds")
        auroc_mean, auroc_std = population_stats([float(row["auroc"]) for row in rows])
        auprc_mean, auprc_std = population_stats([float(row["auprc"]) for row in rows])
        dataset_rows.append(
            {
                "setting": setting,
                "target_graph": target,
                "dataset_domain": dataset_domain(target),
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
            }
        )

    setting_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for record in records:
        setting_seed[(str(record["setting"]), int(record["seed"]))].append(record)

    setting_macros: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"auroc": [], "auprc": []}
    )
    domain_macros_c: dict[str, list[float]] = {"auroc": [], "auprc": []}
    for (setting, seed), rows in sorted(setting_seed.items()):
        setting_macros[setting]["auroc"].append(
            float(np.mean([float(row["auroc"]) for row in rows]))
        )
        setting_macros[setting]["auprc"].append(
            float(np.mean([float(row["auprc"]) for row in rows]))
        )
        if setting == "C":
            by_domain: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                by_domain[str(row["dataset_domain"])].append(row)
            domain_macros_c["auroc"].append(
                float(
                    np.mean(
                        [
                            np.mean([float(row["auroc"]) for row in domain_rows])
                            for domain_rows in by_domain.values()
                        ]
                    )
                )
            )
            domain_macros_c["auprc"].append(
                float(
                    np.mean(
                        [
                            np.mean([float(row["auprc"]) for row in domain_rows])
                            for domain_rows in by_domain.values()
                        ]
                    )
                )
            )

    macro_rows = []
    for setting in ("OFO", "A", "B", "C"):
        auroc_mean, auroc_std = population_stats(setting_macros[setting]["auroc"])
        auprc_mean, auprc_std = population_stats(setting_macros[setting]["auprc"])
        macro_rows.append(
            {
                "setting": setting,
                "aggregation": "dataset_macro",
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
            }
        )
    c_auroc_mean, c_auroc_std = population_stats(domain_macros_c["auroc"])
    c_auprc_mean, c_auprc_std = population_stats(domain_macros_c["auprc"])
    macro_rows.append(
        {
            "setting": "C",
            "aggregation": "domain_macro",
            "auroc_mean": c_auroc_mean,
            "auroc_std": c_auroc_std,
            "auprc_mean": c_auprc_mean,
            "auprc_std": c_auprc_std,
        }
    )
    return dataset_rows, macro_rows


def validate_records(records: list[dict], manifest: list[RunSpec]) -> dict:
    expected_keys = {
        (spec.run_id, target, spec.seed)
        for spec in manifest
        for target in spec.target_graphs
    }
    actual_keys = {
        (str(row["run_id"]), str(row["target_graph"]), int(row["seed"]))
        for row in records
    }
    duplicates = len(actual_keys) != len(records)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    nonfinite = []
    missing_paths = []
    for row in records:
        for metric in ("auroc", "auprc", "train_seconds", "inference_seconds"):
            if not math.isfinite(float(row[metric])):
                nonfinite.append((row["run_id"], row["target_graph"], metric))
        for field in ("config_path", "checkpoint_path", "community_output_path"):
            if not Path(row[field]).exists():
                missing_paths.append((row["run_id"], row["target_graph"], field))
    passed = (
        len(records) == 87
        and not duplicates
        and not missing
        and not extra
        and not nonfinite
        and not missing_paths
    )
    return {
        "passed": passed,
        "expected_records": 87,
        "actual_records": len(records),
        "duplicate_keys": duplicates,
        "missing_keys": missing,
        "extra_keys": extra,
        "nonfinite": nonfinite,
        "missing_paths": missing_paths,
    }


def validate_run_artifacts(
    output_root: Path,
    manifest: list[RunSpec],
    records: list[dict],
) -> dict:
    """Audit every formal run artifact against the locked execution protocol."""
    problems: list[str] = []
    warnings: list[str] = []
    expected_evaluations = sum(len(spec.target_graphs) for spec in manifest)
    actual_diagnostic_rows = 0
    actual_checkpoints = 0
    actual_logs = 0
    passed_reload_audits = 0
    exact_knn_runs = 0

    expected_run_ids = {spec.run_id for spec in manifest}
    runs_root = output_root / "runs"
    actual_run_ids = {
        path.name for path in runs_root.iterdir() if path.is_dir()
    } if runs_root.exists() else set()
    if actual_run_ids != expected_run_ids:
        problems.append(
            "run directory mismatch: "
            f"missing={sorted(expected_run_ids - actual_run_ids)}, "
            f"extra={sorted(actual_run_ids - expected_run_ids)}"
        )

    records_by_run: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        records_by_run[str(record["run_id"])].append(record)

    expected_model_values = {
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
        "lambda_bal": 0.1,
        "lambda_E": 0.0,
        "beta": 0.02,
        "gamma": 0.01,
        "knn_cache_enabled": True,
    }

    for spec in manifest:
        prefix = spec.run_id
        run_dir = runs_root / spec.run_id
        required_files = (
            "complete.json",
            "status.json",
            "resolved_config.json",
            "data_metadata.json",
            "training_history.json",
            "training_diagnostics.csv",
            "result_records.json",
            "result_records.csv",
            "checkpoint_reload_audit.json",
            "stdout.log",
            "stderr.log",
        )
        missing_files = [name for name in required_files if not (run_dir / name).exists()]
        if missing_files:
            problems.append(f"{prefix}: missing files {missing_files}")
            continue
        actual_logs += 2

        try:
            complete = json.loads((run_dir / "complete.json").read_text())
            status = json.loads((run_dir / "status.json").read_text())
            config = json.loads((run_dir / "resolved_config.json").read_text())
            history = json.loads((run_dir / "training_history.json").read_text())
            reload_audit = json.loads(
                (run_dir / "checkpoint_reload_audit.json").read_text()
            )
            result_records = json.loads((run_dir / "result_records.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{prefix}: invalid JSON artifact: {error}")
            continue

        if complete.get("status") != "complete" or status.get("status") != "complete":
            problems.append(f"{prefix}: non-complete status")
        if complete.get("run_id") != spec.run_id or status.get("run_id") != spec.run_id:
            problems.append(f"{prefix}: run ID mismatch in status artifacts")
        if int(complete.get("record_count", -1)) != len(spec.target_graphs):
            problems.append(f"{prefix}: wrong complete.json record_count")
        if complete.get("config_hash") != config.get("config_hash"):
            problems.append(f"{prefix}: config hash mismatch")
        if config.get("base_commit") != "c94c4d7985d2cb1438c430173ad868d68d0c1efe":
            problems.append(f"{prefix}: wrong scientific base commit")
        if config.get("standard_deviation_ddof") != 0:
            problems.append(f"{prefix}: standard deviation convention is not ddof=0")
        if config.get("label_isolation") != (
            "sanitized Data objects; labels accessed after scores"
        ):
            problems.append(f"{prefix}: label-isolation declaration mismatch")
        if tuple(config.get("diagnostic_epochs", ())) != DIAGNOSTIC_EPOCHS:
            problems.append(f"{prefix}: diagnostic epoch lock mismatch")
        if tuple(config.get("checkpoint_epochs", ())) != (25, 50, 75, 100):
            problems.append(f"{prefix}: checkpoint epoch lock mismatch")

        run_spec = config.get("run_spec", {})
        if (
            run_spec.get("run_id") != spec.run_id
            or int(run_spec.get("seed", -1)) != spec.seed
            or tuple(run_spec.get("source_graphs", ())) != spec.source_graphs
            or tuple(run_spec.get("target_graphs", ())) != spec.target_graphs
        ):
            problems.append(f"{prefix}: resolved run specification mismatch")

        train_config = config.get("train_config", {})
        if (
            int(train_config.get("epochs", -1)) != 100
            or bool(train_config.get("early_stop", True))
            or int(train_config.get("seed", -1)) != spec.seed
        ):
            problems.append(f"{prefix}: training lock mismatch")
        model_config = config.get("model_config", {})
        for key, expected in expected_model_values.items():
            if model_config.get(key) != expected:
                problems.append(
                    f"{prefix}: model_config.{key}={model_config.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if "ann" in json.dumps(model_config).lower():
            problems.append(f"{prefix}: ANN field found in formal model config")

        checkpoint_names = (
            "resume_epoch_25.pt",
            "resume_epoch_50.pt",
            "resume_epoch_75.pt",
            "resume_epoch_100.pt",
            "final.pt",
        )
        for name in checkpoint_names:
            checkpoint = run_dir / "checkpoints" / name
            if checkpoint.exists() and checkpoint.stat().st_size > 0:
                actual_checkpoints += 1
            else:
                problems.append(f"{prefix}: missing or empty checkpoint {name}")

        losses = history.get("losses", [])
        if int(history.get("epochs", -1)) != 100 or len(losses) != 100:
            problems.append(f"{prefix}: expected 100 recorded training losses")
        elif not all(math.isfinite(float(loss)) for loss in losses):
            problems.append(f"{prefix}: non-finite training loss")
        knn_cache = history.get("knn_cache", {})
        if set(knn_cache) != set(spec.source_graphs):
            problems.append(f"{prefix}: training exact-KNN provenance mismatch")
        elif all(
            "exact_knn" in str(value.get("path", ""))
            and value.get("key_sha256")
            for value in knn_cache.values()
        ):
            exact_knn_runs += 1
        else:
            problems.append(f"{prefix}: incomplete exact-KNN cache provenance")

        if bool(reload_audit.get("passed")) and math.isfinite(
            float(reload_audit.get("max_abs_score_difference", math.nan))
        ):
            passed_reload_audits += 1
        else:
            problems.append(f"{prefix}: checkpoint-reload audit failed")
        if complete.get("checkpoint_reload_audit") != reload_audit:
            problems.append(f"{prefix}: checkpoint audit copy mismatch")

        try:
            with (run_dir / "training_diagnostics.csv").open(newline="") as handle:
                diagnostics = list(csv.DictReader(handle))
        except OSError as error:
            problems.append(f"{prefix}: cannot read diagnostics: {error}")
            diagnostics = []
        expected_per_epoch = len(spec.source_graphs) + (
            1 if spec.paradigm == "one-for-all" else 0
        )
        expected_rows = len(DIAGNOSTIC_EPOCHS) * expected_per_epoch
        actual_diagnostic_rows += len(diagnostics)
        if len(diagnostics) != expected_rows:
            problems.append(
                f"{prefix}: diagnostic rows={len(diagnostics)}, expected={expected_rows}"
            )
        for epoch in DIAGNOSTIC_EPOCHS:
            epoch_rows = [
                row for row in diagnostics if int(float(row.get("epoch", -1))) == epoch
            ]
            sources = {
                row.get("dataset")
                for row in epoch_rows
                if row.get("row_type") == "source"
            }
            macros = [
                row for row in epoch_rows if row.get("row_type") == "macro"
            ]
            if sources != set(spec.source_graphs):
                problems.append(f"{prefix}: source diagnostics mismatch at epoch {epoch}")
            if spec.paradigm == "one-for-all":
                if len(macros) != 1 or macros[0].get("dataset") != "__source_macro__":
                    problems.append(
                        f"{prefix}: source-macro diagnostic mismatch at epoch {epoch}"
                    )
            elif macros:
                problems.append(f"{prefix}: unexpected macro diagnostic at epoch {epoch}")
        numeric_diagnostic_fields = (
            "total_loss",
            "optimizer_step_loss",
            "assignment_entropy",
            "effective_communities",
        )
        for row in diagnostics:
            if any(
                not math.isfinite(float(row.get(field, math.nan)))
                for field in numeric_diagnostic_fields
            ):
                problems.append(f"{prefix}: non-finite diagnostic value")
                break

        expected_targets = set(spec.target_graphs)
        actual_targets = {str(row.get("target_graph")) for row in result_records}
        if len(result_records) != len(spec.target_graphs) or actual_targets != expected_targets:
            problems.append(f"{prefix}: result-record targets mismatch")
        if len(records_by_run.get(spec.run_id, ())) != len(spec.target_graphs):
            problems.append(f"{prefix}: raw aggregate target count mismatch")

    expected_diagnostic_rows = sum(
        len(DIAGNOSTIC_EPOCHS)
        * (len(spec.source_graphs) + (1 if spec.paradigm == "one-for-all" else 0))
        for spec in manifest
    )
    failure_log = output_root / "failure_log.jsonl"
    historical_failure_entries = 0
    if failure_log.exists():
        historical_failure_entries = sum(
            1 for line in failure_log.read_text().splitlines() if line.strip()
        )
    if historical_failure_entries:
        warnings.append(
            f"{historical_failure_entries} historical tooling failure entries retained; "
            "all corresponding formal run statuses are now complete"
        )

    summary = {
        "passed": not problems,
        "formal_runs_expected": len(manifest),
        "formal_run_directories": len(actual_run_ids),
        "final_evaluations_expected": expected_evaluations,
        "final_evaluations_actual": len(records),
        "diagnostic_rows_expected": expected_diagnostic_rows,
        "diagnostic_rows_actual": actual_diagnostic_rows,
        "checkpoints_expected": len(manifest) * 5,
        "checkpoints_actual": actual_checkpoints,
        "log_files_expected": len(manifest) * 2,
        "log_files_actual": actual_logs,
        "checkpoint_reload_audits_expected": len(manifest),
        "checkpoint_reload_audits_passed": passed_reload_audits,
        "exact_knn_provenance_runs_expected": len(manifest),
        "exact_knn_provenance_runs_actual": exact_knn_runs,
        "active_failures": sum(
            1
            for spec in manifest
            if not (runs_root / spec.run_id / "complete.json").exists()
        ),
        "historical_failure_entries": historical_failure_entries,
        "problems": problems,
        "warnings": warnings,
    }
    return summary


def detailed_timing_summary(records: list[dict]) -> tuple[dict, list[dict]]:
    """Return protocol-aware timing totals without repeating OFA training costs."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_run[str(record["run_id"])].append(record)

    rows: list[dict] = []
    for setting in ("OFO", "A", "B", "C"):
        run_rows = [
            group[0]
            for group in by_run.values()
            if str(group[0]["setting"]) == setting
        ]
        evaluation_rows = [
            record for record in records if str(record["setting"]) == setting
        ]
        train_values = [float(row["train_seconds"]) for row in run_rows]
        prepare_values = [float(row["data_prepare_seconds"]) for row in run_rows]
        diagnostic_values = [float(row["diagnostic_seconds"]) for row in run_rows]
        inference_values = [
            float(row["inference_seconds"]) for row in evaluation_rows
        ]
        peak_values = [float(row["peak_gpu_memory_mb"]) for row in run_rows]
        rows.append(
            {
                "setting": setting,
                "training_runs": len(run_rows),
                "evaluations": len(evaluation_rows),
                "data_prepare_seconds_total": float(sum(prepare_values)),
                "train_seconds_total": float(sum(train_values)),
                "diagnostic_seconds_total": float(sum(diagnostic_values)),
                "inference_seconds_total": float(sum(inference_values)),
                "train_seconds_mean": population_stats(train_values)[0],
                "train_seconds_std": population_stats(train_values)[1],
                "inference_seconds_mean": population_stats(inference_values)[0],
                "inference_seconds_std": population_stats(inference_values)[1],
                "peak_gpu_memory_mb_mean": population_stats(peak_values)[0],
                "peak_gpu_memory_mb_max": float(max(peak_values)),
            }
        )

    overall = {
        "formal_training_runs": len(by_run),
        "final_evaluations": len(records),
        "data_prepare_seconds_total": float(
            sum(float(group[0]["data_prepare_seconds"]) for group in by_run.values())
        ),
        "train_seconds_total": float(
            sum(float(group[0]["train_seconds"]) for group in by_run.values())
        ),
        "diagnostic_seconds_total": float(
            sum(float(group[0]["diagnostic_seconds"]) for group in by_run.values())
        ),
        "inference_seconds_total": float(
            sum(float(record["inference_seconds"]) for record in records)
        ),
        "peak_gpu_memory_mb_max": float(
            max(float(group[0]["peak_gpu_memory_mb"]) for group in by_run.values())
        ),
    }
    overall["accounted_seconds_total"] = float(
        overall["data_prepare_seconds_total"]
        + overall["train_seconds_total"]
        + overall["diagnostic_seconds_total"]
        + overall["inference_seconds_total"]
    )
    return overall, rows


def analyze(output_root: Path, manifest_path: Path, allow_partial: bool) -> dict:
    manifest = load_manifest(manifest_path)
    records = collect_records(output_root, manifest)
    validation = validate_records(records, manifest)
    atomic_json(output_root / "analysis" / "record_validation.json", validation)
    if not validation["passed"]:
        if allow_partial:
            return {"record_validation": validation, "partial": True}
        raise ValueError(f"Raw record validation failed: {validation}")

    dataset_rows, macro_rows = metric_summaries(records)
    pair_rows, stability_summary = stability_rows(records)
    analysis_dir = output_root / "analysis"
    artifact_validation = validate_run_artifacts(output_root, manifest, records)
    atomic_json(analysis_dir / "artifact_validation.json", artifact_validation)
    if not artifact_validation["passed"]:
        raise ValueError(f"Run artifact validation failed: {artifact_validation}")
    atomic_csv(
        analysis_dir / "metrics_by_dataset.csv",
        dataset_rows,
        list(dataset_rows[0]),
    )
    atomic_csv(
        analysis_dir / "metric_macros.csv",
        macro_rows,
        list(macro_rows[0]),
    )
    atomic_csv(
        analysis_dir / "stability_pairs.csv",
        pair_rows,
        PAIR_FIELDS,
    )
    atomic_csv(
        analysis_dir / "stability_summary.csv",
        stability_summary,
        list(stability_summary[0]),
    )
    timing_overall, timing_by_setting = detailed_timing_summary(records)
    atomic_csv(
        analysis_dir / "timing_by_setting.csv",
        timing_by_setting,
        list(timing_by_setting[0]),
    )
    timing_records = [row for row in records if not bool(row.get("resumed", False))]
    timing_summary = {
        "records_included": len(timing_records),
        "records_excluded_resumed": len(records) - len(timing_records),
        "train_seconds_total_unique_runs": float(
            sum(
                float(next(rows)["train_seconds"])
                for _, rows in _group_iter(timing_records, "run_id")
            )
        ),
        "inference_seconds_total": float(
            sum(float(row["inference_seconds"]) for row in timing_records)
        ),
        **timing_overall,
    }
    atomic_json(analysis_dir / "timing_summary.json", timing_summary)
    final_validation = {
        "passed": validation["passed"] and artifact_validation["passed"],
        "training_runs": 42,
        "final_evaluations": len(records),
        "stability_pair_rows": len(pair_rows),
        "stability_summary_rows": len(stability_summary),
        "dataset_metric_rows": len(dataset_rows),
        "macro_metric_rows": len(macro_rows),
        "ddof": 0,
        "artifact_validation": "analysis/artifact_validation.json",
    }
    atomic_json(analysis_dir / "final_validation.json", final_validation)
    return {
        "record_validation": validation,
        "artifact_validation": artifact_validation,
        "final_validation": final_validation,
        "timing_summary": timing_summary,
    }


def _group_iter(rows: list[dict], key: str):
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    for group_key, group_rows in grouped.items():
        yield group_key, iter(group_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--manifest",
        default=str(REBUTTAL_ROOT / "phase1_manifest.json"),
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        output_root=Path(args.output_root),
        manifest_path=Path(args.manifest),
        allow_partial=args.allow_partial,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
