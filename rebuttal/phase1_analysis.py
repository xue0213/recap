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
if str(REBUTTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(REBUTTAL_ROOT))

from phase1_protocol import RunSpec, build_manifest, dataset_domain  # noqa: E402
from phase1_runner import (  # noqa: E402
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
    }
    atomic_json(analysis_dir / "timing_summary.json", timing_summary)
    final_validation = {
        "passed": True,
        "training_runs": 42,
        "final_evaluations": len(records),
        "stability_pair_rows": len(pair_rows),
        "stability_summary_rows": len(stability_summary),
        "dataset_metric_rows": len(dataset_rows),
        "macro_metric_rows": len(macro_rows),
        "ddof": 0,
    }
    atomic_json(analysis_dir / "final_validation.json", final_validation)
    return {
        "record_validation": validation,
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
