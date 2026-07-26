"""Audit and aggregate the locked Phase 2 OFA baseline reproductions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score

from .baseline_common import atomic_json, sha256_array, sha256_file
from .baseline_protocol import (
    DATASETS,
    SEEDS,
    SETTINGS,
    BaselineRunSpec,
    build_manifest,
    validate_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "rebuttal" / "artifacts" / "phase2_baselines"
)
PROTOCOL_PATH = PROJECT_ROOT / "rebuttal" / "BASELINE_OFA_REPROTOCOL.md"
UPSTREAM_MANIFEST_PATH = (
    PROJECT_ROOT / "rebuttal" / "baselines" / "upstream_manifest.json"
)
REPORT_PATH = (
    PROJECT_ROOT / "rebuttal" / "reports" / "PHASE2_OFA_BASELINE_REPORT.md"
)
PHASE1_MACRO_PATH = (
    PROJECT_ROOT
    / "rebuttal"
    / "artifacts"
    / "phase1"
    / "analysis"
    / "metric_macros.csv"
)

DATASET_ORDER = {
    setting: {name: index for index, name in enumerate(definition["targets"])}
    for setting, definition in SETTINGS.items()
}
METHOD_ORDER = {
    setting: {name: index for index, name in enumerate(definition["methods"])}
    for setting, definition in SETTINGS.items()
}


def _population(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size != len(SEEDS) or not np.isfinite(array).all():
        raise ValueError(f"Expected {len(SEEDS)} finite seed values, got {array}")
    return float(array.mean()), float(array.std(ddof=0))


def _labels(dataset_dir: Path, name: str) -> np.ndarray:
    path = dataset_dir / DATASETS[name]["file"]
    raw = sio.loadmat(path, variable_names=["Label", "gnd"])
    value = raw["Label"] if "Label" in raw else raw["gnd"]
    labels = np.asarray(value, dtype=np.float32).reshape(-1)
    if not set(np.unique(labels).tolist()).issubset({0.0, 1.0}):
        raise ValueError(f"{name}: labels are not binary")
    return labels


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _expected_context(method: str) -> int:
    return 10 if method in {"ARC", "IA-GGAD"} else 0


def audit_runs(
    output_root: Path,
    dataset_dir: Path,
    specs: list[BaselineRunSpec],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Independently verify all formal run and score artifacts."""

    problems: list[str] = []
    records: list[dict[str, Any]] = []
    expected_protocol_hash = sha256_file(PROTOCOL_PATH)
    expected_upstream_hash = sha256_file(UPSTREAM_MANIFEST_PATH)
    expected_ids = {spec.run_id for spec in specs}
    run_root = output_root / "runs"
    actual_ids = (
        {path.name for path in run_root.iterdir() if path.is_dir()}
        if run_root.exists()
        else set()
    )
    if actual_ids != expected_ids:
        problems.append(
            "formal run directories differ from manifest: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"extra={sorted(actual_ids - expected_ids)}"
        )

    protocol_hashes: set[str] = set()
    upstream_hashes: set[str] = set()
    checkpoint_hashes: set[str] = set()
    score_hashes: set[str] = set()
    dataset_hash_by_name: dict[str, str] = {}
    label_events = 0
    target_score_files = 0
    checkpoint_reload_max = 0.0

    for spec in specs:
        prefix = spec.run_id
        run_dir = run_root / prefix
        complete_path = run_dir / "complete.json"
        result_path = run_dir / "result.json"
        checkpoint_path = run_dir / "checkpoint.pt"
        if not complete_path.exists():
            problems.append(f"{prefix}: missing complete.json")
            continue
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f"{prefix}: unreadable complete.json: {error}")
            continue

        if complete.get("status") != "complete":
            problems.append(f"{prefix}: status is not complete")
        expected_run = json.loads(json.dumps(spec.to_dict()))
        if complete.get("run") != expected_run:
            problems.append(f"{prefix}: embedded run specification drift")
        if not result_path.exists() or result_path.read_bytes() != complete_path.read_bytes():
            problems.append(f"{prefix}: result.json and complete.json differ")

        protocol_hash = str(complete.get("protocol_sha256", ""))
        upstream_hash = str(complete.get("upstream_manifest_sha256", ""))
        protocol_hashes.add(protocol_hash)
        upstream_hashes.add(upstream_hash)
        if protocol_hash != expected_protocol_hash:
            problems.append(f"{prefix}: protocol hash mismatch")
        if upstream_hash != expected_upstream_hash:
            problems.append(f"{prefix}: upstream manifest hash mismatch")

        dataset_hashes = complete.get("dataset_hashes", {})
        expected_dataset_hashes = set(spec.source_graphs) | set(spec.target_graphs)
        if set(dataset_hashes) != expected_dataset_hashes:
            problems.append(f"{prefix}: source/target dataset hash map differs")
        for name, value in dataset_hashes.items():
            if (
                name in dataset_hash_by_name
                and dataset_hash_by_name[name] != value
            ):
                problems.append(f"{prefix}: {name} dataset hash differs")
            dataset_hash_by_name[name] = value

        checkpoint = complete.get("checkpoint", {})
        if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
            problems.append(f"{prefix}: missing or empty checkpoint")
        else:
            actual_checkpoint_hash = sha256_file(checkpoint_path)
            checkpoint_hashes.add(actual_checkpoint_hash)
            if checkpoint.get("sha256") != actual_checkpoint_hash:
                problems.append(f"{prefix}: checkpoint SHA-256 mismatch")
        reload_diff = float(checkpoint.get("reload_max_abs_diff", math.inf))
        checkpoint_reload_max = max(checkpoint_reload_max, reload_diff)
        if not checkpoint.get("reload_passed") or not math.isfinite(reload_diff):
            problems.append(f"{prefix}: checkpoint reload audit failed")
        reload_targets = checkpoint.get("reload_target_diffs", {})
        if set(reload_targets) != set(spec.target_graphs):
            problems.append(f"{prefix}: reload target set mismatch")

        calibration_key = {
            "IA-GGAD": "fusion_calibration",
            "AnomalyGFM-ZS": "score_calibration",
        }.get(spec.method)
        if calibration_key is not None:
            calibration = complete.get(calibration_key, {})
            calibration_file = Path(str(calibration.get("path", "")))
            expected_status = (
                "created_and_locked" if spec.seed == 0 else "reused_seed0_lock"
            )
            if calibration.get("status") != expected_status:
                problems.append(f"{prefix}: calibration status differs")
            if not calibration_file.exists():
                problems.append(f"{prefix}: locked calibration file is missing")
            else:
                locked_calibration = json.loads(
                    calibration_file.read_text(encoding="utf-8")
                )
                if (
                    int(locked_calibration.get("calibration_seed", -1)) != 0
                    or float(locked_calibration.get("selected_weight", math.nan))
                    != float(calibration.get("selected_weight", math.inf))
                    or locked_calibration.get("grid_records")
                    != calibration.get("grid_records")
                ):
                    problems.append(f"{prefix}: source-only calibration lock differs")

        audit = complete.get("label_audit", {})
        label_events += len(audit.get("events", []))
        if not audit.get("passed") or audit.get("invalid_events"):
            problems.append(f"{prefix}: label audit failed")
        frozen = audit.get("frozen_scores", {})
        if set(frozen) != set(spec.target_graphs):
            problems.append(f"{prefix}: frozen-score target set mismatch")

        target_results = complete.get("target_results", {})
        if set(target_results) != set(spec.target_graphs):
            problems.append(f"{prefix}: target result set mismatch")
            continue

        for target in spec.target_graphs:
            result = target_results[target]
            score_path = run_dir / "scores" / f"{target}.npz"
            if not score_path.exists() or score_path.stat().st_size == 0:
                problems.append(f"{prefix}/{target}: missing score artifact")
                continue
            target_score_files += 1
            try:
                with np.load(score_path, allow_pickle=False) as score_file:
                    if not {
                        "scores",
                        "query_mask",
                        "context_indices",
                    }.issubset(score_file.files):
                        problems.append(
                            f"{prefix}/{target}: required score keys are missing"
                        )
                    scores = np.asarray(score_file["scores"], dtype=np.float32)
                    query_mask = np.asarray(score_file["query_mask"], dtype=bool)
                    context = np.asarray(
                        score_file["context_indices"], dtype=np.int64
                    )
            except Exception as error:  # corrupted npz should be an audit failure
                problems.append(f"{prefix}/{target}: unreadable score npz: {error}")
                continue

            if scores.ndim != 1 or not np.isfinite(scores).all():
                problems.append(f"{prefix}/{target}: invalid score vector")
            if query_mask.ndim != 1:
                problems.append(f"{prefix}/{target}: invalid query mask")
            if len(context) != _expected_context(spec.method):
                problems.append(
                    f"{prefix}/{target}: context size {len(context)} is not "
                    f"{_expected_context(spec.method)}"
                )
            if len(np.unique(context)) != len(context):
                problems.append(f"{prefix}/{target}: duplicate context indices")
            if len(query_mask) and (
                np.any(context < 0) or np.any(context >= len(query_mask))
            ):
                problems.append(f"{prefix}/{target}: context index out of range")
            if len(context) and np.any(query_mask[context]):
                problems.append(f"{prefix}/{target}: context remains in query")

            query_nodes = int(query_mask.sum())
            nodes = int(result.get("nodes", -1))
            context_result_key = (
                "internal_reference_nodes"
                if spec.method == "IA-GGAD"
                else "context_nodes"
            )
            if (
                nodes != len(query_mask)
                or query_nodes != len(scores)
                or query_nodes != int(result.get("query_nodes", -1))
                or len(context) != int(result.get(context_result_key, -1))
            ):
                problems.append(f"{prefix}/{target}: score/query shape mismatch")

            score_hash = sha256_array(scores)
            query_hash = sha256_array(query_mask)
            score_hashes.add(score_hash)
            frozen_target = frozen.get(target, {})
            if (
                frozen_target.get("score_sha256") != score_hash
                or frozen_target.get("query_mask_sha256") != query_hash
            ):
                problems.append(f"{prefix}/{target}: frozen score hash mismatch")

            labels = _labels(dataset_dir, target)
            if len(labels) != len(query_mask):
                problems.append(f"{prefix}/{target}: label/query length mismatch")
                continue
            query_labels = labels[query_mask]
            recomputed_auroc = float(roc_auc_score(query_labels, scores))
            recomputed_auprc = float(average_precision_score(query_labels, scores))
            recorded_auroc = float(result["AUROC"])
            recorded_auprc = float(result["AUPRC"])
            if not (
                math.isclose(recomputed_auroc, recorded_auroc, abs_tol=1e-12)
                and math.isclose(recomputed_auprc, recorded_auprc, abs_tol=1e-12)
            ):
                problems.append(f"{prefix}/{target}: independently recomputed metric differs")

            records.append(
                {
                    "setting": spec.setting,
                    "method": spec.method,
                    "seed": spec.seed,
                    "target_graph": target,
                    "domain": DATASETS[target]["domain"],
                    "nodes": nodes,
                    "query_nodes": query_nodes,
                    "context_nodes": len(context),
                    "auroc": recomputed_auroc,
                    "auprc": recomputed_auprc,
                    "train_seconds": float(complete["training_seconds"]),
                    "preparation_seconds": float(complete["preparation_seconds"]),
                    "evaluation_seconds": float(complete["evaluation_seconds"]),
                    "peak_gpu_memory_bytes": int(complete["peak_gpu_memory_bytes"]),
                    "run_id": prefix,
                    "score_path": str(score_path),
                    "score_sha256": score_hash,
                    "checkpoint_path": str(checkpoint_path),
                }
            )

    expected_keys = {
        (spec.setting, spec.method, spec.seed, target)
        for spec in specs
        for target in spec.target_graphs
    }
    actual_keys = {
        (
            str(record["setting"]),
            str(record["method"]),
            int(record["seed"]),
            str(record["target_graph"]),
        )
        for record in records
    }
    if len(records) != 156 or len(actual_keys) != len(records):
        problems.append(f"expected 156 unique evaluation records, got {len(records)}")
    if actual_keys != expected_keys:
        problems.append(
            f"evaluation key mismatch: missing={len(expected_keys - actual_keys)}, "
            f"extra={len(actual_keys - expected_keys)}"
        )
    if set(dataset_hash_by_name) != set(DATASETS):
        problems.append("not every locked dataset was hashed across the manifest")

    summary = {
        "format": "recap_phase2_baseline_artifact_audit_v1",
        "passed": not problems,
        "training_runs_expected": 24,
        "training_runs_found": len(actual_ids & expected_ids),
        "evaluations_expected": 156,
        "evaluations_verified": len(records),
        "target_score_files_verified": target_score_files,
        "unique_score_hashes": len(score_hashes),
        "unique_checkpoint_hashes": len(checkpoint_hashes),
        "label_audit_events": label_events,
        "protocol_hashes": sorted(protocol_hashes),
        "upstream_manifest_hashes": sorted(upstream_hashes),
        "checkpoint_reload_max_abs_diff": checkpoint_reload_max,
        "problems": problems,
    }
    return records, summary


def aggregate(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    run_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        dataset_groups[
            (str(row["setting"]), str(row["method"]), str(row["target_graph"]))
        ].append(row)
        run_groups[
            (str(row["setting"]), str(row["method"]), int(row["seed"]))
        ].append(row)

    dataset_rows: list[dict[str, Any]] = []
    for (setting, method, target), rows in sorted(
        dataset_groups.items(),
        key=lambda value: (
            value[0][0],
            METHOD_ORDER[value[0][0]][value[0][1]],
            DATASET_ORDER[value[0][0]][value[0][2]],
        ),
    ):
        if sorted(int(row["seed"]) for row in rows) != list(SEEDS):
            raise ValueError(f"{setting}/{method}/{target}: incomplete seeds")
        auroc_mean, auroc_std = _population(float(row["auroc"]) for row in rows)
        auprc_mean, auprc_std = _population(float(row["auprc"]) for row in rows)
        dataset_rows.append(
            {
                "setting": setting,
                "method": method,
                "target_graph": target,
                "display_name": DATASETS[target]["display"],
                "domain": DATASETS[target]["domain"],
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
            }
        )

    seed_macro: dict[
        tuple[str, str, str], dict[int, tuple[float, float]]
    ] = defaultdict(dict)
    for (setting, method, seed), rows in sorted(run_groups.items()):
        if len(rows) != len(SETTINGS[setting]["targets"]):
            raise ValueError(f"{setting}/{method}/seed{seed}: target count differs")
        seed_macro[(setting, method, "dataset_macro")][seed] = (
            float(np.mean([float(row["auroc"]) for row in rows])),
            float(np.mean([float(row["auprc"]) for row in rows])),
        )
        if setting == "C":
            domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                domains[str(row["domain"])].append(row)
            seed_macro[(setting, method, "domain_macro")][seed] = (
                float(
                    np.mean(
                        [
                            np.mean([float(row["auroc"]) for row in domain_rows])
                            for domain_rows in domains.values()
                        ]
                    )
                ),
                float(
                    np.mean(
                        [
                            np.mean([float(row["auprc"]) for row in domain_rows])
                            for domain_rows in domains.values()
                        ]
                    )
                ),
            )

    macro_rows: list[dict[str, Any]] = []
    for (setting, method, aggregation), by_seed in sorted(
        seed_macro.items(),
        key=lambda value: (
            value[0][0],
            METHOD_ORDER[value[0][0]][value[0][1]],
            value[0][2],
        ),
    ):
        if sorted(by_seed) != list(SEEDS):
            raise ValueError(f"{setting}/{method}/{aggregation}: incomplete seeds")
        auroc_mean, auroc_std = _population(by_seed[s][0] for s in SEEDS)
        auprc_mean, auprc_std = _population(by_seed[s][1] for s in SEEDS)
        macro_rows.append(
            {
                "setting": setting,
                "method": method,
                "aggregation": aggregation,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
                "seed_values": {
                    str(seed): {
                        "auroc": by_seed[seed][0],
                        "auprc": by_seed[seed][1],
                    }
                    for seed in SEEDS
                },
            }
        )
    return dataset_rows, macro_rows


def _metric(mean: float, std: float) -> str:
    return f"{100 * mean:.2f}±{100 * std:.2f}"


def _paired(row: dict[str, Any]) -> str:
    return (
        f"{_metric(float(row['auroc_mean']), float(row['auroc_std']))} / "
        f"{_metric(float(row['auprc_mean']), float(row['auprc_std']))}"
    )


def _lookup(
    rows: list[dict[str, Any]], **keys: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == value for key, value in keys.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one aggregate for {keys}, got {len(matches)}")
    return matches[0]


def _phase1_recap_macros(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {
                key: (
                    float(value)
                    if key in {
                        "auroc_mean",
                        "auroc_std",
                        "auprc_mean",
                        "auprc_std",
                    }
                    else value
                )
                for key, value in row.items()
            }
            for row in csv.DictReader(handle)
            if row["setting"] in {"A", "B", "C"}
        ]


def _calibrations(output_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output_root / "calibration").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": path.stem,
                "selected_weight": float(value["selected_weight"]),
                "calibration_seed": int(value["calibration_seed"]),
                "grid_size": len(value["grid_records"]),
            }
        )
    return rows


def _timings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = {str(row["run_id"]): row for row in records}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs.values():
        groups[str(row["method"])].append(row)
    return [
        {
            "method": method,
            "runs": len(rows),
            "preparation_seconds": sum(
                float(row["preparation_seconds"]) for row in rows
            ),
            "training_seconds": sum(float(row["train_seconds"]) for row in rows),
            "evaluation_seconds": sum(
                float(row["evaluation_seconds"]) for row in rows
            ),
            "peak_gpu_gib": max(
                float(row["peak_gpu_memory_bytes"]) for row in rows
            )
            / 2**30,
        }
        for method, rows in sorted(groups.items())
    ]


def render_report(
    dataset_rows: list[dict[str, Any]],
    macro_rows: list[dict[str, Any]],
    audit: dict[str, Any],
    recap_macros: list[dict[str, Any]],
    calibrations: list[dict[str, Any]],
    timings: list[dict[str, Any]],
) -> str:
    target_a = SETTINGS["A"]["targets"]
    setting_a_method_order = ("AnomalyGFM-ZS", "IA-GGAD", "ARC", "UNPrompt")
    lines = [
        "# Phase 2 RECAP-OFA Baseline Reproduction Report",
        "",
        "Date: 2026-07-26",
        "",
        "Status: **PASS**" if audit["passed"] else "Status: **FAIL**",
        "",
        "All values are mean±population standard deviation over seeds 0/1/2,",
        "in percent. Target labels were read only after score freeze, except",
        "ARC's protocol-authorized 10 labeled-normal target contexts.",
        "",
        "## Table 3 baseline rows — Setting A AUROC (%)",
        "",
        "| Method | " + " | ".join(DATASETS[name]["display"] for name in target_a) + " | Avg. |",
        "|---|" + "---:|" * (len(target_a) + 1),
    ]
    for method in setting_a_method_order:
        cells = [
            _metric(
                float(
                    _lookup(
                        dataset_rows,
                        setting="A",
                        method=method,
                        target_graph=target,
                    )["auroc_mean"]
                ),
                float(
                    _lookup(
                        dataset_rows,
                        setting="A",
                        method=method,
                        target_graph=target,
                    )["auroc_std"]
                ),
            )
            for target in target_a
        ]
        macro = _lookup(
            macro_rows,
            setting="A",
            method=method,
            aggregation="dataset_macro",
        )
        lines.append(
            f"| {method} | " + " | ".join(cells) + f" | {_metric(macro['auroc_mean'], macro['auroc_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Table 4 baseline rows — Setting A AUPRC (%)",
            "",
            "| Method | " + " | ".join(DATASETS[name]["display"] for name in target_a) + " | Avg. |",
            "|---|" + "---:|" * (len(target_a) + 1),
        ]
    )
    for method in setting_a_method_order:
        cells = [
            _metric(
                float(
                    _lookup(
                        dataset_rows,
                        setting="A",
                        method=method,
                        target_graph=target,
                    )["auprc_mean"]
                ),
                float(
                    _lookup(
                        dataset_rows,
                        setting="A",
                        method=method,
                        target_graph=target,
                    )["auprc_std"]
                ),
            )
            for target in target_a
        ]
        macro = _lookup(
            macro_rows,
            setting="A",
            method=method,
            aggregation="dataset_macro",
        )
        lines.append(
            f"| {method} | " + " | ".join(cells) + f" | {_metric(macro['auprc_mean'], macro['auprc_std'])} |"
        )

    lines.extend(
        [
            "",
            "## Table 5 baseline rows — Source-target robustness",
            "",
            "| Method | Setting A Dataset-Macro | Setting B Social-Macro | Setting C Dataset-Macro | Setting C Domain-Macro |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in ("ARC", "IA-GGAD"):
        cells = [
            _paired(
                _lookup(
                    macro_rows,
                    setting=setting,
                    method=method,
                    aggregation=aggregation,
                )
            )
            for setting, aggregation in (
                ("A", "dataset_macro"),
                ("B", "dataset_macro"),
                ("C", "dataset_macro"),
                ("C", "domain_macro"),
            )
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")

    for setting, title, macro_label in (
        ("B", "Table 6 baseline rows — Leave-Social-Domain-Out", "Social Macro"),
        ("C", "Table 7 baseline rows — Citation-only Source Transfer", "Dataset-Macro"),
    ):
        targets = SETTINGS[setting]["targets"]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Method | "
                + " | ".join(DATASETS[name]["display"] for name in targets)
                + f" | {macro_label}"
                + (" | Domain-Macro |" if setting == "C" else " |"),
                "|---|" + "---:|" * (len(targets) + 1 + (setting == "C")),
            ]
        )
        for method in SETTINGS[setting]["methods"]:
            cells = [
                _paired(
                    _lookup(
                        dataset_rows,
                        setting=setting,
                        method=method,
                        target_graph=target,
                    )
                )
                for target in targets
            ]
            cells.append(
                _paired(
                    _lookup(
                        macro_rows,
                        setting=setting,
                        method=method,
                        aggregation="dataset_macro",
                    )
                )
            )
            if setting == "C":
                cells.append(
                    _paired(
                        _lookup(
                            macro_rows,
                            setting=setting,
                            method=method,
                            aggregation="domain_macro",
                        )
                    )
                )
            lines.append(f"| {method} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Independent artifact audit",
            "",
            f"- Training runs: {audit['training_runs_found']}/{audit['training_runs_expected']}",
            f"- Recomputed evaluations: {audit['evaluations_verified']}/{audit['evaluations_expected']}",
            f"- Frozen score files verified: {audit['target_score_files_verified']}",
            f"- Label-audit events checked: {audit['label_audit_events']}",
            f"- Maximum checkpoint reload difference: {audit['checkpoint_reload_max_abs_diff']:.8g}",
            f"- Audit problems: {len(audit['problems'])}",
            "",
            "The failed pre-recovery UNPrompt attempt and pre-amendment ARC results",
            "are retained outside the formal `runs/` directory and are excluded",
            "from every aggregate above.",
            "",
        ]
    )
    if recap_macros:
        lines.extend(
            [
                "## Relation to the locked RECAP results",
                "",
                "This comparison is descriptive: the baselines use source labels;",
                "RECAP uses no source labels, target context, or target tuning.",
                "",
                "| Setting | RECAP | ARC | IA-GGAD |",
                "|---|---:|---:|---:|",
            ]
        )
        for setting, aggregation in (
            ("A", "dataset_macro"),
            ("B", "dataset_macro"),
            ("C", "dataset_macro"),
        ):
            recap = _lookup(
                recap_macros,
                setting=setting,
                aggregation=aggregation,
            )
            arc = _lookup(
                macro_rows,
                setting=setting,
                method="ARC",
                aggregation=aggregation,
            )
            ia = _lookup(
                macro_rows,
                setting=setting,
                method="IA-GGAD",
                aggregation=aggregation,
            )
            lines.append(
                f"| {setting} | {_paired(recap)} | {_paired(arc)} | {_paired(ia)} |"
            )
        lines.extend(
            [
                "",
                "RECAP trails ARC and IA-GGAD in Settings A/B. Under citation-only",
                "sources (C), RECAP has higher dataset-macro AUROC than both while",
                "its AUPRC remains slightly below both. In Setting A, RECAP exceeds",
                "the target-context-free UNPrompt and AnomalyGFM-ZS reproductions",
                "on both dataset-macro metrics.",
                "",
            ]
        )

    lines.extend(
        [
            "## Source-only calibration locks",
            "",
            "| Lock | Selected weight | Seed | Grid size |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in calibrations:
        lines.append(
            f"| {row['name']} | {row['selected_weight']:.6g} | "
            f"{row['calibration_seed']} | {row['grid_size']} |"
        )

    lines.extend(
        [
            "",
            "## Formal resource totals",
            "",
            "Times are summed once per formal training run; peak memory is the",
            "maximum observed allocation for that method.",
            "",
            "| Method | Runs | Preparation (s) | Training (s) | Evaluation (s) | Peak GPU GiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in timings:
        lines.append(
            f"| {row['method']} | {row['runs']} | "
            f"{row['preparation_seconds']:.2f} | {row['training_seconds']:.2f} | "
            f"{row['evaluation_seconds']:.2f} | {row['peak_gpu_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Method-native settings and every compatibility adaptation are frozen",
            "in `rebuttal/BASELINE_OFA_REPROTOCOL.md`; dense/sparse and deterministic",
            "equivalence evidence is recorded in the Phase 2 gate report.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(output_root: Path, dataset_dir: Path) -> dict[str, Any]:
    specs = build_manifest()
    validate_manifest(specs)
    records, audit = audit_runs(output_root, dataset_dir, specs)
    analysis_root = output_root / "analysis"
    atomic_json(analysis_root / "artifact_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(
            f"Phase 2 artifact audit failed with {len(audit['problems'])} problems"
        )
    dataset_rows, macro_rows = aggregate(records)
    recap_macros = _phase1_recap_macros(PHASE1_MACRO_PATH)
    calibrations = _calibrations(output_root)
    timings = _timings(records)
    atomic_json(analysis_root / "raw_records.json", records)
    _csv(analysis_root / "raw_records.csv", records)
    atomic_json(analysis_root / "dataset_summary.json", dataset_rows)
    _csv(analysis_root / "dataset_summary.csv", dataset_rows)
    atomic_json(analysis_root / "macro_summary.json", macro_rows)
    _csv(
        analysis_root / "macro_summary.csv",
        [{k: v for k, v in row.items() if k != "seed_values"} for row in macro_rows],
    )
    report = render_report(
        dataset_rows,
        macro_rows,
        audit,
        recap_macros,
        calibrations,
        timings,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    payload = {
        "audit": audit,
        "dataset_rows": len(dataset_rows),
        "macro_rows": len(macro_rows),
        "report": str(REPORT_PATH),
    }
    atomic_json(analysis_root / "analysis_complete.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--dataset-dir", default="/root/autodl-tmp/recap/dataset"
    )
    args = parser.parse_args()
    result = analyze(Path(args.output_root), Path(args.dataset_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
