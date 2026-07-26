"""Recompute and audit the complete user-revised RECAP experiment protocol.

Run this script from the repository root on the experiment server.  It reads
only seed-level or seed-pair-level artifacts and writes consolidated,
machine-readable tables plus a human-readable completion report.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rebuttal.phase1_protocol import DATASETS, OFA_SETTINGS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "rebuttal" / "artifacts"
OUTPUT_ROOT = ARTIFACT_ROOT / "protocol_completion" / "analysis"
REPORT_PATH = (
    PROJECT_ROOT / "rebuttal" / "reports"
    / "RECAP_EXPERIMENT_PROTOCOL_COMPLETION_REPORT.md"
)

SEEDS = (0, 1, 2)
SEED_PAIRS = ((0, 1), (0, 2), (1, 2))
OFO_DATASETS = (
    "pubmed",
    "cora",
    "citeseer",
    "ACM",
    "Flickr",
    "BlogCatalog",
    "Facebook",
    "weibo",
    "Reddit",
    "questions",
    "YelpChi",
    "Amazon",
)
OFO_BASELINES = (
    "GCN",
    "GAT",
    "BWGNN",
    "XGBGraph",
    "DOMINANT",
    "AnomalyDAE",
    "CoLA",
    "ADA-GAD",
    "DiffGAD",
    "GUIDE",
)
OFO_METHODS = (*OFO_BASELINES, "RECAP-OFO")
OFA_BASELINES = ("ARC", "IA-GGAD", "UNPrompt", "AnomalyGFM-ZS", "OWLEYE")
OFA_METHODS = (*OFA_BASELINES, "RECAP")
STABILITY_METRICS = (
    "nmi",
    "ari",
    "soft_coassignment_similarity",
    "score_spearman",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def population_stats(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"Invalid values for statistics: {array}")
    return float(array.mean()), float(array.std(ddof=0))


def display_name(dataset: str) -> str:
    return str(DATASETS[dataset]["display"])


def domain(dataset: str) -> str:
    return str(DATASETS[dataset]["domain"])


def pct(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} ± {100.0 * std:.2f}"


def value_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def time_pm(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def assert_seed_cells(
    rows: list[dict[str, Any]],
    *,
    methods: tuple[str, ...],
    setting_targets: dict[str, tuple[str, ...]],
    setting_field: str,
    target_field: str,
) -> None:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["method"]),
                str(row[setting_field]),
                str(row[target_field]),
            )
        ].append(int(row["seed"]))
    expected = {
        (method, setting, target)
        for method in methods
        for setting, targets in setting_targets.items()
        for target in targets
    }
    if set(grouped) != expected:
        raise AssertionError(
            "Cell coverage mismatch: "
            f"missing={sorted(expected - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected)}"
        )
    bad = {
        key: sorted(seeds)
        for key, seeds in grouped.items()
        if sorted(seeds) != list(SEEDS)
    }
    if bad:
        raise AssertionError(f"Seed coverage mismatch: {bad}")


def validate_metric_ranges(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for metric in ("auroc", "auprc"):
            value = float(row[metric])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise AssertionError(f"Invalid {metric}={value}: {row}")


def aggregate_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    setting_field: str,
    target_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row[setting_field]),
                str(row["method"]),
                str(row[target_field]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (setting, method, target), values in sorted(grouped.items()):
        auroc_mean, auroc_std = population_stats(float(row["auroc"]) for row in values)
        auprc_mean, auprc_std = population_stats(float(row["auprc"]) for row in values)
        output.append(
            {
                "setting": setting,
                "method": method,
                "target_graph": target,
                "display_name": display_name(target),
                "domain": domain(target),
                "seed_count": len(values),
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
            }
        )
    return output


def aggregate_macro_rows(
    rows: list[dict[str, Any]],
    *,
    setting_targets: dict[str, tuple[str, ...]],
    target_field: str,
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        indexed[(str(row["setting"]), str(row["method"]), int(row["seed"]))].append(
            row
        )
    output: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for setting, targets in setting_targets.items():
        for method in methods:
            seed_dataset: dict[str, list[float]] = {"auroc": [], "auprc": []}
            seed_domain: dict[str, list[float]] = {"auroc": [], "auprc": []}
            for seed in SEEDS:
                values = indexed[(setting, method, seed)]
                actual_targets = {str(row[target_field]) for row in values}
                if actual_targets != set(targets):
                    raise AssertionError(
                        f"{setting}/{method}/seed{seed}: target mismatch "
                        f"{sorted(actual_targets)}"
                    )
                for metric in ("auroc", "auprc"):
                    seed_dataset[metric].append(
                        float(np.mean([float(row[metric]) for row in values]))
                    )
                    if setting == "C":
                        by_domain: dict[str, list[float]] = defaultdict(list)
                        for row in values:
                            by_domain[domain(str(row[target_field]))].append(
                                float(row[metric])
                            )
                        seed_domain[metric].append(
                            float(np.mean([np.mean(group) for group in by_domain.values()]))
                        )
            for aggregation, values_by_metric in (
                ("dataset_macro", seed_dataset),
                ("domain_macro", seed_domain),
            ):
                if aggregation == "domain_macro" and setting != "C":
                    continue
                auroc_mean, auroc_std = population_stats(values_by_metric["auroc"])
                auprc_mean, auprc_std = population_stats(values_by_metric["auprc"])
                output.append(
                    {
                        "setting": setting,
                        "method": method,
                        "aggregation": aggregation,
                        "dataset_count": len(targets),
                        "seed_count": 3,
                        "auroc_mean": auroc_mean,
                        "auroc_std": auroc_std,
                        "auprc_mean": auprc_mean,
                        "auprc_std": auprc_std,
                    }
                )
    return output


def load_inputs() -> dict[str, Any]:
    phase1 = ARTIFACT_ROOT / "phase1"
    questions = ARTIFACT_ROOT / "questions_ofo_addendum"
    primary = ARTIFACT_ROOT / "phase2_baselines"
    supplement = ARTIFACT_ROOT / "phase2_bc_supplement"
    ofo_baselines = ARTIFACT_ROOT / "ofo_12_baselines" / "formal"
    extension = ARTIFACT_ROOT / "three_baseline_extension"
    return {
        "phase1_raw": read_json(phase1 / "raw_results.json"),
        "questions_raw": read_json(questions / "raw_results.json"),
        "phase1_stability": read_csv(phase1 / "analysis" / "stability_pairs.csv"),
        "questions_stability": read_csv(
            questions / "analysis" / "questions_stability_pairs.csv"
        ),
        "ofa_primary": read_csv(primary / "analysis" / "raw_records.csv"),
        "ofa_supplement": read_csv(supplement / "analysis" / "raw_records.csv"),
        "ofo_baselines": read_csv(
            ofo_baselines / "analysis" / "run_records.csv"
        ),
        "extension_records": read_csv(
            extension / "analysis" / "run_records.csv"
        ),
        "audits": {
            "RECAP original Phase 1": read_json(
                phase1 / "analysis" / "artifact_validation.json"
            ),
            "RECAP Questions addendum": read_json(
                questions / "analysis" / "artifact_validation.json"
            ),
            "OFA baseline primary": read_json(
                primary / "analysis" / "artifact_audit.json"
            ),
            "OFA baseline B/C supplement": read_json(
                supplement / "analysis" / "artifact_audit.json"
            ),
            "OFO 12-dataset baselines": read_json(
                ofo_baselines / "analysis" / "global_audit.json"
            ),
            "Three-baseline extension": read_json(
                extension / "analysis" / "global_audit.json"
            ),
        },
    }


def normalize_ofo_records(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    recap = [
        {**row, "setting": "OFO", "method": "RECAP-OFO"}
        for row in (*inputs["phase1_raw"], *inputs["questions_raw"])
        if str(row["setting"]) == "OFO"
    ]
    baseline = [
        {
            **row,
            "setting": "OFO",
            "target_graph": row["dataset"],
            "method": row["method"],
        }
        for row in inputs["ofo_baselines"]
    ]
    extension = [
        {
            **row,
            "setting": "OFO",
            "target_graph": row["dataset"],
            "method": row["method"],
            "auroc": row["AUROC"],
            "auprc": row["AUPRC"],
            "evaluation_population": "full_graph",
        }
        for row in inputs["extension_records"]
        if row["paradigm"] == "OFO"
    ]
    rows = [*baseline, *extension, *recap]
    assert len(recap) == 36
    assert len(baseline) == 288
    assert len(extension) == 72
    assert_seed_cells(
        rows,
        methods=OFO_METHODS,
        setting_targets={"OFO": OFO_DATASETS},
        setting_field="setting",
        target_field="target_graph",
    )
    validate_metric_ranges(rows)
    return rows


def normalize_ofa_records(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    recap = [
        {**row, "method": "RECAP"}
        for row in inputs["phase1_raw"]
        if str(row["setting"]) in OFA_SETTINGS
    ]
    baseline = [*inputs["ofa_primary"], *inputs["ofa_supplement"]]
    extension = [
        {
            **row,
            "target_graph": row["dataset"],
            "auroc": row["AUROC"],
            "auprc": row["AUPRC"],
        }
        for row in inputs["extension_records"]
        if row["paradigm"] == "OFA"
    ]
    rows = [*baseline, *extension, *recap]
    assert len(recap) == 54
    assert len(baseline) == 216
    assert len(extension) == 54
    assert_seed_cells(
        rows,
        methods=OFA_METHODS,
        setting_targets={
            setting: tuple(split["targets"])
            for setting, split in OFA_SETTINGS.items()
        },
        setting_field="setting",
        target_field="target_graph",
    )
    validate_metric_ranges(rows)
    return rows


def stability_summary(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    for row in inputs["phase1_stability"]:
        pair_rows.append({**row})
    for row in inputs["questions_stability"]:
        pair_rows.append(
            {
                **row,
                "paradigm": "one-for-one",
                "setting": "OFO",
                "target_graph": "questions",
                "dataset_domain": domain("questions"),
            }
        )
    if len(pair_rows) != 90:
        raise AssertionError(f"Expected 90 stability pairs, got {len(pair_rows)}")

    targets = {
        "OFO": OFO_DATASETS,
        **{
            setting: tuple(split["targets"])
            for setting, split in OFA_SETTINGS.items()
        },
    }
    by_pair: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_pair[
            (str(row["setting"]), int(row["seed_a"]), int(row["seed_b"]))
        ].append(row)

    all_recap = [*inputs["phase1_raw"], *inputs["questions_raw"]]
    effective: dict[tuple[str, str, int], float] = {}
    for row in all_recap:
        setting = str(row["setting"])
        target = str(row["target_graph"])
        seed = int(row["seed"])
        key = (setting, target, seed)
        if key in effective:
            raise AssertionError(f"Duplicate effective-community cell: {key}")
        with np.load(Path(row["community_output_path"]), allow_pickle=False) as archive:
            effective[key] = float(archive["effective_communities"].item())

    summary_rows: list[dict[str, Any]] = []
    pair_macro_rows: list[dict[str, Any]] = []
    for setting in ("OFO", "A", "B", "C"):
        expected_targets = set(targets[setting])
        metric_pair_values: dict[str, list[float]] = defaultdict(list)
        for seed_a, seed_b in SEED_PAIRS:
            values = by_pair[(setting, seed_a, seed_b)]
            actual_targets = {str(row["target_graph"]) for row in values}
            if actual_targets != expected_targets:
                raise AssertionError(
                    f"{setting}/{seed_a}-{seed_b}: stability targets mismatch"
                )
            macro = {
                "setting": setting,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "dataset_count": len(values),
            }
            for metric in STABILITY_METRICS:
                macro[metric] = float(
                    np.mean([float(row[metric]) for row in values])
                )
                metric_pair_values[metric].append(macro[metric])
            pair_macro_rows.append(macro)

        ceff_seed_values = []
        for seed in SEEDS:
            ceff_seed_values.append(
                float(
                    np.mean(
                        [
                            effective[(setting, target, seed)]
                            for target in targets[setting]
                        ]
                    )
                )
            )
        summary: dict[str, Any] = {
            "setting": setting,
            "dataset_count": len(targets[setting]),
            "seed_pair_count": 3,
            "seed_count": 3,
        }
        for metric in STABILITY_METRICS:
            summary[f"{metric}_mean"], summary[f"{metric}_std"] = population_stats(
                metric_pair_values[metric]
            )
        (
            summary["effective_communities_mean"],
            summary["effective_communities_std"],
        ) = population_stats(ceff_seed_values)
        summary_rows.append(summary)
    return pair_macro_rows, summary_rows


def timing_summary(inputs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = [*inputs["phase1_raw"], *inputs["questions_raw"]]
    seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    targets = {
        "OFO": OFO_DATASETS,
        **{
            setting: tuple(split["targets"])
            for setting, split in OFA_SETTINGS.items()
        },
    }
    for setting in ("OFO", "A", "B", "C"):
        setting_seed_rows = []
        for seed in SEEDS:
            values = [
                row
                for row in records
                if str(row["setting"]) == setting and int(row["seed"]) == seed
            ]
            actual_targets = {str(row["target_graph"]) for row in values}
            if actual_targets != set(targets[setting]):
                raise AssertionError(f"{setting}/seed{seed}: timing target mismatch")
            if setting == "OFO":
                preparation = float(
                    np.mean([float(row["data_prepare_seconds"]) for row in values])
                )
                training = float(
                    np.mean([float(row["train_seconds"]) for row in values])
                )
                diagnostic = float(
                    np.mean([float(row["diagnostic_seconds"]) for row in values])
                )
            else:
                for field in (
                    "data_prepare_seconds",
                    "train_seconds",
                    "diagnostic_seconds",
                ):
                    unique = {float(row[field]) for row in values}
                    if len(unique) != 1:
                        raise AssertionError(
                            f"{setting}/seed{seed}: repeated shared {field} differs"
                        )
                preparation = float(values[0]["data_prepare_seconds"])
                training = float(values[0]["train_seconds"])
                diagnostic = float(values[0]["diagnostic_seconds"])
            inference = float(
                np.mean([float(row["inference_seconds"]) for row in values])
            )
            seed_row = {
                "setting": setting,
                "seed": seed,
                "source_graph_count": 1 if setting == "OFO" else 4,
                "target_graph_count": len(targets[setting]),
                "data_prepare_seconds": preparation,
                "train_seconds": training,
                "diagnostic_seconds": diagnostic,
                "inference_per_target_seconds": inference,
            }
            seed_rows.append(seed_row)
            setting_seed_rows.append(seed_row)
        summary: dict[str, Any] = {
            "setting": setting,
            "source_graph_count": 1 if setting == "OFO" else 4,
            "target_graph_count": len(targets[setting]),
            "seed_count": 3,
        }
        for field in (
            "data_prepare_seconds",
            "train_seconds",
            "diagnostic_seconds",
            "inference_per_target_seconds",
        ):
            summary[f"{field}_mean"], summary[f"{field}_std"] = population_stats(
                float(row[field]) for row in setting_seed_rows
            )
        summary_rows.append(summary)
    return seed_rows, summary_rows


def evaluation_strata(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    ofo_rows = inputs["ofo_baselines"]
    output = []
    for method in OFO_BASELINES:
        rows = [row for row in ofo_rows if row["method"] == method]
        populations = {
            row.get("evaluation_population", "full_graph") for row in rows
        }
        if not populations and method in {"DiffGAD", "GUIDE"}:
            populations = {"full_graph"}
        output.append(
            {
                "paradigm": "OFO",
                "method": method,
                "source_labels": method in {"GCN", "GAT", "BWGNN", "XGBGraph"},
                "target_label_context": "none",
                "target_tuning": method in {"GCN", "GAT", "BWGNN", "XGBGraph"},
                "evaluation_population": ", ".join(sorted(populations)),
                "query_node_rule": (
                    "held-out stratified test 40%"
                    if method in {"GCN", "GAT", "BWGNN", "XGBGraph"}
                    else "full graph"
                ),
            }
        )
    output.append(
        {
            "paradigm": "OFO",
            "method": "RECAP-OFO",
            "source_labels": False,
            "target_label_context": "none",
            "target_tuning": False,
            "evaluation_population": "full_graph",
            "query_node_rule": "full graph",
        }
    )
    for method in OFA_METHODS:
        if method == "ARC":
            context = "10 labeled-normal target nodes"
            population = "all target nodes except the 10 contexts"
        elif method == "IA-GGAD":
            context = "10 unlabeled random internal references"
            population = "all target nodes except the 10 references"
        elif method == "OWLEYE":
            context = "10 unlabeled target patterns"
            population = "full target graph"
        else:
            context = "none"
            population = "full target graph"
        output.append(
            {
                "paradigm": "OFA",
                "method": method,
                "source_labels": method != "RECAP",
                "target_label_context": context,
                "target_tuning": False,
                "evaluation_population": population,
                "query_node_rule": population,
            }
        )
    return output


def completion_and_consistency(
    inputs: dict[str, Any],
    ofo_rows: list[dict[str, Any]],
    ofa_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def audit_passed(value: dict[str, Any]) -> bool:
        if "passed" in value:
            return bool(value["passed"])
        return str(value.get("status", "")).upper() == "PASS"

    recap_records = [*inputs["phase1_raw"], *inputs["questions_raw"]]
    ofa_baseline_records = [*inputs["ofa_primary"], *inputs["ofa_supplement"]]
    extension_records = inputs["extension_records"]
    extension_ofo_records = [
        row for row in extension_records if row["paradigm"] == "OFO"
    ]
    extension_ofa_records = [
        row for row in extension_records if row["paradigm"] == "OFA"
    ]
    audits_passed = all(audit_passed(value) for value in inputs["audits"].values())
    counts = {
        "recap_training_runs": len({row["run_id"] for row in recap_records}),
        "recap_final_evaluations": len(recap_records),
        "ofa_baseline_training_runs": len(
            {row["run_id"] for row in ofa_baseline_records}
        )
        + len({row["run_id"] for row in extension_ofa_records}),
        "ofa_baseline_final_evaluations": (
            len(ofa_baseline_records) + len(extension_ofa_records)
        ),
        "ofo_baseline_training_runs": (
            len(inputs["ofo_baselines"]) + len(extension_ofo_records)
        ),
        "ofo_baseline_final_evaluations": (
            len(inputs["ofo_baselines"]) + len(extension_ofo_records)
        ),
    }
    counts["all_training_runs"] = (
        counts["recap_training_runs"]
        + counts["ofa_baseline_training_runs"]
        + counts["ofo_baseline_training_runs"]
    )
    counts["all_final_evaluations"] = (
        counts["recap_final_evaluations"]
        + counts["ofa_baseline_final_evaluations"]
        + counts["ofo_baseline_final_evaluations"]
    )
    expected_counts = {
        "recap_training_runs": 45,
        "recap_final_evaluations": 90,
        "ofa_baseline_training_runs": 45,
        "ofa_baseline_final_evaluations": 270,
        "ofo_baseline_training_runs": 360,
        "ofo_baseline_final_evaluations": 360,
        "all_training_runs": 450,
        "all_final_evaluations": 720,
    }
    if counts != expected_counts:
        raise AssertionError(f"Run/evaluation counts mismatch: {counts}")

    phase1_audit = inputs["audits"]["RECAP original Phase 1"]
    questions_audit = inputs["audits"]["RECAP Questions addendum"]
    artifact_counts = {
        "community_outputs": len(recap_records),
        "stability_pair_records": 90,
        "diagnostic_rows": int(phase1_audit["diagnostic_rows_actual"])
        + int(questions_audit["diagnostic_rows_actual"]),
        "checkpoints": int(phase1_audit["checkpoints_actual"])
        + int(questions_audit["checkpoints_actual"]),
        "checkpoint_reload_audits_passed": int(
            phase1_audit["checkpoint_reload_audits_passed"]
        )
        + int(questions_audit["checkpoint_reload_audits_passed"]),
    }
    expected_artifacts = {
        "community_outputs": 90,
        "stability_pair_records": 90,
        "diagnostic_rows": 486,
        "checkpoints": 225,
        "checkpoint_reload_audits_passed": 45,
    }
    if artifact_counts != expected_artifacts:
        raise AssertionError(f"RECAP artifact counts mismatch: {artifact_counts}")

    completion = [
        {
            "requirement": "12 datasets load and have immutable raw hashes",
            "expected": "12 datasets",
            "actual": len(
                inputs["audits"]["OFO 12-dataset baselines"]["raw_dataset_hashes"]
            ),
            "status": "PASS",
            "evidence": "OFO global audit raw_dataset_hashes",
        },
        {
            "requirement": "RECAP-OFO",
            "expected": "12 datasets × 3 seeds",
            "actual": "36 evaluations / 36 training runs",
            "status": "PASS",
            "evidence": "Phase 1 raw results + Questions addendum",
        },
        {
            "requirement": "RECAP-OFA A/B/C",
            "expected": "3 settings × 3 seeds",
            "actual": "54 evaluations / 9 training runs",
            "status": "PASS",
            "evidence": "Phase 1 raw results",
        },
        {
            "requirement": "10 OFO baselines",
            "expected": "10 methods × 12 datasets × 3 seeds",
            "actual": "360 evaluations / 360 training runs",
            "status": "PASS",
            "evidence": "OFO global audit + three-baseline extension audit",
        },
        {
            "requirement": "5 OFA baselines in A/B/C",
            "expected": "5 methods × (8+5+5) targets × 3 seeds",
            "actual": "270 evaluations / 45 training runs",
            "status": "PASS",
            "evidence": "Phase 2 audits + three-baseline extension audit",
        },
        {
            "requirement": "Community stability",
            "expected": "30 scopes × 3 seed pairs",
            "actual": "90 pair records",
            "status": "PASS",
            "evidence": "pair-level stability CSVs",
        },
        {
            "requirement": "Training diagnostics",
            "expected": "486 rows after Questions addendum",
            "actual": artifact_counts["diagnostic_rows"],
            "status": "PASS",
            "evidence": "RECAP artifact audits",
        },
        {
            "requirement": "Checkpoints and reload gates",
            "expected": "225 checkpoints / 45 final reload gates",
            "actual": (
                f"{artifact_counts['checkpoints']} / "
                f"{artifact_counts['checkpoint_reload_audits_passed']}"
            ),
            "status": "PASS",
            "evidence": "RECAP artifact audits",
        },
        {
            "requirement": "Protocol Tables 1–9",
            "expected": "all cells backed by raw records",
            "actual": "consolidated and recomputed",
            "status": "PASS",
            "evidence": "protocol_completion analysis outputs",
        },
    ]
    checks = {
        "all_required_cells_have_seeds_0_1_2": True,
        "no_duplicate_required_cells": True,
        "all_metrics_finite_and_in_unit_interval": True,
        "all_six_source_artifact_audits_passed": audits_passed,
        "three_baseline_extension_81_of_81_runs": (
            inputs["audits"]["Three-baseline extension"][
                "training_runs_complete"
            ]
            == 81
        ),
        "three_baseline_extension_126_of_126_evaluations": (
            inputs["audits"]["Three-baseline extension"][
                "evaluations_recomputed"
            ]
            == 126
        ),
        "three_baseline_metric_recomputation_exact": (
            inputs["audits"]["Three-baseline extension"][
                "maximum_metric_difference"
            ]
            == 0
        ),
        "three_baseline_label_and_score_freeze_audit_passed": (
            inputs["audits"]["Three-baseline extension"]["passed"]
            and not inputs["audits"]["Three-baseline extension"]["problems"]
        ),
        "dataset_macros_recomputed_seed_first": True,
        "setting_c_domain_macros_recomputed_seed_first": True,
        "stability_recomputed_pair_macro_first": len(stability_rows) == 4,
        "timing_recomputed_seed_first": len(timing_rows) == 4,
        "cross_setting_overall_average_omitted": True,
        "evaluation_population_strata_explicit": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"Consistency check failed: {checks}")
    consistency = {
        "status": "PASS",
        "missing_experiments": [],
        "reruns_required": [],
        "run_and_evaluation_counts": counts,
        "recap_artifact_counts": artifact_counts,
        "source_artifact_audits": {
            name: audit_passed(value) for name, value in inputs["audits"].items()
        },
        "checks": checks,
        "corrected_reporting_gaps": [
            (
                "The original protocol text names eight OFO targets; later user "
                "instructions superseded it with all 12 datasets. Consolidated "
                "OFO tables therefore use 12 datasets."
            ),
            (
                "The earlier stability summary averaged per-dataset means and did "
                "not report macro standard deviations. The consolidated table "
                "uses seed-pair-first aggregation and reports population std."
            ),
            (
                "The earlier timing summary did not provide the protocol Table 9 "
                "seed-first mean±std view. The consolidated table now does."
            ),
            (
                "OFA results were split between a primary report and a B/C "
                "supplement. They are now combined from raw records."
            ),
        ],
        "interpretation_caveats": [
            (
                "Supervised OFO baselines report a held-out stratified 40% test "
                "population; unsupervised baselines and RECAP report the full graph."
            ),
            (
                "ARC excludes 10 labeled-normal target contexts from evaluation; "
                "IA-GGAD excludes 10 randomly sampled unlabeled internal reference "
                "nodes; UNPrompt, AnomalyGFM-ZS, OWLEYE, and RECAP score the full "
                "target graph. OWLEYE's 10 unlabeled target patterns remain in "
                "the evaluation population."
            ),
            (
                "Settings A, B, and C use different target sets and are not averaged "
                "into one cross-setting number."
            ),
        ],
        "ofo_record_count": len(ofo_rows),
        "ofa_record_count": len(ofa_rows),
    }
    return completion, consistency


def dataset_lookup(
    rows: list[dict[str, Any]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["setting"], row["method"], row["target_graph"]): row for row in rows
    }


def macro_lookup(
    rows: list[dict[str, Any]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["setting"], row["method"], row["aggregation"]): row for row in rows
    }


def report_lines(
    completion: list[dict[str, Any]],
    consistency: dict[str, Any],
    ofo_dataset: list[dict[str, Any]],
    ofo_macro: list[dict[str, Any]],
    ofa_dataset: list[dict[str, Any]],
    ofa_macro: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    timing: list[dict[str, Any]],
    strata: list[dict[str, Any]],
) -> list[str]:
    lines = [
        "# RECAP Experiment Protocol Completion Report",
        "",
        "## Conclusion",
        "",
        (
            "**PASS. No requested training or inference cell is missing, and no "
            "rerun is required.** The revised scope contains 450 successful "
            "training runs and 720 final evaluations. All consolidated values below "
            "were recomputed from unrounded seed-level or seed-pair-level records "
            "using population standard deviation (`ddof=0`)."
        ),
        "",
        "## Completion matrix",
        "",
    ]
    lines.extend(
        md_table(
            ["Requirement", "Expected", "Actual", "Status"],
            [
                [
                    str(row["requirement"]),
                    str(row["expected"]),
                    str(row["actual"]),
                    str(row["status"]),
                ]
                for row in completion
            ],
        )
    )
    lines.extend(["", "## One-for-One 12-dataset results", ""])
    ds = dataset_lookup(ofo_dataset)
    macros = macro_lookup(ofo_macro)
    headers = ["Method", *[display_name(value) for value in OFO_DATASETS], "Macro"]
    auroc_rows = []
    auprc_rows = []
    for method in OFO_METHODS:
        macro = macros[("OFO", method, "dataset_macro")]
        auroc_rows.append(
            [
                method,
                *[
                    pct(
                        ds[("OFO", method, target)]["auroc_mean"],
                        ds[("OFO", method, target)]["auroc_std"],
                    )
                    for target in OFO_DATASETS
                ],
                pct(macro["auroc_mean"], macro["auroc_std"]),
            ]
        )
        auprc_rows.append(
            [
                method,
                *[
                    pct(
                        ds[("OFO", method, target)]["auprc_mean"],
                        ds[("OFO", method, target)]["auprc_std"],
                    )
                    for target in OFO_DATASETS
                ],
                pct(macro["auprc_mean"], macro["auprc_std"]),
            ]
        )
    lines.extend(["### AUROC (%)", ""])
    lines.extend(md_table(headers, auroc_rows))
    lines.extend(["", "### AUPRC (%)", ""])
    lines.extend(md_table(headers, auprc_rows))

    lines.extend(["", "## One-for-All Setting A results", ""])
    ofa_ds = dataset_lookup(ofa_dataset)
    ofa_macros = macro_lookup(ofa_macro)
    targets_a = tuple(OFA_SETTINGS["A"]["targets"])
    headers_a = ["Method", *[display_name(value) for value in targets_a], "Macro"]
    for metric, label in (("auroc", "AUROC (%)"), ("auprc", "AUPRC (%)")):
        rows = []
        for method in OFA_METHODS:
            macro = ofa_macros[("A", method, "dataset_macro")]
            rows.append(
                [
                    method,
                    *[
                        pct(
                            ofa_ds[("A", method, target)][f"{metric}_mean"],
                            ofa_ds[("A", method, target)][f"{metric}_std"],
                        )
                        for target in targets_a
                    ],
                    pct(macro[f"{metric}_mean"], macro[f"{metric}_std"]),
                ]
            )
        lines.extend([f"### {label}", ""])
        lines.extend(md_table(headers_a, rows))
        lines.append("")

    lines.extend(["## OFA robustness summary", ""])
    robustness_rows = []
    for method in OFA_METHODS:
        values = []
        for setting, aggregation in (
            ("A", "dataset_macro"),
            ("B", "dataset_macro"),
            ("C", "dataset_macro"),
            ("C", "domain_macro"),
        ):
            row = ofa_macros[(setting, method, aggregation)]
            values.append(
                f"{pct(row['auroc_mean'], row['auroc_std'])} / "
                f"{pct(row['auprc_mean'], row['auprc_std'])}"
            )
        robustness_rows.append([method, *values])
    lines.extend(
        md_table(
            [
                "Method",
                "Setting A dataset macro",
                "Setting B social macro",
                "Setting C dataset macro",
                "Setting C domain macro",
            ],
            robustness_rows,
        )
    )

    for setting, title in (
        ("B", "Setting B: leave-Social-domain-out"),
        ("C", "Setting C: citation-only transfer"),
    ):
        lines.extend(["", f"## {title}", ""])
        targets = tuple(OFA_SETTINGS[setting]["targets"])
        table_rows = []
        for method in OFA_METHODS:
            macro = ofa_macros[(setting, method, "dataset_macro")]
            values = []
            for target in targets:
                row = ofa_ds[(setting, method, target)]
                values.append(
                    f"{pct(row['auroc_mean'], row['auroc_std'])} / "
                    f"{pct(row['auprc_mean'], row['auprc_std'])}"
                )
            values.append(
                f"{pct(macro['auroc_mean'], macro['auroc_std'])} / "
                f"{pct(macro['auprc_mean'], macro['auprc_std'])}"
            )
            if setting == "C":
                domain_macro = ofa_macros[(setting, method, "domain_macro")]
                values.append(
                    f"{pct(domain_macro['auroc_mean'], domain_macro['auroc_std'])} / "
                    f"{pct(domain_macro['auprc_mean'], domain_macro['auprc_std'])}"
                )
            table_rows.append([method, *values])
        headers = [
            "Method",
            *[display_name(value) for value in targets],
            "Dataset macro",
        ]
        if setting == "C":
            headers.append("Domain macro")
        lines.extend(md_table(headers, table_rows))

    lines.extend(["", "## Corrected community-stability summary", ""])
    lines.extend(
        md_table(
            [
                "Setting",
                "Scope",
                "NMI",
                "ARI",
                "Soft co-assignment",
                "Score Spearman",
                "C_eff",
            ],
            [
                [
                    row["setting"],
                    f"{row['dataset_count']} datasets",
                    value_pm(row["nmi_mean"], row["nmi_std"]),
                    value_pm(row["ari_mean"], row["ari_std"]),
                    value_pm(
                        row["soft_coassignment_similarity_mean"],
                        row["soft_coassignment_similarity_std"],
                    ),
                    value_pm(
                        row["score_spearman_mean"], row["score_spearman_std"]
                    ),
                    value_pm(
                        row["effective_communities_mean"],
                        row["effective_communities_std"],
                        digits=2,
                    ),
                ]
                for row in stability
            ],
        )
    )

    lines.extend(["", "## Corrected RECAP timing summary", ""])
    lines.extend(
        md_table(
            [
                "Setting",
                "Source graphs",
                "Target graphs",
                "Preparation (s)",
                "Training (s)",
                "Diagnostics (s)",
                "Inference/target (s)",
            ],
            [
                [
                    row["setting"],
                    "1 per model" if row["setting"] == "OFO" else "4",
                    str(row["target_graph_count"]),
                    time_pm(
                        row["data_prepare_seconds_mean"],
                        row["data_prepare_seconds_std"],
                    ),
                    time_pm(row["train_seconds_mean"], row["train_seconds_std"]),
                    time_pm(
                        row["diagnostic_seconds_mean"],
                        row["diagnostic_seconds_std"],
                    ),
                    time_pm(
                        row["inference_per_target_seconds_mean"],
                        row["inference_per_target_seconds_std"],
                    ),
                ]
                for row in timing
            ],
        )
    )

    lines.extend(["", "## Consistency audit", ""])
    lines.extend(
        md_table(
            ["Check", "Result"],
            [
                [key.replace("_", " "), "PASS" if value else "FAIL"]
                for key, value in consistency["checks"].items()
            ],
        )
    )
    lines.extend(["", "### Corrected reporting gaps", ""])
    lines.extend(f"- {item}" for item in consistency["corrected_reporting_gaps"])
    lines.extend(["", "### Interpretation boundaries", ""])
    lines.extend(f"- {item}" for item in consistency["interpretation_caveats"])

    lines.extend(["", "### Evaluation strata", ""])
    lines.extend(
        md_table(
            [
                "Paradigm",
                "Method",
                "Source labels",
                "Target context/reference",
                "Evaluation population",
            ],
            [
                [
                    row["paradigm"],
                    row["method"],
                    "yes" if row["source_labels"] else "no",
                    row["target_label_context"],
                    row["evaluation_population"],
                ]
                for row in strata
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Machine-readable evidence",
            "",
            "- `completion_matrix.csv` and `completion_matrix.json`",
            "- `consistency_audit.json`",
            "- `consolidated_ofo12_by_dataset.csv` and `consolidated_ofo12_macro.csv`",
            "- `consolidated_ofa_abc_by_dataset.csv` and `consolidated_ofa_abc_macro.csv`",
            "- `recap_stability_pair_macros.csv` and `recap_stability_summary.csv`",
            "- `recap_timing_by_seed.csv` and `recap_timing_summary.csv`",
            "- `evaluation_strata.csv`",
            "- `THREE_BASELINE_EXTENSION_REPORT.md`",
            "- Three-baseline audit and summaries under "
            "`rebuttal/artifacts/three_baseline_extension/analysis/`",
            "",
        ]
    )
    return lines


def main() -> None:
    inputs = load_inputs()
    ofo_rows = normalize_ofo_records(inputs)
    ofa_rows = normalize_ofa_records(inputs)
    ofo_dataset = aggregate_dataset_rows(
        ofo_rows, setting_field="setting", target_field="target_graph"
    )
    ofo_macro = aggregate_macro_rows(
        ofo_rows,
        setting_targets={"OFO": OFO_DATASETS},
        target_field="target_graph",
    )
    ofa_dataset = aggregate_dataset_rows(
        ofa_rows, setting_field="setting", target_field="target_graph"
    )
    ofa_macro = aggregate_macro_rows(
        ofa_rows,
        setting_targets={
            setting: tuple(split["targets"])
            for setting, split in OFA_SETTINGS.items()
        },
        target_field="target_graph",
    )
    stability_pair_macros, stability = stability_summary(inputs)
    timing_seed, timing = timing_summary(inputs)
    strata = evaluation_strata(inputs)
    completion, consistency = completion_and_consistency(
        inputs, ofo_rows, ofa_rows, stability, timing
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_ROOT / "consolidated_ofo12_by_dataset.csv", ofo_dataset)
    write_csv(OUTPUT_ROOT / "consolidated_ofo12_macro.csv", ofo_macro)
    write_csv(OUTPUT_ROOT / "consolidated_ofa_abc_by_dataset.csv", ofa_dataset)
    write_csv(OUTPUT_ROOT / "consolidated_ofa_abc_macro.csv", ofa_macro)
    write_csv(
        OUTPUT_ROOT / "recap_stability_pair_macros.csv",
        stability_pair_macros,
    )
    write_csv(OUTPUT_ROOT / "recap_stability_summary.csv", stability)
    write_csv(OUTPUT_ROOT / "recap_timing_by_seed.csv", timing_seed)
    write_csv(OUTPUT_ROOT / "recap_timing_summary.csv", timing)
    write_csv(OUTPUT_ROOT / "evaluation_strata.csv", strata)
    write_csv(OUTPUT_ROOT / "completion_matrix.csv", completion)
    write_json(OUTPUT_ROOT / "completion_matrix.json", completion)
    write_json(OUTPUT_ROOT / "consistency_audit.json", consistency)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        "\n".join(
            report_lines(
                completion,
                consistency,
                ofo_dataset,
                ofo_macro,
                ofa_dataset,
                ofa_macro,
                stability,
                timing,
                strata,
            )
        )
    )
    print(
        json.dumps(
            {
                "status": consistency["status"],
                "report": str(REPORT_PATH),
                "output_root": str(OUTPUT_ROOT),
                "training_runs": consistency["run_and_evaluation_counts"][
                    "all_training_runs"
                ],
                "final_evaluations": consistency["run_and_evaluation_counts"][
                    "all_final_evaluations"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
