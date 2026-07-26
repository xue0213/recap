"""Independent artifact audit and report generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
from sklearn.metrics import average_precision_score, roc_auc_score

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from rebuttal.new_baselines.common import (  # type: ignore
        DEFAULT_DATASET_DIR,
        DEFAULT_OUTPUT_ROOT,
        atomic_json,
        dataset_path,
        sha256_array,
    )
    from rebuttal.new_baselines.protocol import (  # type: ignore
        SEEDS,
        ExtensionRunSpec,
        build_manifest,
        expected_evaluations,
    )
else:
    from .common import (
        DEFAULT_DATASET_DIR,
        DEFAULT_OUTPUT_ROOT,
        atomic_json,
        dataset_path,
        sha256_array,
    )
    from .protocol import SEEDS, ExtensionRunSpec, build_manifest, expected_evaluations


DISPLAY = {
    "pubmed": "PubMed",
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "ACM": "ACM",
    "Flickr": "Flickr",
    "BlogCatalog": "BlogCatalog",
    "Facebook": "Facebook",
    "weibo": "Weibo",
    "Reddit": "Reddit",
    "questions": "Questions",
    "YelpChi": "YelpChi",
    "Amazon": "Amazon",
}
OFO_ORDER = tuple(DISPLAY)
DOMAIN = {
    "pubmed": "Citation",
    "cora": "Citation",
    "citeseer": "Citation",
    "ACM": "Citation",
    "Flickr": "Social",
    "BlogCatalog": "Social",
    "Facebook": "Social",
    "weibo": "Social",
    "Reddit": "Social",
    "questions": "Q&A",
    "YelpChi": "E-commerce",
    "Amazon": "E-commerce",
}


def _labels(dataset_dir: Path, name: str) -> np.ndarray:
    raw = sio.loadmat(
        dataset_path(dataset_dir, name), variable_names=["Label", "gnd"]
    )
    value = raw["Label"] if "Label" in raw else raw["gnd"]
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _format(values: list[float], scale: float = 100.0) -> str:
    mean, std = _stats(values)
    return f"{scale * mean:.2f} ± {scale * std:.2f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def audit(
    *,
    output_root: Path,
    dataset_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = build_manifest()
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    maximum_metric_difference = 0.0
    maximum_reload_difference = 0.0
    label_events = 0
    score_files = 0
    for spec in specs:
        directory = output_root / "runs" / spec.run_id
        complete_path = directory / "complete.json"
        if not complete_path.exists():
            problems.append(f"missing complete: {spec.run_id}")
            continue
        complete = json.loads(complete_path.read_text())
        if complete.get("smoke"):
            problems.append(f"formal run marked smoke: {spec.run_id}")
        audit_value = complete["label_audit"]
        label_events += len(audit_value["events"])
        if not audit_value["passed"] or audit_value["invalid_events"]:
            problems.append(f"label audit failed: {spec.run_id}")
        if spec.dataset is not None:
            score_path = Path(complete["score_path"])
            with np.load(score_path, allow_pickle=False) as archive:
                scores = np.asarray(archive["scores"], dtype=np.float32)
                mask = np.asarray(archive["evaluation_mask"], dtype=np.bool_)
            score_files += 1
            labels = _labels(dataset_dir, spec.dataset)
            recomputed = {
                "AUROC": float(roc_auc_score(labels[mask], scores[mask])),
                "AUPRC": float(
                    average_precision_score(labels[mask], scores[mask])
                ),
            }
            for metric in recomputed:
                maximum_metric_difference = max(
                    maximum_metric_difference,
                    abs(recomputed[metric] - complete["metrics"][metric]),
                )
            if sha256_array(scores) != complete["score_sha256"]:
                problems.append(f"score hash mismatch: {spec.run_id}")
            maximum_reload_difference = max(
                maximum_reload_difference,
                float(complete["reload_max_abs_difference"]),
            )
            records.append(
                {
                    "run_id": spec.run_id,
                    "method": spec.method,
                    "paradigm": "OFO",
                    "setting": "",
                    "dataset": spec.dataset,
                    "seed": spec.seed,
                    **recomputed,
                    "nodes": int(labels.shape[0]),
                    "evaluation_nodes": int(mask.sum()),
                    "preprocessing_seconds": complete["timing_seconds"][
                        "preprocessing"
                    ],
                    "training_seconds": complete["timing_seconds"]["training"],
                    "inference_seconds": complete["timing_seconds"]["inference"],
                    "total_seconds": complete["timing_seconds"]["total"],
                    "gpu_allocated_mb": complete["resources"]["allocated_mb"],
                    "gpu_reserved_mb": complete["resources"]["reserved_mb"],
                    "peak_rss_mb": complete["resources"]["peak_rss_mb"],
                }
            )
        else:
            for name in spec.target_graphs:
                score_path = directory / "scores" / f"{name}.npz"
                with np.load(score_path, allow_pickle=False) as archive:
                    scores = np.asarray(archive["scores"], dtype=np.float32)
                    mask = np.asarray(
                        archive["evaluation_mask"], dtype=np.bool_
                    )
                    indices = np.asarray(
                        archive["target_pattern_indices"], dtype=np.int64
                    )
                score_files += 1
                if len(indices) != 10 or mask.sum() != len(mask):
                    problems.append(
                        f"OWLEYE population/pattern mismatch: {spec.run_id}/{name}"
                    )
                labels = _labels(dataset_dir, name)
                recomputed = {
                    "AUROC": float(roc_auc_score(labels[mask], scores[mask])),
                    "AUPRC": float(
                        average_precision_score(labels[mask], scores[mask])
                    ),
                }
                stored = complete["target_results"][name]
                for metric in recomputed:
                    maximum_metric_difference = max(
                        maximum_metric_difference,
                        abs(recomputed[metric] - stored[metric]),
                    )
                if sha256_array(scores) != stored["score_sha256"]:
                    problems.append(
                        f"score hash mismatch: {spec.run_id}/{name}"
                    )
                reload_value = complete["reload"][name]
                maximum_reload_difference = max(
                    maximum_reload_difference,
                    float(reload_value["max_abs_difference"]),
                )
                records.append(
                    {
                        "run_id": spec.run_id,
                        "method": spec.method,
                        "paradigm": "OFA",
                        "setting": spec.setting,
                        "dataset": name,
                        "seed": spec.seed,
                        **recomputed,
                        "nodes": int(labels.shape[0]),
                        "evaluation_nodes": int(mask.sum()),
                        "preprocessing_seconds": complete["timing_seconds"][
                            "preprocessing"
                        ],
                        "training_seconds": complete["timing_seconds"][
                            "training"
                        ],
                        "inference_seconds": complete["timing_seconds"][
                            "inference"
                        ],
                        "total_seconds": complete["timing_seconds"]["total"],
                        "gpu_allocated_mb": complete["resources"]["allocated_mb"],
                        "gpu_reserved_mb": complete["resources"]["reserved_mb"],
                        "peak_rss_mb": complete["resources"]["peak_rss_mb"],
                    }
                )
    expected_runs = len(specs)
    if len(records) != expected_evaluations():
        problems.append(
            f"record count {len(records)} != {expected_evaluations()}"
        )
    if maximum_metric_difference > 1e-12:
        problems.append(
            f"metric recomputation difference {maximum_metric_difference}"
        )
    keys = {
        (record["method"], record["setting"], record["dataset"], record["seed"])
        for record in records
    }
    if len(keys) != len(records):
        problems.append("duplicate evaluation cell")
    for method in ("DiffGAD", "GUIDE"):
        for dataset in OFO_ORDER:
            found = {
                record["seed"]
                for record in records
                if record["method"] == method and record["dataset"] == dataset
            }
            if found != set(SEEDS):
                problems.append(f"seed coverage {method}/{dataset}: {found}")
    for setting in ("A", "B", "C"):
        setting_specs = [
            spec for spec in specs if spec.method == "OWLEYE" and spec.setting == setting
        ]
        targets = setting_specs[0].target_graphs
        for dataset in targets:
            found = {
                record["seed"]
                for record in records
                if record["method"] == "OWLEYE"
                and record["setting"] == setting
                and record["dataset"] == dataset
            }
            if found != set(SEEDS):
                problems.append(
                    f"seed coverage OWLEYE/{setting}/{dataset}: {found}"
                )
    report = {
        "format": "recap_three_baseline_global_audit_v1",
        "passed": not problems,
        "training_runs_expected": expected_runs,
        "training_runs_complete": sum(
            (
                output_root
                / "runs"
                / spec.run_id
                / "complete.json"
            ).exists()
            for spec in specs
        ),
        "evaluations_expected": expected_evaluations(),
        "evaluations_recomputed": len(records),
        "score_files_verified": score_files,
        "label_events_checked": label_events,
        "maximum_metric_difference": maximum_metric_difference,
        "maximum_reload_difference": maximum_reload_difference,
        "problems": problems,
    }
    return records, report


def summarize(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_rows = []
    for key in sorted(
        {
            (
                record["method"],
                record["paradigm"],
                record["setting"],
                record["dataset"],
            )
            for record in records
        }
    ):
        method, paradigm, setting, dataset = key
        values = [
            record
            for record in records
            if (
                record["method"],
                record["paradigm"],
                record["setting"],
                record["dataset"],
            )
            == key
        ]
        auroc_mean, auroc_std = _stats([value["AUROC"] for value in values])
        auprc_mean, auprc_std = _stats([value["AUPRC"] for value in values])
        dataset_rows.append(
            {
                "method": method,
                "paradigm": paradigm,
                "setting": setting,
                "dataset": dataset,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
                "seeds": len(values),
            }
        )

    macro_rows = []
    for method, paradigm, setting in sorted(
        {
            (record["method"], record["paradigm"], record["setting"])
            for record in records
        }
    ):
        selected = [
            record
            for record in records
            if (
                record["method"],
                record["paradigm"],
                record["setting"],
            )
            == (method, paradigm, setting)
        ]
        per_seed = []
        for seed in SEEDS:
            seed_values = [record for record in selected if record["seed"] == seed]
            per_seed.append(
                {
                    "seed": seed,
                    "AUROC": float(
                        np.mean([value["AUROC"] for value in seed_values])
                    ),
                    "AUPRC": float(
                        np.mean([value["AUPRC"] for value in seed_values])
                    ),
                }
            )
        auroc_mean, auroc_std = _stats(
            [value["AUROC"] for value in per_seed]
        )
        auprc_mean, auprc_std = _stats(
            [value["AUPRC"] for value in per_seed]
        )
        macro_rows.append(
            {
                "method": method,
                "paradigm": paradigm,
                "setting": setting,
                "macro_kind": "dataset",
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
            }
        )
        if method == "OWLEYE" and setting == "C":
            domain_seed_values = []
            for seed in SEEDS:
                by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for value in selected:
                    if value["seed"] == seed:
                        by_domain[DOMAIN[value["dataset"]]].append(value)
                domain_seed_values.append(
                    {
                        "AUROC": float(
                            np.mean(
                                [
                                    np.mean([item["AUROC"] for item in group])
                                    for group in by_domain.values()
                                ]
                            )
                        ),
                        "AUPRC": float(
                            np.mean(
                                [
                                    np.mean([item["AUPRC"] for item in group])
                                    for group in by_domain.values()
                                ]
                            )
                        ),
                    }
                )
            domain_auroc_mean, domain_auroc_std = _stats(
                [value["AUROC"] for value in domain_seed_values]
            )
            domain_auprc_mean, domain_auprc_std = _stats(
                [value["AUPRC"] for value in domain_seed_values]
            )
            macro_rows.append(
                {
                    "method": method,
                    "paradigm": paradigm,
                    "setting": setting,
                    "macro_kind": "domain",
                    "auroc_mean": domain_auroc_mean,
                    "auroc_std": domain_auroc_std,
                    "auprc_mean": domain_auprc_mean,
                    "auprc_std": domain_auprc_std,
                }
            )

    timing_rows = []
    for method in ("DiffGAD", "GUIDE", "OWLEYE"):
        run_ids = sorted(
            {record["run_id"] for record in records if record["method"] == method}
        )
        run_records = [
            next(record for record in records if record["run_id"] == run_id)
            for run_id in run_ids
        ]
        timing_rows.append(
            {
                "method": method,
                "runs": len(run_ids),
                "preprocessing_sum_seconds": sum(
                    value["preprocessing_seconds"] for value in run_records
                ),
                "training_sum_seconds": sum(
                    value["training_seconds"] for value in run_records
                ),
                "inference_sum_seconds": sum(
                    value["inference_seconds"] for value in run_records
                ),
                "total_sum_seconds": sum(
                    value["total_seconds"] for value in run_records
                ),
                "mean_total_seconds": float(
                    np.mean([value["total_seconds"] for value in run_records])
                ),
                "max_gpu_allocated_gib": max(
                    value["gpu_allocated_mb"] for value in run_records
                )
                / 1024.0,
                "max_gpu_reserved_gib": max(
                    value["gpu_reserved_mb"] for value in run_records
                )
                / 1024.0,
                "max_peak_rss_gib": max(
                    value["peak_rss_mb"] for value in run_records
                )
                / 1024.0,
            }
        )
    return dataset_rows, macro_rows, timing_rows


def make_report(
    *,
    dataset_rows: list[dict[str, Any]],
    macro_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    audit_report: dict[str, Any],
) -> str:
    by_key = {
        (row["method"], row["setting"], row["dataset"]): row
        for row in dataset_rows
    }
    macros = {
        (row["method"], row["setting"], row["macro_kind"]): row
        for row in macro_rows
    }
    lines = [
        "# RECAP Three-Baseline Extension Report",
        "",
        "Status: **PASS**",
        "",
        "DiffGAD and GUIDE are unsupervised one-for-one methods evaluated on "
        "the full graph. OWLEYE is source-label supervised and target-label-free "
        "zero-shot; its ten uniformly sampled unlabeled target pattern nodes "
        "remain in the full-target evaluation population.",
        "",
        "All values are mean ± population standard deviation over seeds 0/1/2, "
        "in percent. Dataset macros are computed within each seed before the "
        "three-seed mean and standard deviation.",
        "",
        "## OFO AUROC (%)",
        "",
        "| Method | "
        + " | ".join(DISPLAY[name] for name in OFO_ORDER)
        + " | Macro |",
        "|" + "---|" * (len(OFO_ORDER) + 2),
    ]
    for method in ("DiffGAD", "GUIDE"):
        cells = []
        for dataset in OFO_ORDER:
            row = by_key[(method, "", dataset)]
            cells.append(
                f"{100*row['auroc_mean']:.2f} ± {100*row['auroc_std']:.2f}"
            )
        macro = macros[(method, "", "dataset")]
        cells.append(
            f"{100*macro['auroc_mean']:.2f} ± {100*macro['auroc_std']:.2f}"
        )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## OFO AUPRC (%)",
            "",
            "| Method | "
            + " | ".join(DISPLAY[name] for name in OFO_ORDER)
            + " | Macro |",
            "|" + "---|" * (len(OFO_ORDER) + 2),
        ]
    )
    for method in ("DiffGAD", "GUIDE"):
        cells = []
        for dataset in OFO_ORDER:
            row = by_key[(method, "", dataset)]
            cells.append(
                f"{100*row['auprc_mean']:.2f} ± {100*row['auprc_std']:.2f}"
            )
        macro = macros[(method, "", "dataset")]
        cells.append(
            f"{100*macro['auprc_mean']:.2f} ± {100*macro['auprc_std']:.2f}"
        )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")

    setting_targets = {
        "A": (
            "cora",
            "citeseer",
            "ACM",
            "BlogCatalog",
            "Facebook",
            "weibo",
            "Reddit",
            "Amazon",
        ),
        "B": ("Flickr", "BlogCatalog", "Facebook", "weibo", "Reddit"),
        "C": ("BlogCatalog", "Flickr", "Reddit", "Amazon", "questions"),
    }
    for setting, targets in setting_targets.items():
        lines.extend(
            [
                "",
                f"## OWLEYE Setting {setting} (AUROC / AUPRC, %)",
                "",
                "| Method | "
                + " | ".join(DISPLAY[name] for name in targets)
                + " | Dataset macro"
                + (" | Domain macro" if setting == "C" else "")
                + " |",
                "|"
                + "---|"
                * (
                    len(targets)
                    + 2
                    + (1 if setting == "C" else 0)
                ),
            ]
        )
        cells = []
        for dataset in targets:
            row = by_key[("OWLEYE", setting, dataset)]
            cells.append(
                f"{100*row['auroc_mean']:.2f} ± {100*row['auroc_std']:.2f} / "
                f"{100*row['auprc_mean']:.2f} ± {100*row['auprc_std']:.2f}"
            )
        macro = macros[("OWLEYE", setting, "dataset")]
        cells.append(
            f"{100*macro['auroc_mean']:.2f} ± {100*macro['auroc_std']:.2f} / "
            f"{100*macro['auprc_mean']:.2f} ± {100*macro['auprc_std']:.2f}"
        )
        if setting == "C":
            domain = macros[("OWLEYE", setting, "domain")]
            cells.append(
                f"{100*domain['auroc_mean']:.2f} ± {100*domain['auroc_std']:.2f} / "
                f"{100*domain['auprc_mean']:.2f} ± {100*domain['auprc_std']:.2f}"
            )
        lines.append("| OWLEYE | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Timing and resources",
            "",
            "| Method | Runs | Preprocess sum (s) | Train sum (s) | "
            "Inference sum (s) | Total sum (s) | Mean/run (s) | "
            "Max GPU allocated (GiB) | Peak RSS (GiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in timing_rows:
        lines.append(
            f"| {row['method']} | {row['runs']} | "
            f"{row['preprocessing_sum_seconds']:.2f} | "
            f"{row['training_sum_seconds']:.2f} | "
            f"{row['inference_sum_seconds']:.2f} | "
            f"{row['total_sum_seconds']:.2f} | "
            f"{row['mean_total_seconds']:.2f} | "
            f"{row['max_gpu_allocated_gib']:.2f} | "
            f"{row['max_peak_rss_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Independent audit",
            "",
            f"- Training runs: {audit_report['training_runs_complete']}/"
            f"{audit_report['training_runs_expected']}",
            f"- Recomputed evaluations: {audit_report['evaluations_recomputed']}/"
            f"{audit_report['evaluations_expected']}",
            f"- Frozen score files verified: {audit_report['score_files_verified']}",
            f"- Label-audit events checked: {audit_report['label_events_checked']}",
            f"- Maximum metric recomputation difference: "
            f"{audit_report['maximum_metric_difference']:.3g}",
            f"- Maximum checkpoint reload score difference: "
            f"{audit_report['maximum_reload_difference']:.9g}",
            f"- Problems: {len(audit_report['problems'])}",
            "",
            "## Fidelity notes",
            "",
            "- DiffGAD removes the released target-label selection over "
            "autoencoder trials and 500 diffusion levels. It uses the locked "
            "ten-level label-free average and exact non-quadratic structure loss.",
            "- GUIDE uses exact ORCA order-four node orbits. The mapping was "
            "validated against independent induced-subgraph enumeration.",
            "- OWLEYE uses source labels and must not be described as fully "
            "unsupervised. Target labels are unavailable until every target's "
            "full-node score vector is frozen.",
            "- No weak or high-variance result was replaced, tuned, or selectively "
            "rerun.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "reports"
            / "THREE_BASELINE_EXTENSION_REPORT.md"
        ),
    )
    args = parser.parse_args()
    records, audit_report = audit(
        output_root=args.output_root, dataset_dir=args.dataset_dir
    )
    dataset_rows, macro_rows, timing_rows = summarize(records)
    analysis_dir = args.output_root / "analysis"
    _write_csv(analysis_dir / "run_records.csv", records)
    _write_csv(analysis_dir / "dataset_summary.csv", dataset_rows)
    _write_csv(analysis_dir / "macro_summary.csv", macro_rows)
    _write_csv(analysis_dir / "timing_resource_summary.csv", timing_rows)
    atomic_json(analysis_dir / "global_audit.json", audit_report)
    if not audit_report["passed"]:
        raise SystemExit(json.dumps(audit_report, indent=2))
    report = make_report(
        dataset_rows=dataset_rows,
        macro_rows=macro_rows,
        timing_rows=timing_rows,
        audit_report=audit_report,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
