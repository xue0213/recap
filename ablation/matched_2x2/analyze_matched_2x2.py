#!/usr/bin/env python3
"""Merge, validate, and summarize the matched 2x2 ablation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SHARD_ROOT = ROOT / "ablation" / "matched_2x2" / "shards"
OUTPUT_ROOT = ROOT / "ablation" / "matched_2x2" / "results"
SIMPLE_FORMAL = ROOT / "ablation" / "simple_combine" / "results" / "simple_combine_raw.json"
MODULE_FORMAL = ROOT / "ablation" / "without_module" / "results" / "without_module_raw.json"

DATASETS = [
    "Facebook",
    "cora",
    "citeseer",
    "ACM",
    "BlogCatalog",
    "weibo",
    "Reddit",
    "Amazon",
]
SEEDS = list(range(5))
METHODS = [
    "Non-residual + KMeans",
    "Residual + KMeans",
    "RECAP w/o Residual",
    "RECAP",
]
EQUIVALENCE_TOLERANCE = 1e-5


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def index_records(records: list[dict]) -> dict[tuple[str, int, str], dict]:
    indexed = {}
    for row in records:
        key = (row["method"], int(row["trial"]), row["dataset"])
        if key in indexed:
            raise ValueError(f"duplicate record: {key}")
        indexed[key] = row
    return indexed


def validate_complete(indexed: dict[tuple[str, int, str], dict]) -> None:
    expected = {
        (method, seed, dataset)
        for method in METHODS
        for seed in SEEDS
        for dataset in DATASETS
    }
    observed = set(indexed)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(f"record scope mismatch: missing={missing}, extra={extra}")
    failed = [
        (key, row.get("status"), row.get("error"))
        for key, row in indexed.items()
        if row.get("status", "ok") != "ok"
    ]
    if failed:
        raise ValueError(f"failed records: {failed}")


def macro_rows(indexed: dict[tuple[str, int, str], dict]) -> list[dict]:
    rows = []
    for method in METHODS:
        seed_auroc = []
        seed_auprc = []
        for seed in SEEDS:
            auroc = [
                float(indexed[(method, seed, dataset)]["AUROC"])
                for dataset in DATASETS
            ]
            auprc = [
                float(indexed[(method, seed, dataset)]["AUPRC"])
                for dataset in DATASETS
            ]
            seed_auroc.append(float(np.mean(auroc)))
            seed_auprc.append(float(np.mean(auprc)))
        rows.append(
            {
                "method": method,
                "seeds": len(SEEDS),
                "targets": len(DATASETS),
                "AUROC_mean": float(np.mean(seed_auroc)),
                "AUROC_std": float(np.std(seed_auroc)),
                "AUPRC_mean": float(np.mean(seed_auprc)),
                "AUPRC_std": float(np.std(seed_auprc)),
            }
        )
    return rows


def dataset_rows(indexed: dict[tuple[str, int, str], dict]) -> list[dict]:
    rows = []
    for method in METHODS:
        for dataset in DATASETS:
            auroc = [
                float(indexed[(method, seed, dataset)]["AUROC"])
                for seed in SEEDS
            ]
            auprc = [
                float(indexed[(method, seed, dataset)]["AUPRC"])
                for seed in SEEDS
            ]
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "seeds": len(SEEDS),
                    "AUROC_mean": float(np.mean(auroc)),
                    "AUROC_std": float(np.std(auroc)),
                    "AUPRC_mean": float(np.mean(auprc)),
                    "AUPRC_std": float(np.std(auprc)),
                }
            )
    return rows


def paired_effect(
    indexed: dict[tuple[str, int, str], dict],
    high: str,
    low: str,
    label: str,
) -> dict:
    seed_differences = defaultdict(list)
    for seed in SEEDS:
        for metric in ("AUROC", "AUPRC"):
            high_values = [
                float(indexed[(high, seed, dataset)][metric])
                for dataset in DATASETS
            ]
            low_values = [
                float(indexed[(low, seed, dataset)][metric])
                for dataset in DATASETS
            ]
            seed_differences[metric].append(
                float(np.mean(high_values) - np.mean(low_values))
            )
    return {
        "effect": label,
        "high": high,
        "low": low,
        "AUROC_mean_difference": float(np.mean(seed_differences["AUROC"])),
        "AUROC_std_difference": float(np.std(seed_differences["AUROC"])),
        "AUPRC_mean_difference": float(np.mean(seed_differences["AUPRC"])),
        "AUPRC_std_difference": float(np.std(seed_differences["AUPRC"])),
    }


def interaction_effect(
    indexed: dict[tuple[str, int, str], dict],
) -> dict:
    differences = defaultdict(list)
    for seed in SEEDS:
        for metric in ("AUROC", "AUPRC"):
            macro = {}
            for method in METHODS:
                macro[method] = float(
                    np.mean(
                        [
                            float(indexed[(method, seed, dataset)][metric])
                            for dataset in DATASETS
                        ]
                    )
                )
            differences[metric].append(
                (macro["RECAP"] - macro["RECAP w/o Residual"])
                - (
                    macro["Residual + KMeans"]
                    - macro["Non-residual + KMeans"]
                )
            )
    return {
        "effect": "Difference-in-differences interaction",
        "high": "(RECAP - RECAP w/o Residual)",
        "low": "(Residual KMeans - Non-residual KMeans)",
        "AUROC_mean_difference": float(np.mean(differences["AUROC"])),
        "AUROC_std_difference": float(np.std(differences["AUROC"])),
        "AUPRC_mean_difference": float(np.mean(differences["AUPRC"])),
        "AUPRC_std_difference": float(np.std(differences["AUPRC"])),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    new_records = []
    shard_hashes = {}
    for dataset in DATASETS:
        path = SHARD_ROOT / dataset / "simple_combine_raw.json"
        payload = path.read_bytes()
        shard_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
        rows = read_json(path)["records"]
        new_records.extend(rows)

    new_index = index_records(new_records)
    expected_new = {
        (method, seed, dataset)
        for method in ("Non-residual + KMeans", "Residual + KMeans")
        for seed in SEEDS
        for dataset in DATASETS
    }
    if set(new_index) != expected_new:
        raise ValueError("new KMeans shard scope is incomplete")

    accepted_simple = index_records(read_json(SIMPLE_FORMAL)["records"])
    residual_differences = []
    for key in sorted(expected_new):
        if key[0] != "Residual + KMeans":
            continue
        for metric in ("AUROC", "AUPRC"):
            residual_differences.append(
                abs(float(new_index[key][metric]) - float(accepted_simple[key][metric]))
            )
    max_residual_difference = max(residual_differences, default=0.0)
    if max_residual_difference > EQUIVALENCE_TOLERANCE:
        raise ValueError(
            "Residual + KMeans equivalence gate failed: "
            f"max difference={max_residual_difference}"
        )

    module_index = index_records(read_json(MODULE_FORMAL)["records"])
    combined = {}
    for key, row in new_index.items():
        if key[0] == "Non-residual + KMeans":
            combined[key] = row
    for seed in SEEDS:
        for dataset in DATASETS:
            key = ("Residual + KMeans", seed, dataset)
            combined[key] = accepted_simple[key]
    for method in ("RECAP w/o Residual", "RECAP"):
        for seed in SEEDS:
            for dataset in DATASETS:
                key = (method, seed, dataset)
                combined[key] = module_index[key]

    validate_complete(combined)
    macros = macro_rows(combined)
    per_dataset = dataset_rows(combined)
    effects = [
        paired_effect(
            combined,
            "Residual + KMeans",
            "Non-residual + KMeans",
            "Residual representation effect with KMeans",
        ),
        paired_effect(
            combined,
            "RECAP",
            "RECAP w/o Residual",
            "Residual representation effect with RECAP",
        ),
        paired_effect(
            combined,
            "RECAP w/o Residual",
            "Non-residual + KMeans",
            "Community framework effect with non-residual features",
        ),
        paired_effect(
            combined,
            "RECAP",
            "Residual + KMeans",
            "Community framework effect with residual features",
        ),
    ]
    effects.append(interaction_effect(combined))

    write_csv(OUTPUT_ROOT / "matched_2x2_macro.csv", macros)
    write_csv(OUTPUT_ROOT / "matched_2x2_per_dataset.csv", per_dataset)
    write_csv(OUTPUT_ROOT / "matched_2x2_effects.csv", effects)

    validation = {
        "status": "pass",
        "datasets": DATASETS,
        "seeds": SEEDS,
        "methods": METHODS,
        "combined_record_count": len(combined),
        "new_record_count": len(new_index),
        "residual_kmeans_metric_values_compared": len(residual_differences),
        "residual_kmeans_max_abs_difference": max_residual_difference,
        "residual_kmeans_equivalence_tolerance": EQUIVALENCE_TOLERANCE,
        "residual_kmeans_final_source": str(SIMPLE_FORMAL.relative_to(ROOT)),
        "shard_sha256": shard_hashes,
    }
    (OUTPUT_ROOT / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    )

    lines = [
        "# Matched 2x2 Ablation",
        "",
        "| Representation | Conventional detector | RECAP community learning/scoring |",
        "|---|---:|---:|",
    ]
    macro_by_method = {row["method"]: row for row in macros}

    def cell(method: str) -> str:
        row = macro_by_method[method]
        return (
            f"{row['AUROC_mean']:.4f}±{row['AUROC_std']:.4f} / "
            f"{row['AUPRC_mean']:.4f}±{row['AUPRC_std']:.4f}"
        )

    lines.extend(
        [
            f"| Non-residual | {cell('Non-residual + KMeans')} | "
            f"{cell('RECAP w/o Residual')} |",
            f"| Residual | {cell('Residual + KMeans')} | {cell('RECAP')} |",
            "",
            "Values are five-seed dataset-macro AUROC/AUPRC over the eight "
            "Setting-A target graphs.",
            "",
            "## Paired macro effects",
            "",
            "| Effect | AUROC difference | AUPRC difference |",
            "|---|---:|---:|",
        ]
    )
    for row in effects:
        lines.append(
            f"| {row['effect']} | {row['AUROC_mean_difference']:+.4f} | "
            f"{row['AUPRC_mean_difference']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Complete records: {len(combined)}/160.",
            "- Failed runs: 0.",
            f"- Residual + KMeans maximum absolute difference versus the accepted "
            f"formal record: {max_residual_difference:.3g} "
            f"(tolerance: {EQUIVALENCE_TOLERANCE:.1g}).",
        ]
    )
    (OUTPUT_ROOT / "MATCHED_2X2_RESULTS.md").write_text("\n".join(lines) + "\n")

    print((OUTPUT_ROOT / "MATCHED_2X2_RESULTS.md").read_text())
    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
