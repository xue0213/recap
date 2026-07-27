"""Independent hash, freeze-order, completeness, and metric audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.large_target_inference.common import (  # noqa: E402
    atomic_json,
    sha256_file,
    utc_now,
)


TARGETS = ("tfinance", "dgraphfin", "tsocial")


def _labels(dataset_root: Path, target: str) -> tuple[np.ndarray, np.ndarray]:
    base = dataset_root / target
    labels = np.asarray(np.load(base / "labels.npy"), dtype=np.int64).reshape(-1)
    mask_path = base / "evaluation_mask.npy"
    mask = (
        np.asarray(np.load(mask_path), dtype=np.bool_).reshape(-1)
        if mask_path.exists()
        else np.ones(len(labels), dtype=np.bool_)
    )
    return labels, mask


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_source_scan(
    root: Path, dataset_root: Path, target: str
) -> dict:
    base = root / "source_scan" / target
    with (base / "all_scores_frozen.json").open(
        "r", encoding="utf-8"
    ) as handle:
        frozen = json.load(handle)
    records = list(frozen["records"]) + list(frozen.get("ensembles", []))
    if len(frozen["records"]) != 45 or len(frozen.get("ensembles", [])) != 9:
        raise AssertionError(f"{target}: incomplete source scan")
    if frozen.get("all_labels_accessed_before_freeze") is not False:
        raise AssertionError(f"{target}: invalid global freeze declaration")
    score_hashes = 0
    for record in records:
        if record.get("labels_accessed") is not False:
            raise AssertionError(f"{target}: score record accessed labels")
        for route, path in record["score_paths"].items():
            if sha256_file(Path(path)) != record["score_sha256"][route]:
                raise AssertionError(f"{target}: score hash mismatch {path}")
            score_hashes += 1
    rows = _read_csv(base / "per_checkpoint_results.csv")
    if len(rows) != len(records) * 3:
        raise AssertionError(f"{target}: result row mismatch")
    labels, mask = _labels(dataset_root, target)
    by_id = {record["checkpoint_id"]: record for record in records}
    max_metric_difference = 0.0
    for row in rows:
        record = by_id[row["checkpoint_id"]]
        score = np.load(record["score_paths"][row["route"]], mmap_mode="r")
        auroc = float(roc_auc_score(labels[mask], score[mask]))
        auprc = float(average_precision_score(labels[mask], score[mask]))
        max_metric_difference = max(
            max_metric_difference,
            abs(auroc - float(row["auroc"])),
            abs(auprc - float(row["auprc"])),
        )
    if max_metric_difference > 1e-12:
        raise AssertionError(f"{target}: metric recomputation mismatch")
    return {
        "target": target,
        "checkpoint_records": 45,
        "ensemble_records": 9,
        "score_hashes": score_hashes,
        "metric_rows": len(rows),
        "max_metric_difference": max_metric_difference,
        "passed": True,
    }


def audit_three_seed_result(
    base: Path, dataset_root: Path, target: str
) -> dict:
    with (base / "all_scores_frozen.json").open(
        "r", encoding="utf-8"
    ) as handle:
        frozen = json.load(handle)
    records = frozen["records"]
    if len(records) != 3:
        raise AssertionError(f"{base}: expected three frozen seeds")
    if frozen.get("labels_accessed_before_global_freeze") is not False:
        raise AssertionError(f"{base}: invalid freeze declaration")
    rows = _read_csv(base / "results.csv")
    if len(rows) != 9:
        raise AssertionError(f"{base}: expected nine metric rows")
    labels, mask = _labels(dataset_root, target)
    max_metric_difference = 0.0
    score_hashes = 0
    for seed, record in enumerate(records):
        for route, path in record["score_paths"].items():
            if sha256_file(Path(path)) != record["score_sha256"][route]:
                raise AssertionError(f"{base}: score hash mismatch")
            row = next(
                item
                for item in rows
                if int(item["seed"]) == seed and item["route"] == route
            )
            score = np.load(path, mmap_mode="r")
            auroc = float(roc_auc_score(labels[mask], score[mask]))
            auprc = float(average_precision_score(labels[mask], score[mask]))
            max_metric_difference = max(
                max_metric_difference,
                abs(auroc - float(row["auroc"])),
                abs(auprc - float(row["auprc"])),
            )
            score_hashes += 1
    if max_metric_difference > 1e-12:
        raise AssertionError(f"{base}: metric recomputation mismatch")
    return {
        "target": target,
        "base": str(base.resolve()),
        "score_hashes": score_hashes,
        "metric_rows": len(rows),
        "max_metric_difference": max_metric_difference,
        "passed": True,
    }


def run(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    report = {
        "format": "recap_large_target_optimization_audit_v1",
        "source_scans": [],
        "target_adaptation": [],
        "shared_transfer": None,
        "created_at": utc_now(),
    }
    for target in TARGETS:
        report["source_scans"].append(
            audit_source_scan(root, dataset_root, target)
        )
        report["target_adaptation"].append(
            audit_three_seed_result(
                root / "target_adapt" / target, dataset_root, target
            )
        )
    shared = root / "shared_transfer"
    report["shared_transfer"] = audit_three_seed_result(
        shared, dataset_root, "tsocial"
    )
    report["passed"] = True
    atomic_json(root / "independent_audit.json", report)
    print(json.dumps(report, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--root", required=True)
    value.add_argument("--dataset-root", required=True)
    return value


if __name__ == "__main__":
    run(parser().parse_args())

