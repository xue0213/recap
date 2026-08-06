"""Independent audit and aggregation for the locked 12-dataset OFO study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from rebuttal.baselines.baseline_protocol import DATASETS
from rebuttal.ofo_baselines.common import (
    load_labels,
    sha256_array,
    sha256_file,
)
from rebuttal.ofo_baselines.protocol import (
    DATASET_ORDER,
    METHODS,
    SEEDS,
    build_manifest,
    validate_manifest,
)


REBUTTAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = (
    REBUTTAL_ROOT / "artifacts" / "ofo_12_baselines" / "formal" / "runs"
)
DEFAULT_ANALYSIS_ROOT = (
    REBUTTAL_ROOT / "artifacts" / "ofo_12_baselines" / "formal" / "analysis"
)
DEFAULT_RECAP_DATASET_SUMMARY = (
    REBUTTAL_ROOT
    / "artifacts"
    / "questions_ofo_addendum"
    / "analysis"
    / "combined_ofo_12_metrics_by_dataset.csv"
)
DEFAULT_RECAP_MACRO_SUMMARY = (
    REBUTTAL_ROOT
    / "artifacts"
    / "questions_ofo_addendum"
    / "analysis"
    / "combined_ofo_12_summary.json"
)
DEFAULT_REPORT = REBUTTAL_ROOT / "reports" / "OFO_12_BASELINE_REPORT.md"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(materialized[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot aggregate an empty sequence")
    return float(array.mean()), float(array.std(ddof=0))


def validate_label_audit(path: Path, supervised: bool) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("format") != "recap_ofo12_label_audit_v1":
        raise ValueError(f"{path}: unexpected label-audit format")
    if not audit.get("passed") or audit.get("invalid_events"):
        raise ValueError(f"{path}: label audit did not pass")
    if bool(audit.get("supervised")) != supervised:
        raise ValueError(f"{path}: supervision flag mismatch")
    actions = [event["action"] for event in audit["events"]]
    if actions.count("freeze_scores") != 1:
        raise ValueError(f"{path}: expected exactly one score-freeze event")
    if actions.count("load_evaluation_labels") != 1:
        raise ValueError(f"{path}: expected exactly one evaluation-label event")
    if actions.index("freeze_scores") > actions.index("load_evaluation_labels"):
        raise ValueError(f"{path}: evaluation labels were read before score freeze")
    if supervised and actions[0] != "create_stratified_split":
        raise ValueError(f"{path}: supervised split audit is missing")
    if not supervised and "create_stratified_split" in actions:
        raise ValueError(f"{path}: unsupervised run created a supervised split")
    return audit


def validate_split(
    path: Path, labels: np.ndarray, query_mask: np.ndarray
) -> None:
    split = np.load(path)
    expected_keys = {"train", "validation", "test"}
    if set(split.files) != expected_keys:
        raise ValueError(f"{path}: split keys drifted: {split.files}")
    masks = {
        key: np.asarray(split[key], dtype=np.bool_) for key in expected_keys
    }
    if any(mask.shape != labels.shape for mask in masks.values()):
        raise ValueError(f"{path}: split shape mismatch")
    if not np.array_equal(masks["test"], query_mask):
        raise ValueError(f"{path}: frozen query mask is not the test split")
    total = sum(mask.astype(np.int8) for mask in masks.values())
    if not np.all(total == 1):
        raise ValueError(f"{path}: split does not partition every node exactly once")
    for key, mask in masks.items():
        if set(np.unique(labels[mask]).tolist()) != {0, 1}:
            raise ValueError(f"{path}: {key} does not contain both classes")
    fractions = {key: float(mask.mean()) for key, mask in masks.items()}
    if abs(fractions["train"] - 0.4) > 0.01:
        raise ValueError(f"{path}: training fraction drifted")
    if abs(fractions["validation"] - 0.2) > 0.01:
        raise ValueError(f"{path}: validation fraction drifted")
    if abs(fractions["test"] - 0.4) > 0.01:
        raise ValueError(f"{path}: test fraction drifted")


def audit_runs(
    run_root: Path, dataset_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = build_manifest()
    validate_manifest(manifest)
    expected_ids = {spec.run_id for spec in manifest}
    actual_ids = {
        path.name for path in run_root.iterdir() if path.is_dir()
    }
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"Formal run directory drift: missing={missing}, extra={extra}"
        )

    raw_hashes = {
        dataset: sha256_file(dataset_root / DATASETS[dataset]["file"])
        for dataset in DATASET_ORDER
    }
    records: list[dict[str, Any]] = []
    maximum_reload_difference = 0.0
    environment_signatures: set[tuple[str, ...]] = set()

    for spec in manifest:
        run_dir = run_root / spec.run_id
        required = {
            "_SUCCESS.json",
            "history.json",
            "label_audit.json",
            "resolved_config.json",
            "result.json",
            "scores.npz",
            "status.json",
        }
        missing_files = sorted(
            name for name in required if not (run_dir / name).exists()
        )
        if missing_files:
            raise ValueError(f"{spec.run_id}: missing {missing_files}")
        model_candidates = [run_dir / "checkpoint.pt", run_dir / "model.json"]
        if sum(path.exists() for path in model_candidates) != 1:
            raise ValueError(f"{spec.run_id}: expected exactly one saved model")

        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        success = json.loads(
            (run_dir / "_SUCCESS.json").read_text(encoding="utf-8")
        )
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if result.get("format") != "recap_ofo12_run_result_v1":
            raise ValueError(f"{spec.run_id}: result format drift")
        if result.get("status") != "complete" or status.get("status") != "complete":
            raise ValueError(f"{spec.run_id}: non-complete status")
        if result.get("smoke") or status.get("smoke"):
            raise ValueError(f"{spec.run_id}: smoke artifact in formal directory")
        for key, expected in (
            ("run_id", spec.run_id),
            ("method", spec.method),
            ("dataset", spec.dataset),
            ("seed", spec.seed),
            ("supervised", spec.supervised),
        ):
            if result.get(key) != expected:
                raise ValueError(
                    f"{spec.run_id}: {key}={result.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if result["raw_sha256"] != raw_hashes[spec.dataset]:
            raise ValueError(f"{spec.run_id}: raw data hash mismatch")
        if success.get("result_sha256") != sha256_file(run_dir / "result.json"):
            raise ValueError(f"{spec.run_id}: result success hash mismatch")

        frozen = np.load(run_dir / "scores.npz")
        if set(frozen.files) != {"scores", "query_mask"}:
            raise ValueError(f"{spec.run_id}: unexpected frozen arrays")
        scores = np.asarray(frozen["scores"], dtype=np.float32)
        query_mask = np.asarray(frozen["query_mask"], dtype=np.bool_)
        labels = load_labels(dataset_root, spec.dataset)
        expected_shape = (int(result["node_count"]),)
        if scores.shape != expected_shape or query_mask.shape != expected_shape:
            raise ValueError(f"{spec.run_id}: score/mask shape mismatch")
        if labels.shape != expected_shape:
            raise ValueError(f"{spec.run_id}: label shape mismatch")
        if not np.isfinite(scores).all():
            raise ValueError(f"{spec.run_id}: non-finite scores")
        if sha256_array(scores) != result["score_sha256"]:
            raise ValueError(f"{spec.run_id}: score hash mismatch")
        if sha256_array(query_mask) != result["query_mask_sha256"]:
            raise ValueError(f"{spec.run_id}: query-mask hash mismatch")
        if int(query_mask.sum()) != int(result["query_nodes"]):
            raise ValueError(f"{spec.run_id}: query-node count mismatch")

        label_audit = validate_label_audit(
            run_dir / "label_audit.json", spec.supervised
        )
        if label_audit["frozen"]["score_sha256"] != result["score_sha256"]:
            raise ValueError(f"{spec.run_id}: label-audit score hash mismatch")
        if (
            label_audit["frozen"]["query_mask_sha256"]
            != result["query_mask_sha256"]
        ):
            raise ValueError(
                f"{spec.run_id}: label-audit query-mask hash mismatch"
            )
        if spec.supervised:
            validate_split(run_dir / "split_masks.npz", labels, query_mask)
            evaluation_population = "stratified_test_40pct"
        else:
            if not query_mask.all():
                raise ValueError(f"{spec.run_id}: unsupervised query is not full graph")
            if (run_dir / "split_masks.npz").exists():
                raise ValueError(f"{spec.run_id}: unsupervised split file exists")
            evaluation_population = "full_graph"

        recomputed_auroc = float(
            roc_auc_score(labels[query_mask], scores[query_mask])
        )
        recomputed_auprc = float(
            average_precision_score(labels[query_mask], scores[query_mask])
        )
        if abs(recomputed_auroc - float(result["auroc"])) > 1e-12:
            raise ValueError(f"{spec.run_id}: AUROC recomputation mismatch")
        if abs(recomputed_auprc - float(result["auprc"])) > 1e-12:
            raise ValueError(f"{spec.run_id}: AUPRC recomputation mismatch")
        reload_difference = float(result["reload_max_abs_diff"])
        reload_tolerance = float(result["reload_tolerance"])
        if reload_difference > reload_tolerance:
            raise ValueError(f"{spec.run_id}: checkpoint reload audit failed")
        maximum_reload_difference = max(
            maximum_reload_difference, reload_difference
        )

        environment = result["environment"]
        environment_signatures.add(
            tuple(
                str(environment.get(key))
                for key in (
                    "python",
                    "torch",
                    "numpy",
                    "cuda_runtime",
                    "device",
                )
            )
        )
        records.append(
            {
                "run_id": spec.run_id,
                "method": spec.method,
                "supervised": spec.supervised,
                "dataset": spec.dataset,
                "display_name": DATASETS[spec.dataset]["display"],
                "domain": DATASETS[spec.dataset]["domain"],
                "seed": spec.seed,
                "evaluation_population": evaluation_population,
                "query_nodes": int(query_mask.sum()),
                "node_count": int(result["node_count"]),
                "auroc": recomputed_auroc,
                "auprc": recomputed_auprc,
                "train_seconds": float(result["train_seconds"]),
                "inference_seconds": float(result["inference_seconds"]),
                "total_seconds": float(result["total_seconds"]),
                "peak_gpu_allocated_bytes": int(
                    result["peak_gpu_allocated_bytes"]
                ),
                "peak_gpu_reserved_bytes": int(
                    result["peak_gpu_reserved_bytes"]
                ),
                "peak_process_rss_kib": int(result["peak_process_rss_kib"]),
                "reload_max_abs_diff": reload_difference,
                "reload_tolerance": reload_tolerance,
                "score_sha256": result["score_sha256"],
                "query_mask_sha256": result["query_mask_sha256"],
                "raw_sha256": result["raw_sha256"],
                "label_audit_passed": True,
                "checkpoint_reload_passed": True,
            }
        )

    if len(records) != 288:
        raise ValueError(f"Expected 288 audited records, found {len(records)}")
    audit = {
        "format": "recap_ofo12_global_audit_v1",
        "status": "PASS",
        "expected_runs": 288,
        "audited_runs": len(records),
        "recomputed_metrics": 2 * len(records),
        "label_audits_passed": len(records),
        "checkpoint_reloads_passed": len(records),
        "maximum_reload_abs_difference": maximum_reload_difference,
        "environment_signature_count": len(environment_signatures),
        "environment_signatures": sorted(environment_signatures),
        "raw_dataset_hashes": raw_hashes,
        "evaluation_populations": {
            "supervised": "stratified_test_40pct",
            "unsupervised": "full_graph",
        },
    }
    return records, audit


def aggregate_dataset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["method"], row["dataset"])].append(row)
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for dataset in DATASET_ORDER:
            rows = grouped[(method, dataset)]
            if {row["seed"] for row in rows} != set(SEEDS):
                raise ValueError(f"{method}/{dataset}: seed coverage drift")
            auroc_mean, auroc_std = mean_std(row["auroc"] for row in rows)
            auprc_mean, auprc_std = mean_std(row["auprc"] for row in rows)
            output.append(
                {
                    "method": method,
                    "supervised": rows[0]["supervised"],
                    "dataset": dataset,
                    "display_name": DATASETS[dataset]["display"],
                    "domain": DATASETS[dataset]["domain"],
                    "evaluation_population": rows[0]["evaluation_population"],
                    "seed_count": len(rows),
                    "auroc_mean": auroc_mean,
                    "auroc_std": auroc_std,
                    "auprc_mean": auprc_mean,
                    "auprc_std": auprc_std,
                }
            )
    return output


def aggregate_method(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in records if row["method"] == method]
        seed_macros = []
        for seed in SEEDS:
            seed_rows = [row for row in method_rows if row["seed"] == seed]
            if len(seed_rows) != len(DATASET_ORDER):
                raise ValueError(f"{method}/seed{seed}: dataset coverage drift")
            seed_macros.append(
                {
                    "seed": seed,
                    "auroc": float(np.mean([row["auroc"] for row in seed_rows])),
                    "auprc": float(np.mean([row["auprc"] for row in seed_rows])),
                }
            )
        auroc_mean, auroc_std = mean_std(row["auroc"] for row in seed_macros)
        auprc_mean, auprc_std = mean_std(row["auprc"] for row in seed_macros)
        output.append(
            {
                "method": method,
                "supervised": method_rows[0]["supervised"],
                "evaluation_population": method_rows[0]["evaluation_population"],
                "dataset_count": len(DATASET_ORDER),
                "seed_count": len(SEEDS),
                "run_count": len(method_rows),
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
                "seed_macros": seed_macros,
            }
        )
    return output


def aggregate_domain(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    domains = tuple(dict.fromkeys(DATASETS[name]["domain"] for name in DATASET_ORDER))
    for method in METHODS:
        for domain in domains:
            domain_seed_macros = []
            expected_datasets = {
                name
                for name in DATASET_ORDER
                if DATASETS[name]["domain"] == domain
            }
            for seed in SEEDS:
                rows = [
                    row
                    for row in records
                    if row["method"] == method
                    and row["domain"] == domain
                    and row["seed"] == seed
                ]
                if {row["dataset"] for row in rows} != expected_datasets:
                    raise ValueError(
                        f"{method}/{domain}/seed{seed}: dataset coverage drift"
                    )
                domain_seed_macros.append(
                    (
                        float(np.mean([row["auroc"] for row in rows])),
                        float(np.mean([row["auprc"] for row in rows])),
                    )
                )
            array = np.asarray(domain_seed_macros, dtype=np.float64)
            method_row = next(row for row in records if row["method"] == method)
            output.append(
                {
                    "method": method,
                    "supervised": method_row["supervised"],
                    "evaluation_population": method_row["evaluation_population"],
                    "domain": domain,
                    "dataset_count": len(expected_datasets),
                    "seed_count": len(SEEDS),
                    "auroc_mean": float(array[:, 0].mean()),
                    "auroc_std": float(array[:, 0].std(ddof=0)),
                    "auprc_mean": float(array[:, 1].mean()),
                    "auprc_std": float(array[:, 1].std(ddof=0)),
                }
            )
    return output


def aggregate_timing(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        output.append(
            {
                "method": method,
                "run_count": len(rows),
                "train_seconds_sum": float(
                    sum(row["train_seconds"] for row in rows)
                ),
                "inference_seconds_sum": float(
                    sum(row["inference_seconds"] for row in rows)
                ),
                "total_seconds_sum": float(
                    sum(row["total_seconds"] for row in rows)
                ),
                "total_seconds_mean_per_run": float(
                    np.mean([row["total_seconds"] for row in rows])
                ),
                "total_seconds_max_single_run": float(
                    max(row["total_seconds"] for row in rows)
                ),
                "peak_gpu_allocated_gib": float(
                    max(row["peak_gpu_allocated_bytes"] for row in rows) / 2**30
                ),
                "peak_gpu_reserved_gib": float(
                    max(row["peak_gpu_reserved_bytes"] for row in rows) / 2**30
                ),
                "peak_process_rss_gib": float(
                    max(row["peak_process_rss_kib"] for row in rows) / 2**20
                ),
            }
        )
    return output


def load_recap_comparison(
    dataset_path: Path, macro_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if not dataset_path.exists() or not macro_path.exists():
        return None
    dataset_rows = []
    for row in read_csv(dataset_path):
        dataset_rows.append(
            {
                "method": "RECAP-OFO",
                "supervised": False,
                "dataset": row["dataset"],
                "display_name": row["display_name"],
                "domain": DATASETS[row["dataset"]]["domain"],
                "evaluation_population": "full_graph",
                "seed_count": 3,
                "auroc_mean": float(row["auroc_mean"]),
                "auroc_std": float(row["auroc_std"]),
                "auprc_mean": float(row["auprc_mean"]),
                "auprc_std": float(row["auprc_std"]),
            }
        )
    if {row["dataset"] for row in dataset_rows} != set(DATASET_ORDER):
        raise ValueError("RECAP comparison does not cover all twelve datasets")
    macro = json.loads(macro_path.read_text(encoding="utf-8"))
    macro_row = {
        "method": "RECAP-OFO",
        "supervised": False,
        "evaluation_population": "full_graph",
        "dataset_count": 12,
        "seed_count": 3,
        "run_count": 36,
        "auroc_mean": float(macro["auroc_mean"]),
        "auroc_std": float(macro["auroc_std"]),
        "auprc_mean": float(macro["auprc_mean"]),
        "auprc_std": float(macro["auprc_std"]),
        "classification": macro["classification"],
    }
    return dataset_rows, macro_row


def percent_cell(mean: float, std: float) -> str:
    return f"{100 * mean:.2f} ± {100 * std:.2f}"


def markdown_matrix(
    dataset_rows: list[dict[str, Any]],
    methods: list[str],
    metric: str,
) -> list[str]:
    lookup = {
        (row["method"], row["dataset"]): row for row in dataset_rows
    }
    lines = [
        "| Method | " + " | ".join(DATASETS[name]["display"] for name in DATASET_ORDER) + " |",
        "|---|" + "|".join("---:" for _ in DATASET_ORDER) + "|",
    ]
    for method in methods:
        cells = [
            percent_cell(
                lookup[(method, dataset)][f"{metric}_mean"],
                lookup[(method, dataset)][f"{metric}_std"],
            )
            for dataset in DATASET_ORDER
        ]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return lines


def write_report(
    report_path: Path,
    audit: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]],
    recap: tuple[list[dict[str, Any]], dict[str, Any]] | None,
    analysis_root: Path,
) -> None:
    comparison_datasets = list(dataset_rows)
    comparison_methods = list(METHODS)
    comparison_macros = list(method_rows)
    if recap is not None:
        recap_datasets, recap_macro = recap
        comparison_datasets.extend(recap_datasets)
        comparison_methods.append("RECAP-OFO")
        comparison_macros.append(recap_macro)

    macro_lookup = {row["method"]: row for row in comparison_macros}
    lines = [
        "# RECAP-OFO 12 数据集八基线复现实验报告",
        "",
        "## 1. 验收结论",
        "",
        (
            f"正式运行与独立验收均完成：{audit['audited_runs']}/"
            f"{audit['expected_runs']} 个训练与推理运行，"
            f"{audit['recomputed_metrics']} 个 AUROC/AUPRC 指标重新计算，"
            f"{audit['label_audits_passed']} 个标签隔离审计和 "
            f"{audit['checkpoint_reloads_passed']} 个 checkpoint 重载审计全部通过。"
        ),
        "",
        (
            "监督方法 GCN/GAT/BWGNN/XGBGraph 使用每个数据集固定 seed 的 "
            "40/20/40 分层 train/validation/test 划分，表中仅评估 40% test 节点；"
            "无监督方法 DOMINANT/AnomalyDAE/CoLA/ADA-GAD 及 RECAP-OFO "
            "在标签不可见训练后评估全图。因监督权利和评估节点集合不同，"
            "两组结果可以并列报告，但不能解释为完全同条件的算法优劣。"
        ),
        "",
        "## 2. 12 数据集宏平均",
        "",
        "| Method | Supervision | Evaluation population | AUROC | AUPRC |",
        "|---|---|---|---:|---:|",
    ]
    for method in comparison_methods:
        row = macro_lookup[method]
        lines.append(
            f"| {method} | {'supervised' if row['supervised'] else 'unsupervised'} "
            f"| {row['evaluation_population']} "
            f"| {percent_cell(row['auroc_mean'], row['auroc_std'])} "
            f"| {percent_cell(row['auprc_mean'], row['auprc_std'])} |"
        )
    lines.extend(
        [
            "",
            "均值与标准差先在每个 seed 内对 12 个数据集做等权宏平均，"
            "再对 seed 0/1/2 报告总体均值和总体标准差（`ddof=0`）。",
            "",
            "## 3. 逐数据集 AUROC（%）",
            "",
            *markdown_matrix(comparison_datasets, comparison_methods, "auroc"),
            "",
            "## 4. 逐数据集 AUPRC（%）",
            "",
            *markdown_matrix(comparison_datasets, comparison_methods, "auprc"),
            "",
            "## 5. 训练、推理与资源",
            "",
            "| Method | Runs | Train sum (s) | Inference sum (s) | Total sum (s) | "
            "Mean/run (s) | Max GPU allocated (GiB) | Peak RSS (GiB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in timing_rows:
        lines.append(
            f"| {row['method']} | {row['run_count']} "
            f"| {row['train_seconds_sum']:.2f} "
            f"| {row['inference_seconds_sum']:.2f} "
            f"| {row['total_seconds_sum']:.2f} "
            f"| {row['total_seconds_mean_per_run']:.2f} "
            f"| {row['peak_gpu_allocated_gib']:.2f} "
            f"| {row['peak_process_rss_gib']:.2f} |"
        )
    lines.extend(
        [
            "",
            "这些时间来自每个正式运行内部记录，不含 SSH、巡检等待、失败诊断、"
            "smoke、独立审计和报告生成时间。八方法的正式运行按顺序执行，"
            "没有通过并发争抢 GPU 来缩短墙钟时间。",
            "",
            "## 6. 实现适配与协议偏差",
            "",
            "- GCN/GAT/BWGNN 使用当前 PyTorch/PyG 稀疏算子表达发布版的 DGL 操作；"
            "正式前已通过稠密/稀疏及 BWGNN 多项式等价门禁。",
            "- XGBGraph 使用发布版的两层 mean-GIN 特征构造；为兼容 "
            "XGBoost 3.0.2 与 scikit-learn 1.8，checkpoint 改用原生 Booster JSON，"
            "算法、树数和分数未改变。",
            "- DOMINANT 的结构重构误差使用锁定协议中的精确代数恒等式，"
            "避免物化 N×N 矩阵；这不是负采样近似。",
            "- AnomalyDAE 保留全部正边，使用确定性 1:1 非边采样和逆概率加权"
            "估计原稠密结构项；最终分数平均四轮固定采样。",
            "- CoLA 使用 PyGOD 1.1.0 记录的随机邻居上下文采样，"
            "最终分数平均 64 个确定性推理轮次。",
            "- ADA-GAD 保留三视图预训练、编码器平均/冻结和检测器重训两阶段；"
            "结构项同样使用已锁定的 1:1 加权非边适配和四轮最终评分。",
            "- checkpoint 接受阈值在正式结果前锁定为 "
            "`1e-5 + 5e-6 × max(abs(score))`，用于容纳 CUDA scatter 的末位浮点差；"
            f"288 次中的最大绝对重载差为 "
            f"{audit['maximum_reload_abs_difference']:.8g}。",
            "- Questions 完整包含在全部八方法中；没有删数据、减 epoch、"
            "选择性重跑弱结果或使用 target-test 指标调参。",
            "",
            "## 7. 可复查产物",
            "",
            f"- 全局审计：`{analysis_root / 'global_audit.json'}`",
            f"- 逐运行复算：`{analysis_root / 'run_records.csv'}`",
            f"- 逐数据集统计：`{analysis_root / 'dataset_summary.csv'}`",
            f"- 方法宏平均：`{analysis_root / 'method_macro_summary.json'}`",
            f"- 领域宏平均：`{analysis_root / 'domain_summary.csv'}`",
            f"- 时间与资源：`{analysis_root / 'timing_resource_summary.csv'}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)


def analyze(
    *,
    run_root: Path,
    dataset_root: Path,
    analysis_root: Path,
    report_path: Path,
    recap_dataset_summary: Path,
    recap_macro_summary: Path,
) -> dict[str, Any]:
    records, audit = audit_runs(run_root, dataset_root)
    dataset_rows = aggregate_dataset(records)
    method_rows = aggregate_method(records)
    domain_rows = aggregate_domain(records)
    timing_rows = aggregate_timing(records)
    recap = load_recap_comparison(
        recap_dataset_summary, recap_macro_summary
    )

    atomic_csv(analysis_root / "run_records.csv", records)
    atomic_csv(analysis_root / "dataset_summary.csv", dataset_rows)
    atomic_json(
        analysis_root / "method_macro_summary.json",
        {
            "format": "recap_ofo12_method_macro_v1",
            "aggregation": (
                "dataset macro within seed, then mean and population std "
                "over seeds 0/1/2"
            ),
            "methods": method_rows,
        },
    )
    atomic_csv(analysis_root / "domain_summary.csv", domain_rows)
    atomic_csv(
        analysis_root / "timing_resource_summary.csv", timing_rows
    )
    atomic_json(analysis_root / "global_audit.json", audit)
    if recap is not None:
        recap_datasets, recap_macro = recap
        atomic_csv(
            analysis_root / "comparison_with_recap_by_dataset.csv",
            [*dataset_rows, *recap_datasets],
        )
        atomic_json(
            analysis_root / "comparison_with_recap_macro.json",
            {
                "warning": (
                    "Supervised baselines evaluate a stratified 40% test split; "
                    "unsupervised baselines and RECAP evaluate the full graph."
                ),
                "methods": [*method_rows, recap_macro],
            },
        )
    write_report(
        report_path,
        audit,
        dataset_rows,
        method_rows,
        timing_rows,
        recap,
        analysis_root,
    )
    return {
        "status": "PASS",
        "audited_runs": audit["audited_runs"],
        "analysis_root": str(analysis_root),
        "report_path": str(report_path),
        "method_macro": method_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT
    )
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--recap-dataset-summary",
        type=Path,
        default=DEFAULT_RECAP_DATASET_SUMMARY,
    )
    parser.add_argument(
        "--recap-macro-summary",
        type=Path,
        default=DEFAULT_RECAP_MACRO_SUMMARY,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        run_root=args.run_root,
        dataset_root=args.dataset_root,
        analysis_root=args.analysis_root,
        report_path=args.report_path,
        recap_dataset_summary=args.recap_dataset_summary,
        recap_macro_summary=args.recap_macro_summary,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
