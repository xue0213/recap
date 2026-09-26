"""Run and audit the locked three-seed RECAP-OFO Questions addendum."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from itertools import combinations
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

from rebuttal.phase1_analysis import (  # noqa: E402
    exact_soft_coassignment_similarity,
    load_community,
    population_stats,
    validate_run_artifacts,
)
from rebuttal.phase1_protocol import RunSpec, display_name  # noqa: E402
from rebuttal.phase1_runner import (  # noqa: E402
    BASE_COMMIT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATASET_DIR,
    DEFAULT_OUTPUT_ROOT,
    RAW_FIELDS,
    atomic_csv,
    atomic_json,
    collect_records,
    hardware_metadata,
    run_one,
    utc_now,
)


DEFAULT_ADDENDUM_ROOT = REBUTTAL_ROOT / "artifacts" / "questions_ofo_addendum"
DEFAULT_MANIFEST_PATH = REBUTTAL_ROOT / "questions_ofo_manifest.json"


def locked_specs() -> list[RunSpec]:
    return [
        RunSpec(
            run_id=f"ofo__questions__seed{seed}",
            method="RECAP-OFO",
            paradigm="one-for-one",
            setting="OFO",
            seed=seed,
            source_graphs=("questions",),
            target_graphs=("questions",),
        )
        for seed in (0, 1, 2)
    ]


def load_addendum_manifest(path: Path) -> list[RunSpec]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    expected_runs = json.loads(json.dumps([spec.to_dict() for spec in locked_specs()]))
    if payload.get("format") != "recap_questions_ofo_addendum_manifest_v1":
        raise ValueError("Questions addendum manifest format mismatch")
    if payload.get("base_commit") != BASE_COMMIT:
        raise ValueError("Questions addendum base commit mismatch")
    if payload.get("locked_protocol") != (
        "rebuttal/QUESTIONS_OFO_ADDENDUM_PROTOCOL.md"
    ):
        raise ValueError("Questions addendum protocol path mismatch")
    if payload.get("runs") != expected_runs:
        raise ValueError("Questions addendum manifest differs from locked runs")
    if (
        int(payload.get("training_run_count", -1)) != 3
        or int(payload.get("final_evaluation_count", -1)) != 3
    ):
        raise ValueError("Questions addendum manifest count mismatch")
    return [RunSpec.from_dict(value) for value in payload["runs"]]


def write_progress(output_root: Path, specs: list[RunSpec]) -> dict[str, Any]:
    complete: list[str] = []
    failed: list[str] = []
    running: list[str] = []
    pending: list[str] = []
    for spec in specs:
        status_path = output_root / "runs" / spec.run_id / "status.json"
        if not status_path.exists():
            pending.append(spec.run_id)
            continue
        status = json.loads(status_path.read_text())
        state = status.get("status")
        if state == "complete":
            complete.append(spec.run_id)
        elif state == "failed":
            failed.append(spec.run_id)
        elif state == "running":
            running.append(spec.run_id)
        else:
            pending.append(spec.run_id)
    records = collect_records(output_root, specs)
    progress = {
        "captured_at": utc_now(),
        "training_runs_total": 3,
        "training_runs_complete": len(complete),
        "training_runs_failed": len(failed),
        "training_runs_running": len(running),
        "training_runs_pending": len(pending),
        "final_evaluations_total": 3,
        "final_evaluations_complete": len(records),
        "complete_run_ids": complete,
        "failed_run_ids": failed,
        "running_run_ids": running,
        "pending_run_ids": pending,
    }
    atomic_json(output_root / "progress.json", progress)
    if records:
        atomic_json(output_root / "raw_results.json", records)
        atomic_csv(output_root / "raw_results.csv", records, RAW_FIELDS)
    return progress


def execute(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)
    specs = load_addendum_manifest(manifest_path)
    hardware_path = output_root / "preflight" / "hardware.json"
    if not hardware_path.exists():
        atomic_json(hardware_path, hardware_metadata(args.device))

    for index, spec in enumerate(specs, start=1):
        run_dir = output_root / "runs" / spec.run_id
        complete_path = run_dir / "complete.json"
        if complete_path.exists():
            print(f"[{index}/3] SKIP complete {spec.run_id}", flush=True)
        else:
            print(f"[{index}/3] START {spec.run_id}", flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--output-root",
                str(output_root),
                "--dataset-dir",
                str(Path(args.dataset_dir)),
                "--config",
                str(Path(args.config)),
                "--manifest",
                str(manifest_path),
                "--phase1-root",
                str(Path(args.phase1_root)),
                "--device",
                args.device,
                "run-one",
                spec.run_id,
            ]
            if args.no_resume:
                command.append("--no-resume")
            with (run_dir / "stdout.log").open("a", encoding="utf-8") as stdout:
                with (run_dir / "stderr.log").open("a", encoding="utf-8") as stderr:
                    completed = subprocess.run(
                        command,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                    )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{spec.run_id} failed with exit code {completed.returncode}; "
                    f"see {run_dir / 'stderr.log'}"
                )
            print(f"[{index}/3] COMPLETE {spec.run_id}", flush=True)
        progress = write_progress(output_root, specs)
        print(
            "PROGRESS "
            + json.dumps(
                {
                    "complete": progress["training_runs_complete"],
                    "failed": progress["training_runs_failed"],
                    "evaluations": progress["final_evaluations_complete"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    analyze(output_root, Path(args.phase1_root), specs)


def validate_records(records: list[dict], specs: list[RunSpec]) -> dict[str, Any]:
    expected = {(spec.run_id, "questions", spec.seed) for spec in specs}
    actual = {
        (str(row["run_id"]), str(row["target_graph"]), int(row["seed"]))
        for row in records
    }
    problems: list[str] = []
    if len(records) != 3 or len(actual) != 3:
        problems.append(f"expected 3 unique records, got {len(records)}")
    if actual != expected:
        problems.append(
            f"record keys mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for row in records:
        for field in (
            "auroc",
            "auprc",
            "data_prepare_seconds",
            "train_seconds",
            "diagnostic_seconds",
            "inference_seconds",
            "peak_gpu_memory_mb",
        ):
            if not math.isfinite(float(row[field])):
                problems.append(f"{row['run_id']}: non-finite {field}")
        for field in ("config_path", "checkpoint_path", "community_output_path"):
            path = Path(row[field])
            if not path.exists() or path.stat().st_size == 0:
                problems.append(f"{row['run_id']}: missing or empty {field}")
    return {
        "passed": not problems,
        "records_expected": 3,
        "records_actual": len(records),
        "problems": problems,
    }


def stability_analysis(records: list[dict]) -> tuple[list[dict], dict[str, Any]]:
    by_seed = {int(row["seed"]): row for row in records}
    if set(by_seed) != {0, 1, 2}:
        raise ValueError(f"Expected seeds 0,1,2; got {sorted(by_seed)}")
    communities = {
        seed: load_community(Path(row["community_output_path"]))
        for seed, row in by_seed.items()
    }
    for seed in (1, 2):
        if not np.array_equal(
            communities[0]["node_indices"],
            communities[seed]["node_indices"],
        ):
            raise ValueError(f"Questions node order differs for seed {seed}")

    pair_rows: list[dict] = []
    for seed_a, seed_b in combinations((0, 1, 2), 2):
        first = communities[seed_a]
        second = communities[seed_b]
        pair_rows.append(
            {
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
                "score_spearman": float(
                    spearmanr(
                        first["final_scores"],
                        second["final_scores"],
                    ).statistic
                ),
            }
        )
    if any(
        not math.isfinite(float(row[field]))
        for row in pair_rows
        for field in (
            "nmi",
            "ari",
            "soft_coassignment_similarity",
            "score_spearman",
        )
    ):
        raise ValueError("Non-finite Questions stability metric")

    summary: dict[str, Any] = {
        "dataset": "questions",
        "node_count": int(communities[0]["H"].shape[0]),
        "num_communities": int(communities[0]["H"].shape[1]),
        "pair_count": 3,
    }
    for field in (
        "nmi",
        "ari",
        "soft_coassignment_similarity",
        "score_spearman",
    ):
        mean, std = population_stats([float(row[field]) for row in pair_rows])
        summary[f"{field}_mean"] = mean
        summary[f"{field}_std"] = std
    effective = [
        float(communities[seed]["effective_communities"]) for seed in (0, 1, 2)
    ]
    summary["effective_communities_mean"], summary["effective_communities_std"] = (
        population_stats(effective)
    )
    return pair_rows, summary


def combined_ofo_summary(
    phase1_root: Path,
    questions_records: list[dict],
) -> tuple[list[dict], dict[str, Any], list[dict]]:
    original_path = phase1_root / "raw_results.json"
    original = json.loads(original_path.read_text())
    original_ofo = [
        row
        for row in original
        if str(row["setting"]) == "OFO"
        and str(row["target_graph"]) != "questions"
    ]
    if len(original_ofo) != 33:
        raise ValueError(f"Expected 33 original OFO records, got {len(original_ofo)}")
    combined_records = [*original_ofo, *questions_records]
    seed_rows = []
    for seed in (0, 1, 2):
        rows = [row for row in combined_records if int(row["seed"]) == seed]
        if len(rows) != 12 or len({row["target_graph"] for row in rows}) != 12:
            raise ValueError(f"Seed {seed}: expected 12 distinct OFO datasets")
        seed_rows.append(
            {
                "seed": seed,
                "dataset_count": 12,
                "auroc_macro": float(
                    np.mean([float(row["auroc"]) for row in rows])
                ),
                "auprc_macro": float(
                    np.mean([float(row["auprc"]) for row in rows])
                ),
            }
        )
    auroc_mean, auroc_std = population_stats(
        [row["auroc_macro"] for row in seed_rows]
    )
    auprc_mean, auprc_std = population_stats(
        [row["auprc_macro"] for row in seed_rows]
    )
    summary = {
        "classification": "post_hoc_user_requested_combined_ofo_12",
        "dataset_count": 12,
        "seed_count": 3,
        "auroc_mean": auroc_mean,
        "auroc_std": auroc_std,
        "auprc_mean": auprc_mean,
        "auprc_std": auprc_std,
        "original_phase1_ofo_dataset_count": 11,
        "questions_addendum_dataset_count": 1,
    }
    return seed_rows, summary, combined_records


def dataset_metric_rows(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(str(row["target_graph"]), []).append(row)
    output = []
    for target, rows in sorted(grouped.items()):
        auroc_mean, auroc_std = population_stats(
            [float(row["auroc"]) for row in rows]
        )
        auprc_mean, auprc_std = population_stats(
            [float(row["auprc"]) for row in rows]
        )
        output.append(
            {
                "dataset": target,
                "display_name": display_name(target),
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
            }
        )
    return output


def report_text(
    records: list[dict],
    metrics: dict[str, Any],
    combined: dict[str, Any],
    stability: dict[str, Any],
    artifact_validation: dict[str, Any],
    output_root: Path,
) -> str:
    def pct(mean: float, std: float) -> str:
        return f"{100.0 * mean:.2f} ± {100.0 * std:.2f}"

    rows = sorted(records, key=lambda row: int(row["seed"]))
    lines = [
        "# RECAP OFO–Questions 补充实验报告",
        "",
        f"生成时间（UTC）：{utc_now()}",
        "",
        "状态：**CONFIRMATORY USER-REQUESTED ADDENDUM / PASSED**",
        "",
        "## 结论",
        "",
        (
            "Questions 的三个独立 OFO 训练和推理全部完成。三种子结果为 "
            f"AUROC {pct(metrics['auroc_mean'], metrics['auroc_std'])}、"
            f"AUPRC {pct(metrics['auprc_mean'], metrics['auprc_std'])}。"
        ),
        "",
        (
            "把该补充结果与原 11 个 OFO 数据集合并后，12 数据集宏平均为 "
            f"AUROC {pct(combined['auroc_mean'], combined['auroc_std'])}、"
            f"AUPRC {pct(combined['auprc_mean'], combined['auprc_std'])}。"
            "这个 12 数据集数字是用户在原 Phase 1 完成后追加得到的结果，"
            "不覆盖原锁定报告。"
        ),
        "",
        "## 逐种子结果",
        "",
        "| Seed | AUROC (%) | AUPRC (%) | 训练 (s) | 推理 (s) | 峰值显存 (GiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    f"{100.0 * float(row['auroc']):.2f}",
                    f"{100.0 * float(row['auprc']):.2f}",
                    f"{float(row['train_seconds']):.2f}",
                    f"{float(row['inference_seconds']):.3f}",
                    f"{float(row['peak_gpu_memory_mb']) / 1024.0:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 跨种子社区稳定性",
            "",
            "| NMI | ARI | 软共分配相似度 | 分数 Spearman | 有效社区数 |",
            "| ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {stability['nmi_mean']:.3f} ± {stability['nmi_std']:.3f} "
                f"| {stability['ari_mean']:.3f} ± {stability['ari_std']:.3f} "
                f"| {stability['soft_coassignment_similarity_mean']:.3f} ± "
                f"{stability['soft_coassignment_similarity_std']:.3f} "
                f"| {stability['score_spearman_mean']:.3f} ± "
                f"{stability['score_spearman_std']:.3f} "
                f"| {stability['effective_communities_mean']:.2f} ± "
                f"{stability['effective_communities_std']:.2f} |"
            ),
            "",
            "## 验收",
            "",
            f"- 正式运行：{artifact_validation['formal_run_directories']}/3。",
            f"- 最终评估：{artifact_validation['final_evaluations_actual']}/3。",
            (
                "- 固定点诊断："
                f"{artifact_validation['diagnostic_rows_actual']}/"
                f"{artifact_validation['diagnostic_rows_expected']}。"
            ),
            (
                "- 中间点和最终模型："
                f"{artifact_validation['checkpoints_actual']}/"
                f"{artifact_validation['checkpoints_expected']}。"
            ),
            (
                "- 检查点重载："
                f"{artifact_validation['checkpoint_reload_audits_passed']}/3。"
            ),
            (
                "- 精确 KNN 溯源："
                f"{artifact_validation['exact_knn_provenance_runs_actual']}/3。"
            ),
            f"- 活动失败：{artifact_validation['active_failures']}。",
            "",
            "所有运行均为 100 epochs、精确 KNN K=64、无目标标签训练、无早停；Questions 的三个 seed 分别从新模型开始。",
            "",
            "## 产物",
            "",
            f"- 原始结果：`{output_root / 'raw_results.csv'}`",
            f"- 验收：`{output_root / 'analysis' / 'artifact_validation.json'}`",
            f"- 社区稳定性：`{output_root / 'analysis' / 'questions_stability_summary.json'}`",
            f"- 12 数据集宏平均：`{output_root / 'analysis' / 'combined_ofo_12_summary.json'}`",
            f"- 检查点：`{output_root / 'runs'}`",
            f"- 逐节点社区输出：`{output_root / 'community_stability'}`",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    output_root: Path,
    phase1_root: Path,
    specs: list[RunSpec],
) -> dict[str, Any]:
    records = collect_records(output_root, specs)
    record_validation = validate_records(records, specs)
    if not record_validation["passed"]:
        raise ValueError(f"Questions raw record validation failed: {record_validation}")

    artifact_validation = validate_run_artifacts(output_root, specs, records)
    if not artifact_validation["passed"]:
        raise ValueError(
            f"Questions run artifact validation failed: {artifact_validation}"
        )

    auroc_mean, auroc_std = population_stats(
        [float(row["auroc"]) for row in records]
    )
    auprc_mean, auprc_std = population_stats(
        [float(row["auprc"]) for row in records]
    )
    metrics = {
        "dataset": "questions",
        "seed_count": 3,
        "auroc_mean": auroc_mean,
        "auroc_std": auroc_std,
        "auprc_mean": auprc_mean,
        "auprc_std": auprc_std,
    }
    pair_rows, stability = stability_analysis(records)
    seed_macro_rows, combined, combined_records = combined_ofo_summary(
        phase1_root,
        records,
    )

    analysis_dir = output_root / "analysis"
    atomic_json(analysis_dir / "record_validation.json", record_validation)
    atomic_json(analysis_dir / "artifact_validation.json", artifact_validation)
    atomic_json(analysis_dir / "questions_metrics.json", metrics)
    atomic_csv(
        analysis_dir / "questions_stability_pairs.csv",
        pair_rows,
        list(pair_rows[0]),
    )
    atomic_json(analysis_dir / "questions_stability_summary.json", stability)
    atomic_csv(
        analysis_dir / "combined_ofo_12_by_seed.csv",
        seed_macro_rows,
        list(seed_macro_rows[0]),
    )
    atomic_json(analysis_dir / "combined_ofo_12_summary.json", combined)
    combined_dataset_metrics = dataset_metric_rows(combined_records)
    atomic_csv(
        analysis_dir / "combined_ofo_12_metrics_by_dataset.csv",
        combined_dataset_metrics,
        list(combined_dataset_metrics[0]),
    )
    final_validation = {
        "passed": True,
        "training_runs": 3,
        "final_evaluations": 3,
        "stability_pairs": 3,
        "combined_ofo_datasets": 12,
        "ddof": 0,
    }
    atomic_json(analysis_dir / "final_validation.json", final_validation)
    report = report_text(
        records,
        metrics,
        combined,
        stability,
        artifact_validation,
        output_root,
    )
    report_path = analysis_dir / "QUESTIONS_OFO_ADDENDUM_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    temp_path.write_text(report)
    os.replace(temp_path, report_path)
    return {
        "record_validation": record_validation,
        "artifact_validation": artifact_validation,
        "questions_metrics": metrics,
        "combined_ofo_12": combined,
        "stability": stability,
        "final_validation": final_validation,
        "report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_ADDENDUM_ROOT))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--phase1-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda:0")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("analyze")
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--no-resume", action="store_true")
    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("run_id")
    run_parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    specs = load_addendum_manifest(Path(args.manifest))
    if args.command == "status":
        print(json.dumps(write_progress(output_root, specs), indent=2, sort_keys=True))
    elif args.command == "execute":
        execute(args)
    elif args.command == "run-one":
        by_id = {spec.run_id: spec for spec in specs}
        if args.run_id not in by_id:
            raise KeyError(f"Unknown addendum run ID: {args.run_id}")
        run_one(
            spec=by_id[args.run_id],
            dataset_dir=Path(args.dataset_dir),
            output_root=output_root,
            config_path=Path(args.config),
            device=args.device,
            resume=not args.no_resume,
        )
        write_progress(output_root, specs)
    elif args.command == "analyze":
        result = analyze(output_root, Path(args.phase1_root), specs)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
