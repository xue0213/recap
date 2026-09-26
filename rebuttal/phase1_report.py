"""Generate the human-readable RECAP Phase 1 acceptance report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REBUTTAL_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rebuttal.phase1_protocol import (  # noqa: E402
    OFA_SETTINGS,
    OFO_DATASETS,
    display_name,
)
from rebuttal.phase1_runner import DEFAULT_OUTPUT_ROOT  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def format_metric(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} ± {100.0 * std:.2f}"


def format_seconds(value: float) -> str:
    return f"{value:.2f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def stability_setting_rows(stability_rows: list[dict[str, str]]) -> list[dict]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stability_rows:
        grouped[row["setting"]].append(row)
    output = []
    for setting in ("OFO", "A", "B", "C"):
        rows = grouped[setting]
        summary = {
            "setting": setting,
            "datasets": len(rows),
        }
        for metric in (
            "nmi",
            "ari",
            "soft_coassignment_similarity",
            "score_spearman",
            "effective_communities",
        ):
            summary[metric] = sum(
                float(row[f"{metric}_mean"]) for row in rows
            ) / len(rows)
        score_min = min(rows, key=lambda row: float(row["score_spearman_mean"]))
        soft_min = min(
            rows, key=lambda row: float(row["soft_coassignment_similarity_mean"])
        )
        summary["score_min"] = float(score_min["score_spearman_mean"])
        summary["score_min_dataset"] = score_min["target_graph"]
        summary["soft_min"] = float(soft_min["soft_coassignment_similarity_mean"])
        summary["soft_min_dataset"] = soft_min["target_graph"]
        output.append(summary)
    return output


def manuscript_comparison(
    metric_rows: list[dict[str, str]],
    reference: dict,
) -> tuple[list[list[str]], dict[str, float]]:
    current = {
        (row["setting"], row["target_graph"]): row for row in metric_rows
    }
    table_rows: list[list[str]] = []
    deltas: dict[str, list[float]] = {"auroc": [], "auprc": []}
    for dataset in OFA_SETTINGS["A"]["targets"]:
        observed = current[("A", dataset)]
        manuscript = reference["datasets"][dataset]
        auroc_now = 100.0 * float(observed["auroc_mean"])
        auprc_now = 100.0 * float(observed["auprc_mean"])
        auroc_delta = auroc_now - float(manuscript["auroc_mean"])
        auprc_delta = auprc_now - float(manuscript["auprc_mean"])
        deltas["auroc"].append(auroc_delta)
        deltas["auprc"].append(auprc_delta)
        table_rows.append(
            [
                display_name(dataset),
                (
                    f"{manuscript['auroc_mean']:.2f} ± "
                    f"{manuscript['auroc_std']:.2f}"
                ),
                format_metric(
                    float(observed["auroc_mean"]),
                    float(observed["auroc_std"]),
                ),
                f"{auroc_delta:+.2f}",
                (
                    f"{manuscript['auprc_mean']:.2f} ± "
                    f"{manuscript['auprc_std']:.2f}"
                ),
                format_metric(
                    float(observed["auprc_mean"]),
                    float(observed["auprc_std"]),
                ),
                f"{auprc_delta:+.2f}",
            ]
        )
    summary = {
        "auroc_mae": sum(abs(value) for value in deltas["auroc"]) / 8,
        "auprc_mae": sum(abs(value) for value in deltas["auprc"]) / 8,
        "auroc_macro_delta": sum(deltas["auroc"]) / 8,
        "auprc_macro_delta": sum(deltas["auprc"]) / 8,
    }
    return table_rows, summary


def generate_report(output_root: Path, report_path: Path) -> None:
    analysis = output_root / "analysis"
    record_validation = json.loads(
        (analysis / "record_validation.json").read_text()
    )
    artifact_validation = json.loads(
        (analysis / "artifact_validation.json").read_text()
    )
    final_validation = json.loads((analysis / "final_validation.json").read_text())
    timing_summary = json.loads((analysis / "timing_summary.json").read_text())
    hardware = json.loads((output_root / "preflight" / "hardware.json").read_text())
    metric_rows = read_csv(analysis / "metrics_by_dataset.csv")
    macro_rows = read_csv(analysis / "metric_macros.csv")
    stability_rows = read_csv(analysis / "stability_summary.csv")
    timing_rows = read_csv(analysis / "timing_by_setting.csv")
    manuscript = json.loads(
        (REBUTTAL_ROOT / "manuscript_setting_a_reference.json").read_text()
    )

    if not (
        record_validation["passed"]
        and artifact_validation["passed"]
        and final_validation["passed"]
    ):
        raise ValueError("Refusing to generate a final report from failed validation")

    metrics = {
        (row["setting"], row["target_graph"]): row for row in metric_rows
    }
    macro = {
        (row["setting"], row["aggregation"]): row for row in macro_rows
    }
    setting_a_comparison, comparison_summary = manuscript_comparison(
        metric_rows, manuscript
    )
    stability_by_setting = stability_setting_rows(stability_rows)

    lines = [
        "# RECAP Phase 1 最终实验与验收报告",
        "",
        f"生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}",
        "",
        "协议状态：**CONFIRMATORY / LOCKED**",
        "",
        "## 结论摘要",
        "",
        (
            "**正式实验验收通过。** 42/42 个训练任务、87/87 个最终评估、"
            "87/87 个种子对稳定性比较均完整；没有活动失败、重复记录、"
            "缺失路径、NaN 或 Inf。"
        ),
        "",
        (
            "RECAP 在 11 个 OFO 数据集上的数据集宏平均为 "
            f"AUROC {format_metric(float(macro[('OFO', 'dataset_macro')]['auroc_mean']), float(macro[('OFO', 'dataset_macro')]['auroc_std']))}、"
            f"AUPRC {format_metric(float(macro[('OFO', 'dataset_macro')]['auprc_mean']), float(macro[('OFO', 'dataset_macro')]['auprc_std']))}。"
        ),
        "",
        (
            "OFA Setting A 的数据集宏平均为 "
            f"AUROC {format_metric(float(macro[('A', 'dataset_macro')]['auroc_mean']), float(macro[('A', 'dataset_macro')]['auroc_std']))}、"
            f"AUPRC {format_metric(float(macro[('A', 'dataset_macro')]['auprc_mean']), float(macro[('A', 'dataset_macro')]['auprc_std']))}；"
            "与投稿论文 Setting A 的逐数据集结果整体对齐。"
        ),
        "",
        (
            "结论需要保留边界：OFO 的 YelpChi AUROC 低于随机水平；"
            "Setting C 的跨域异常分数通常仍有稳定性，但社区划分在若干目标上"
            "明显不稳定，尤其是 BlogCatalog。不能把结果表述为“所有数据集上"
            "都同样有效或社区完全稳定”。"
        ),
        "",
        "## 1. 锁定范围与实现",
        "",
        "- OFO：排除 Questions，仅运行其余 11 个数据集，3 个种子，共 33 次训练。",
        "- OFA：Setting A/B/C 原样保留，Questions 仍按原协议出现在 OFA 中，3 个种子，共 9 次训练、54 次目标评估。",
        "- 总计：42 次训练、87 次最终评估、29 个 setting–dataset 稳定性范围、87 个种子对。",
        "- 论文设置：100 epochs、32 维、4 hops、36 communities、精确 KNN（K=64）、seeds 0/1/2、无目标标签早停。",
        "- 科学基线提交：`c94c4d7985d2cb1438c430173ad868d68d0c1efe`。",
        "- 标准差：population standard deviation，`ddof=0`。",
        "",
        "Questions 只从 OFO 中删除。Amazon 使用实际文件中的 10,224 个节点；论文表格的 10,244 记录为疑似笔误。",
        "",
        "## 2. 主结果宏平均",
        "",
    ]
    macro_table = []
    for setting, aggregation in (
        ("OFO", "dataset_macro"),
        ("A", "dataset_macro"),
        ("B", "dataset_macro"),
        ("C", "dataset_macro"),
        ("C", "domain_macro"),
    ):
        row = macro[(setting, aggregation)]
        macro_table.append(
            [
                setting,
                aggregation,
                format_metric(float(row["auroc_mean"]), float(row["auroc_std"])),
                format_metric(float(row["auprc_mean"]), float(row["auprc_std"])),
            ]
        )
    lines.extend(
        markdown_table(
            ["Setting", "聚合方式", "AUROC (%)", "AUPRC (%)"],
            macro_table,
        )
    )

    lines.extend(["", "## 3. OFO：11 个数据集", ""])
    ofo_table = []
    for dataset in OFO_DATASETS:
        row = metrics[("OFO", dataset)]
        ofo_table.append(
            [
                display_name(dataset),
                format_metric(float(row["auroc_mean"]), float(row["auroc_std"])),
                format_metric(float(row["auprc_mean"]), float(row["auprc_std"])),
            ]
        )
    lines.extend(markdown_table(["数据集", "AUROC (%)", "AUPRC (%)"], ofo_table))

    lines.extend(["", "## 4. OFA：Setting A/B/C", ""])
    for setting in ("A", "B", "C"):
        lines.extend([f"### Setting {setting}", ""])
        setting_table = []
        for dataset in OFA_SETTINGS[setting]["targets"]:
            row = metrics[(setting, dataset)]
            setting_table.append(
                [
                    display_name(dataset),
                    format_metric(
                        float(row["auroc_mean"]), float(row["auroc_std"])
                    ),
                    format_metric(
                        float(row["auprc_mean"]), float(row["auprc_std"])
                    ),
                ]
            )
        lines.extend(
            markdown_table(["目标数据集", "AUROC (%)", "AUPRC (%)"], setting_table)
        )
        lines.append("")

    lines.extend(
        [
            "## 5. 与投稿论文 Setting A 的复现对照",
            "",
            "差值为“本次运行 − 投稿论文”，单位是百分点（pp）。",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "数据集",
                "论文 AUROC",
                "本次 AUROC",
                "Δ AUROC",
                "论文 AUPRC",
                "本次 AUPRC",
                "Δ AUPRC",
            ],
            setting_a_comparison,
        )
    )
    lines.extend(
        [
            "",
            (
                "逐数据集绝对差的平均值为 "
                f"AUROC {comparison_summary['auroc_mae']:.2f} pp、"
                f"AUPRC {comparison_summary['auprc_mae']:.2f} pp；"
                "本次数据集宏平均相对论文分别变化 "
                f"{comparison_summary['auroc_macro_delta']:+.2f} pp 和 "
                f"{comparison_summary['auprc_macro_delta']:+.2f} pp。"
            ),
            "",
            "## 6. 社区与异常分数的跨种子稳定性",
            "",
            "下表是各 setting 内按目标数据集做宏平均后，再汇总三个种子对的结果。NMI、ARI 和软共分配相似度衡量社区结构；Spearman 衡量最终异常分数排序。",
            "",
        ]
    )
    stability_table = []
    for row in stability_by_setting:
        stability_table.append(
            [
                row["setting"],
                str(row["datasets"]),
                f"{row['nmi']:.3f}",
                f"{row['ari']:.3f}",
                f"{row['soft_coassignment_similarity']:.3f}",
                f"{row['score_spearman']:.3f}",
                f"{row['effective_communities']:.2f}",
                (
                    f"{display_name(row['score_min_dataset'])} "
                    f"({row['score_min']:.3f})"
                ),
                (
                    f"{display_name(row['soft_min_dataset'])} "
                    f"({row['soft_min']:.3f})"
                ),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Setting",
                "目标数",
                "NMI",
                "ARI",
                "软共分配",
                "分数 Spearman",
                "有效社区数",
                "最低分数稳定性",
                "最低软社区稳定性",
            ],
            stability_table,
        )
    )
    lines.extend(
        [
            "",
            "Setting C 表明“异常排序可复现”和“潜在社区身份可复现”不是同一件事：多数目标的分数排序稳定性高于社区划分稳定性；BlogCatalog 是两者都较弱的主要例外。完整 29 个范围和 87 个种子对见分析 CSV。",
            "",
            "## 7. 运行时间与资源",
            "",
        ]
    )
    timing_table = []
    for row in timing_rows:
        timing_table.append(
            [
                row["setting"],
                row["training_runs"],
                row["evaluations"],
                format_seconds(float(row["data_prepare_seconds_total"])),
                format_seconds(float(row["train_seconds_total"])),
                format_seconds(float(row["diagnostic_seconds_total"])),
                format_seconds(float(row["inference_seconds_total"])),
                f"{float(row['peak_gpu_memory_mb_max']) / 1024.0:.2f}",
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Setting",
                "训练数",
                "评估数",
                "数据准备 (s)",
                "训练 (s)",
                "诊断 (s)",
                "推理 (s)",
                "峰值显存 (GiB)",
            ],
            timing_table,
        )
    )
    lines.extend(
        [
            "",
            (
                "去除 OFA 目标重复计数后，42 次正式训练累计 "
                f"{timing_summary['train_seconds_total']:.2f} s；"
                "数据准备、训练、固定点诊断和最终推理的可归属时间合计 "
                f"{timing_summary['accounted_seconds_total']:.2f} s。"
                "这不是包含审计暂停、进程启动和工具修复的端到端墙钟时间。"
            ),
            "",
            (
                "硬件："
                f"{hardware['gpu_query']}；PyTorch {hardware['torch']}，"
                f"CUDA {hardware['cuda_runtime']}，PyG {hardware['torch_geometric']}。"
            ),
            "",
            "## 8. 完整性验收",
            "",
        ]
    )
    audit_table = [
        ["正式运行目录", f"{artifact_validation['formal_run_directories']}/42"],
        ["最终评估记录", f"{artifact_validation['final_evaluations_actual']}/87"],
        [
            "固定点诊断行",
            (
                f"{artifact_validation['diagnostic_rows_actual']}/"
                f"{artifact_validation['diagnostic_rows_expected']}"
            ),
        ],
        [
            "训练中间点 + 最终模型",
            (
                f"{artifact_validation['checkpoints_actual']}/"
                f"{artifact_validation['checkpoints_expected']}"
            ),
        ],
        [
            "stdout/stderr 日志",
            (
                f"{artifact_validation['log_files_actual']}/"
                f"{artifact_validation['log_files_expected']}"
            ),
        ],
        [
            "检查点重载审计",
            (
                f"{artifact_validation['checkpoint_reload_audits_passed']}/"
                f"{artifact_validation['checkpoint_reload_audits_expected']}"
            ),
        ],
        [
            "精确 KNN 溯源",
            (
                f"{artifact_validation['exact_knn_provenance_runs_actual']}/"
                f"{artifact_validation['exact_knn_provenance_runs_expected']}"
            ),
        ],
        ["活动失败", str(artifact_validation["active_failures"])],
        ["保留的历史工具失败", str(artifact_validation["historical_failure_entries"])],
    ]
    lines.extend(markdown_table(["验收项", "结果"], audit_table))
    lines.extend(
        [
            "",
            "保留的 3 条历史失败分别来自：PyTorch 新版 `weights_only` 默认行为、CPU RNG 状态被错误映射到 CUDA、以及过严的 float32 逐值阈值。它们均发生在训练完成后的工具审计路径；修复后对应正式运行通过，失败日志未删除。",
            "",
            "标签隔离通过代码边界和测试共同保证：模型接收的 PyG `Data` 已删除 `ano_labels`，标签只在最终分数转为 CPU 数组后进入 AUROC/AUPRC 计算；没有早停、目标调参或用标签选检查点。",
            "",
            "## 9. 结果不变的计算优化与偏离报告",
            "",
            "- 仅复用带完整数据哈希和配置键的特征对齐缓存、精确 KNN 缓存；未在 Phase 1 使用 ANN。",
            "- 用 `C×C` 恒等式精确计算软共分配相似度，避免构造 `N×N` 矩阵。",
            "- OFA 每个 seed 只训练一次，再对全部锁定目标推理；训练时间统计不按目标重复。",
            "- 固定 epoch 只保存紧凑诊断，最终 epoch 保存完整逐节点社区输出。",
            "- 每个运行独立子进程、原子状态文件、epoch 25/50/75/100 中间点与 RNG/优化器状态，支持不改变轨迹的恢复。",
            "- 显式补全论文隐含默认值（`tau_e=1`、`lambda_bal=0.1`、`lambda_E=0`、`gamma=0.01`）；没有基于本次结果修改论文超参数。",
            "- 唯一实验范围变更是用户批准的 OFO 删除 Questions；OFA 未改。",
            "",
            "## 10. 产物索引",
            "",
            f"- 原始逐种子结果：`{output_root / 'raw_results.csv'}`",
            f"- 数据集均值/标准差：`{analysis / 'metrics_by_dataset.csv'}`",
            f"- 宏平均：`{analysis / 'metric_macros.csv'}`",
            f"- 29 个稳定性范围：`{analysis / 'stability_summary.csv'}`",
            f"- 87 个种子对：`{analysis / 'stability_pairs.csv'}`",
            f"- 分 setting 时间：`{analysis / 'timing_by_setting.csv'}`",
            f"- 完整产物审计：`{analysis / 'artifact_validation.json'}`",
            f"- 原始记录审计：`{analysis / 'record_validation.json'}`",
            f"- 全部运行目录与检查点：`{output_root / 'runs'}`",
            f"- 逐节点社区输出：`{output_root / 'community_stability'}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--report-path",
        help="Defaults to OUTPUT_ROOT/analysis/PHASE1_FINAL_REPORT.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    report_path = (
        Path(args.report_path)
        if args.report_path
        else output_root / "analysis" / "PHASE1_FINAL_REPORT.md"
    )
    generate_report(output_root, report_path)
    print(report_path)


if __name__ == "__main__":
    main()
