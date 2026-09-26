# RECAP OFO–Questions 补充实验报告

生成时间（UTC）：2026-07-26T07:27:45.942634+00:00

状态：**CONFIRMATORY USER-REQUESTED ADDENDUM / PASSED**

## 结论

Questions 的三个独立 OFO 训练和推理全部完成。三种子结果为 AUROC 63.47 ± 0.05、AUPRC 4.78 ± 0.05。

把该补充结果与原 11 个 OFO 数据集合并后，12 数据集宏平均为 AUROC 71.08 ± 0.35、AUPRC 23.61 ± 0.26。这个 12 数据集数字是用户在原 Phase 1 完成后追加得到的结果，不覆盖原锁定报告。

## 逐种子结果

| Seed | AUROC (%) | AUPRC (%) | 训练 (s) | 推理 (s) | 峰值显存 (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 63.48 | 4.85 | 12.86 | 0.073 | 37.21 |
| 1 | 63.41 | 4.75 | 12.72 | 0.073 | 37.21 |
| 2 | 63.52 | 4.73 | 12.71 | 0.073 | 37.21 |

## 跨种子社区稳定性

| NMI | ARI | 软共分配相似度 | 分数 Spearman | 有效社区数 |
| ---: | ---: | ---: | ---: | ---: |
| 0.386 ± 0.032 | 0.504 ± 0.020 | 0.848 ± 0.027 | 0.993 ± 0.000 | 28.63 ± 1.17 |

## 验收

- 正式运行：3/3。
- 最终评估：3/3。
- 固定点诊断：18/18。
- 中间点和最终模型：15/15。
- 检查点重载：3/3。
- 精确 KNN 溯源：3/3。
- 活动失败：0。

所有运行均为 100 epochs、精确 KNN K=64、无目标标签训练、无早停；Questions 的三个 seed 分别从新模型开始。

## 产物

- 原始结果：`rebuttal/artifacts/questions_ofo_addendum/raw_results.csv`
- 验收：`rebuttal/artifacts/questions_ofo_addendum/analysis/artifact_validation.json`
- 社区稳定性：`rebuttal/artifacts/questions_ofo_addendum/analysis/questions_stability_summary.json`
- 12 数据集宏平均：`rebuttal/artifacts/questions_ofo_addendum/analysis/combined_ofo_12_summary.json`
- 检查点：`rebuttal/artifacts/questions_ofo_addendum/runs`
- 逐节点社区输出：`rebuttal/artifacts/questions_ofo_addendum/community_stability`
