# Quick Start

```bash
cd /Users/coco/Desktop/temp/recap
./tuning_hyperparams/run_sensitivity.sh
```

```bash
# 完整，重名强制重跑
RUN_NAME=sensitivity_paper_final \
EPOCHS=200 \
TRIALS=3 \
DEVICE=cuda:0 \
FORCE=1 \
./tuning_hyperparams/run_sensitivity.sh
```

```bash
# 快速检验（更少数据集、更少trial）
TRAIN_DATASETS="pubmed Flickr" \
TEST_DATASETS="cora ACM weibo" \
EPOCHS=80 TRIALS=1 RUN_NAME=recap_sens_quick \
./tuning_hyperparams/run_sensitivity.sh
```

```bash
# 仅重画图（不重跑训练）
PLOT_ONLY=1 OUTPUT_DIR=/autodl-fs/data/recap/tuning_hyperparams/sensitivity_results/sensitivity_quick_final/raw_results.csv \
./tuning_hyperparams/run_sensitivity.sh
```

## lambda_E=0 default check

```bash
# 快速扫描原 sensitivity_default_v2 日志，确认 L_var 在敏感度实验中是否实际活跃
/root/miniconda3/bin/python tuning_hyperparams/lambda_e_activity_check.py
```

```bash
# 更严格的配对补充实验：只在 lambda_E=0 下重跑主敏感参数的少量代表点，
# 并和 sensitivity_default_v2 的 lambda_E=0.1 结果自动生成差值表。
RUN_NAME=lambda_e0_invariance_default_v2 \
EPOCHS=100 \
TRIALS=3 \
DEVICE=cuda:0 \
bash tuning_hyperparams/run_lambda_e0_invariance.sh
```
