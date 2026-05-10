# Interaction Tuning

```bash
cd /autodl-fs/data/recap
RUN_NAME=interaction_paper_v1 bash tuning_hyperparams/interaction_tuning/run_interaction_tuning.sh
```

```bash
# 重新完整跑同名实验
RUN_NAME=interaction_paper_v1 FORCE=1 bash tuning_hyperparams/interaction_tuning/run_interaction_tuning.sh
```

```bash
# 快速预检
TRAIN_DATASETS="pubmed Flickr" \
TEST_DATASETS="cora ACM weibo" \
EPOCHS=80 TRIALS=1 RUN_NAME=interaction_quick \
bash tuning_hyperparams/interaction_tuning/run_interaction_tuning.sh
```

```bash
# 只重新汇总已有结果
PLOT_ONLY=1 \
OUTPUT_DIR=/autodl-fs/data/recap/tuning_hyperparams/interaction_tuning/results/<RUN_NAME> \
bash tuning_hyperparams/interaction_tuning/run_interaction_tuning.sh
```

```bash
# k x C 联合调优并生成热力图
nohup bash -c 'BASE_CONFIG=/root/autodl-fs/recap/params/recap_auprc_best.json RUN_NAME=k_c_paper_v2_auprc_best DEVICE=cuda:0 EPOCHS=100 TRIALS=3 K_VALUES=16,24,30,36,48,64,80,96 C_VALUES=16,20,24,28,32,36,40,48 bash tuning_hyperparams/interaction_tuning/run_k_c_heatmap.sh' > k_c_paper_v2_auprc_best.out 2>&1 &
```