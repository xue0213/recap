# RECAP

## 环境与数据

```bash
pip install -r requirements.txt
```

- 数据集：将 `.mat` 数据放在 `./dataset/`（与 `utils.Dataset` 的默认 `prefix` 一致）；首次运行会生成特征对齐缓存 `*_dims.npz`。
- **Python 建议 3.10+**（代码使用 `|` 类型联合等语法）。
- GPU：训练/推理默认 `cuda:0`；CPU 请将命令中的 `--device` 改为对应 CPU 写法并安装匹配的 PyTorch。

## 仓库脚本一览

| 位置 | 说明 |
|------|------|
| `train.py` | 端到端训练与评测，写入 `./checkpoints/<model>/trial_*/`。 |
| `inference.py` | 载入 `model.pt` 在指定或默认测试集上推理。 |
| `config.py` | `TrainConfig` / `ModelConfig` 及 JSON 读写。 |
| `model.py`, `detector.py`, `utils.py`, `model_checkpoint.py` | 模型、检测器管线、数据与检查点。 |
| `params/*.json` | 模型初始超参（`train.py` / `inference.py` 默认从 `--json-dir` 读取，与 `--model` 名称对应）。 |
| `tuning_hyperparams/` | 敏感性分析、脚本化调参流水线（含 `run_sensitivity.sh` 等）。 |
| `tuning_hyperparams/interaction_tuning/` | 多超参耦合搜索与 K×C 热力图脚本；细节见该目录 README。 |
| `ablation/`, `interpretability/` | 消融与可解释性实验入口脚本。 |

## 一键训练（默认配置）

在项目**根目录**执行（默认 4 个训练集 + 8 个单视图测试集，3 次 trial，`./params/recap.json` 需存在或与 `--model` 同名 JSON 就位；若仅有 `params/recap_auprc_best.json`，可复制为 `params/recap.json` 或使用 `train.py` 所期望的文件名）。

```bash
python train.py --device cuda:0
```

精简示例（单次 trial、自定义输出目录与数据子集）：

```bash
python train.py --device cuda:0 --trials 1 --epochs 100 \
  --output-dir ./checkpoints \
  --train-datasets pubmed Flickr --test-datasets cora citeseer \
  --json-dir ./params --model recap --dims 32
```

训练结束后，权重典型路径：**`./checkpoints/recap/trial_<n>/model.pt`**（随行 `train_config.json` / `model_config.json`）。

## 一键推理

对某一 trial 的检查点评测（不写 `--datasets` 时使用脚本内置默认列表，含 `tfinance` 等与训练默认略有不同，可按需显式传入）：

```bash
python inference.py \
  --checkpoint ./checkpoints/recap/trial_0/model.pt \
  --device cuda:0 \
  --output-dir ./inference_results
```

只对部分数据集评测：

```bash
python inference.py \
  --checkpoint ./checkpoints/recap/trial_0/model.pt \
  --datasets cora weibo \
  --device cuda:0
```

对 `./checkpoints/recap/` 下**全部 trial** 批量推理：

```bash
python inference.py --checkpoint ./checkpoints/recap --batch --device cuda:0
```
