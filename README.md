# RECAP

RECAP is a label-free generalist graph anomaly detector. It trains on one or
more source graphs without anomaly labels, freezes the model, and applies it to
unseen target graphs without target labels, prompt nodes, or fine-tuning.

The model has five stages: robust feature alignment, multi-hop residual
encoding, residual-similarity graph construction, transferable soft-community
learning, and anomaly scoring from prototype adhesion and community-context
inconsistency.

## Paper-aligned results

The released configuration is `params/recap_auprc_best.json`.

| Protocol | AUROC (%) | AUPRC (%) |
|---|---:|---:|
| RECAP-OFO, 12-dataset macro | 71.08 ± 0.35 | 23.61 ± 0.26 |
| RECAP-OFA, Setting A | 74.65 ± 0.23 | 27.04 ± 0.34 |
| RECAP-OFA, Setting B | 67.75 ± 0.22 | 21.98 ± 0.32 |
| RECAP-OFA, Setting C | 67.31 ± 0.43 | 17.50 ± 0.09 |

The OFA protocol is the main deployment setting: one frozen model is trained
on source graphs and evaluated on unseen targets. OFO trains a separate model
for each target and is included as a target-specific reference.

## Repository layout

```text
model.py, detector.py, utils.py       core model and data pipeline
train.py, inference.py                training and frozen-checkpoint inference
config.py, params/                    configuration and paper hyperparameters
ablation/                             ablation scripts
interpretability/                     optional explanation script
tuning_hyperparams/                   sensitivity-analysis scripts
docs/                                 large-graph and ANN documentation
tools/                                dataset preparation utilities
```

## Installation

The reference environment uses Python 3.12, PyTorch 2.11 with CUDA 12.8,
PyTorch Geometric 2.7, NumPy 2.1, SciPy 1.17, and scikit-learn 1.8.
Install the pinned dependencies with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data

Datasets are not included because of size and licensing restrictions. Place the
12 standard `.mat` files in `dataset/`: `pubmed`, `cora`, `citeseer`, `ACM`,
`Flickr`, `BlogCatalog`, `Facebook`, `weibo`, `Reddit`, `questions`, `YelpChi`,
and `Amazon`. Each file must provide adjacency, node attributes, and anomaly
labels under the field names accepted by `utils.py`.

## Training and inference

Run commands from the repository root. The following reproduces the paper's
OFA Setting A split:

```bash
python train.py \
  --model recap_auprc_best \
  --device cuda:0 \
  --trials 3 \
  --epochs 100 \
  --train-datasets pubmed Flickr questions YelpChi \
  --test-datasets cora citeseer ACM BlogCatalog Facebook weibo Reddit Amazon
```

For target-specific OFO, use the same dataset for training and evaluation:

```bash
python train.py \
  --model recap_auprc_best \
  --device cuda:0 \
  --trials 3 \
  --epochs 100 \
  --train-datasets cora \
  --test-datasets cora
```

A saved checkpoint can be evaluated with:

```bash
python inference.py \
  --checkpoint checkpoints/recap_auprc_best/trial_0/model.pt \
  --datasets cora citeseer ACM \
  --device cuda:0 \
  --output-dir inference_results
```

Large-graph preparation and approximate-nearest-neighbor options are documented
in [`docs/LARGE_DATASETS.md`](docs/LARGE_DATASETS.md) and
[`docs/APPROXIMATE_KNN.md`](docs/APPROXIMATE_KNN.md).
