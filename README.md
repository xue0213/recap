# RECAP

RECAP is a label-free generalist graph anomaly detector. It trains on one or
more source graphs without anomaly labels, freezes the model, and transfers it
to unseen target graphs without target labels, prompt nodes, or fine-tuning.

The implementation follows five stages:

1. robust feature alignment into a shared coordinate space;
2. multi-hop ego-neighbor residual encoding;
3. residual-similarity graph construction;
4. transferable soft-community learning; and
5. anomaly scoring by prototype adhesion and community-context inconsistency.

This repository contains the model, the complete experimental protocol,
baseline adapters, ablations, large-graph inference tools, and audited result
tables used during the project.

## Experimental settings

- **OFO (one-for-one):** training and evaluation use the same graph. A separate
  model is trained for every dataset.
- **OFA (one-for-all):** one model is trained on source graphs and directly
  applied to unseen targets. This is RECAP's main setting.
- **Unsupervised:** anomaly labels are unavailable during training, tuning, and
  model selection. Labels are unlocked only after the complete score vector is
  frozen.

RECAP-OFO is an evaluation variant trained independently on each target graph.
RECAP-OFA is the proposed deployment protocol and uses neither source labels
nor target-side context.

| Track | Training vs. target | Label/context rights | Deployment | Evaluation |
|---|---|---|---|---|
| Unsupervised OFO | same graph | no labels | one model per target | full graph |
| Supervised OFO reference | same graph | target train/validation labels | one model per target | held-out 40% test nodes |
| Supervised/target-assisted OFA | different graphs | source labels; method-dependent target context | one model per split | full target or declared exclusion |
| **RECAP-OFA** | **different graphs** | **no labels, target context, or fine-tuning** | **one frozen model per split** | **full target graph** |

Results from different supervision or evaluation populations are reported as
separate strata and should not be interpreted as strictly equivalent rankings.

### OFA source-target splits

| Setting | Source graphs | Target graphs | Purpose |
|---|---|---|---|
| A | PubMed, Flickr, Questions, YelpChi | Cora, CiteSeer, ACM, BlogCatalog, Facebook, Weibo, Reddit, Amazon | original multi-domain split |
| B | PubMed, Cora, Questions, YelpChi | Flickr, BlogCatalog, Facebook, Weibo, Reddit | leave-Social-domain-out |
| C | PubMed, Cora, CiteSeer, ACM | BlogCatalog, Flickr, Reddit, Amazon, Questions | citation-only sources |

Unless explicitly noted, formal results use seeds `0/1/2`, population standard
deviation (`ddof=0`), and a seed-first dataset macro.

## Main results

### RECAP

| Setting | AUROC (%) | AUPRC (%) |
|---|---:|---:|
| RECAP-OFO, 12-dataset macro | 71.08 ± 0.35 | 23.61 ± 0.26 |
| RECAP-OFA Setting A | 74.65 ± 0.23 | 27.04 ± 0.34 |
| RECAP-OFA Setting B | 67.75 ± 0.22 | 21.98 ± 0.32 |
| RECAP-OFA Setting C | 67.31 ± 0.43 | 17.50 ± 0.09 |

The consolidated protocol audit covers 450 successful training runs and 720
final evaluations. Complete per-dataset tables, timing, stability, supervision
rights, and consistency checks are in
[`rebuttal/reports/RECAP_EXPERIMENT_PROTOCOL_COMPLETION_REPORT.md`](rebuttal/reports/RECAP_EXPERIMENT_PROTOCOL_COMPLETION_REPORT.md).

### Unsupervised OFO baselines

| Method | AUROC (%) | AUPRC (%) |
|---|---:|---:|
| DOMINANT | 53.32 ± 0.18 | 10.40 ± 0.01 |
| AnomalyDAE | 56.24 ± 0.33 | 9.97 ± 0.11 |
| CoLA | 54.36 ± 0.19 | 11.88 ± 0.24 |
| ADA-GAD | 57.57 ± 0.75 | 7.17 ± 0.06 |
| DiffGAD | 55.03 ± 2.17 | 10.61 ± 0.24 |
| GUIDE | **75.19 ± 0.14** | **33.09 ± 0.09** |
| RECAP-OFO | 71.08 ± 0.35 | 23.61 ± 0.26 |

GUIDE is stronger when every method can be retrained on each target. RECAP's
intended advantage is source-only cross-graph transfer, not unconditional
dominance over target-specific detectors.

### OFA baselines

Dataset-macro AUROC/AUPRC (%):

| Method | Source labels | Target context | Setting A | Setting B | Setting C |
|---|:---:|---|---:|---:|---:|
| ARC | yes | 10 labeled normal nodes | 78.69/36.49 | 72.76/29.49 | 66.18/18.86 |
| IA-GGAD | yes | 10 unlabeled references | 77.22/35.51 | 72.84/30.74 | 63.03/17.74 |
| UNPrompt | yes | none | 58.02/10.22 | 57.92/11.69 | 59.25/11.37 |
| AnomalyGFM-ZS | yes | none | 53.07/8.02 | 49.64/8.43 | 51.03/5.57 |
| OWLEYE | yes | 10 unlabeled patterns | 76.04/35.67 | 71.79/28.66 | 60.33/18.01 |
| **RECAP-OFA** | **no** | **none** | **74.65/27.04** | **67.75/21.98** | **67.31/17.50** |

We additionally construct protocol-aligned, non-official OFA adaptations of
GUIDE and DiffGAD by training one model on the same four source graphs and
freezing it for every target:

| Adapted method | Setting A | Setting B | Setting C |
|---|---:|---:|---:|
| GUIDE-OFA-adapted | 63.86/27.74 | 50.66/6.90 | 55.68/7.01 |
| DiffGAD-OFA-adapted | 66.94/21.11 | 59.61/18.47 | **69.61/17.99** |
| **RECAP-OFA** | **74.65/27.04** | **67.75/21.98** | 67.31/17.50 |

See the [complete experiment index](rebuttal/README.md) for every protocol,
runner, result table, audit, and interpretation boundary.

## Repository layout

```text
.
├── model.py, detector.py, utils.py      # RECAP model and data pipeline
├── train.py, inference.py               # standard training and inference
├── config.py, params/                    # configuration and paper hyperparameters
├── ablation/                             # module, simple-combination, and 2×2 ablations
├── interpretability/                     # community cards and node explanations
├── tuning_hyperparams/                   # sensitivity and interaction studies
├── rebuttal/                             # locked protocols, baseline adapters, audits, reports
├── tools/                                # large-dataset conversion and ANN-cache tools
├── docs/                                 # large-data and approximate-KNN documentation
└── ara/                                  # research provenance and decision log
```

## Environment

The formal RECAP environment used Python 3.12.3, PyTorch 2.11.0+cu128,
PyTorch Geometric 2.7.0, NumPy 2.1.3, SciPy 1.17.1, and scikit-learn 1.8.0.
`requirements.txt` is the exact Linux/CUDA lock from that environment, not a
portable CPU dependency list.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the optional experiment dependencies only when needed:

```bash
python -m pip install -r requirements-baselines.txt
python -m pip install -r requirements-large.txt
```

`faiss-cpu` is loaded lazily and is required only by the large-graph ANN route
and the context-neighborhood ablation. T-Finance/T-Social conversion requires
DGL 1.1.3 in a separate Python 3.10 environment. Exact baseline versions and
upstream revisions are recorded in `rebuttal/*/environment_matrix.md` and
`rebuttal/*/upstream_manifest.json`.

## Data

Datasets are not committed because of their size and licensing conditions.
Place the 12 standard `.mat` files in `dataset/`:

```text
pubmed, cora, citeseer, ACM, Flickr, BlogCatalog,
Facebook, weibo, Reddit, questions, YelpChi, Amazon
```

Each file must contain adjacency (`Network` or `A`), node attributes, and
labels (`Label` or `gnd`). The first load creates a versioned feature-alignment
cache. The accepted file identities are recorded in
[`rebuttal/data_manifest.json`](rebuttal/data_manifest.json).

For T-Finance, T-Social, and DGraph-Fin, follow
[`docs/LARGE_DATASETS.md`](docs/LARGE_DATASETS.md). The canonical layout is:

```text
dataset/<name>/
├── adjacency.npz
├── features.npy
├── labels.npy
├── metadata.json
└── evaluation_mask.npy    # DGraph-Fin only
```

## Train and infer

Run all commands from the repository root. The paper-aligned configuration is
`params/recap_auprc_best.json`.

### OFA Setting A

```bash
python train.py \
  --model recap_auprc_best \
  --device cuda:0 \
  --trials 3 \
  --epochs 100 \
  --train-datasets pubmed Flickr questions YelpChi \
  --test-datasets cora citeseer ACM BlogCatalog Facebook weibo Reddit Amazon
```

The checkpoints are written to
`checkpoints/recap_auprc_best/trial_<seed>/model.pt`.

### Target-specific RECAP-OFO

```bash
python train.py \
  --model recap_auprc_best \
  --device cuda:0 \
  --trials 3 \
  --epochs 100 \
  --train-datasets cora \
  --test-datasets cora
```

### Frozen-checkpoint inference

```bash
python inference.py \
  --checkpoint checkpoints/recap_auprc_best/trial_0/model.pt \
  --datasets cora citeseer ACM \
  --device cuda:0 \
  --output-dir inference_results
```

Use `--batch` with the model directory to evaluate all trials. Neither training
nor inference uses target labels for model selection.

## Large-graph inference

T-Finance retains exact blockwise KNN. T-Social and DGraph-Fin automatically
use FAISS IVF-PQ candidate search plus exact candidate reranking and
memory-bounded scoring. No ordinary dataset is routed through ANN.

```bash
python tools/build_large_knn_cache.py \
  --dataset tsocial --dims 32 --num-hops 4 --knn-k 64 --device cuda:0

python tools/build_large_knn_cache.py \
  --dataset dgraphfin --dims 32 --num-hops 4 --knn-k 64 --device cuda:0
```

The accepted full-graph run scores all 39,357 T-Finance nodes, all 3,700,550
DGraph-Fin nodes, and all 5,781,065 T-Social nodes. This establishes
**target-side inference scalability only**. The unadapted Setting-A checkpoints
produce below-random mean predictive results on all three targets, so the large
experiment must not be presented as evidence of predictive effectiveness. See
[`rebuttal/reports/LARGE_TARGET_INFERENCE_REPORT.md`](rebuttal/reports/LARGE_TARGET_INFERENCE_REPORT.md).

## Reproducibility and audits

- Formal runners write resumable, per-run artifacts and freeze scores before
  labels are opened.
- Dataset, checkpoint, candidate-cache, mask, and score hashes are recorded.
- AUROC/AUPRC are independently recomputed from frozen scores.
- Reload consistency, finite-score, missing-cell, duplicate-cell, and aggregate
  consistency gates are enforced.
- Raw datasets, checkpoints, vendor archives, KNN caches, and bulky logs are
  intentionally excluded from Git. Compact analysis tables and final audits are
  retained.

To regenerate the consolidated audit after all underlying artifacts are present:

```bash
python -m rebuttal.protocol_completion_audit
```

## Result interpretation

The repository intentionally retains weak, high-variance, and negative results.
Do not average Settings A/B/C because their target sets differ. Do not directly
rank supervised held-out OFO, full-graph unsupervised OFO, and zero-shot OFA as
if their label rights and task difficulty were identical. Adapted GUIDE/DiffGAD
OFA results are protocol-aligned research adaptations, not official variants.
