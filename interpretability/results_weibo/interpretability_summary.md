# RECAP Interpretability Summary

This directory contains community-level and node-level interpretability artifacts.

## Configuration

- Model: `recap_auprc_best`
- Epochs: `100`
- Trials: `3`
- Source-train datasets: `['pubmed', 'Flickr', 'questions', 'YelpChi']`
- Analysis datasets: `['weibo']`
- Checkpoint dir: `interpretability/results_weibo/checkpoints`
- Target usage: source-trained checkpoints are reloaded and directly forwarded on analysis graphs; target labels are post-hoc only.

## Dataset Metrics

| Dataset | Trial | Total AUPRC | Adhesion AUPRC | Context AUPRC | Top Lift | Context Lift |
|---|---:|---:|---:|---:|---:|---:|
| weibo | 0 | 0.3479 | 0.3065 | 0.3381 | 3.7031 | 21.9089 |
| weibo | 1 | 0.3425 | 0.3148 | 0.2841 | 3.8871 | 14.3818 |
| weibo | 2 | 0.3378 | 0.2904 | 0.3137 | 3.3121 | 15.6069 |

## Training / Inference Manifest

| Trial | Mode | Seed | Epochs | Checkpoint |
|---:|---|---:|---:|---|
| 0 | loaded_checkpoint | 0 | 100 | `interpretability/results_weibo/checkpoints/recap_auprc_best/trial_0/model.pt` |
| 1 | loaded_checkpoint | 1 | 100 | `interpretability/results_weibo/checkpoints/recap_auprc_best/trial_1/model.pt` |
| 2 | loaded_checkpoint | 2 | 100 | `interpretability/results_weibo/checkpoints/recap_auprc_best/trial_2/model.pt` |

## Stability Across Seeds

| Dataset | Trials | Top-Node Jaccard | Score Spearman |
|---|---:|---:|---:|
| weibo | 3 | 0.8697 +/- 0.0052 | 0.8245 +/- 0.0053 |

## Files

- `community_cards.csv`: community-level diagnostic cards.
- `node_explanations.csv`: top-node diagnosis types and node-level explanations.
- `explanation_metrics.csv`: faithfulness and concentration metrics.
- `stability_summary.csv`: seed-level explanation stability, when multiple trials are used.
- `training_manifest.csv`: source-training checkpoints and target-inference provenance.
- `reports/`: per-dataset Markdown reports.
- `figures/`: paper-ready community lift/mass/response plots, t-SNE confident-community maps, pre-training embeddings, trained community embeddings, community risk maps, score maps, component-percentile plots, and lift bars.
