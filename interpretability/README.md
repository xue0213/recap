# RECAP Interpretability

This directory contains experiments for showing what the RECAP community layer
contributes beyond average AUROC/AUPRC ranking.

The main script generates four kinds of artifacts, ordered by importance:

- `Community Cards`: community-level diagnostics such as risk tier, soft mass,
  anomaly lift, average score components, and representative high-risk nodes.
- `Node Explanation Reports`: diagnosis-specific explanations for top anomalies,
  including prototype distance percentile, community risk, assignment ambiguity,
  and KNN community-context mismatch.
- `Explanation Metrics`: component AUROC/AUPRC, top-risk anomaly lift,
  context-mismatch lift, and community concentration of top anomalies.
- `Residual Community Visualizations`: pre-training residual embeddings,
  trained residual-community embeddings, PCA-based community risk maps, anomaly
  score landscapes, score-component percentile plots, and community lift bars.

The default analysis dataset is `weibo`, because the current ablation results
show that community/context information matters most clearly there. Other
datasets can be added with `--analysis-datasets`.

## Quick Smoke Run

```bash
python interpretability/run_interpretability.py \
  --device cuda:0 \
  --quick \
  --no-diagnostics \
  --output-dir /tmp/recap_interpretability_smoke
```

## Suggested Case Study Run

```bash
python interpretability/run_interpretability.py \
  --device cuda:0 \
  --model recap_auprc_best \
  --trials 3 \
  --epochs 100 \
  --analysis-datasets weibo \
  --top-nodes 30 \
  --top-fraction 0.05 \
  --no-diagnostics \
  --output-dir interpretability/results_weibo
```

This run trains RECAP only on source graphs, saves the source-trained checkpoint
under `interpretability/results_weibo/checkpoints/`, reloads that checkpoint,
and then directly forwards Weibo for interpretability analysis. Weibo labels are
used only for post-hoc validation metrics and figure annotations.

If checkpoints already exist, skip source training and run target-only
interpretability:

```bash
python interpretability/run_interpretability.py \
  --device cuda:0 \
  --checkpoint-paths \
    interpretability/results_weibo/checkpoints/recap_auprc_best/trial_0/model.pt \
    interpretability/results_weibo/checkpoints/recap_auprc_best/trial_1/model.pt \
    interpretability/results_weibo/checkpoints/recap_auprc_best/trial_2/model.pt \
  --analysis-datasets weibo \
  --top-nodes 30 \
  --top-fraction 0.05 \
  --no-diagnostics \
  --output-dir interpretability/results_weibo_from_checkpoints
```

## Broader Report

```bash
python interpretability/run_interpretability.py \
  --device cuda:0 \
  --model recap_auprc_best \
  --trials 3 \
  --epochs 100 \
  --analysis-datasets Facebook cora citeseer ACM BlogCatalog weibo Reddit Amazon \
  --top-nodes 20 \
  --top-fraction 0.05 \
  --no-diagnostics \
  --output-dir interpretability/results_all
```

## Outputs

- `community_cards.csv`: one row per community.
- `node_explanations.csv`: one row per top-ranked anomaly, with diagnosis type
  and concrete evidence fields.
- `explanation_metrics.csv`: faithfulness/concentration proxy metrics.
- `stability_summary.csv`: pairwise seed stability when `--trials > 1`.
- `training_manifest.csv`: saved checkpoints and target-inference provenance.
- `reports/*.md`: per-dataset case-study reports.
- `figures/*`: `pretrain_residual_embedding`,
  `trained_residual_communities`, `paper_confident_communities_tsne`,
  `paper_high_risk_response_maps`, `paper_community_mass_lift`,
  `paper_community_lift_bars`, `community_risk_map`, `anomaly_score_map`,
  `score_component_percentiles`, and `community_lift_bars` plots.
- `interpretability_summary.md`: short index of all generated outputs.

## Interpretation Angle

These artifacts support a cautious, evidence-aligned claim:

> The RECAP community layer is not primarily responsible for average ranking
> gains. Its key contribution is an interpretable soft-prototype and
> neighborhood-context layer that explains each anomaly as prototype deviation,
> community-context mismatch, or both.
