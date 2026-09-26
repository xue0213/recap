# RECAP Experiment Index

This directory is the reproducibility layer for the complete RECAP experiment
suite. Protocol files lock the scientific question and label rights before a
run. Runners create resumable artifacts. Analysis scripts independently
recompute metrics. Reports contain the compact, publication-facing results.

## Experiment map

| Experiment | Scope | Protocol | Runner/analysis | Final report |
|---|---|---|---|---|
| RECAP-OFO and RECAP-OFA A/B/C | 45 training runs, 90 target evaluations | [`RECAP_EXPERIMENT_PROTOCOL.md`](RECAP_EXPERIMENT_PROTOCOL.md) | `phase1_runner.py`, `phase1_analysis.py`, `phase1_report.py` | [`PHASE1_FINAL_REPORT.md`](reports/PHASE1_FINAL_REPORT.md) |
| Questions OFO addendum | 3 training/evaluation runs | [`QUESTIONS_OFO_ADDENDUM_PROTOCOL.md`](QUESTIONS_OFO_ADDENDUM_PROTOCOL.md) | `questions_ofo_addendum.py` | [`QUESTIONS_OFO_ADDENDUM_REPORT.md`](reports/QUESTIONS_OFO_ADDENDUM_REPORT.md) |
| Original OFA baselines | ARC, IA-GGAD, UNPrompt, AnomalyGFM-ZS on A/B/C | [`BASELINE_OFA_REPROTOCOL.md`](BASELINE_OFA_REPROTOCOL.md), [`BASELINE_BC_SUPPLEMENT_PROTOCOL.md`](BASELINE_BC_SUPPLEMENT_PROTOCOL.md) | `baselines/baseline_runner.py`, `baselines/baseline_bc_runner.py` | [`PHASE2_OFA_BASELINE_REPORT.md`](reports/PHASE2_OFA_BASELINE_REPORT.md), [`PHASE2_BC_BASELINE_SUPPLEMENT_REPORT.md`](reports/PHASE2_BC_BASELINE_SUPPLEMENT_REPORT.md) |
| Twelve-dataset OFO baselines | GCN, GAT, BWGNN, XGBGraph, DOMINANT, AnomalyDAE, CoLA, ADA-GAD | [`OFO_12_BASELINE_PROTOCOL.md`](OFO_12_BASELINE_PROTOCOL.md) | `ofo_baselines/runner.py`, `ofo_baselines/analysis.py` | [`OFO_12_BASELINE_REPORT.md`](reports/OFO_12_BASELINE_REPORT.md) |
| Recent baselines | DiffGAD and GUIDE OFO; OWLEYE OFA | [`THREE_BASELINE_EXTENSION_PROTOCOL.md`](THREE_BASELINE_EXTENSION_PROTOCOL.md) | `new_baselines/runner.py`, `new_baselines/analysis.py` | [`THREE_BASELINE_EXTENSION_REPORT.md`](reports/THREE_BASELINE_EXTENSION_REPORT.md) |
| GUIDE/DiffGAD source-only adaptations | 2 methods × 3 settings × 3 seeds | [`OFA_ADAPTED_UNSUPERVISED_PROTOCOL.md`](OFA_ADAPTED_UNSUPERVISED_PROTOCOL.md) | `new_baselines/ofa_adapted.py` | [`OFA_ADAPTED_UNSUPERVISED_REPORT.md`](reports/OFA_ADAPTED_UNSUPERVISED_REPORT.md) |
| Full protocol audit | core 450 runs and 720 evaluations | [`EXPERIMENT_PROTOCOL_COMPLETION_AUDIT.md`](EXPERIMENT_PROTOCOL_COMPLETION_AUDIT.md) | `protocol_completion_audit.py` | [`RECAP_EXPERIMENT_PROTOCOL_COMPLETION_REPORT.md`](reports/RECAP_EXPERIMENT_PROTOCOL_COMPLETION_REPORT.md) |
| Context-neighborhood ablation | residual KNN vs aligned-feature KNN | controlled in `context_neighbor_ablation.py` | `context_neighbor_ablation.py` | [`context_neighbor_ablation/RESULTS.md`](reports/context_neighbor_ablation/RESULTS.md) |
| Large-target inference | T-Finance, DGraph-Fin, T-Social | [`LARGE_TARGET_INFERENCE_PROTOCOL.md`](LARGE_TARGET_INFERENCE_PROTOCOL.md) | `large_target_inference/runner.py`, `large_target_inference/analysis.py` | [`LARGE_TARGET_INFERENCE_REPORT.md`](reports/LARGE_TARGET_INFERENCE_REPORT.md) |
| Large-target diagnostics/optimization | source scans and label-free adaptations | `LARGE_TARGET_*_PROTOCOL.md` | `large_target_optimization/` | [`LARGE_TARGET_OPTIMIZATION_REPORT.md`](reports/LARGE_TARGET_OPTIMIZATION_REPORT.md) |
| T-Finance source transfer | T-Finance → three large targets | [`TFINANCE_SOURCE_TRANSFER_PROTOCOL.md`](TFINANCE_SOURCE_TRANSFER_PROTOCOL.md) | `large_target_optimization/tfinance_transfer.py` | [`TFINANCE_SOURCE_TRANSFER_REPORT.md`](reports/TFINANCE_SOURCE_TRANSFER_REPORT.md) |

The matched residual/community factorial ablation is indexed separately under
[`../ablation/matched_2x2/`](../ablation/matched_2x2/).

## Core completion status

| Requirement | Completed |
|---|---:|
| RECAP-OFO | 12 datasets × 3 seeds |
| RECAP-OFA | 3 splits × 3 seeds, 54 target evaluations |
| OFO baselines | 10 methods × 12 datasets × 3 seeds |
| OFA baselines | 5 methods × 18 targets across A/B/C × 3 seeds |
| Community stability | 30 scopes × 3 seed pairs |
| Checkpoints/reload gates | 225 checkpoints / 45 final reload gates |

All required core cells pass the consolidated audit. Later adapted-OFA,
context-neighborhood, matched-2×2, and large-target experiments are reported
separately and are not retroactively added to the `450/720` count.

## Recommended execution order

The commands below assume the repository root is the working directory and
that datasets/vendor sources are available at explicit paths.

### 1. RECAP

```bash
python -m rebuttal.phase1_runner manifest
python -m rebuttal.phase1_runner \
  --dataset-dir dataset --device cuda:0 preflight
python -m rebuttal.phase1_runner \
  --dataset-dir dataset --device cuda:0 execute
python -m rebuttal.phase1_analysis
python -m rebuttal.phase1_report
```

The runner is resumable. Use `status` to inspect completion and `run-one` for a
specific manifest cell.

### 2. Original OFA baselines

```bash
python -m rebuttal.baselines.baseline_runner \
  --dataset-dir dataset --vendor-root /path/to/vendor manifest

python -m rebuttal.baselines.baseline_runner \
  --dataset-dir dataset --vendor-root /path/to/vendor \
  --device cuda:0 run-pending
```

Run the B/C supplement with `rebuttal.baselines.baseline_bc_runner` using the
same explicit data, vendor, output, and device arguments.

### 3. OFO baselines

The OFO runner executes one manifest cell at a time:

```bash
python -m rebuttal.ofo_baselines.runner \
  --run-id <manifest-run-id> \
  --dataset-dir dataset \
  --device cuda:0
```

The exact environment and method-specific adaptations are documented in
[`ofo_baselines/environment_matrix.md`](ofo_baselines/environment_matrix.md)
and [`OFO_12_BASELINE_PROTOCOL.md`](OFO_12_BASELINE_PROTOCOL.md).

### 4. DiffGAD, GUIDE, and OWLEYE

```bash
python -m rebuttal.new_baselines.runner manifest
python -m rebuttal.new_baselines.runner prepare-vendor \
  --dataset-dir dataset --vendor-root /path/to/vendor
python -m rebuttal.new_baselines.runner run \
  --dataset-dir dataset --vendor-root /path/to/vendor --device cuda:0
```

Exact upstream commits and archive hashes are in
[`new_baselines/upstream_manifest.json`](new_baselines/upstream_manifest.json).

### 5. Source-only GUIDE/DiffGAD adaptations

```bash
python -m rebuttal.new_baselines.ofa_adapted preflight \
  --dataset-dir dataset --vendor-root /path/to/vendor
python -m rebuttal.new_baselines.ofa_adapted run \
  --dataset-dir dataset --vendor-root /path/to/vendor --device cuda:0
python -m rebuttal.new_baselines.ofa_adapted analyze \
  --dataset-dir dataset --vendor-root /path/to/vendor
```

These are our protocol-aligned adaptations, not variants released by the
original GUIDE or DiffGAD authors.

### 6. Consolidated audit

```bash
python -m rebuttal.protocol_completion_audit
```

The audit must be run only after all expected underlying artifacts are present.
It checks missing/duplicate cells, seeds, finite metrics, label rights, reload
consistency, aggregation order, and evaluation strata.

### 7. Large-target inference

After preparing canonical bundles and installing `requirements-large.txt`:

```bash
python -m rebuttal.large_target_inference.runner preflight \
  --dataset-root dataset \
  --checkpoint-root /path/to/setting_a_checkpoints \
  --output-root /path/to/large_target_artifacts \
  --device cuda:0

python -m rebuttal.large_target_inference.runner gates \
  --output-root /path/to/large_target_artifacts --device cuda:0

python -m rebuttal.large_target_inference.runner run-target \
  --dataset-root dataset \
  --checkpoint-root /path/to/setting_a_checkpoints \
  --output-root /path/to/large_target_artifacts \
  --target tfinance --device cuda:0
```

Repeat `run-target` for `dgraphfin` and `tsocial`, then run the analysis script
with explicit dataset, checkpoint, artifact, and report paths.

## Result hierarchy

1. `reports/RECAP_EXPERIMENT_PROTOCOL_COMPLETION_REPORT.md` is the canonical
   compact report for the core experiment suite.
2. `artifacts/protocol_completion/analysis/` contains its machine-readable
   source tables and audit output.
3. Experiment-specific reports provide method adaptation details and additional
   per-dataset/resource tables.
4. `findings.md`, `research-log.md`, and `research-state.yaml` preserve the
   chronological interpretation, including failed hypotheses and negative
   results.

## Scientific boundaries

- OFO and OFA answer different questions and are never collapsed into one
  overall ranking.
- Supervised OFO uses a held-out 40% population. Unsupervised OFO uses the full
  graph.
- OFA baseline target contexts and evaluation exclusions are declared per
  method in the consolidated evaluation-strata table.
- Settings A/B/C have different target sets and are not averaged together.
- DGraph-Fin background labels remain graph context but are excluded from
  metrics through an immutable evaluation mask.
- Large-target experiments establish target-side inference scalability only.
  Their predictive results are retained as negative evidence.
- No target labels are used to invert scores, select checkpoints, choose an ANN
  configuration, or tune RECAP.
