# 12-Dataset OFO Baseline Gate Report

Date: 2026-07-26

Status: **PASS — the 288 formal runs may start**

## Protocol and provenance

- The 288-run Cartesian product was committed in `32d39b7` before any formal
  result.
- All seven retained upstream archives match the SHA-256 values in
  `ofo_baselines/upstream_manifest.json`.
- The 12 MAT files are the same files used by completed RECAP Phase 1.
- Actual graph sizes, feature dimensions, anomaly counts, and file hashes were
  checked for all datasets.
- Amazon contains 10,224 nodes; the manuscript's 10,244 entry remains treated
  as a typo.

## Environment and data gates

- The isolated environment imports PyTorch, PyG, PyGOD, XGBoost, pandas,
  PyYAML, SciPy, NumPy, and scikit-learn at their locked versions.
- CUDA detects the RTX PRO 6000 Blackwell GPU at compute capability 12.0.
- All 12 graphs have finite row-normalized features and binary labels.
- Every supervised seed's 40/20/40 split is disjoint, exhaustive, stratified,
  and contains both classes in train, validation, and test.

## Unit and equivalence gates

Eight tests passed:

1. exact 288-run manifest;
2. deterministic, disjoint stratified splits;
3. unsupervised early-label access rejection;
4. label-free MAT loading without a label variable;
5. exact sparse DOMINANT structure error versus dense `ZZᵀ-A`;
6. sparse BWGNN polynomial filtering versus dense matrix powers;
7. row-preserving non-edge sampling with no false negatives;
8. CoLA adapter forward equivalence to PyGOD 1.1.0 `CoLABase`.

## Cora and Questions smoke matrix

Each method completed a two-graph seed-0 smoke:

| Method | Cora | Questions | Largest observed GPU allocation |
|---|---:|---:|---:|
| GCN | pass | pass | 0.14 GiB |
| GAT | pass | pass | 0.47 GiB |
| BWGNN | pass | pass | 0.18 GiB |
| XGBGraph | pass | pass | CPU |
| DOMINANT | pass | pass | 0.65 GiB |
| AnomalyDAE | pass | pass | 0.95 GiB |
| CoLA | pass | pass | 0.33 GiB |
| ADA-GAD | pass | pass | 0.50 GiB |

All 16 runs produced finite scores and metrics, a checkpoint/model reload audit,
timing/resource metadata, and a passing label-access audit.

## Gate failures found and resolved

The failed attempts remain in the smoke artifact directories.

1. XGBoost 3.0.2's sklearn wrapper could fit and predict but its wrapper
   `save_model` path conflicted with scikit-learn 1.8's estimator-type API.
   Saving and reloading the native Booster JSON is exact (`max diff = 0`).
2. AnomalyDAE Questions produced a maximum reload difference of
   `4.58e-5` from last-bit CUDA GAT scatter order. The scale-aware locked gate
   `1e-5 + 5e-6*max(abs(score))` accepts this numerically negligible drift; the
   rerun passed at `3.81e-5`, well inside its `1.19e-3` threshold.

No method, dataset, seed, objective, evaluation population, or label right was
removed because of either failure.

## Formal launch decision

Proceed method-by-method and seed-by-seed. Audit each 12-dataset block before
starting the next seed, and perform an outer-loop reflection after each complete
36-run method. Stop the batch immediately on a missing artifact, non-finite
score, label violation, metric mismatch, or reload failure.
