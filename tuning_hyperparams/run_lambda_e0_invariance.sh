#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-100}"
TRIALS="${TRIALS:-3}"
RUN_NAME="${RUN_NAME:-lambda_e0_invariance_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/tuning_hyperparams/lambda_e0_invariance_results/${RUN_NAME}}"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_ROOT}/tuning_hyperparams/sensitivity_results/sensitivity_default_v2/configs/baseline/recap.json}"
REFERENCE_SUMMARY="${REFERENCE_SUMMARY:-${PROJECT_ROOT}/tuning_hyperparams/sensitivity_results/sensitivity_default_v2/sensitivity_summary.csv}"
KNN_CACHE_DIR="${KNN_CACHE_DIR:-${PROJECT_ROOT}/knn_cache}"
KNN_SEARCH_DTYPE="${KNN_SEARCH_DTYPE:-auto}"
FORCE="${FORCE:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
PARAMS="${PARAMS:-beta lambda_H cluster_init_gain tau_c tau_s num_hops}"
TRAIN_DATASETS="${TRAIN_DATASETS:-pubmed Flickr questions YelpChi}"
TEST_DATASETS="${TEST_DATASETS:-Facebook cora citeseer ACM BlogCatalog weibo Reddit Amazon}"

read -r -a PARAM_ARGS <<< "${PARAMS}"
read -r -a TRAIN_DATASET_ARGS <<< "${TRAIN_DATASETS}"
read -r -a TEST_DATASET_ARGS <<< "${TEST_DATASETS}"

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/tuning_hyperparams/lambda_e0_invariance.py"
  --base-config "${BASE_CONFIG}"
  --reference-summary "${REFERENCE_SUMMARY}"
  --params "${PARAM_ARGS[@]}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --trials "${TRIALS}"
  --knn-cache-dir "${KNN_CACHE_DIR}"
  --knn-search-dtype "${KNN_SEARCH_DTYPE}"
  --train-datasets "${TRAIN_DATASET_ARGS[@]}"
  --test-datasets "${TEST_DATASET_ARGS[@]}"
)

if [[ "${FORCE}" == "1" ]]; then
  CMD+=(--force)
fi

if [[ "${PLOT_ONLY}" == "1" ]]; then
  CMD+=(--plot-only)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi

cd "${PROJECT_ROOT}"
echo "Running lambda_E=0 invariance check:"
printf '  %q' "${CMD[@]}"
echo
exec "${CMD[@]}"
