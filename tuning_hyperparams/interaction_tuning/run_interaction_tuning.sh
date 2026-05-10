#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
PLAN="${PLAN:-performance_v2}"
EPOCHS="${EPOCHS:-200}"
TRIALS="${TRIALS:-3}"
RUN_NAME="${RUN_NAME:-interaction_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/tuning_hyperparams/interaction_tuning/results/${RUN_NAME}}"
BASE_CONFIG="${BASE_CONFIG:-${PROJECT_ROOT}/params/recap.json}"
KNN_CACHE_DIR="${KNN_CACHE_DIR:-${PROJECT_ROOT}/knn_cache}"
KNN_SEARCH_DTYPE="${KNN_SEARCH_DTYPE:-auto}"
FORCE="${FORCE:-0}"
PLOT_ONLY="${PLOT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
AUROC_WEIGHT="${AUROC_WEIGHT:-1.0}"
AUPRC_WEIGHT="${AUPRC_WEIGHT:-1.0}"
TRAIN_DATASETS="${TRAIN_DATASETS:-pubmed Flickr questions YelpChi}"
TEST_DATASETS="${TEST_DATASETS:-Facebook cora citeseer ACM BlogCatalog weibo Reddit Amazon}"

read -r -a TRAIN_DATASET_ARGS <<< "${TRAIN_DATASETS}"
read -r -a TEST_DATASET_ARGS <<< "${TEST_DATASETS}"

CMD=(
  "${PYTHON_BIN}"
  "${PROJECT_ROOT}/tuning_hyperparams/interaction_tuning/interaction_tuning.py"
  --base-config "${BASE_CONFIG}"
  --plan "${PLAN}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --trials "${TRIALS}"
  --knn-cache-dir "${KNN_CACHE_DIR}"
  --knn-search-dtype "${KNN_SEARCH_DTYPE}"
  --auroc-weight "${AUROC_WEIGHT}"
  --auprc-weight "${AUPRC_WEIGHT}"
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
echo "Running interaction tuning:"
printf '  %q' "${CMD[@]}"
echo
exec "${CMD[@]}"
