#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
TASK=${TASK:-regression}
TUNED_BASE=${TUNED_BASE:-${REPO}/data/tuned_wide}
OUTPUT_DIR=${OUTPUT_DIR:-${TUNED_BASE}/model_intercomparison}
FIG_DIR=${FIG_DIR:-${REPO}/figures/tuned_wide/model_intercomparison}
REFERENCE_MODEL=${REFERENCE_MODEL:-cnn}
TOP_N=${TOP_N:-12}

mkdir -p "${REPO}/logs" "${OUTPUT_DIR}" "${FIG_DIR}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

python scripts/compare_tuned_models.py \
    --config "${CONFIG}" \
    --task "${TASK}" \
    --tuned-base "${TUNED_BASE}" \
    --output-dir "${OUTPUT_DIR}" \
    --fig-dir "${FIG_DIR}" \
    --models cnn lstm xgb \
    --leads 3 6 12 \
    --reference-model "${REFERENCE_MODEL}" \
    --top-n "${TOP_N}"

echo "Model intercomparison complete."
