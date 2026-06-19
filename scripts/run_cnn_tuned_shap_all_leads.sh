#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
TASK=${TASK:-regression}
DEVICE=${DEVICE:-cuda}
TUNED_ROOT=${TUNED_ROOT:-${REPO}/data/tuned_wide/cnn}
GLOBAL_SHAP_DIR=${GLOBAL_SHAP_DIR:-${REPO}/data/shap_tuned_wide/cnn}
FIG_ROOT=${FIG_ROOT:-${REPO}/figures/tuned_wide/cnn}

# 0 means full test period. For smoke tests, set MAX_EVAL_SAMPLES=40.
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
BACKGROUND_SAMPLES=${BACKGROUND_SAMPLES:-80}
DEEP_BATCH_SIZE=${DEEP_BATCH_SIZE:-4}

# Predictor maps to include in the multi-panel seasonal map gallery.
MAP_VARIABLES=${MAP_VARIABLES:-"sst d20 tauu olr slp"}

mkdir -p "${REPO}/logs"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

for LEAD in 3 6 12; do
    LEAD_PAD=$(printf "%02d" "${LEAD}")
    RUN_DIR="${TUNED_ROOT}/lead${LEAD_PAD}_${TASK}"

    echo "=== Computing tuned CNN SHAP | lead=${LEAD} task=${TASK} ==="

    python scripts/compute_shap_tuned_cnn.py \
        --config "${CONFIG}" \
        --lead "${LEAD}" \
        --task "${TASK}" \
        --device "${DEVICE}" \
        --tuned-root "${TUNED_ROOT}" \
        --model-dir "${RUN_DIR}/models" \
        --output-dir "${RUN_DIR}/shap" \
        --global-shap-dir "${GLOBAL_SHAP_DIR}" \
        --fig-dir "${FIG_ROOT}/lead${LEAD_PAD}_${TASK}" \
        --model-module cnn_model_tuned \
        --max-eval-samples "${MAX_EVAL_SAMPLES}" \
        --background-samples "${BACKGROUND_SAMPLES}" \
        --deep-batch-size "${DEEP_BATCH_SIZE}" \
        --map-variables ${MAP_VARIABLES}
done

echo "Tuned CNN SHAP complete."
