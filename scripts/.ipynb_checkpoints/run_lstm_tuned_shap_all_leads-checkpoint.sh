#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
TASK=${TASK:-regression}
DEVICE=${DEVICE:-cuda}
TUNED_ROOT=${TUNED_ROOT:-${REPO}/data/tuned_wide/lstm}
GLOBAL_SHAP_DIR=${GLOBAL_SHAP_DIR:-${REPO}/data/shap_tuned_wide/lstm}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}      # 0 means full test period
BACKGROUND_SAMPLES=${BACKGROUND_SAMPLES:-100}
BATCH_SIZE=${BATCH_SIZE:-50}
TOP_N=${TOP_N:-25}
MODEL_MODULE=${MODEL_MODULE:-lstm_model_tuned}

mkdir -p "${REPO}/logs" "${GLOBAL_SHAP_DIR}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

for LEAD in 3 6 12; do
    LEAD_PAD=$(printf "%02d" "${LEAD}")
    RUN_DIR="${TUNED_ROOT}/lead${LEAD_PAD}_${TASK}"
    MODEL_DIR="${RUN_DIR}/models"
    OUT_DIR="${RUN_DIR}/shap"

    echo "=== Computing tuned LSTM SHAP | lead=${LEAD} task=${TASK} ==="
    python scripts/compute_shap_tuned_lstm.py \
        --config "${CONFIG}" \
        --lead "${LEAD}" \
        --task "${TASK}" \
        --device "${DEVICE}" \
        --tuned-root "${TUNED_ROOT}" \
        --model-dir "${MODEL_DIR}" \
        --model-module "${MODEL_MODULE}" \
        --output-dir "${OUT_DIR}" \
        --global-shap-dir "${GLOBAL_SHAP_DIR}" \
        --max-eval-samples "${MAX_EVAL_SAMPLES}" \
        --background-samples "${BACKGROUND_SAMPLES}" \
        --batch-size "${BATCH_SIZE}" \
        --top-n "${TOP_N}"
done

echo "Tuned LSTM SHAP complete."
