#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
TASK=${TASK:-regression}
TUNED_ROOT=${TUNED_ROOT:-${REPO}/data/tuned_wide/xgb}
GLOBAL_SHAP_DIR=${GLOBAL_SHAP_DIR:-${REPO}/data/shap_tuned_wide/xgb}
FIG_ROOT=${FIG_ROOT:-${REPO}/figures/tuned_wide/xgb}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-0}
TOP_N=${TOP_N:-25}
MODEL_MODULE=${MODEL_MODULE:-xgb_model_tuned}

mkdir -p "${REPO}/logs" "${GLOBAL_SHAP_DIR}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

for LEAD in 3 6 12; do
    LEAD_PAD=$(printf "%02d" "${LEAD}")
    RUN_DIR="${TUNED_ROOT}/lead${LEAD_PAD}_${TASK}"
    echo "=== Computing tuned XGB SHAP | lead=${LEAD} task=${TASK} ==="
    python scripts/compute_shap_tuned_xgb.py \
        --config "${CONFIG}" \
        --lead "${LEAD}" \
        --task "${TASK}" \
        --tuned-root "${TUNED_ROOT}" \
        --model-dir "${RUN_DIR}/models" \
        --output-dir "${RUN_DIR}/shap" \
        --global-shap-dir "${GLOBAL_SHAP_DIR}" \
        --fig-dir "${FIG_ROOT}/lead${LEAD_PAD}_${TASK}" \
        --model-module "${MODEL_MODULE}" \
        --max-eval-samples "${MAX_EVAL_SAMPLES}" \
        --top-n "${TOP_N}"
done

echo "Tuned XGB SHAP complete."
