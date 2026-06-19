#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
TASK=${TASK:-regression}
TUNED_ROOT=${TUNED_ROOT:-${REPO}/data/tuned_wide/lstm}
GLOBAL_SHAP_DIR=${GLOBAL_SHAP_DIR:-${REPO}/data/shap_tuned_wide/lstm}
OUT_DIR=${OUT_DIR:-${TUNED_ROOT}/compiled_analysis}
FIG_DIR=${FIG_DIR:-${REPO}/figures/tuned_wide/lstm}
TOP_N=${TOP_N:-25}

mkdir -p "${REPO}/logs" "${OUT_DIR}" "${FIG_DIR}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

echo "=== Compiling tuned LSTM metrics and SHAP figure gallery ==="
python scripts/compile_metrics_tuned_lstm.py \
    --config "${CONFIG}" \
    --task "${TASK}" \
    --leads 3 6 12 \
    --tuned-root "${TUNED_ROOT}" \
    --shap-root "${GLOBAL_SHAP_DIR}" \
    --output-dir "${OUT_DIR}" \
    --fig-dir "${FIG_DIR}" \
    --top-n "${TOP_N}"

echo "Compiled analysis complete. Outputs: ${OUT_DIR}; figures: ${FIG_DIR}"
