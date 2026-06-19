#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
TASK=${TASK:-regression}
TUNED_ROOT=${TUNED_ROOT:-${REPO}/data/tuned_wide/xgb}
OUTPUT_DIR=${OUTPUT_DIR:-${TUNED_ROOT}/compiled_analysis}
FIG_DIR=${FIG_DIR:-${REPO}/figures/tuned_wide/xgb}
TOP_N=${TOP_N:-25}

mkdir -p "${REPO}/logs" "${OUTPUT_DIR}" "${FIG_DIR}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

python scripts/compile_metrics_tuned_xgb.py \
    --config "${CONFIG}" \
    --task "${TASK}" \
    --tuned-root "${TUNED_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --fig-dir "${FIG_DIR}" \
    --leads 3 6 12 \
    --top-n "${TOP_N}"

echo "Tuned XGB metrics and figures complete."
