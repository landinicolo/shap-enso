#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
TASK=${TASK:-regression}
TUNED_ROOT=${TUNED_ROOT:-${REPO}/data/tuned_wide/cnn}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO}/data/tuned_wide/cnn/compiled_analysis}
FIG_DIR=${FIG_DIR:-${REPO}/figures/tuned_wide/cnn}

MAP_VARIABLES=${MAP_VARIABLES:-"sst d20 tauu olr slp"}
MAP_LAG=${MAP_LAG:-0}

mkdir -p "${OUTPUT_DIR}" "${FIG_DIR}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

python scripts/compile_metrics_tuned_cnn.py \
    --config "${CONFIG}" \
    --task "${TASK}" \
    --tuned-root "${TUNED_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --fig-dir "${FIG_DIR}" \
    --leads 3 6 12 \
    --map-lag "${MAP_LAG}" \
    --map-variables ${MAP_VARIABLES}

echo "Tuned CNN metrics and figures complete."
