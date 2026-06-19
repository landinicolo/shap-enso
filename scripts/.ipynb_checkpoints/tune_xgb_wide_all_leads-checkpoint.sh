#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
MAX_TRIALS=${MAX_TRIALS:-256}
CV_FOLDS=${CV_FOLDS:-3}

mkdir -p "${REPO}/logs"
mkdir -p "${REPO}/data/tuned_wide/xgb"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

for LEAD in 3 6 12; do
    LEAD_PAD=$(printf "%02d" "${LEAD}")

    echo "=== XGB lead=${LEAD} regression | tuned_wide ==="
    python scripts/train_xgb_tuned.py \
        --config "${CONFIG}" \
        --lead "${LEAD}" \
        --task regression \
        --grid-preset wide \
        --cv-folds "${CV_FOLDS}" \
        --max-trials "${MAX_TRIALS}" \
        --selection-metric r2 \
        --run-output-dir "${REPO}/data/tuned_wide/xgb/lead${LEAD_PAD}_regression" \
        2>&1 | tee "${REPO}/logs/xgb_tuned_wide_lead${LEAD_PAD}_regression.log"

    echo "=== XGB lead=${LEAD} classification | tuned_wide ==="
    python scripts/train_xgb_tuned.py \
        --config "${CONFIG}" \
        --lead "${LEAD}" \
        --task classification \
        --grid-preset wide \
        --cv-folds "${CV_FOLDS}" \
        --max-trials "${MAX_TRIALS}" \
        --selection-metric bss \
        --run-output-dir "${REPO}/data/tuned_wide/xgb/lead${LEAD_PAD}_classification" \
        2>&1 | tee "${REPO}/logs/xgb_tuned_wide_lead${LEAD_PAD}_classification.log"
done

echo "XGB wide tuning complete."
