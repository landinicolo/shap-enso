#!/usr/bin/env bash
set -euo pipefail

# Combined launcher. You can override variables when running, e.g.:
#   REPO=/work/ext/st12/shap-enso CNN_MAX_TRIALS=64 XGB_MAX_TRIALS=256 bash tune_cnn_xgb_wide_all_leads.sh

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
CV_FOLDS=${CV_FOLDS:-3}
DEVICE=${DEVICE:-cuda}
CNN_MAX_TRIALS=${CNN_MAX_TRIALS:-128}
XGB_MAX_TRIALS=${XGB_MAX_TRIALS:-256}
TUNING_MAX_EPOCHS=${TUNING_MAX_EPOCHS:-60}

mkdir -p "${REPO}/logs"
mkdir -p "${REPO}/data/tuned_wide/cnn" "${REPO}/data/tuned_wide/xgb"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

for LEAD in 3 6 12; do
    LEAD_PAD=$(printf "%02d" "${LEAD}")

    echo "=== CNN lead=${LEAD} regression | tuned_wide ==="
    python scripts/train_cnn_tuned.py \
        --config "${CONFIG}" --lead "${LEAD}" --task regression --device "${DEVICE}" \
        --grid-preset wide --cv-folds "${CV_FOLDS}" --max-trials "${CNN_MAX_TRIALS}" \
        --tuning-max-epochs "${TUNING_MAX_EPOCHS}" --selection-metric r2 \
        --run-output-dir "${REPO}/data/tuned_wide/cnn/lead${LEAD_PAD}_regression" \
        2>&1 | tee "${REPO}/logs/cnn_tuned_wide_lead${LEAD_PAD}_regression.log"

    echo "=== CNN lead=${LEAD} classification | tuned_wide ==="
    python scripts/train_cnn_tuned.py \
        --config "${CONFIG}" --lead "${LEAD}" --task classification --device "${DEVICE}" \
        --grid-preset wide --cv-folds "${CV_FOLDS}" --max-trials "${CNN_MAX_TRIALS}" \
        --tuning-max-epochs "${TUNING_MAX_EPOCHS}" --selection-metric bss \
        --run-output-dir "${REPO}/data/tuned_wide/cnn/lead${LEAD_PAD}_classification" \
        2>&1 | tee "${REPO}/logs/cnn_tuned_wide_lead${LEAD_PAD}_classification.log"

    echo "=== XGB lead=${LEAD} regression | tuned_wide ==="
    python scripts/train_xgb_tuned.py \
        --config "${CONFIG}" --lead "${LEAD}" --task regression \
        --grid-preset wide --cv-folds "${CV_FOLDS}" --max-trials "${XGB_MAX_TRIALS}" \
        --selection-metric r2 \
        --run-output-dir "${REPO}/data/tuned_wide/xgb/lead${LEAD_PAD}_regression" \
        2>&1 | tee "${REPO}/logs/xgb_tuned_wide_lead${LEAD_PAD}_regression.log"

    echo "=== XGB lead=${LEAD} classification | tuned_wide ==="
    python scripts/train_xgb_tuned.py \
        --config "${CONFIG}" --lead "${LEAD}" --task classification \
        --grid-preset wide --cv-folds "${CV_FOLDS}" --max-trials "${XGB_MAX_TRIALS}" \
        --selection-metric bss \
        --run-output-dir "${REPO}/data/tuned_wide/xgb/lead${LEAD_PAD}_classification" \
        2>&1 | tee "${REPO}/logs/xgb_tuned_wide_lead${LEAD_PAD}_classification.log"
done

echo "CNN + XGB wide tuning complete."
