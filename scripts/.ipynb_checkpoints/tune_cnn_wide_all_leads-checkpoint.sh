#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
CONFIG=${CONFIG:-${REPO}/configs/default.yaml}
MAX_TRIALS=${MAX_TRIALS:-128}
CV_FOLDS=${CV_FOLDS:-3}
TUNING_MAX_EPOCHS=${TUNING_MAX_EPOCHS:-60}
DEVICE=${DEVICE:-cuda}

mkdir -p "${REPO}/logs"
mkdir -p "${REPO}/data/tuned_wide/cnn"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
cd "${REPO}"

for LEAD in 3 6 12; do
    LEAD_PAD=$(printf "%02d" "${LEAD}")

    echo "=== CNN lead=${LEAD} regression | tuned_wide ==="
    python scripts/train_cnn_tuned.py \
        --config "${CONFIG}" \
        --lead "${LEAD}" \
        --task regression \
        --device "${DEVICE}" \
        --grid-preset wide \
        --cv-folds "${CV_FOLDS}" \
        --max-trials "${MAX_TRIALS}" \
        --tuning-max-epochs "${TUNING_MAX_EPOCHS}" \
        --selection-metric r2 \
        --run-output-dir "${REPO}/data/tuned_wide/cnn/lead${LEAD_PAD}_regression" \
        2>&1 | tee "${REPO}/logs/cnn_tuned_wide_lead${LEAD_PAD}_regression.log"

    echo "=== CNN lead=${LEAD} classification | tuned_wide ==="
    python scripts/train_cnn_tuned.py \
        --config "${CONFIG}" \
        --lead "${LEAD}" \
        --task classification \
        --device "${DEVICE}" \
        --grid-preset wide \
        --cv-folds "${CV_FOLDS}" \
        --max-trials "${MAX_TRIALS}" \
        --tuning-max-epochs "${TUNING_MAX_EPOCHS}" \
        --selection-metric bss \
        --run-output-dir "${REPO}/data/tuned_wide/cnn/lead${LEAD_PAD}_classification" \
        2>&1 | tee "${REPO}/logs/cnn_tuned_wide_lead${LEAD_PAD}_classification.log"
done

echo "CNN wide tuning complete."
