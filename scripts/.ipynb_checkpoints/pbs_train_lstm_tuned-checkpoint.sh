set -euo pipefail

REPO=/work/ext/st12/shap-enso
CONFIG=${REPO}/configs/default.yaml

mkdir -p "${REPO}/logs"
mkdir -p "${REPO}/data/tuned_wide/lstm"

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

cd "${REPO}"

for LEAD in 3 6 12; do
    LEAD_PAD=$(printf "%02d" "${LEAD}")

    echo "=== LSTM lead=${LEAD} regression | tuned_wide ==="

    python scripts/train_lstm_tuned.py \
        --config "${CONFIG}" \
        --lead "${LEAD}" \
        --task regression \
        --grid-preset wide \
        --cv-folds 3 \
        --max-trials 256 \
        --tuning-max-epochs 60 \
        --selection-metric r2 \
        --run-output-dir "${REPO}/data/tuned_wide/lstm/lead${LEAD_PAD}_regression"

    echo "=== LSTM lead=${LEAD} classification | tuned_wide ==="

    python scripts/train_lstm_tuned.py \
        --config "${CONFIG}" \
        --lead "${LEAD}" \
        --task classification \
        --grid-preset wide \
        --cv-folds 3 \
        --max-trials 256 \
        --tuning-max-epochs 60 \
        --selection-metric bss \
        --run-output-dir "${REPO}/data/tuned_wide/lstm/lead${LEAD_PAD}_classification"
done

echo "LSTM wide tuning complete."