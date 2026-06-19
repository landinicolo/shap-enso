#!/bin/bash
source /users_home/ext/st19/venvs/shapenso/bin/activate
set -euo pipefail

REPO=/work/ext/st12/shap-enso
CONFIG=${REPO}/configs/default.yaml

mkdir -p ${REPO}/logs


export PYTHONPATH=${REPO}:${PYTHONPATH:-}

cd ${REPO}

# -----------------------------------------------------------------------
# XGBoost SHAP already completed — skipping
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# LSTM — GradientExplainer, GPU
# -----------------------------------------------------------------------
for LEAD in 3 6 12; do
    echo "=== LSTM SHAP  lead=${LEAD}  regression ==="
    python scripts/compute_shap.py --config ${CONFIG} \
        --model lstm --lead ${LEAD} --task regression --device cuda
done

# -----------------------------------------------------------------------
# CNN — DeepExplainer, GPU; --save-spatial writes (var, lag, lat, lon) map
# -----------------------------------------------------------------------
for LEAD in 3 6 12; do
    echo "=== CNN SHAP  lead=${LEAD}  regression ==="
    python scripts/compute_shap.py --config ${CONFIG} \
        --model cnn --lead ${LEAD} --task regression --device cuda --save-spatial
done

echo "All SHAP computation complete."
