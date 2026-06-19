#!/bin/bash

source /users_home/ext/st19/venvs/shapenso/bin/activate

cd /work/ext/st12/shap-enso

echo "Using Python:"
which python
python --version

echo "Testing dependencies:"
python -c "import torch, shap; print('torch OK:', torch.__version__); print('cuda:', torch.cuda.is_available())"

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
