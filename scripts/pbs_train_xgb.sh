
set -euo pipefail

REPO=/work/ext/st12/shap-enso
CONFIG=${REPO}/configs/default.yaml

mkdir -p ${REPO}/logs

export PYTHONPATH=${REPO}:${PYTHONPATH:-}

cd ${REPO}

for LEAD in 3 6 12; do
    echo "=== XGBoost  lead=${LEAD}  regression ==="
    python scripts/train_xgb.py --config ${CONFIG} --lead ${LEAD} --task regression

    echo "=== XGBoost  lead=${LEAD}  classification ==="
    python scripts/train_xgb.py --config ${CONFIG} --lead ${LEAD} --task classification
done

echo "XGBoost training complete."
