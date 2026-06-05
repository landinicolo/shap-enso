#!/bin/bash
#PBS -N shap_enso_xgb
#PBS -A UCUB0143
#PBS -l select=1:ncpus=8:mem=32GB
#PBS -l walltime=04:00:00
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/acsubram/GitRepos/shap-enso/logs/train_xgb.log

set -euo pipefail

REPO=/glade/work/acsubram/GitRepos/shap-enso
CONFIG=${REPO}/configs/default.yaml

mkdir -p ${REPO}/logs

module load conda
conda activate shap-enso
export PYTHONPATH=${REPO}:${PYTHONPATH:-}

cd ${REPO}

for LEAD in 3 6 12; do
    echo "=== XGBoost  lead=${LEAD}  regression ==="
    python scripts/train_xgb.py --config ${CONFIG} --lead ${LEAD} --task regression

    echo "=== XGBoost  lead=${LEAD}  classification ==="
    python scripts/train_xgb.py --config ${CONFIG} --lead ${LEAD} --task classification
done

echo "XGBoost training complete."
