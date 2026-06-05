#!/bin/bash
#PBS -N shap_enso_lstm
#PBS -A UCUB0143
#PBS -l select=1:ncpus=4:ngpus=1:mem=32GB:gpu_type=a100
#PBS -l walltime=06:00:00
#PBS -q gpudev
#PBS -j oe
#PBS -o /glade/work/acsubram/GitRepos/shap-enso/logs/train_lstm.log

set -euo pipefail

REPO=/glade/work/acsubram/GitRepos/shap-enso
CONFIG=${REPO}/configs/default.yaml

mkdir -p ${REPO}/logs

module load conda
conda activate shap-enso
export PYTHONPATH=${REPO}:${PYTHONPATH:-}

cd ${REPO}

for LEAD in 3 6 12; do
    echo "=== LSTM  lead=${LEAD}  regression ==="
    python scripts/train_lstm.py --config ${CONFIG} --lead ${LEAD} --task regression --device cuda

    echo "=== LSTM  lead=${LEAD}  classification ==="
    python scripts/train_lstm.py --config ${CONFIG} --lead ${LEAD} --task classification --device cuda
done

echo "LSTM training complete."
