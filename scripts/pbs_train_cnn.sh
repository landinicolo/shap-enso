#!/bin/bash
#PBS -N shap_enso_cnn
#PBS -A UCUB0143
#PBS -l select=1:ncpus=4:ngpus=1:mem=64GB:gpu_type=a100
#PBS -l walltime=08:00:00
#PBS -q nvgpu
#PBS -j oe
#PBS -o /glade/work/acsubram/GitRepos/shap-enso/logs/train_cnn.log

set -euo pipefail

REPO=/glade/work/acsubram/GitRepos/shap-enso
CONFIG=${REPO}/configs/default.yaml

mkdir -p ${REPO}/logs

module load conda
conda activate shap-enso
export PYTHONPATH=${REPO}:${PYTHONPATH:-}

cd ${REPO}

for LEAD in 3 6 12; do
    echo "=== CNN  lead=${LEAD}  regression ==="
    python scripts/train_cnn.py --config ${CONFIG} --lead ${LEAD} --task regression

    echo "=== CNN  lead=${LEAD}  classification ==="
    python scripts/train_cnn.py --config ${CONFIG} --lead ${LEAD} --task classification
done

echo "CNN training complete."
