#!/bin/bash
#PBS -N shap_enso_shap
#PBS -A UCUB0143
#PBS -l select=1:ncpus=4:ngpus=1:mem=64GB:gpu_type=a100
#PBS -l walltime=12:00:00
#PBS -q nvgpu
#PBS -j oe
#PBS -o /glade/work/acsubram/GitRepos/shap-enso/logs/compute_shap.log

set -euo pipefail

REPO=/glade/work/acsubram/GitRepos/shap-enso
CONFIG=${REPO}/configs/default.yaml

mkdir -p ${REPO}/logs

module load conda
conda activate shap-enso
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
