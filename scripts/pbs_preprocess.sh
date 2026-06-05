#!/bin/bash
#PBS -N shap_enso_preprocess
#PBS -A NAML0001
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -o /glade/work/acsubram/GitRepos/shap-enso/logs/preprocess.log

set -euo pipefail

REPO=/glade/work/acsubram/GitRepos/shap-enso
CONFIG=${REPO}/configs/default.yaml

mkdir -p ${REPO}/logs

module load conda
conda activate shap-enso

cd ${REPO}

# Full pipeline: download then preprocess
# Add --no-download if raw files are already present
python scripts/run_preprocess.py --config ${CONFIG}
