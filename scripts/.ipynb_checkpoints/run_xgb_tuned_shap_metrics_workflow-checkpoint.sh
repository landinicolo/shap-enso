#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/work/ext/st12/shap-enso}
export REPO

bash "${REPO}/scripts/run_xgb_tuned_shap_all_leads.sh"
bash "${REPO}/scripts/run_xgb_tuned_compile_metrics_figures.sh"

echo "Full tuned XGB SHAP + metrics workflow complete."
