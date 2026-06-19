#!/usr/bin/env bash
set -euo pipefail

# End-to-end tuned LSTM XAI workflow:
# 1) recompute SHAP from tuned_wide/lstm/leadXX_regression/models
# 2) compile metrics, persistence baseline, and plot gallery

REPO=${REPO:-/work/ext/st12/shap-enso}
export REPO

bash "${REPO}/scripts/run_lstm_tuned_shap_all_leads.sh"
bash "${REPO}/scripts/run_lstm_tuned_compile_metrics_figures.sh"
