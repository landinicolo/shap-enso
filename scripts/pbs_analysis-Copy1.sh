#!/bin/bash

set -euo pipefail

REPO=/work/ext/st12/shap-enso
CONFIG=${REPO}/configs/default.yaml
FIGURES=${REPO}/figures

mkdir -p ${REPO}/logs ${FIGURES}


export PYTHONPATH=${REPO}:${PYTHONPATH:-}

cd ${REPO}

# -----------------------------------------------------------------------
# Step 1: compile test-set skill metrics for all models + both tasks
# -----------------------------------------------------------------------
echo "=== Compiling metrics ==="
python scripts/compile_metrics.py --config ${CONFIG} --task regression classification

# -----------------------------------------------------------------------
# Step 2: generate publication figures and save to figures/
# -----------------------------------------------------------------------
echo "=== Generating figures ==="
python - <<'EOF'
import sys
sys.path.insert(0, "/work/ext/st12/shap-enso")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.config import load_config
from src.shap_analysis.aggregate import (
    load_shap_store, global_mean_abs_shap, seasonal_shap_mean,
    spring_barrier_stats, enso_composite_shap, lead_importance_table,
    load_spatial_shap_store,
)
from src.shap_analysis.plotting import (
    plot_feature_importance_bar, plot_seasonal_heatmap,
    plot_spring_barrier, plot_enso_asymmetry,
    plot_lead_importance_heatmap, plot_spatial_shap, plot_shap_beeswarm
)

cfg      = load_config("configs/default.yaml")
shap_dir = cfg["shap"]["output_dir"]
fig_dir  = Path("figures")
fig_dir.mkdir(exist_ok=True)

MODELS = ["xgboost", "lstm", "cnn"]
LEADS  = [3, 6, 12]


def save(fig, name):
    path = fig_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


for model in MODELS:
    for lead in LEADS:
        try:
            ds = load_shap_store(shap_dir, model, lead, "regression")
        except FileNotFoundError:
            print(f"  skip {model} lead={lead} (store not found)")
            continue


        # SHAP distribution violin plot
        try:
            shap_values = ds["shap_values"].values
            feature_names = ds["feature"].values.tolist()

            # MODIFICA
            # load original feature matrix
            from src.utils.io_utils import load_feature_matrix
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            feat_dir  = Path(cfg["data"]["processed_dir"]) / "features"

            feat_path = feat_dir / f"features_lead{lead:02d}.npz"
            X, y_reg, feat_names, times = load_feature_matrix(feat_path)

            print(shap_values.shape)
            print(X.shape)
            
            fig, _ = plot_shap_beeswarm(
                shap_values=shap_values,
                X=X,
                feature_names=feature_names,
                top_n=15,
            )
            #
        
            save(fig, f"distribution_{model}_lead{lead:02d}.png")
        
        except Exception as e:
            print(f" skip distribution: {e}")


print("All figures saved.")
EOF

echo "Analysis complete. Figures in: ${FIGURES}"
