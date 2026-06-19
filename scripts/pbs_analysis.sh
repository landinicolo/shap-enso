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

        # Feature importance bar
        imp = global_mean_abs_shap(ds)
        fig, _ = plot_feature_importance_bar(
            imp, title=f"Feature importance — {model} lead {lead}mo", top_n=15
        )
        save(fig, f"importance_{model}_lead{lead:02d}.png")


        # SHAP distribution violin plot
        try:
            shap_values = ds["shap_values"].values
            feature_names = ds["feature"].values.tolist()

            # MODIFICA
            # load original feature matrix
            X, y, _, times = load_feature_matrix(
                feat_dir / f"features_lead{lead:02d}.npz"
            )

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

        # Seasonal heatmap
        seas = seasonal_shap_mean(ds)
        fig, _ = plot_seasonal_heatmap(
            seas, title=f"Seasonal SHAP — {model} lead {lead}mo", top_n=10
        )
        save(fig, f"seasonal_{model}_lead{lead:02d}.png")

        # Spring barrier
        fig, _ = plot_spring_barrier(
            seas, title=f"Spring barrier — {model} lead {lead}mo"
        )
        save(fig, f"spring_barrier_{model}_lead{lead:02d}.png")

        # ENSO asymmetry
        comp = enso_composite_shap(ds, threshold=0.5)
        fig, _ = plot_enso_asymmetry(
            comp, title=f"El Niño vs. La Niña — {model} lead {lead}mo"
        )
        save(fig, f"asymmetry_{model}_lead{lead:02d}.png")

    # Lead-dependence heatmap (one per model)
    lead_df = lead_importance_table(shap_dir, model, "regression", leads=LEADS)
    if not lead_df.dropna(how="all").empty:
        fig, _ = plot_lead_importance_heatmap(
            lead_df.dropna(axis=1, how="all"),
            title=f"Importance vs. lead — {model}"
        )
        save(fig, f"lead_dep_{model}.png")

# CNN spatial SHAP maps
for lead in LEADS:
    try:
        sds = load_spatial_shap_store(shap_dir, "cnn", lead, "regression")
        for var in ["sst", "d20", "tauu", "olr", "slp"]:
            try:
                fig, _ = plot_spatial_shap(sds, var_name=var, lag=0,
                                           title=f"Spatial SHAP CNN lead {lead}mo {var}")
                save(fig, f"spatial_cnn_lead{lead:02d}_{var}.png")
            except Exception as e:
                print(f"  skip spatial {var}: {e}")
    except FileNotFoundError:
        print(f"  skip spatial CNN lead={lead} (store not found)")

print("All figures saved.")
EOF

echo "Analysis complete. Figures in: ${FIGURES}"
