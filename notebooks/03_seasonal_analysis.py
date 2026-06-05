import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import sys
    from pathlib import Path

    repo = Path(__file__).parent.parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.config import load_config
    from src.shap_analysis.aggregate import (
        load_shap_store, seasonal_shap_mean, spring_barrier_stats,
    )
    from src.shap_analysis.plotting import plot_seasonal_heatmap, plot_spring_barrier

    return (mo, np, pd, plt, repo, Path, load_config,
            load_shap_store, seasonal_shap_mean, spring_barrier_stats,
            plot_seasonal_heatmap, plot_spring_barrier)


@app.cell
def __(mo):
    mo.md(
        """
        # Seasonal SHAP Analysis & Spring Predictability Barrier

        The **spring predictability barrier (SPB)** is a well-known drop in
        ENSO predictability for initialization months in boreal spring (MAM).
        Here we examine whether SHAP importances also exhibit this barrier —
        reduced feature influence when initializing in spring.

        **SPB ratio** = mean |SHAP| in MAM / mean |SHAP| in SON.
        Values < 1 indicate features that the model relies on less in spring.
        """
    )
    return ()


@app.cell
def __(mo):
    model_dd = mo.ui.dropdown(
        options={"XGBoost": "xgboost", "LSTM": "lstm", "CNN": "cnn"},
        value="XGBoost",
        label="Model",
    )
    lead_dd = mo.ui.dropdown(
        options={"3 months": 3, "6 months": 6, "12 months": 12},
        value="6 months",
        label="Lead time",
    )
    task_dd = mo.ui.dropdown(
        options={"Regression": "regression"},
        value="Regression",
        label="Task",
    )
    top_n_sl = mo.ui.slider(start=3, stop=15, step=1, value=8, label="Top features (heatmap)")
    mo.hstack([model_dd, lead_dd, task_dd, top_n_sl])
    return model_dd, lead_dd, task_dd, top_n_sl


@app.cell
def __(repo):
    from src.utils.config import load_config as _lc
    _cfg = _lc(str(repo / "configs" / "default.yaml"))
    shap_dir = _cfg["shap"]["output_dir"]
    return (shap_dir,)


@app.cell
def __(shap_dir, model_dd, lead_dd, task_dd,
       load_shap_store, seasonal_shap_mean, mo):
    try:
        _ds  = load_shap_store(shap_dir, model_dd.value, lead_dd.value, task_dd.value)
        seas_df  = seasonal_shap_mean(_ds)
        _status  = mo.md(
            f"Loaded {_ds.sizes['time']} samples across "
            f"{len(_ds.coords['feature'])} features."
        )
    except FileNotFoundError as _e:
        seas_df = None
        _status = mo.callout(mo.md(f"**SHAP store not found:** `{_e}`"), kind="warn")
    except Exception as _e:
        seas_df = None
        _status = mo.callout(mo.md(f"**Error:** `{_e}`"), kind="danger")
    _status
    return (seas_df,)


@app.cell
def __(seas_df, top_n_sl, model_dd, lead_dd, task_dd,
       plot_seasonal_heatmap, plt, mo):
    if seas_df is not None:
        _fig, _ = plot_seasonal_heatmap(
            seas_df,
            title=(f"Seasonal SHAP importance — {model_dd.value.upper()}  "
                   f"lead {lead_dd.value} mo"),
            top_n=top_n_sl.value,
        )
        _out = mo.as_html(_fig)
        plt.close(_fig)
    else:
        _out = mo.md("")
    _out
    return ()


@app.cell
def __(seas_df, model_dd, lead_dd, plot_spring_barrier, plt, mo):
    if seas_df is not None:
        _fig, _ = plot_spring_barrier(
            seas_df,
            title=(f"Spring barrier in SHAP — {model_dd.value.upper()}  "
                   f"lead {lead_dd.value} mo"),
        )
        _out = mo.as_html(_fig)
        plt.close(_fig)
    else:
        _out = mo.md("")
    _out
    return ()


@app.cell
def __(seas_df, spring_barrier_stats, mo, pd):
    if seas_df is not None:
        _spb = spring_barrier_stats(seas_df).reset_index()
        _spb.columns = ["feature", "spb_ratio (MAM/SON)"]
        _spb["spb_ratio (MAM/SON)"] = _spb["spb_ratio (MAM/SON)"].round(4)
        mo.vstack([
            mo.md("### SPB ratio per feature (< 1 = spring suppression)"),
            mo.ui.table(_spb, label="Spring barrier ratios"),
        ])
    else:
        mo.md("")
    return ()


if __name__ == "__main__":
    app.run()
