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
    from src.shap_analysis.aggregate import lead_importance_table
    from src.shap_analysis.plotting import plot_lead_importance_heatmap

    return (mo, np, pd, plt, repo, Path, load_config,
            lead_importance_table, plot_lead_importance_heatmap)


@app.cell
def __(mo):
    mo.md(
        """
        # Feature Importance vs. Lead Time

        How the model's reliance on each predictor changes as the forecast
        lead time grows from 3 → 6 → 12 months.

        **Interpretation:** Features that remain important at 12 months
        likely encode slow oceanic memory (D20, basin-wide SST).
        Features that drop off at longer leads may capture fast
        atmospheric bridge signals (OLR, τx, SLP).

        The heatmap is row-normalised so you can compare the *relative*
        shift in importance, not the absolute magnitude.
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
    task_dd = mo.ui.dropdown(
        options={"Regression": "regression"},
        value="Regression",
        label="Task",
    )
    top_n_sl = mo.ui.slider(start=5, stop=38, step=1, value=12, label="Top N features")
    mo.hstack([model_dd, task_dd, top_n_sl])
    return model_dd, task_dd, top_n_sl


@app.cell
def __(repo):
    from src.utils.config import load_config as _lc
    _cfg = _lc(str(repo / "configs" / "default.yaml"))
    shap_dir = _cfg["shap"]["output_dir"]
    return (shap_dir,)


@app.cell
def __(shap_dir, model_dd, task_dd,
       lead_importance_table, mo):
    try:
        lead_df = lead_importance_table(
            shap_dir, model_dd.value, task_dd.value, leads=[3, 6, 12]
        )
        if lead_df.dropna(how="all").empty:
            lead_df = None
            _status = mo.callout(
                mo.md("No SHAP stores found for any lead. Run `scripts/compute_shap.py` first."),
                kind="warn",
            )
        else:
            _found = lead_df.dropna(axis=1, how="all").columns.tolist()
            _status = mo.md(f"Loaded leads: **{_found}** months")
    except Exception as _e:
        lead_df = None
        _status = mo.callout(mo.md(f"**Error:** `{_e}`"), kind="danger")
    _status
    return (lead_df,)


@app.cell
def __(lead_df, top_n_sl, model_dd, task_dd,
       plot_lead_importance_heatmap, plt, mo):
    if lead_df is not None and not lead_df.dropna(how="all").empty:
        _fig, _ = plot_lead_importance_heatmap(
            lead_df.dropna(axis=1, how="all"),
            title=f"Importance vs. lead — {model_dd.value.upper()}  ({task_dd.value})",
            top_n=top_n_sl.value,
        )
        _out = mo.as_html(_fig)
        plt.close(_fig)
    else:
        _out = mo.md("")
    _out
    return ()


@app.cell
def __(lead_df, mo, pd):
    if lead_df is not None:
        _df = lead_df.dropna(how="all").round(6)
        mo.ui.table(
            _df.reset_index().rename(columns={"feature": "Feature"}),
            label="Raw importance by lead",
        )
    else:
        mo.md("")
    return ()


if __name__ == "__main__":
    app.run()
