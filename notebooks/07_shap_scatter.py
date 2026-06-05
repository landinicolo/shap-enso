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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.config import load_config
    from src.shap_analysis.aggregate import (
        load_shap_store, global_mean_abs_shap, shap_prediction_corr,
    )
    from src.shap_analysis.plotting import plot_shap_scatter

    return (mo, np, plt, repo, Path, load_config,
            load_shap_store, global_mean_abs_shap, shap_prediction_corr,
            plot_shap_scatter)


@app.cell
def __(mo):
    mo.md(
        """
        # SHAP Scatter — Individual Feature Attribution

        Scatter of each sample's **signed SHAP value** (y-axis) against
        the model's **Niño3.4 prediction** (x-axis, colour).

        - Points above zero → feature pushed the forecast **higher** (warmer)
        - Points below zero → feature pushed the forecast **lower** (cooler)
        - Strong diagonal trend → feature is a consistent linear driver
        - S-shaped or clustered pattern → non-linear / phase-dependent role

        This is a proxy for the classic SHAP beeswarm plot, using
        the model's own prediction as the sorting axis.
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
    mo.hstack([model_dd, lead_dd, task_dd])
    return model_dd, lead_dd, task_dd


@app.cell
def __(repo):
    from src.utils.config import load_config as _lc
    _cfg = _lc(str(repo / "configs" / "default.yaml"))
    shap_dir = _cfg["shap"]["output_dir"]
    return (shap_dir,)


@app.cell
def __(shap_dir, model_dd, lead_dd, task_dd,
       load_shap_store, global_mean_abs_shap, mo):
    try:
        _ds         = load_shap_store(shap_dir, model_dd.value, lead_dd.value, task_dd.value)
        _feat_list  = list(_ds.coords["feature"].values.astype(str))
        _imp        = global_mean_abs_shap(_ds)
        shap_ds     = _ds
        feat_list   = _feat_list
        top_feature = _imp.index[0]
        _status     = mo.md(
            f"Loaded {_ds.sizes['time']} samples | "
            f"{len(_feat_list)} features | "
            f"Top feature: **{top_feature}**"
        )
    except FileNotFoundError as _e:
        shap_ds     = None
        feat_list   = []
        top_feature = ""
        _status     = mo.callout(mo.md(f"**SHAP store not found:** `{_e}`"), kind="warn")
    except Exception as _e:
        shap_ds     = None
        feat_list   = []
        top_feature = ""
        _status     = mo.callout(mo.md(f"**Error:** `{_e}`"), kind="danger")
    _status
    return shap_ds, feat_list, top_feature


@app.cell
def __(feat_list, top_feature, mo):
    if feat_list:
        feat_options = {f: f for f in feat_list}
        feat_dd = mo.ui.dropdown(
            options=feat_options,
            value=top_feature,
            label="Feature",
        )
    else:
        feat_dd = mo.ui.dropdown(
            options={"(no data)": "(no data)"},
            value="(no data)",
            label="Feature",
        )
    feat_dd
    return (feat_dd,)


@app.cell
def __(shap_ds, feat_dd, model_dd, lead_dd,
       plot_shap_scatter, plt, mo):
    if shap_ds is not None and feat_dd.value != "(no data)":
        try:
            _fig, _ = plot_shap_scatter(
                shap_ds,
                feature=feat_dd.value,
                title=(f"SHAP vs. prediction — {feat_dd.value}  |  "
                       f"{model_dd.value.upper()}  lead {lead_dd.value} mo"),
            )
            _out = mo.as_html(_fig)
            plt.close(_fig)
        except Exception as _e:
            _out = mo.callout(mo.md(f"**Plot error:** `{_e}`"), kind="danger")
    else:
        _out = mo.md("")
    _out
    return ()


@app.cell
def __(shap_ds, shap_prediction_corr, mo, np):
    if shap_ds is not None:
        _corr = shap_prediction_corr(shap_ds).reset_index()
        _corr.columns = ["feature", "SHAP–pred correlation"]
        _corr["SHAP–pred correlation"] = _corr["SHAP–pred correlation"].round(4)
        mo.vstack([
            mo.md(
                "### SHAP–prediction correlation per feature  \n"
                "Positive → feature drives El Niño; negative → drives La Niña."
            ),
            mo.ui.table(_corr, label="Correlation table"),
        ])
    else:
        mo.md("")
    return ()


if __name__ == "__main__":
    app.run()
