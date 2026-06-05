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
    from src.shap_analysis.aggregate import load_shap_store, global_mean_abs_shap
    from src.shap_analysis.plotting import plot_feature_importance_bar

    return mo, np, plt, repo, Path, load_config, load_shap_store, global_mean_abs_shap, plot_feature_importance_bar


@app.cell
def __(mo):
    mo.md(
        """
        # SHAP Feature Importance

        Global mean |SHAP| value per feature — averaged across all test-set samples.
        Larger bars indicate features the model relies on most for this lead / task.
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
        value="3 months",
        label="Lead time",
    )
    task_dd = mo.ui.dropdown(
        options={"Regression": "regression", "Classification": "classification"},
        value="Regression",
        label="Task",
    )
    top_n_sl = mo.ui.slider(start=5, stop=38, step=1, value=15, label="Top N features")
    mo.hstack([model_dd, lead_dd, task_dd, top_n_sl])
    return model_dd, lead_dd, task_dd, top_n_sl


@app.cell
def __(repo):
    from src.utils.config import load_config as _lc
    _cfg = _lc(str(repo / "configs" / "default.yaml"))
    shap_dir = _cfg["shap"]["output_dir"]
    return (shap_dir,)


@app.cell
def __(shap_dir, model_dd, lead_dd, task_dd, top_n_sl,
       load_shap_store, global_mean_abs_shap, plot_feature_importance_bar,
       plt, mo):
    try:
        _ds  = load_shap_store(shap_dir, model_dd.value, lead_dd.value, task_dd.value)
        _imp = global_mean_abs_shap(_ds)
        _fig, _ = plot_feature_importance_bar(
            _imp,
            title=(f"Feature importance — {model_dd.value.upper()}  "
                   f"lead {lead_dd.value} mo  ({task_dd.value})"),
            top_n=top_n_sl.value,
        )
        _out = mo.as_html(_fig)
        plt.close(_fig)
    except FileNotFoundError as _e:
        _out = mo.callout(mo.md(f"**SHAP store not found:** `{_e}`"), kind="warn")
    except Exception as _e:
        _out = mo.callout(mo.md(f"**Error:** `{_e}`"), kind="danger")
    _out
    return ()


@app.cell
def __(shap_dir, model_dd, lead_dd, task_dd,
       load_shap_store, global_mean_abs_shap, mo):
    try:
        _ds  = load_shap_store(shap_dir, model_dd.value, lead_dd.value, task_dd.value)
        _imp = global_mean_abs_shap(_ds).reset_index()
        _imp.columns = ["feature", "mean_abs_shap"]
        _imp["mean_abs_shap"] = _imp["mean_abs_shap"].round(6)
        _tbl = mo.ui.table(_imp, label="Full importance table")
    except Exception:
        _tbl = mo.md("")
    _tbl
    return ()


if __name__ == "__main__":
    app.run()
