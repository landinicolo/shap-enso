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
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.utils.config import load_config
    from src.shap_analysis.plotting import plot_skill_vs_lead

    return mo, pd, np, plt, repo, Path, load_config, plot_skill_vs_lead


@app.cell
def __(mo):
    mo.md(
        """
        # ENSO Prediction — Model Skill Scores

        Loads `data/metrics.csv` compiled by `scripts/compile_metrics.py`.
        Run that script after training models and computing SHAP values.
        """
    )
    return ()


@app.cell
def __(mo):
    metric_dd = mo.ui.dropdown(
        options={
            "Correlation": "corr",
            "RMSE (°C)":   "rmse",
            "R²":          "r2",
            "MAE (°C)":    "mae",
        },
        value="Correlation",
        label="Skill metric",
    )
    task_dd = mo.ui.dropdown(
        options={"Regression": "regression", "Classification": "classification"},
        value="Regression",
        label="Task",
    )
    mo.hstack([metric_dd, task_dd])
    return metric_dd, task_dd


@app.cell
def __(repo, pd, mo):
    metrics_path = repo / "data" / "metrics.csv"
    if metrics_path.exists():
        _df = pd.read_csv(metrics_path)
        metrics_status = mo.md(f"Loaded {len(_df)} rows from `{metrics_path.name}`.")
        metrics_df = _df
    else:
        metrics_df = pd.DataFrame()
        metrics_status = mo.callout(
            mo.md(
                f"**metrics.csv not found.**  "
                f"Run `python scripts/compile_metrics.py` after computing SHAP values."
            ),
            kind="warn",
        )
    return metrics_df, metrics_status


@app.cell
def __(mo, metrics_status):
    metrics_status
    return ()


@app.cell
def __(metrics_df, metric_dd, task_dd, plot_skill_vs_lead, plt, mo, pd):
    if metrics_df.empty:
        _out = mo.md("No data — run `scripts/compile_metrics.py` first.")
    else:
        _sub = metrics_df[metrics_df["task"] == task_dd.value]
        if _sub.empty or metric_dd.value not in _sub.columns:
            _out = mo.md(f"No data for task=**{task_dd.value}** / metric=**{metric_dd.value}**.")
        else:
            _fig, _ = plot_skill_vs_lead(
                _sub,
                metric=metric_dd.value,
                title=f"{metric_dd.value.upper()} vs. lead time ({task_dd.value})",
            )
            _out = mo.as_html(_fig)
            plt.close(_fig)
    _out
    return ()


@app.cell
def __(metrics_df, task_dd, mo, pd):
    if not metrics_df.empty:
        _sub = metrics_df[metrics_df["task"] == task_dd.value]
        mo.ui.table(
            _sub[["model_type", "lead_months", "corr", "rmse", "r2", "n_samples"]].round(4),
            label="Full metrics table",
        )
    else:
        mo.md("")
    return ()


if __name__ == "__main__":
    app.run()
