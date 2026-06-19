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
        load_shap_store, enso_composite_shap, global_mean_abs_shap,
    )
    from src.shap_analysis.plotting import plot_enso_asymmetry

    return (mo, np, pd, plt, repo, Path, load_config,
            load_shap_store, enso_composite_shap, global_mean_abs_shap,
            plot_enso_asymmetry)


@app.cell
def __(mo):
    mo.md(
        """
        # El Niño vs. La Niña SHAP Asymmetry

        ENSO exhibits a well-documented **asymmetry**: El Niño events tend to be
        stronger and shorter-lived than La Niña events, and different physical
        mechanisms dominate each phase. Here we test whether the model's
        attribution (SHAP) reflects this asymmetry.

        Features with larger El Niño bars are more informative for predicting
        warm events; larger La Niña bars for cold events.
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
    thresh_sl = mo.ui.number(
        start=0.3, stop=1.0, step=0.1, value=0.5,
        label="ENSO threshold (°C)",
    )
    top_n_sl = mo.ui.slider(start=5, stop=20, step=1, value=10, label="Top N features")
    mo.hstack([model_dd, lead_dd, task_dd, thresh_sl, top_n_sl])
    return model_dd, lead_dd, task_dd, thresh_sl, top_n_sl


@app.cell
def __(repo):
    from src.utils.config import load_config as _lc
    _cfg = _lc(str(repo / "configs" / "default.yaml"))
    shap_dir = _cfg["shap"]["output_dir"]
    return (shap_dir,)


@app.cell
def __(shap_dir, model_dd, lead_dd, task_dd, thresh_sl,
       load_shap_store, enso_composite_shap, mo):
    try:
        _ds      = load_shap_store(shap_dir, model_dd.value, lead_dd.value, task_dd.value)
        composite = enso_composite_shap(_ds, threshold=thresh_sl.value)
        _n_en    = int((_ds["prediction"].values >  thresh_sl.value).sum())
        _n_ln    = int((_ds["prediction"].values < -thresh_sl.value).sum())
        _status  = mo.md(
            f"El Niño samples: **{_n_en}** | La Niña samples: **{_n_ln}** "
            f"(threshold ±{thresh_sl.value}°C)"
        )
    except FileNotFoundError as _e:
        composite = None
        _status   = mo.callout(mo.md(f"**SHAP store not found:** `{_e}`"), kind="warn")
    except Exception as _e:
        composite = None
        _status   = mo.callout(mo.md(f"**Error:** `{_e}`"), kind="danger")
    _status
    return (composite,)


@app.cell
def __(composite, top_n_sl, model_dd, lead_dd,
       plot_enso_asymmetry, plt, mo):
    if composite is not None:
        _fig, _ = plot_enso_asymmetry(
            composite,
            title=(f"El Niño vs. La Niña importance — {model_dd.value.upper()}  "
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
def __(composite, mo, pd, np):
    if composite is not None:
        _en = composite["elnino"].rename("El Niño")
        _ln = composite["lanina"].rename("La Niña")
        _asym = pd.DataFrame({"El Niño": _en, "La Niña": _ln})
        _asym["ratio (EN/LN)"] = (_en / _ln.where(_ln > 1e-9, other=np.nan)).round(3)
        _asym = _asym.sort_values("El Niño", ascending=False).round(6)
        mo.vstack([
            mo.md("### Asymmetry table (ratio > 1 → more important for El Niño)"),
            mo.ui.table(_asym.reset_index().rename(columns={"index": "feature"}),
                        label="El Niño / La Niña asymmetry"),
        ])
    else:
        mo.md("")
    return ()


if __name__ == "__main__":
    app.run()
