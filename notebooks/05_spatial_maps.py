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
    from src.shap_analysis.aggregate import load_spatial_shap_store
    from src.shap_analysis.plotting import plot_spatial_shap

    return (mo, np, plt, repo, Path, load_config,
            load_spatial_shap_store, plot_spatial_shap)


@app.cell
def __(mo):
    mo.md(
        """
        # CNN Spatial SHAP Maps

        Mean absolute SHAP value at each grid point, showing which
        geographic regions the CNN relies on most for ENSO prediction.

        Spatial SHAP is computed only for the CNN model
        (`--save-spatial` flag in `scripts/compute_shap.py`).
        The map is averaged over all test-set samples.
        """
    )
    return ()


@app.cell
def __(mo):
    lead_dd = mo.ui.dropdown(
        options={"3 months": 3, "6 months": 6, "12 months": 12},
        value="6 months",
        label="Lead time",
    )
    var_dd = mo.ui.dropdown(
        options={
            "SST":  "sst",
            "D20":  "d20",
            "Wind stress (τx)": "tauu",
            "OLR":  "olr",
            "SLP":  "slp",
        },
        value="SST",
        label="Variable",
    )
    lag_sl = mo.ui.slider(start=0, stop=3, step=1, value=0, label="Lag (months)")
    mo.hstack([lead_dd, var_dd, lag_sl])
    return lead_dd, var_dd, lag_sl


@app.cell
def __(repo):
    from src.utils.config import load_config as _lc
    _cfg = _lc(str(repo / "configs" / "default.yaml"))
    shap_dir = _cfg["shap"]["output_dir"]
    return (shap_dir,)


@app.cell
def __(shap_dir, lead_dd, var_dd, lag_sl,
       load_spatial_shap_store, plot_spatial_shap, plt, mo):
    try:
        _ds  = load_spatial_shap_store(shap_dir, "cnn", lead_dd.value, "regression")
        _available_vars = list(_ds["mean_abs_shap"].coords["var"].values.astype(str))

        if var_dd.value not in _available_vars:
            _out = mo.callout(
                mo.md(
                    f"Variable **{var_dd.value}** not in spatial SHAP store. "
                    f"Available: {_available_vars}"
                ),
                kind="warn",
            )
        else:
            _fig, _ = plot_spatial_shap(
                _ds,
                var_name=var_dd.value,
                lag=lag_sl.value,
                title=(f"Spatial SHAP — CNN  lead {lead_dd.value} mo  "
                       f"{var_dd.value}  lag {lag_sl.value}"),
            )
            _out = mo.as_html(_fig)
            plt.close(_fig)

    except FileNotFoundError as _e:
        _out = mo.callout(
            mo.md(
                f"**Spatial SHAP store not found:** `{_e}`  \n"
                "Run `scripts/compute_shap.py --model cnn --save-spatial` first."
            ),
            kind="warn",
        )
    except Exception as _e:
        _out = mo.callout(mo.md(f"**Error:** `{_e}`"), kind="danger")
    _out
    return ()


@app.cell
def __(shap_dir, lead_dd, load_spatial_shap_store, mo):
    try:
        _ds = load_spatial_shap_store(shap_dir, "cnn", lead_dd.value, "regression")
        _vars = list(_ds["mean_abs_shap"].coords["var"].values.astype(str))
        _n_lags = int(_ds.sizes.get("lag", 0))
        mo.md(
            f"Store contains variables: **{_vars}**  |  "
            f"lags 0–{_n_lags - 1}  |  "
            f"grid {int(_ds.sizes['lat'])}×{int(_ds.sizes['lon'])}"
        )
    except Exception:
        mo.md("")
    return ()


if __name__ == "__main__":
    app.run()
