"""Smoke tests for Phase 4 SHAP analysis and plotting.

All tests use synthetic data — no actual SHAP Zarr stores required.
Plotting tests check that figures are created without exceptions.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

RNG     = np.random.default_rng(42)
N       = 100
N_FEAT  = 38
N_CLS   = 3


# ---------------------------------------------------------------------------
# Synthetic SHAP dataset helpers
# ---------------------------------------------------------------------------

FEAT_NAMES = (
    [f"sst_nino34_lag{i}" for i in range(4)] +
    [f"sst_nino4_lag{i}"  for i in range(4)] +
    [f"sst_nino3_lag{i}"  for i in range(4)] +
    [f"sst_nino12_lag{i}" for i in range(4)] +
    [f"d20_nino34_lag{i}" for i in range(4)] +
    [f"d20_nino4_lag{i}"  for i in range(4)] +
    [f"tauu_eq_lag{i}"    for i in range(4)] +
    [f"olr_eq_lag{i}"     for i in range(4)] +
    [f"slp_eq_lag{i}"     for i in range(4)] +
    ["sin_month", "cos_month"]
)[:N_FEAT]


def _make_shap_ds(n: int = N, classification: bool = False) -> xr.Dataset:
    times = pd.date_range("2015-01", periods=n, freq="MS")
    preds = RNG.standard_normal(n).astype(np.float32)

    if classification:
        sv = RNG.standard_normal((n, N_FEAT, N_CLS)).astype(np.float32)
        sv_da = xr.DataArray(sv, dims=["time", "feature", "class"],
                             coords={"time": times, "feature": FEAT_NAMES,
                                     "class": [0, 1, 2]})
        abs_da = xr.DataArray(np.abs(sv).mean(axis=-1),
                              dims=["time", "feature"],
                              coords={"time": times, "feature": FEAT_NAMES})
    else:
        sv = RNG.standard_normal((n, N_FEAT)).astype(np.float32)
        sv_da = xr.DataArray(sv, dims=["time", "feature"],
                             coords={"time": times, "feature": FEAT_NAMES})
        abs_da = xr.DataArray(np.abs(sv), dims=["time", "feature"],
                              coords={"time": times, "feature": FEAT_NAMES})

    pred_da = xr.DataArray(preds, dims=["time"], coords={"time": times})
    ds = xr.Dataset({"shap_values": sv_da, "abs_shap": abs_da, "prediction": pred_da})
    ds.attrs["lead_months"] = 6
    ds.attrs["model_type"]  = "xgboost"
    ds.attrs["task"]        = "classification" if classification else "regression"
    return ds


# ---------------------------------------------------------------------------
# aggregate.py tests
# ---------------------------------------------------------------------------

def test_global_mean_abs_shap():
    from src.shap_analysis.aggregate import global_mean_abs_shap
    ds  = _make_shap_ds()
    imp = global_mean_abs_shap(ds)
    assert len(imp) == N_FEAT
    assert (imp.values >= 0).all()
    # sorted descending
    assert imp.iloc[0] >= imp.iloc[-1]
    log.info("global_mean_abs_shap OK  top=%s  val=%.4f", imp.index[0], imp.iloc[0])


def test_global_mean_abs_shap_classification():
    from src.shap_analysis.aggregate import global_mean_abs_shap
    ds  = _make_shap_ds(classification=True)
    imp = global_mean_abs_shap(ds)
    assert len(imp) == N_FEAT
    log.info("global_mean_abs_shap (classification) OK")


def test_seasonal_shap_mean():
    from src.shap_analysis.aggregate import seasonal_shap_mean
    ds  = _make_shap_ds(n=120)
    df  = seasonal_shap_mean(ds)
    assert df.shape == (12, N_FEAT), f"Expected (12, {N_FEAT}), got {df.shape}"
    assert df.index.name == "month"
    assert list(df.index) == list(range(1, 13))
    log.info("seasonal_shap_mean OK  shape=%s", df.shape)


def test_spring_barrier_stats():
    from src.shap_analysis.aggregate import seasonal_shap_mean, spring_barrier_stats
    ds   = _make_shap_ds(n=240)
    seas = seasonal_shap_mean(ds)
    spb  = spring_barrier_stats(seas)
    assert len(spb) == N_FEAT
    assert spb.name == "spb_ratio"
    assert np.isfinite(spb.dropna()).all()
    log.info("spring_barrier_stats OK  min=%.3f  max=%.3f", spb.min(), spb.max())


def test_enso_composite_shap():
    from src.shap_analysis.aggregate import enso_composite_shap
    ds   = _make_shap_ds(n=200)
    comp = enso_composite_shap(ds, threshold=0.5)
    assert set(comp.keys()) == {"elnino", "lanina", "neutral"}
    for phase, series in comp.items():
        assert len(series) == N_FEAT, f"{phase} length mismatch"
    log.info("enso_composite_shap OK  keys=%s", list(comp.keys()))


def test_shap_prediction_corr():
    from src.shap_analysis.aggregate import shap_prediction_corr
    ds   = _make_shap_ds(n=150)
    corr = shap_prediction_corr(ds)
    assert len(corr) == N_FEAT
    assert (corr.abs() <= 1.0 + 1e-9).all()
    log.info("shap_prediction_corr OK  top=%s  r=%.3f", corr.index[0], corr.iloc[0])


def test_lead_importance_table():
    from src.shap_analysis.aggregate import load_shap_store, lead_importance_table
    from src.utils.io_utils import save_zarr

    with tempfile.TemporaryDirectory() as tmp:
        for lead in [3, 6]:
            ds = _make_shap_ds(n=80)
            path = Path(tmp) / f"xgboost_lead{lead:02d}_regression_shap.zarr"
            save_zarr(ds, path)

        df = lead_importance_table(tmp, "xgboost", "regression", leads=[3, 6, 12])
        assert set(df.columns).issuperset({3, 6}), f"Missing leads: {df.columns.tolist()}"
        assert df.shape[0] == N_FEAT
        assert df[3].notna().any()
        assert df[12].isna().all()   # lead=12 not written → all NaN
    log.info("lead_importance_table OK  shape=%s", df.shape)


# ---------------------------------------------------------------------------
# plotting.py tests (check figures created; don't render to screen)
# ---------------------------------------------------------------------------

def test_plot_feature_importance_bar():
    from src.shap_analysis.aggregate import global_mean_abs_shap
    from src.shap_analysis.plotting import plot_feature_importance_bar
    import matplotlib.pyplot as plt

    ds  = _make_shap_ds()
    imp = global_mean_abs_shap(ds)
    fig, ax = plot_feature_importance_bar(imp, top_n=10)
    assert fig is not None and ax is not None
    # width must not exceed 11
    assert fig.get_figwidth() <= 11.1
    plt.close(fig)
    log.info("plot_feature_importance_bar OK")


def test_plot_seasonal_heatmap():
    from src.shap_analysis.aggregate import seasonal_shap_mean
    from src.shap_analysis.plotting import plot_seasonal_heatmap
    import matplotlib.pyplot as plt

    ds   = _make_shap_ds(n=120)
    seas = seasonal_shap_mean(ds)
    fig, ax = plot_seasonal_heatmap(seas, top_n=8)
    assert fig is not None
    assert fig.get_figwidth() <= 11.1
    plt.close(fig)
    log.info("plot_seasonal_heatmap OK")


def test_plot_spring_barrier():
    from src.shap_analysis.aggregate import seasonal_shap_mean
    from src.shap_analysis.plotting import plot_spring_barrier
    import matplotlib.pyplot as plt

    ds   = _make_shap_ds(n=120)
    seas = seasonal_shap_mean(ds)
    fig, ax = plot_spring_barrier(seas)
    assert fig is not None
    plt.close(fig)
    log.info("plot_spring_barrier OK")


def test_plot_enso_asymmetry():
    from src.shap_analysis.aggregate import enso_composite_shap
    from src.shap_analysis.plotting import plot_enso_asymmetry
    import matplotlib.pyplot as plt

    ds   = _make_shap_ds(n=200)
    comp = enso_composite_shap(ds, threshold=0.5)
    fig, ax = plot_enso_asymmetry(comp, top_n=8)
    assert fig is not None
    assert fig.get_figwidth() <= 11.1
    plt.close(fig)
    log.info("plot_enso_asymmetry OK")


def test_plot_lead_heatmap():
    from src.shap_analysis.aggregate import lead_importance_table
    from src.shap_analysis.plotting import plot_lead_importance_heatmap
    from src.utils.io_utils import save_zarr
    import matplotlib.pyplot as plt

    with tempfile.TemporaryDirectory() as tmp:
        for lead in [3, 6, 12]:
            ds   = _make_shap_ds(n=60)
            path = Path(tmp) / f"xgboost_lead{lead:02d}_regression_shap.zarr"
            save_zarr(ds, path)

        df  = lead_importance_table(tmp, "xgboost", "regression", leads=[3, 6, 12])
        fig, ax = plot_lead_importance_heatmap(df, top_n=10)
        assert fig is not None
        plt.close(fig)
    log.info("plot_lead_importance_heatmap OK")


def test_plot_shap_scatter():
    from src.shap_analysis.plotting import plot_shap_scatter
    import matplotlib.pyplot as plt

    ds  = _make_shap_ds(n=80)
    fig, ax = plot_shap_scatter(ds, feature=FEAT_NAMES[0])
    assert fig is not None
    plt.close(fig)
    log.info("plot_shap_scatter OK")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_global_mean_abs_shap,
        test_global_mean_abs_shap_classification,
        test_seasonal_shap_mean,
        test_spring_barrier_stats,
        test_enso_composite_shap,
        test_shap_prediction_corr,
        test_lead_importance_table,
        test_plot_feature_importance_bar,
        test_plot_seasonal_heatmap,
        test_plot_spring_barrier,
        test_plot_enso_asymmetry,
        test_plot_lead_heatmap,
        test_plot_shap_scatter,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            log.error("FAIL %s: %s", t.__name__, exc)
            import traceback; traceback.print_exc()
            failed.append(t.__name__)

    if failed:
        log.error("%d test(s) failed: %s", len(failed), failed)
        sys.exit(1)
    else:
        log.info("All Phase 4 tests passed.")
