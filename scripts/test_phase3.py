"""Smoke tests for Phase 3 SHAP computation.

Aggregation and Zarr I/O tests run in any environment (numpy/xarray only).
Explainer tests (TreeSHAP, GradientSHAP, DeepSHAP) are skipped if shap/xgboost/torch
are not installed — run the full suite inside the shap-enso conda environment.
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
from src.utils.preprocessing import TARGET_LAT, TARGET_LON
from src.shap_analysis.compute_shap import (
    aggregate_cnn_shap_spatial,
    aggregate_lstm_shap,
    project_shap_to_grid,
    save_shap_dataset,
    save_spatial_shap,
    select_background,
    _concat_batches,
)

log = get_logger(__name__)

RNG = np.random.default_rng(99)
N_FEAT  = 38
N_LAT, N_LON = len(TARGET_LAT), len(TARGET_LON)
N_VARS, N_LAGS = 5, 4
N_CHANNELS = N_VARS * N_LAGS
SEQ_LEN = 12
N_VARS_BASIN = 9


# ---------------------------------------------------------------------------
# Utility / aggregation tests (no heavy deps)
# ---------------------------------------------------------------------------

def test_select_background():
    X = RNG.standard_normal((300, N_FEAT)).astype(np.float32)
    bg = select_background(X, n_samples=100, seed=42)
    assert bg.shape == (100, N_FEAT)
    # Reproducible
    bg2 = select_background(X, n_samples=100, seed=42)
    np.testing.assert_array_equal(bg, bg2)
    # Capped at len(X) when n_samples > len(X)
    bg3 = select_background(X, n_samples=500, seed=42)
    assert bg3.shape[0] == 300
    log.info("select_background OK")


def test_aggregate_lstm_shap():
    shap_3d = RNG.standard_normal((50, SEQ_LEN, N_VARS_BASIN)).astype(np.float32)
    agg = aggregate_lstm_shap(shap_3d)
    assert agg.shape == (50, N_VARS_BASIN), f"Expected (50, {N_VARS_BASIN}), got {agg.shape}"
    assert np.all(agg >= 0), "aggregated values should be |SHAP|, i.e. non-negative"
    # Already 2-D input should pass through unchanged
    shap_2d = RNG.standard_normal((50, N_VARS_BASIN)).astype(np.float32)
    np.testing.assert_array_equal(aggregate_lstm_shap(shap_2d), shap_2d)
    log.info("aggregate_lstm_shap OK  output shape %s", agg.shape)


def test_aggregate_cnn_shap_spatial():
    shap_4d = RNG.standard_normal((50, N_CHANNELS, N_LAT, N_LON)).astype(np.float32)
    agg = aggregate_cnn_shap_spatial(shap_4d)
    assert agg.shape == (50, N_CHANNELS)
    assert np.all(agg >= 0)
    log.info("aggregate_cnn_shap_spatial OK  output shape %s", agg.shape)


def test_project_shap_to_grid():
    shap_4d = RNG.standard_normal((50, N_CHANNELS, N_LAT, N_LON)).astype(np.float32)
    variables = ["sst", "d20", "tauu", "olr", "slp"]
    da = project_shap_to_grid(shap_4d, variables, N_LAGS, TARGET_LAT, TARGET_LON)

    assert da.dims == ("var", "lag", "lat", "lon")
    assert da.shape == (N_VARS, N_LAGS, N_LAT, N_LON)
    np.testing.assert_array_equal(da.coords["var"].values, np.array(variables))
    assert np.all(da.values >= 0)
    log.info("project_shap_to_grid OK  shape %s", da.shape)


def test_concat_batches_regression():
    # Simulate two batches of (batch, n_feat) SHAP arrays
    b1 = RNG.standard_normal((10, N_FEAT)).astype(np.float32)
    b2 = RNG.standard_normal((8,  N_FEAT)).astype(np.float32)
    result = _concat_batches([b1, b2])
    assert result.shape == (18, N_FEAT)
    log.info("_concat_batches regression OK")


def test_concat_batches_classification():
    # Simulate two batches of list-per-class SHAP arrays
    b1 = [RNG.standard_normal((10, N_FEAT)).astype(np.float32) for _ in range(3)]
    b2 = [RNG.standard_normal((8,  N_FEAT)).astype(np.float32) for _ in range(3)]
    result = _concat_batches([b1, b2])
    assert result.shape == (18, N_FEAT, 3), f"Got {result.shape}"
    log.info("_concat_batches classification OK  shape %s", result.shape)


# ---------------------------------------------------------------------------
# Zarr save / load roundtrip tests
# ---------------------------------------------------------------------------

def test_save_shap_dataset_regression():
    n, nf = 80, N_FEAT
    shap_vals = RNG.standard_normal((n, nf)).astype(np.float32)
    preds     = RNG.standard_normal(n).astype(np.float32)
    feat_names = [f"feat_{i}" for i in range(nf)]
    times     = pd.date_range("2015-01", periods=n, freq="MS")

    with tempfile.TemporaryDirectory() as tmp:
        path = save_shap_dataset(
            shap_vals, preds, base_value=0.42, feature_names=feat_names,
            times=times, model_type="xgboost", lead=6, task="regression",
            output_dir=tmp,
        )
        assert path.exists()

        ds = xr.open_zarr(str(path))
        assert "shap_values" in ds
        assert "abs_shap"    in ds
        assert "prediction"  in ds
        assert ds.shap_values.shape == (n, nf)
        assert ds.attrs["lead_months"] == 6
        assert ds.attrs["base_value"]  == 0.42
    log.info("save_shap_dataset regression OK  shape %s", shap_vals.shape)


def test_save_shap_dataset_classification():
    n, nf, nc = 60, N_FEAT, 3
    shap_vals = RNG.standard_normal((n, nf, nc)).astype(np.float32)
    preds     = RNG.integers(0, 3, n).astype(np.float32)
    feat_names = [f"feat_{i}" for i in range(nf)]
    times     = pd.date_range("2015-01", periods=n, freq="MS")

    with tempfile.TemporaryDirectory() as tmp:
        path = save_shap_dataset(
            shap_vals, preds, base_value=0.0, feature_names=feat_names,
            times=times, model_type="xgboost", lead=6, task="classification",
            output_dir=tmp,
        )
        ds = xr.open_zarr(str(path))
        assert ds.shap_values.dims == ("time", "feature", "class")
        assert ds.abs_shap.dims   == ("time", "feature")   # mean across classes
    log.info("save_shap_dataset classification OK")


def test_save_spatial_shap():
    shap_4d = RNG.standard_normal((50, N_CHANNELS, N_LAT, N_LON)).astype(np.float32)
    variables = ["sst", "d20", "tauu", "olr", "slp"]
    da = project_shap_to_grid(shap_4d, variables, N_LAGS, TARGET_LAT, TARGET_LON)

    with tempfile.TemporaryDirectory() as tmp:
        path = save_spatial_shap(da, "cnn", lead=6, task="regression", output_dir=tmp)
        assert path.exists()
        ds = xr.open_zarr(str(path))
        assert "mean_abs_shap" in ds
        assert ds.mean_abs_shap.dims == ("var", "lag", "lat", "lon")
    log.info("save_spatial_shap OK  shape %s", da.shape)


# ---------------------------------------------------------------------------
# TreeSHAP integration test (needs xgboost + shap)
# ---------------------------------------------------------------------------

def test_tree_shap_end_to_end():
    try:
        import xgboost as xgb
        import shap  # noqa: F401
    except ImportError:
        log.warning("xgboost or shap not installed — skipping TreeSHAP integration test")
        return

    from src.shap_analysis.compute_shap import get_tree_explainer, compute_tree_shap

    n_tr, n_te, nf = 200, 50, 20
    X_tr = RNG.standard_normal((n_tr, nf)).astype(np.float32)
    y_tr = RNG.standard_normal(n_tr).astype(np.float32)
    X_te = RNG.standard_normal((n_te, nf)).astype(np.float32)

    model = xgb.XGBRegressor(n_estimators=20, max_depth=3, random_state=0)
    model.fit(X_tr, y_tr)

    class _Wrapper:
        def __init__(self, m):
            self.model = m

    explainer = get_tree_explainer(_Wrapper(model))
    sv, bv = compute_tree_shap(explainer, X_te, batch_size=25)

    assert sv.shape == (n_te, nf), f"Expected ({n_te}, {nf}), got {sv.shape}"
    assert np.isfinite(sv).all(), "SHAP values contain NaN/Inf"
    log.info("TreeSHAP end-to-end OK  shape=%s  base_value=%.4f", sv.shape, bv)


# ---------------------------------------------------------------------------
# GradientSHAP integration test (needs torch + shap)
# ---------------------------------------------------------------------------

def test_gradient_shap_end_to_end():
    try:
        import torch
        import shap  # noqa: F401
    except ImportError:
        log.warning("torch or shap not installed — skipping GradientSHAP integration test")
        return

    import torch.nn as nn
    from src.shap_analysis.compute_shap import get_gradient_explainer, compute_deep_shap

    # Tiny LSTM
    class TinyLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(N_VARS_BASIN, 16, batch_first=True)
            self.fc   = nn.Linear(16, 1)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1]).squeeze(-1)

    net = TinyLSTM().eval()
    n_bg, n_te = 20, 15
    X_bg = torch.randn(n_bg, SEQ_LEN, N_VARS_BASIN)
    X_te = np.random.randn(n_te, SEQ_LEN, N_VARS_BASIN).astype(np.float32)

    explainer = get_gradient_explainer(net, X_bg)
    sv, bv = compute_deep_shap(explainer, X_te, batch_size=5, device="cpu")

    assert sv.shape == (n_te, SEQ_LEN, N_VARS_BASIN), f"Got {sv.shape}"
    log.info("GradientSHAP end-to-end OK  raw shape=%s  (before LSTM aggregation)", sv.shape)

    agg = aggregate_lstm_shap(sv)
    assert agg.shape == (n_te, N_VARS_BASIN)
    log.info("  after LSTM aggregation: shape=%s", agg.shape)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_select_background,
        test_aggregate_lstm_shap,
        test_aggregate_cnn_shap_spatial,
        test_project_shap_to_grid,
        test_concat_batches_regression,
        test_concat_batches_classification,
        test_save_shap_dataset_regression,
        test_save_shap_dataset_classification,
        test_save_spatial_shap,
        test_tree_shap_end_to_end,
        test_gradient_shap_end_to_end,
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
        log.info("All Phase 3 tests passed.")
