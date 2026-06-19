"""Smoke tests for Phase 2 model training.

Metrics tests run in any environment.
Model tests (XGBoost, LSTM, CNN) are skipped with a warning if the relevant
package is not installed — install xgboost / torch to run the full suite.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import get_logger
from src.utils.preprocessing import (
    build_class_labels,
    build_cnn_tensors,
    build_lstm_sequences,
    compute_basin_indices,
    TARGET_LAT,
    TARGET_LON,
)

log = get_logger(__name__)

RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

N_MONTHS   = 360
START      = "1979-01"
TIME       = pd.date_range(START, periods=N_MONTHS, freq="MS")
N_VARS     = 9       # basin indices
N_LAT, N_LON = len(TARGET_LAT), len(TARGET_LON)


def _basin_df(n: int = N_MONTHS) -> pd.DataFrame:
    t = pd.date_range(START, periods=n, freq="MS")
    cols = ["sst_nino34","sst_nino4","sst_nino3","sst_nino12",
            "d20_nino34","d20_nino4","tauu_eq","olr_eq","slp_eq"]
    return pd.DataFrame(RNG.standard_normal((n, len(cols))).astype(np.float32), index=t, columns=cols)


def _nino34(n: int = N_MONTHS):
    import xarray as xr
    t = pd.date_range(START, periods=n, freq="MS")
    v = (RNG.standard_normal(n) * 0.8).astype(np.float32)
    return xr.DataArray(v, coords={"time": t}, dims=["time"], name="nino34")


def _grid_ds(n: int = N_MONTHS):
    import xarray as xr
    t = pd.date_range(START, periods=n, freq="MS")
    data = {
        v: (["time","lat","lon"], RNG.standard_normal((n, N_LAT, N_LON)).astype(np.float32))
        for v in ["sst","d20","tauu","olr","slp"]
    }
    return xr.Dataset(data, coords={"time": t, "lat": TARGET_LAT, "lon": TARGET_LON})


def _fake_cfg(tmp_dir: Path) -> dict:
    return {
        "experiment": {"name": "test", "seed": 42,
                       "output_dir": str(tmp_dir)},
        "data": {
            "train_years": [1979, 2005], "val_years": [2006, 2014],
            "test_years":  [2015, 2023], "lag_months": 3,
        },
        "model": {
            "task": "regression",
            "xgb": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1,
                    "subsample": 0.8, "colsample_bytree": 0.8, "early_stopping_rounds": 5},
            "lstm": {"hidden_size": 16, "num_layers": 1, "dropout": 0.0,
                     "sequence_length": 6, "batch_size": 32, "max_epochs": 3, "lr": 1e-3},
            "cnn":  {"channels": [8, 16], "kernel_size": 3, "dropout": 0.0,
                     "batch_size": 16, "max_epochs": 3, "lr": 1e-3},
        },
        "shap": {},
    }


# ---------------------------------------------------------------------------
# Metrics tests (no heavy deps)
# ---------------------------------------------------------------------------

def test_regression_metrics():
    from src.models.metrics import regression_metrics
    y_true = np.array([1.0, 0.5, -0.5, -1.0])
    y_pred = np.array([0.9, 0.4, -0.6, -0.9])
    m = regression_metrics(y_true, y_pred)
    assert set(m.keys()) == {"rmse", "mae", "corr", "r2"}
    assert 0.0 < m["rmse"] < 0.2
    assert m["corr"] > 0.99
    log.info("regression_metrics OK  rmse=%.4f  corr=%.4f", m["rmse"], m["corr"])


def test_classification_metrics():
    from src.models.metrics import classification_metrics
    y_true = np.array([0, 1, 2, 0, 1, 2])
    proba  = np.eye(3)[[0, 1, 2, 0, 1, 2]].astype(float)  # perfect predictions
    m = classification_metrics(y_true, proba)
    assert m["accuracy"] == 1.0
    assert m["bss"] > 0.0
    log.info("classification_metrics OK  acc=%.2f  bss=%.4f", m["accuracy"], m["bss"])


def test_skill_vs_lead():
    from src.models.metrics import skill_vs_lead
    results = {3: {"rmse": 0.5, "corr": 0.8}, 6: {"rmse": 0.7, "corr": 0.65}}
    df = skill_vs_lead(results)
    assert list(df.index) == [3, 6]
    assert "rmse" in df.columns
    log.info("skill_vs_lead OK")


# ---------------------------------------------------------------------------
# Sequence / tensor builder tests (numpy only)
# ---------------------------------------------------------------------------

def test_build_lstm_sequences():
    df = _basin_df(120)
    nino = _nino34(120)
    for lead in [3, 6, 12]:
        X, y, vnames, times = build_lstm_sequences(df, nino, lead, sequence_length=6)
        assert X.ndim == 3 and X.shape[1] == 6 and X.shape[2] == len(df.columns)
        assert len(y) == len(times) == X.shape[0]
        assert X.dtype == np.float32
    log.info("build_lstm_sequences OK  X=%s", X.shape)


def test_build_cnn_tensors():
    ds = _grid_ds(60)
    nino = _nino34(60)
    n_lags = 4
    X, y, ch_names, times = build_cnn_tensors(ds, nino, lead_months=3, n_lags=n_lags)
    n_vars = len(list(ds.data_vars))
    assert X.shape[1] == n_vars * n_lags
    assert X.shape[2] == N_LAT and X.shape[3] == N_LON
    assert len(ch_names) == n_vars * n_lags
    log.info("build_cnn_tensors OK  X=%s  channels=%d", X.shape, X.shape[1])


# ---------------------------------------------------------------------------
# XGBoost model test
# ---------------------------------------------------------------------------

def test_xgb_model():
    try:
        import xgboost  # noqa: F401
    except ImportError:
        log.warning("xgboost not installed — skipping XGBoost model test")
        return

    from src.models.xgb_model import ENSOXGBModel
    from src.models.metrics import regression_metrics

    df = _basin_df(360)
    nino = _nino34(360)

    from src.utils.preprocessing import build_feature_matrix, train_val_test_split_temporal
    X, y, names, times = build_feature_matrix(df, nino, lead_months=3, lag_months=3)
    (X_tr, y_tr, t_tr,
     X_val, y_val, t_val,
     X_te, y_te, t_te) = train_val_test_split_temporal(
        X, y, times, (1979, 2005), (2006, 2014), (2015, 2023))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fake_cfg(Path(tmp))
        model = ENSOXGBModel(cfg, lead=3, task="regression")
        val_m = model.fit(X_tr, y_tr, X_val, y_val, names)
        assert "val_rmse" in val_m or "val_corr" in val_m

        y_pred = model.predict(X_te)
        assert y_pred.shape == (len(y_te),)

        model_dir = Path(tmp) / "xgb"
        saved = model.save(model_dir)
        assert saved.exists()

        m2 = ENSOXGBModel(cfg, lead=3, task="regression")
        m2.load(model_dir)
        np.testing.assert_allclose(m2.predict(X_te), y_pred, rtol=1e-5)

    log.info("XGBoost model test OK  val_m=%s", {k: f'{v:.4f}' for k, v in val_m.items()})


# ---------------------------------------------------------------------------
# LSTM model test
# ---------------------------------------------------------------------------

def test_lstm_model():
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("torch not installed — skipping LSTM model test")
        return

    from src.models.lstm_model import ENSOLSTMModel

    df = _basin_df(360)
    nino = _nino34(360)
    X, y, _, times = build_lstm_sequences(df, nino, lead_months=3, sequence_length=6)

    from src.utils.preprocessing import train_val_test_split_temporal
    (X_tr, y_tr, _, X_val, y_val, _, X_te, y_te, _) = train_val_test_split_temporal(
        X, y, times, (1979, 2005), (2006, 2014), (2015, 2023))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fake_cfg(Path(tmp))
        model = ENSOLSTMModel(cfg, lead=3, task="regression", device="cpu")
        val_m = model.fit(X_tr, y_tr, X_val, y_val)
        assert "val_rmse" in val_m or "best_val_loss" in val_m

        y_pred = model.predict(X_te)
        assert y_pred.shape == (len(y_te),)

        model_dir = Path(tmp) / "lstm"
        saved = model.save(model_dir)
        assert saved.exists()

        m2 = ENSOLSTMModel(cfg, lead=3, task="regression", device="cpu")
        m2.load(model_dir)
        np.testing.assert_allclose(m2.predict(X_te), y_pred, rtol=1e-4)

    log.info("LSTM model test OK")


# ---------------------------------------------------------------------------
# CNN model test
# ---------------------------------------------------------------------------

def test_cnn_model():
    try:
        import torch  # noqa: F401
    except ImportError:
        log.warning("torch not installed — skipping CNN model test")
        return

    from src.models.cnn_model import ENSOCNNModel

    ds = _grid_ds(120)
    nino = _nino34(120)
    X, y, _, times = build_cnn_tensors(ds, nino, lead_months=3, n_lags=2)

    from src.utils.preprocessing import train_val_test_split_temporal
    (X_tr, y_tr, _, X_val, y_val, _, X_te, y_te, _) = train_val_test_split_temporal(
        X, y, times, (1979, 1999), (2000, 2005), (2006, 2009))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _fake_cfg(Path(tmp))
        model = ENSOCNNModel(cfg, lead=3, task="regression", device="cpu")
        val_m = model.fit(X_tr, y_tr, X_val, y_val)
        assert "best_val_loss" in val_m

        y_pred = model.predict(X_te)
        assert y_pred.shape == (len(y_te),)

        model_dir = Path(tmp) / "cnn"
        saved = model.save(model_dir)
        assert saved.exists()

        m2 = ENSOCNNModel(cfg, lead=3, task="regression", device="cpu")
        m2.load(model_dir)
        np.testing.assert_allclose(m2.predict(X_te), y_pred, rtol=1e-4)

    log.info("CNN model test OK")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_regression_metrics,
        test_classification_metrics,
        test_skill_vs_lead,
        test_build_lstm_sequences,
        test_build_cnn_tensors,
        test_xgb_model,
        test_lstm_model,
        test_cnn_model,
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
        log.info("All Phase 2 tests passed.")
