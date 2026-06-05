"""Smoke tests for Phase 1 preprocessing — runs entirely on synthetic data.

No ERA5, GODAS, or network access required.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.io_utils import load_feature_matrix, load_zarr, save_feature_matrix, save_zarr
from src.utils.logging_utils import get_logger
from src.utils.preprocessing import (
    REGIONS,
    TARGET_LAT,
    TARGET_LON,
    build_class_labels,
    build_feature_matrix,
    compute_anomalies,
    compute_basin_indices,
    encode_season,
    train_val_test_split_temporal,
)

log = get_logger(__name__)

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

def _make_synthetic_ds(
    n_months: int = 120,
    start: str = "1979-01",
    lat: np.ndarray = TARGET_LAT,
    lon: np.ndarray = TARGET_LON,
) -> xr.Dataset:
    """Create a synthetic xr.Dataset mimicking regridded ERA5 + D20."""
    time = pd.date_range(start=start, periods=n_months, freq="MS")
    nlat, nlon = len(lat), len(lon)

    def _field(scale: float = 1.0) -> np.ndarray:
        return (RNG.standard_normal((n_months, nlat, nlon)) * scale).astype(np.float32)

    return xr.Dataset(
        {
            "sst":  (["time", "lat", "lon"], _field(1.0)),
            "d20":  (["time", "lat", "lon"], _field(20.0)),
            "tauu": (["time", "lat", "lon"], _field(0.05)),
            "olr":  (["time", "lat", "lon"], _field(10.0)),
            "slp":  (["time", "lat", "lon"], _field(5.0)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def _make_nino34(n_months: int = 120, start: str = "1979-01") -> xr.DataArray:
    """Synthetic Niño 3.4 anomaly timeseries."""
    time = pd.date_range(start=start, periods=n_months, freq="MS")
    vals = (RNG.standard_normal(n_months) * 0.8).astype(np.float32)
    return xr.DataArray(vals, coords={"time": time}, dims=["time"], name="nino34")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_anomalies():
    ds = _make_synthetic_ds(n_months=360, start="1979-01")
    ds_anom = compute_anomalies(ds, "1981-01", "2010-12")
    # Monthly mean of anomalies over the clim period should be ~0
    clim_mean = ds_anom.sel(time=slice("1981-01", "2010-12")).groupby("time.month").mean()
    assert float(abs(clim_mean["sst"]).max()) < 1e-4, "Anomalies should be near-zero over clim period"
    log.info("compute_anomalies OK")


def test_compute_basin_indices():
    ds = _make_synthetic_ds(n_months=60, start="1979-01")
    df = compute_basin_indices(ds)
    # Check expected columns
    expected = {"sst_nino34", "sst_nino4", "sst_nino3", "sst_nino12",
                "d20_nino34", "d20_nino4", "tauu_eq", "olr_eq", "slp_eq"}
    assert expected == set(df.columns), f"Unexpected columns: {set(df.columns)}"
    assert len(df) == 60
    assert not df.isnull().any().any(), "Basin indices should have no NaNs"
    log.info("compute_basin_indices OK — shape %s, columns: %s", df.shape, list(df.columns))


def test_encode_season():
    idx = pd.date_range("2000-01", periods=12, freq="MS")
    enc = encode_season(idx)
    assert enc.shape == (12, 2), "Shape should be (12, 2)"
    # sin/cos of annual cycle: Jan and Dec should be continuous
    assert abs(enc[0, 0] - enc[-1, 0]) < 0.6, "Cyclical encoding should be continuous across year boundary"
    # Values in [-1, 1]
    assert enc.min() >= -1.01 and enc.max() <= 1.01
    log.info("encode_season OK — min=%.3f  max=%.3f", enc.min(), enc.max())


def test_build_feature_matrix():
    ds    = _make_synthetic_ds(n_months=120, start="1979-01")
    nino  = _make_nino34(n_months=120, start="1979-01")
    df_b  = compute_basin_indices(ds)

    for lead in [3, 6, 12]:
        X, y, names, times = build_feature_matrix(df_b, nino, lead_months=lead, lag_months=3)
        n_vars = len(df_b.columns)
        expected_feats = n_vars * 4 + 2    # 4 lags (0–3) + sin/cos
        assert X.shape[1] == expected_feats, (
            f"lead={lead}: expected {expected_feats} features, got {X.shape[1]}"
        )
        assert X.shape[0] == y.shape[0] == len(times)
        assert X.dtype == np.float32
        assert not np.any(np.isnan(X)), "Feature matrix should contain no NaNs"
        assert "season_sin" in names and "season_cos" in names
        log.info(
            "build_feature_matrix OK — lead=%02d  X=%s  n_features=%d",
            lead, X.shape, X.shape[1],
        )


def test_build_class_labels():
    y     = np.array([-1.0, -0.3, 0.0, 0.4, 1.2, -0.6])
    labels = build_class_labels(y, thresholds=(-0.5, 0.5))
    expected = np.array([0, 1, 1, 1, 2, 0])
    np.testing.assert_array_equal(labels, expected)
    log.info("build_class_labels OK")


def test_temporal_split():
    ds   = _make_synthetic_ds(n_months=540, start="1979-01")  # 1979–2023
    nino = _make_nino34(n_months=540, start="1979-01")
    df_b = compute_basin_indices(ds)
    X, y, names, times = build_feature_matrix(df_b, nino, lead_months=6, lag_months=3)

    X_tr, y_tr, t_tr, X_val, y_val, t_val, X_te, y_te, t_te = train_val_test_split_temporal(
        X, y, times,
        train_years=(1979, 2005),
        val_years=(2006, 2014),
        test_years=(2015, 2023),
    )
    assert X_tr.shape[0] > 0 and X_val.shape[0] > 0 and X_te.shape[0] > 0
    # No overlap between splits
    assert t_tr.max() < t_val.min(), "Train must end before val starts"
    assert t_val.max() < t_te.min(), "Val must end before test starts"
    # Correct year ranges
    assert t_tr.year.min() >= 1979 and t_tr.year.max() <= 2005
    assert t_val.year.min() >= 2006 and t_val.year.max() <= 2014
    assert t_te.year.min() >= 2015
    log.info(
        "temporal_split OK — train=%d  val=%d  test=%d",
        len(t_tr), len(t_val), len(t_te),
    )


def test_io_roundtrip_zarr():
    ds = _make_synthetic_ds(n_months=24, start="1979-01")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.zarr"
        save_zarr(ds, path)
        ds2 = load_zarr(path)
        for var in ds.data_vars:
            np.testing.assert_allclose(
                ds[var].values, ds2[var].values, rtol=1e-5,
                err_msg=f"Zarr roundtrip mismatch for variable '{var}'",
            )
    log.info("Zarr roundtrip OK")


def test_io_roundtrip_feature_matrix():
    ds    = _make_synthetic_ds(n_months=60, start="1979-01")
    nino  = _make_nino34(n_months=60, start="1979-01")
    df_b  = compute_basin_indices(ds)
    X, y, names, times = build_feature_matrix(df_b, nino, lead_months=3, lag_months=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "features_lead03.npz"
        save_feature_matrix(X, y, names, times.astype(str), path)
        X2, y2, names2, times2 = load_feature_matrix(path)

    np.testing.assert_allclose(X, X2)
    np.testing.assert_allclose(y, y2)
    assert names == names2
    log.info("Feature matrix roundtrip OK — shape %s", X2.shape)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_compute_anomalies,
        test_compute_basin_indices,
        test_encode_season,
        test_build_feature_matrix,
        test_build_class_labels,
        test_temporal_split,
        test_io_roundtrip_zarr,
        test_io_roundtrip_feature_matrix,
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
        log.info("All Phase 1 tests passed.")
