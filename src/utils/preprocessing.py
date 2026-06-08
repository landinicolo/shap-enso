"""Preprocessing pipeline: regridding, anomaly computation, feature construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Standard ENSO monitoring regions (longitude on 0–360 grid)
# ---------------------------------------------------------------------------

REGIONS: dict[str, dict[str, tuple[float, float]]] = {
    "nino12": {"lat": (-10.0,  0.0), "lon": (270.0, 280.0)},
    "nino3":  {"lat": ( -5.0,  5.0), "lon": (210.0, 270.0)},
    "nino34": {"lat": ( -5.0,  5.0), "lon": (190.0, 240.0)},
    "nino4":  {"lat": ( -5.0,  5.0), "lon": (160.0, 210.0)},
    "wpac":   {"lat": ( -5.0,  5.0), "lon": (120.0, 160.0)},
    "eq":     {"lat": ( -5.0,  5.0), "lon": (120.0, 290.0)},
}

# ERA5 short name → CDS API long name mapping
ERA5_VAR_MAP: dict[str, str] = {
    "sst":  "sea_surface_temperature",
    "tauu": "eastward_turbulent_surface_stress",
    "olr":  "top_net_thermal_radiation",   # negated to get OLR (outgoing = positive up)
    "slp":  "mean_sea_level_pressure",
}

# Target 2° grid over the tropical Pacific
TARGET_LAT = np.arange(-30.0, 31.0, 2.0)   # 31 points
TARGET_LON = np.arange(120.0, 292.0, 2.0)  # 86 points


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------

def regrid_to_common(
    ds: xr.Dataset,
    target_lat: np.ndarray = TARGET_LAT,
    target_lon: np.ndarray = TARGET_LON,
    method: str = "bilinear",
) -> xr.Dataset:
    """Regrid all variables in *ds* to a common 2° tropical-Pacific grid.

    Uses xesmf when available; falls back to scipy linear interpolation.

    Args:
        ds: Input dataset with (lat, lon) or (latitude, longitude) dims.
        target_lat: 1-D target latitude array.
        target_lon: 1-D target longitude array.
        method: xesmf regridding method ('bilinear' or 'conservative').

    Returns:
        Dataset on the target grid with dims (time, lat, lon).
    """
    # Normalise coordinate names
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    # Check if already on target grid
    if (
        len(ds.lat) == len(target_lat)
        and np.allclose(ds.lat.values, target_lat, atol=0.1)
        and len(ds.lon) == len(target_lon)
        and np.allclose(ds.lon.values, target_lon, atol=0.1)
    ):
        return ds

    try:
        import xesmf as xe

        # lat/lon must be dimension-indexed coordinates for xesmf to recognise
        # the target as a rectilinear grid (plain variables → ncells flat output).
        ds_out = xr.Dataset(
            coords={
                "lat": xr.DataArray(target_lat, dims=["lat"],
                                    attrs={"units": "degrees_north"}),
                "lon": xr.DataArray(target_lon, dims=["lon"],
                                    attrs={"units": "degrees_east"}),
            }
        )
        regridder = xe.Regridder(ds, ds_out, method=method, periodic=False)
        ds_regrid = regridder(ds)
        # Ensure output dims are named lat/lon regardless of xesmf version
        rename = {}
        for old, new in [("latitude", "lat"), ("longitude", "lon"),
                         ("y", "lat"), ("x", "lon")]:
            if old in ds_regrid.dims and new not in ds_regrid.dims:
                rename[old] = new
        if rename:
            ds_regrid = ds_regrid.rename(rename)
        return ds_regrid
    except ImportError:
        log.warning("xesmf not available — falling back to xarray interp (linear)")
        return ds.interp(lat=target_lat, lon=target_lon, method="linear")


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def _convert_units(ds: xr.Dataset) -> xr.Dataset:
    """Apply physical unit conversions after loading ERA5.

    - sst: K → °C
    - olr: top_net_thermal_radiation (W m⁻²) is net downward LW; OLR = −ttr
    - slp: Pa → hPa
    """
    ds = ds.copy()
    if "sst" in ds:
        ds["sst"] = ds["sst"] - 273.15
        ds["sst"].attrs["units"] = "°C"
    if "olr" in ds:
        ds["olr"] = -ds["olr"]          # outgoing = positive upward
        ds["olr"].attrs["units"] = "W m-2"
    if "slp" in ds:
        ds["slp"] = ds["slp"] / 100.0   # Pa → hPa
        ds["slp"].attrs["units"] = "hPa"
    return ds


# ---------------------------------------------------------------------------
# Anomaly computation
# ---------------------------------------------------------------------------

def compute_anomalies(
    ds: xr.Dataset,
    clim_start: str = "1981-01",
    clim_end: str = "2010-12",
) -> xr.Dataset:
    """Remove the monthly climatology from each variable.

    Climatology is computed over [clim_start, clim_end] and subtracted month-by-month.

    Args:
        ds: Dataset with a 'time' coordinate.
        clim_start: First month of climatology period (YYYY-MM).
        clim_end: Last month of climatology period (YYYY-MM).

    Returns:
        Dataset of anomalies with the same coordinates.
    """
    clim = ds.sel(time=slice(clim_start, clim_end)).groupby("time.month").mean("time")
    return ds.groupby("time.month") - clim


# ---------------------------------------------------------------------------
# Area-weighted means
# ---------------------------------------------------------------------------

def apply_area_weights(da: xr.DataArray) -> xr.DataArray:
    """Return the area-weighted (cos-lat) mean of *da* over its spatial dims."""
    weights = np.cos(np.deg2rad(da.lat))
    weights = weights / weights.sum()
    return (da * weights).sum("lat")


def _region_mean(da: xr.DataArray, region: dict[str, tuple[float, float]]) -> xr.DataArray:
    """Compute area-weighted mean of *da* over a lat/lon region box."""
    lat_min, lat_max = region["lat"]
    lon_min, lon_max = region["lon"]
    sub = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    weights = np.cos(np.deg2rad(sub.lat))
    return sub.weighted(weights).mean(["lat", "lon"])


# ---------------------------------------------------------------------------
# Basin index computation
# ---------------------------------------------------------------------------

#: Which (variable, region) pairs to compute as basin indices
BASIN_INDEX_PAIRS: list[tuple[str, str]] = [
    ("sst",  "nino34"),
    ("sst",  "nino4"),
    ("sst",  "nino3"),
    ("sst",  "nino12"),
    ("d20",  "nino34"),
    ("d20",  "nino4"),
    ("tauu", "eq"),
    ("olr",  "eq"),
    ("slp",  "eq"),
]


def compute_basin_indices(
    ds: xr.Dataset,
    pairs: list[tuple[str, str]] = BASIN_INDEX_PAIRS,
    regions: dict[str, dict] = REGIONS,
) -> pd.DataFrame:
    """Compute area-weighted basin-mean indices for ENSO-relevant variable/region pairs.

    Args:
        ds: Dataset of anomaly fields on the 2° tropical-Pacific grid.
        pairs: List of (variable, region_name) tuples.
        regions: Region definition dict (see REGIONS).

    Returns:
        DataFrame with time as index and columns named "{var}_{region}".
    """
    series: dict[str, np.ndarray] = {}
    for var, region_name in pairs:
        if var not in ds:
            log.warning("Variable '%s' not found in dataset — skipping %s_%s", var, var, region_name)
            continue
        col = f"{var}_{region_name}"
        series[col] = _region_mean(ds[var], regions[region_name]).values

    times = pd.DatetimeIndex(ds.time.values)
    df = pd.DataFrame(series, index=times)
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------
# Season encoding
# ---------------------------------------------------------------------------

def encode_season(time_index: pd.DatetimeIndex) -> np.ndarray:
    """Return (N, 2) array of [sin(2π·month/12), cos(2π·month/12)].

    Cyclical encoding preserves the continuity between December and January.
    """
    months = time_index.month.values.astype(float)
    angle = 2.0 * np.pi * months / 12.0
    return np.column_stack([np.sin(angle), np.cos(angle)])


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df_basin: pd.DataFrame,
    target: xr.DataArray,
    lead_months: int,
    lag_months: int = 3,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Build a lagged feature matrix for XGBoost/LSTM training.

    For each sample at time t the feature vector contains basin indices at
    t, t-1, ..., t-lag_months plus cyclical month encoding. The target is
    the Niño 3.4 anomaly at t + lead_months.

    Args:
        df_basin: DataFrame (time × n_basin_vars) of basin-mean anomalies.
        target: DataArray of the Niño 3.4 anomaly with a 'time' coordinate.
        lead_months: Forecast lead L (months).
        lag_months: Number of prior months included as additional features.

    Returns:
        X:             (n_samples, n_features) float32
        y:             (n_samples,) float32
        feature_names: list of length n_features
        valid_times:   DatetimeIndex of t (initialisation) times for valid samples
    """
    n_vars = df_basin.shape[1]
    base_cols = list(df_basin.columns)

    # Align target onto the same time axis as df_basin
    tgt_series = (
        target
        .to_series()
        .reindex(df_basin.index)
    )

    n = len(df_basin)
    rows_X, rows_y, row_times = [], [], []

    for i in range(lag_months, n):
        t = df_basin.index[i]

        # Target: index i + lead_months
        j = i + lead_months
        if j >= n:
            break
        t_verif = df_basin.index[j]
        y_val = tgt_series.loc[t_verif]
        if np.isnan(y_val):
            continue

        # Features: lags 0..lag_months
        lag_feats = []
        for lag in range(lag_months + 1):       # lag 0 = current month
            lag_feats.append(df_basin.iloc[i - lag].values)
        lag_feats = np.concatenate(lag_feats)   # (n_vars × (lag_months+1),)

        # Season encoding for initialisation month
        season = encode_season(pd.DatetimeIndex([t]))[0]   # (2,)

        rows_X.append(np.concatenate([lag_feats, season]))
        rows_y.append(y_val)
        row_times.append(t)

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)

    # Feature names
    feature_names = []
    for lag in range(lag_months + 1):
        suffix = f"_lag{lag}"
        feature_names.extend(c + suffix for c in base_cols)
    feature_names += ["season_sin", "season_cos"]

    valid_times = pd.DatetimeIndex(row_times)
    return X, y, feature_names, valid_times


# ---------------------------------------------------------------------------
# Classification labels
# ---------------------------------------------------------------------------

def build_class_labels(
    y: np.ndarray,
    thresholds: tuple[float, float] = (-0.5, 0.5),
) -> np.ndarray:
    """Convert continuous Niño 3.4 anomaly to 3-class labels.

    Classes:
        0 = La Niña  (y < thresholds[0])
        1 = Neutral  (thresholds[0] ≤ y ≤ thresholds[1])
        2 = El Niño  (y > thresholds[1])
    """
    labels = np.ones(len(y), dtype=np.int64)     # neutral by default
    labels[y < thresholds[0]] = 0
    labels[y > thresholds[1]] = 2
    return labels


# ---------------------------------------------------------------------------
# Temporal train/val/test split
# ---------------------------------------------------------------------------

def train_val_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    times: pd.DatetimeIndex,
    train_years: tuple[int, int],
    val_years: tuple[int, int],
    test_years: tuple[int, int],
) -> tuple[
    np.ndarray, np.ndarray, pd.DatetimeIndex,
    np.ndarray, np.ndarray, pd.DatetimeIndex,
    np.ndarray, np.ndarray, pd.DatetimeIndex,
]:
    """Strict temporal split — no shuffling, no data leakage.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector (n_samples,).
        times: Initialisation DatetimeIndex of length n_samples.
        train_years: (first_year, last_year) inclusive.
        val_years: (first_year, last_year) inclusive.
        test_years: (first_year, last_year) inclusive.

    Returns:
        (X_tr, y_tr, t_tr, X_val, y_val, t_val, X_te, y_te, t_te)
    """
    def _mask(start: int, end: int) -> np.ndarray:
        return (times.year >= start) & (times.year <= end)

    tr  = _mask(*train_years)
    val = _mask(*val_years)
    te  = _mask(*test_years)

    return (
        X[tr],  y[tr],  times[tr],
        X[val], y[val], times[val],
        X[te],  y[te],  times[te],
    )


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def load_raw_predictors(cfg: dict) -> xr.Dataset:
    """Load all raw ERA5 + D20 NetCDF files into a single xr.Dataset.

    Expects files at cfg['data']['raw_dir']:
      era5/{var}_monthly_{start}_{end}.nc   for sst, tauu, olr, slp
      d20/d20_monthly_{start}_{end}.nc      for D20

    Applies unit conversions and renames coordinates to (lat, lon).
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    start, end = cfg["data"]["time_slice"]
    start_yr = start[:4]
    end_yr   = end[:4]

    arrays: dict[str, xr.DataArray] = {}

    def _snap_to_month_start(da: xr.DataArray) -> xr.DataArray:
        """Snap time coordinate to the first day of each month (e.g. ORAS5 mid-month → start-of-month)."""
        times = pd.DatetimeIndex(da.time.values).to_period("M").to_timestamp()
        return da.assign_coords(time=times)

    for var in cfg["data"]["era5_variables"]:
        fpath = raw_dir / "era5" / f"{var}_monthly_{start_yr}_{end_yr}.nc"
        if not fpath.exists():
            raise FileNotFoundError(
                f"ERA5 file not found: {fpath}\n"
                f"Run: python data/download_era5.py --config configs/default.yaml"
            )
        ds_v = xr.open_dataset(fpath)
        # Drop CDS metadata coords that differ across variables and break xr.Dataset()
        ds_v = ds_v.drop_vars([v for v in ("expver", "number") if v in ds_v.coords or v in ds_v])
        # Normalise coord/dim names: latitude→lat, longitude→lon, valid_time→time
        _rename = {}
        for old, new in (("latitude", "lat"), ("longitude", "lon"), ("valid_time", "time")):
            if old in ds_v.coords and new not in ds_v.coords:
                _rename[old] = new
        if _rename:
            ds_v = ds_v.rename(_rename)
        # Map CDS long name back to short name if needed
        cds_name = {v: k for k, v in ERA5_VAR_MAP.items()}.get(list(ds_v.data_vars)[0])
        da = ds_v[list(ds_v.data_vars)[0]].rename(cds_name or var)
        arrays[cds_name or var] = _snap_to_month_start(da)

    # ERA5-only Dataset — D20 is excluded here so that xr.Dataset() does not
    # align D20's 1° grid to ERA5's 0.25° grid (which NaN-fills D20 everywhere
    # except integer-degree lat/lon and breaks downstream regridding).
    # D20 is loaded and regridded separately via load_raw_d20().
    ds = xr.Dataset(arrays)
    return _convert_units(ds)


def load_raw_d20(cfg: dict) -> xr.Dataset | None:
    """Load the D20/OHC300 file on its native 1° grid.

    Kept separate from load_raw_predictors so that D20 is never merged with
    ERA5 before regridding — merging would align D20 to ERA5's 0.25° lat/lon,
    NaN-filling all non-integer grid points and corrupting the regrid step.

    Returns:
        Dataset with a single 'd20' variable on the native 1° grid,
        or None if the file does not exist.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    start_yr = cfg["data"]["time_slice"][0][:4]
    end_yr   = cfg["data"]["time_slice"][1][:4]
    d20_path = raw_dir / "d20" / f"d20_monthly_{start_yr}_{end_yr}.nc"

    if not d20_path.exists():
        log.warning("D20 file not found: %s — skipping.", d20_path)
        return None

    ds_d20 = xr.open_dataset(str(d20_path))
    da     = ds_d20[list(ds_d20.data_vars)[0]].rename("d20")
    times  = pd.DatetimeIndex(da.time.values).to_period("M").to_timestamp()
    da     = da.assign_coords(time=times)
    return xr.Dataset({"d20": da})


def run_preprocessing(cfg: dict) -> dict[str, Path]:
    """Execute the full Phase 1 preprocessing pipeline.

    Steps:
      1. Load raw predictors
      2. Regrid to 2° tropical-Pacific grid
      3. Slice to configured domain and time range
      4. Compute anomalies
      5. Save gridded predictors as Zarr
      6. Load Niño 3.4 target
      7. Compute basin indices
      8. Save basin-index Zarr and feature matrices for all lead times

    Args:
        cfg: Config dict (from load_config).

    Returns:
        Dict mapping output name → output path.
    """
    from src.utils.io_utils import save_feature_matrix, save_zarr

    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load raw data
    log.info("Loading raw predictors ...")
    ds_raw = load_raw_predictors(cfg)   # ERA5 only (no D20)
    ds_raw_d20 = load_raw_d20(cfg)      # D20 on its native 1° grid

    # 2. Regrid ERA5 and D20 independently to avoid coordinate-alignment NaN.
    # xr.Dataset(era5+d20) would broadcast D20 onto the ERA5 0.25° grid, filling
    # non-integer lat/lon with NaN before xesmf even sees the data.
    log.info("Regridding to 2° grid ...")
    ds_era5 = regrid_to_common(ds_raw)
    if ds_raw_d20 is not None:
        ds_d20  = regrid_to_common(ds_raw_d20)
        ds_grid = xr.merge([ds_era5, ds_d20])
    else:
        ds_grid = ds_era5

    # 3. Domain + time slice
    lat_min, lat_max = cfg["data"]["lat_slice"]
    lon_min, lon_max = cfg["data"]["lon_slice"]
    t_start, t_end   = cfg["data"]["time_slice"]
    ds_grid = ds_grid.sel(
        lat=slice(lat_min, lat_max),
        lon=slice(lon_min, lon_max),
        time=slice(t_start, t_end),
    )

    # 4. Anomalies
    clim_start, clim_end = cfg["data"]["climatology_years"]
    log.info("Computing anomalies (clim %s – %s) ...", clim_start, clim_end)
    ds_anom = compute_anomalies(ds_grid, clim_start, clim_end)

    # 5. Save gridded predictors
    grid_path = processed_dir / "predictors.zarr"
    log.info("Saving gridded predictors → %s", grid_path)
    save_zarr(ds_anom, grid_path)

    # 6. Load Niño 3.4 target
    raw_dir  = Path(cfg["data"]["raw_dir"])
    nino_path = raw_dir / "noaa" / "nino34_monthly.csv"
    if not nino_path.exists():
        raise FileNotFoundError(
            f"Niño 3.4 index not found: {nino_path}\n"
            f"Run: python data/download_noaa_indices.py --config configs/default.yaml"
        )
    df_nino = pd.read_csv(nino_path, index_col=0, parse_dates=True)
    nino34  = xr.DataArray(
        df_nino["nino34_anom"].values,
        coords={"time": df_nino.index},
        dims=["time"],
        name="nino34",
    )
    nino34_path = processed_dir / "target_nino34.zarr"
    log.info("Saving Niño 3.4 target → %s", nino34_path)
    save_zarr(nino34.to_dataset(), nino34_path)

    # 7. Basin indices
    log.info("Computing basin indices ...")
    df_basin = compute_basin_indices(ds_anom)
    basin_path = processed_dir / "predictors_basin.zarr"
    basin_ds   = xr.Dataset.from_dataframe(df_basin)
    save_zarr(basin_ds, basin_path)

    # 8. Feature matrices for each lead time
    outputs: dict[str, Path] = {
        "predictors": grid_path,
        "predictors_basin": basin_path,
        "target_nino34": nino34_path,
    }
    feat_dir = processed_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    for lead in cfg["data"]["lead_months"]:
        log.info("Building feature matrix for lead = %d months ...", lead)
        X, y, feat_names, times = build_feature_matrix(
            df_basin,
            nino34,
            lead_months=lead,
            lag_months=cfg["data"]["lag_months"],
        )
        fpath = feat_dir / f"features_lead{lead:02d}.npz"
        save_feature_matrix(
            X, y, feat_names, times.astype(str), fpath,
            metadata={"lead_months": lead, "lag_months": cfg["data"]["lag_months"]},
        )
        log.info(
            "  lead=%02d  X=%s  y=%s  → %s", lead, X.shape, y.shape, fpath
        )
        outputs[f"features_lead{lead:02d}"] = fpath

    log.info("Preprocessing complete. Outputs:")
    for name, path in outputs.items():
        log.info("  %-30s %s", name, path)

    return outputs


# ---------------------------------------------------------------------------
# LSTM sequence builder
# ---------------------------------------------------------------------------

def build_lstm_sequences(
    df_basin: pd.DataFrame,
    target: xr.DataArray,
    lead_months: int,
    sequence_length: int = 12,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Build overlapping sequences for LSTM input.

    For each initialisation time t the sequence is df_basin[t-seq_len+1 : t+1],
    shaped (sequence_length, n_vars).  The target is Niño 3.4 at t + lead_months.

    Args:
        df_basin:        DataFrame (time × n_vars) of basin-mean anomalies.
        target:          DataArray of Niño 3.4 anomaly.
        lead_months:     Forecast lead L.
        sequence_length: Number of past months fed as input sequence.

    Returns:
        X:            (n_samples, sequence_length, n_vars) float32
        y:            (n_samples,) float32
        var_names:    list of variable names (length n_vars)
        valid_times:  DatetimeIndex of initialisation times
    """
    tgt_series = target.to_series().reindex(df_basin.index)
    n = len(df_basin)

    rows_X, rows_y, row_times = [], [], []
    for i in range(sequence_length - 1, n):
        j = i + lead_months
        if j >= n:
            break
        y_val = tgt_series.iloc[j]
        if np.isnan(y_val):
            continue
        seq = df_basin.iloc[i - sequence_length + 1 : i + 1].values  # (seq_len, n_vars)
        rows_X.append(seq)
        rows_y.append(y_val)
        row_times.append(df_basin.index[i])

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)
    return X, y, list(df_basin.columns), pd.DatetimeIndex(row_times)


# ---------------------------------------------------------------------------
# CNN tensor builder
# ---------------------------------------------------------------------------

def build_cnn_tensors(
    ds_anom: xr.Dataset,
    target: xr.DataArray,
    lead_months: int,
    n_lags: int = 4,
    variables: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DatetimeIndex]:
    """Build multi-channel 2-D tensors for CNN input.

    For each initialisation time t, stacks n_lags consecutive fields for every
    variable into a (n_vars × n_lags, lat, lon) channel tensor.

    Args:
        ds_anom:     Anomaly Dataset on the 2° grid (time, lat, lon per var).
        target:      DataArray of Niño 3.4 anomaly.
        lead_months: Forecast lead L.
        n_lags:      Number of lag time steps (channels per variable).
        variables:   Variables to include; defaults to all data_vars.

    Returns:
        X:             (n_samples, n_vars*n_lags, lat, lon) float32
        y:             (n_samples,) float32
        channel_names: list of "{var}_lag{l}" strings
        valid_times:   DatetimeIndex of initialisation times
    """
    if variables is None:
        variables = [v for v in ds_anom.data_vars]

    times = pd.DatetimeIndex(ds_anom.time.values)
    tgt_series = target.to_series()
    n = len(times)

    # Pre-load all fields into memory as numpy arrays for speed
    arrays: dict[str, np.ndarray] = {v: ds_anom[v].values for v in variables}

    rows_X, rows_y, row_times = [], [], []
    for i in range(n_lags - 1, n):
        j = i + lead_months
        if j >= n:
            break
        t_verif = times[j]
        if t_verif not in tgt_series.index:
            continue
        y_val = tgt_series.loc[t_verif]
        if np.isnan(y_val):
            continue

        channels = [arrays[v][i - lag] for v in variables for lag in range(n_lags)]
        rows_X.append(np.stack(channels, axis=0))   # (n_vars*n_lags, lat, lon)
        rows_y.append(y_val)
        row_times.append(times[i])

    X = np.array(rows_X, dtype=np.float32)
    y = np.array(rows_y, dtype=np.float32)
    channel_names = [f"{v}_lag{lag}" for v in variables for lag in range(n_lags)]
    return X, y, channel_names, pd.DatetimeIndex(row_times)
