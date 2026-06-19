"""SHAP aggregation and analysis utilities.

Functions operate on xr.Dataset objects from save_shap_dataset / save_spatial_shap.
All functions are pure numpy/pandas/xarray — no heavy ML deps.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_shap_store(
    output_dir: str | Path,
    model_type: str,
    lead: int,
    task: str,
) -> xr.Dataset:
    """Load a SHAP Zarr store written by save_shap_dataset."""
    from src.utils.io_utils import load_zarr
    path = Path(output_dir) / f"{model_type}_lead{lead:02d}_{task}_shap.zarr"
    if not path.exists():
        raise FileNotFoundError(f"SHAP store not found: {path}")
    ds = load_zarr(path)
    log.info("Loaded SHAP store  %s  dims=%s", path.name, dict(ds.sizes))
    return ds


def load_spatial_shap_store(
    output_dir: str | Path,
    model_type: str,
    lead: int,
    task: str,
) -> xr.Dataset:
    """Load a spatial SHAP Zarr store written by save_spatial_shap."""
    from src.utils.io_utils import load_zarr
    path = Path(output_dir) / f"{model_type}_lead{lead:02d}_{task}_spatial_shap.zarr"
    if not path.exists():
        raise FileNotFoundError(f"Spatial SHAP store not found: {path}")
    return load_zarr(path)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def global_mean_abs_shap(ds: xr.Dataset) -> pd.Series:
    """Mean absolute SHAP per feature, sorted descending.

    Works for regression (time, feature) and classification (time, feature, class)
    — uses abs_shap which is already collapsed to (time, feature).
    """
    mean_abs = ds["abs_shap"].mean("time").values
    features = ds.coords["feature"].values.astype(str)
    s = pd.Series(mean_abs, index=features, name="mean_abs_shap")
    return s.sort_values(ascending=False)


def lead_importance_table(
    output_dir: str | Path,
    model_type: str,
    task: str,
    leads: list[int] | tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    """Build a (feature × lead) importance table from multiple SHAP stores.

    Missing leads produce NaN columns.
    """
    rows: dict[int, pd.Series] = {}
    for lead in leads:
        try:
            ds = load_shap_store(output_dir, model_type, lead, task)
            rows[lead] = global_mean_abs_shap(ds)
        except (FileNotFoundError, Exception) as exc:
            log.warning("Could not load SHAP for lead=%d: %s", lead, exc)
            rows[lead] = pd.Series(dtype=float, name="mean_abs_shap")

    df = pd.DataFrame(rows)
    df.index.name = "feature"
    df.columns.name = "lead_months"
    return df


# ---------------------------------------------------------------------------
# Seasonal analysis
# ---------------------------------------------------------------------------

def seasonal_shap_mean(ds: xr.Dataset) -> pd.DataFrame:
    """Mean absolute SHAP per feature per calendar month.

    Uses the initialization time dimension. Returns a DataFrame
    with shape (12, n_features) indexed by month 1..12.
    """
    abs_shap = ds["abs_shap"].values          # (time, feature)
    features = ds.coords["feature"].values.astype(str)
    times    = pd.DatetimeIndex(ds.coords["time"].values)

    rows: dict[int, np.ndarray] = {}
    for m in range(1, 13):
        mask = times.month == m
        if mask.sum() == 0:
            rows[m] = np.full(len(features), np.nan)
        else:
            rows[m] = abs_shap[np.where(mask)[0]].mean(axis=0)

    df = pd.DataFrame(rows, index=features).T   # (12, n_features)
    df.index.name = "month"
    return df


def spring_barrier_stats(seasonal_df: pd.DataFrame) -> pd.Series:
    """Spring predictability barrier (SPB) ratio per feature.

    SPB ratio = mean SHAP in boreal spring (MAM) / mean SHAP in autumn (SON).
    Ratio < 1 indicates reduced predictability at the spring barrier.
    """
    spring = seasonal_df.loc[[3, 4, 5]].mean()
    autumn = seasonal_df.loc[[9, 10, 11]].mean()
    ratio = spring / autumn.where(autumn > 1e-9, other=np.nan)
    ratio.name = "spb_ratio"
    return ratio.sort_values()


# ---------------------------------------------------------------------------
# ENSO asymmetry
# ---------------------------------------------------------------------------

def enso_composite_shap(
    ds: xr.Dataset,
    threshold: float = 0.5,
) -> dict[str, pd.Series]:
    """Split mean |SHAP| by ENSO phase inferred from the model's prediction.

    Returns dict with keys 'elnino', 'lanina', 'neutral', each a pd.Series
    of mean abs_shap values for that phase.
    Predictions > threshold are El Niño; < -threshold are La Niña.
    """
    preds    = ds["prediction"].values
    abs_shap = ds["abs_shap"].values          # (time, feature)
    features = ds.coords["feature"].values.astype(str)

    idx_en  = np.where(preds >  threshold)[0]
    idx_ln  = np.where(preds < -threshold)[0]
    idx_neu = np.where((preds >= -threshold) & (preds <= threshold))[0]

    def _mean(idx: np.ndarray) -> pd.Series:
        if len(idx) == 0:
            return pd.Series(np.full(len(features), np.nan), index=features)
        return pd.Series(abs_shap[idx].mean(axis=0), index=features)

    result = {
        "elnino":  _mean(idx_en),
        "lanina":  _mean(idx_ln),
        "neutral": _mean(idx_neu),
    }
    log.info(
        "ENSO composite: El Niño n=%d  La Niña n=%d  Neutral n=%d",
        len(idx_en), len(idx_ln), len(idx_neu),
    )
    return result


# ---------------------------------------------------------------------------
# SHAP–prediction correlation
# ---------------------------------------------------------------------------

def shap_prediction_corr(ds: xr.Dataset) -> pd.Series:
    """Pearson r between each feature's SHAP value and the model's Niño3.4 prediction.

    High positive r  → feature drives El Niño predictions.
    High negative r  → feature drives La Niña predictions.

    For classification, uses El Niño class (class index 2) SHAP values.
    """
    preds    = ds["prediction"].values
    features = list(ds.coords["feature"].values.astype(str))
    corrs    = {}

    for i, feat in enumerate(features):
        if "class" in ds["shap_values"].dims:
            sv = ds["shap_values"].isel(feature=i, **{"class": 2}).values
        else:
            sv = ds["shap_values"].isel(feature=i).values
        mask = np.isfinite(sv) & np.isfinite(preds)
        corrs[feat] = float(np.corrcoef(sv[mask], preds[mask])[0, 1])

    return pd.Series(corrs, name="shap_pred_corr").sort_values(ascending=False)


# ---------------------------------------------------------------------------
# Metrics compilation (no model weights needed — uses stored predictions)
# ---------------------------------------------------------------------------

def compute_metrics_from_shap_store(
    ds: xr.Dataset,
    target_ds: xr.Dataset,
) -> dict[str, float]:
    """Compute skill metrics by matching shap store predictions to observed target.

    Args:
        ds:        SHAP dataset (has coords["time"] and variable "prediction").
        target_ds: Preprocessed target dataset with a single data variable (Niño3.4).

    Returns:
        dict with keys rmse, mae, corr, r2.
    """
    from src.models.metrics import regression_metrics

    preds = ds["prediction"].values
    times = pd.DatetimeIndex(ds.coords["time"].values)

    target = list(target_ds.data_vars.values())[0]
    target_times = pd.DatetimeIndex(target.coords["time"].values)

    # Align on shared times
    common = times.intersection(target_times)
    if len(common) == 0:
        log.warning("No common times between SHAP store and target — returning NaN metrics")
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan, "r2": np.nan}

    pred_aligned = preds[times.get_indexer(common)]
    true_aligned = target.sel(time=common).values

    return regression_metrics(true_aligned, pred_aligned)
