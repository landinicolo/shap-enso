"""Compute SHAP for tuned CNN ENSO models and create spatial XAI figures.

This script is the CNN counterpart of the tuned LSTM XAI workflow. It is
designed for the tuned-wide directory layout created by train_cnn_tuned.py:

    data/tuned_wide/cnn/lead03_regression/
        models/
        metrics/
        predictions/
        tuning/

It loads the tuned CNN model from the per-lead run folder, rebuilds CNN tensors,
applies train-only standardisation, computes DeepSHAP on the test period, and
saves both channel-level SHAP and per-sample signed spatial SHAP.

Key outputs
-----------
Per lead:
    data/tuned_wide/cnn/lead06_regression/shap/
        cnn_lead06_regression_shap.zarr
        cnn_lead06_regression_spatial_samples.zarr
        cnn_lead06_regression_spatial_summary.zarr

    figures/tuned_wide/cnn/lead06_regression/
        channel bars, beeswarm-style plots, violins, hex plots,
        monthly heatmaps, seasonal spatial maps, event maps, phase maps

Examples
--------
python scripts/compute_shap_tuned_cnn.py \
    --config configs/default.yaml \
    --lead 6 \
    --task regression \
    --device cuda \
    --tuned-root data/tuned_wide/cnn

python scripts/compute_shap_tuned_cnn.py \
    --config configs/default.yaml \
    --lead 6 \
    --task regression \
    --model-dir data/tuned_wide/cnn/lead06_regression/models \
    --output-dir data/tuned_wide/cnn/lead06_regression/shap \
    --fig-dir figures/tuned_wide/cnn/lead06_regression
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    ccrs = None
    cfeature = None
    HAS_CARTOPY = False

import torch
import torch.nn as nn

from src.utils.config import load_config
from src.utils.io_utils import load_zarr
from src.utils.logging_utils import get_logger
from src.utils.preprocessing import (
    build_class_labels,
    build_cnn_tensors,
    train_val_test_split_temporal,
)
from src.shap_analysis.compute_shap import (
    select_background,
    get_deep_explainer,
    compute_deep_shap,
)

log = get_logger(__name__)


class _Unsqueeze(nn.Module):
    """Wrap a regression net so DeepExplainer sees output shape (B, 1)."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out.unsqueeze(-1) if out.dim() == 1 else out


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_tuned_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "cnn"


def default_global_shap_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "shap_tuned_wide" / "cnn"


def default_fig_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "figures" / "tuned_wide" / "cnn"


def run_dir(tuned_root: Path, lead: int, task: str) -> Path:
    return tuned_root / f"lead{lead:02d}_{task}"


def write_zarr_safe(ds: xr.Dataset, path: Path) -> None:
    """Write a Zarr store in a way that avoids Zarr v3/numcodecs issues.

    Some HPC environments combine recent zarr/xarray with old-style numcodecs
    compressors. For portability, force Zarr v2 where supported and do not
    pass numcodecs compressor objects.
    """
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds.to_zarr(str(path), mode="w", consolidated=False, zarr_format=2)
    except TypeError:
        ds.to_zarr(str(path), mode="w", consolidated=False, zarr_version=2)


# ---------------------------------------------------------------------------
# Data / model helpers
# ---------------------------------------------------------------------------


def load_tuned_cnn_model(
    cfg: dict,
    lead: int,
    task: str,
    model_dir: Path,
    model_module: str,
    device: str,
):
    module = importlib.import_module(f"src.models.{model_module}")
    cls = getattr(module, "ENSOCNNModel")
    model = cls(cfg, lead, task, device=device)
    model.load(model_dir)
    if model.net is None:
        raise RuntimeError("Loaded CNN model has no network.")
    model.net.eval()
    for m in model.net.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model


def load_cnn_arrays(cfg: dict, lead: int, task: str):
    processed = Path(cfg["data"]["processed_dir"])
    grid_path = processed / "predictors.zarr"
    nino_path = processed / "target_nino34.zarr"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing {grid_path}. Run preprocessing first.")
    if not nino_path.exists():
        raise FileNotFoundError(f"Missing {nino_path}. Run preprocessing first.")

    ds_anom = load_zarr(grid_path).compute()
    ds_nino = load_zarr(nino_path).compute()
    nino34 = ds_nino["nino34"] if "nino34" in ds_nino else list(ds_nino.data_vars.values())[0]

    n_lags = int(cfg["data"]["lag_months"]) + 1
    era5_vars = list(cfg["data"]["era5_variables"])
    if "d20" in ds_anom and "d20" not in era5_vars:
        era5_vars.append("d20")

    X, y_reg, ch_names, times = build_cnn_tensors(ds_anom, nino34, lead, n_lags, era5_vars)

    (X_tr, y_tr, t_tr,
     X_val, y_val, t_val,
     X_te, y_te, t_te) = train_val_test_split_temporal(
        X, y_reg, times,
        train_years=tuple(cfg["data"]["train_years"]),
        val_years=tuple(cfg["data"]["val_years"]),
        test_years=tuple(cfg["data"]["test_years"]),
    )

    X_mean = np.nanmean(X_tr, axis=(0, 2, 3), keepdims=True)
    X_std = np.nanstd(X_tr, axis=(0, 2, 3), keepdims=True)
    X_std = np.where(X_std < 1e-8, 1.0, X_std)

    X_tr_s = np.nan_to_num((X_tr - X_mean) / X_std, nan=0.0)
    X_val_s = np.nan_to_num((X_val - X_mean) / X_std, nan=0.0)
    X_te_s = np.nan_to_num((X_te - X_mean) / X_std, nan=0.0)

    if task == "classification":
        y_tr_fit = build_class_labels(y_tr)
        y_val_fit = build_class_labels(y_val)
        y_te_eval = build_class_labels(y_te)
    else:
        y_tr_fit, y_val_fit, y_te_eval = y_tr, y_val, y_te

    meta = {
        "ds_anom": ds_anom,
        "era5_vars": era5_vars,
        "n_lags": n_lags,
        "ch_names": list(map(str, ch_names)),
        "X_mean": X_mean,
        "X_std": X_std,
        "raw_target_test": y_te,
    }
    return (X_tr_s, y_tr_fit, t_tr,
            X_val_s, y_val_fit, t_val,
            X_te_s, y_te_eval, t_te,
            meta)


def subset_eval(X_te: np.ndarray, y_te: np.ndarray, t_te, max_eval_samples: int, seed: int):
    n = len(X_te)
    if max_eval_samples is None or max_eval_samples <= 0 or max_eval_samples >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=int(max_eval_samples), replace=False))
    return X_te[idx], np.asarray(y_te)[idx], pd.DatetimeIndex(t_te)[idx], idx


def predict_eval(model, X_eval: np.ndarray, task: str) -> np.ndarray:
    pred = model.predict(X_eval)
    if task == "classification":
        return np.argmax(np.asarray(pred), axis=1).astype(np.float32)
    return np.asarray(pred, dtype=np.float32).ravel()


def normalise_shap_array(shap_spatial: Any) -> np.ndarray:
    arr = np.asarray(shap_spatial)
    # SHAP can return (N,C,H,W,1) for regression or sometimes a list/extra output axis.
    while arr.ndim > 4:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr.mean(axis=-1)
    if arr.ndim != 4:
        raise ValueError(f"Expected CNN spatial SHAP with 4 dims (N,C,H,W), got {arr.shape}")
    return arr.astype(np.float32)


def reshape_spatial_by_var_lag(shap_spatial: np.ndarray, era5_vars: list[str], n_lags: int):
    n, c, h, w = shap_spatial.shape
    expected = len(era5_vars) * int(n_lags)
    if c != expected:
        raise ValueError(
            f"Cannot reshape channels into var/lag: C={c}, len(vars)*n_lags={expected}. "
            "Use feature-level spatial output instead."
        )
    return shap_spatial.reshape(n, len(era5_vars), int(n_lags), h, w)


def make_channel_dataset(
    shap_spatial: np.ndarray,
    preds: np.ndarray,
    observed: np.ndarray,
    ch_names: list[str],
    times_eval,
    lead: int,
    task: str,
    base_value: float | np.ndarray | None,
) -> xr.Dataset:
    shap_2d = np.nanmean(shap_spatial, axis=(2, 3)).astype(np.float32)
    abs_2d = np.nanmean(np.abs(shap_spatial), axis=(2, 3)).astype(np.float32)
    ds = xr.Dataset(
        {
            "shap_values": (("time", "feature"), shap_2d),
            "abs_shap": (("time", "feature"), abs_2d),
            "prediction": ("time", np.asarray(preds, dtype=np.float32)),
            "observed": ("time", np.asarray(observed, dtype=np.float32)),
        },
        coords={
            "time": pd.DatetimeIndex(times_eval),
            "feature": np.asarray(ch_names, dtype=str),
        },
        attrs={
            "model_type": "cnn_tuned",
            "lead_months": int(lead),
            "task": str(task),
            "base_value": np.asarray(base_value).tolist() if base_value is not None else None,
            "description": "Channel-level CNN SHAP aggregated over spatial grid cells.",
        },
    )
    return ds


def make_spatial_sample_dataset(
    shap_spatial: np.ndarray,
    preds: np.ndarray,
    observed: np.ndarray,
    times_eval,
    meta: dict,
    lead: int,
    task: str,
) -> xr.Dataset:
    ds_anom = meta["ds_anom"]
    era5_vars = meta["era5_vars"]
    n_lags = int(meta["n_lags"])
    spatial = reshape_spatial_by_var_lag(shap_spatial, era5_vars, n_lags)
    ds = xr.Dataset(
        {
            "shap_values": (("time", "var", "lag", "lat", "lon"), spatial),
            "abs_shap": (("time", "var", "lag", "lat", "lon"), np.abs(spatial)),
            "prediction": ("time", np.asarray(preds, dtype=np.float32)),
            "observed": ("time", np.asarray(observed, dtype=np.float32)),
        },
        coords={
            "time": pd.DatetimeIndex(times_eval),
            "var": np.asarray(era5_vars, dtype=str),
            "lag": np.arange(n_lags, dtype=np.int16),
            "lat": ds_anom.lat.values,
            "lon": ds_anom.lon.values,
        },
        attrs={
            "model_type": "cnn_tuned",
            "lead_months": int(lead),
            "task": str(task),
            "description": "Per-sample signed spatial CNN SHAP values.",
        },
    )
    return ds


def season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def make_spatial_summary_dataset(spatial_ds: xr.Dataset) -> xr.Dataset:
    months = pd.DatetimeIndex(spatial_ds.time.values).month
    seasons = np.array([season_name(int(m)) for m in months], dtype=str)
    work = spatial_ds.assign_coords(month=("time", months), season=("time", seasons))
    mean_abs = work["abs_shap"].mean("time")
    mean_signed = work["shap_values"].mean("time")
    month_abs = work["abs_shap"].groupby("month").mean("time")
    month_signed = work["shap_values"].groupby("month").mean("time")
    season_abs = work["abs_shap"].groupby("season").mean("time").reindex(season=["DJF", "MAM", "JJA", "SON"])
    season_signed = work["shap_values"].groupby("season").mean("time").reindex(season=["DJF", "MAM", "JJA", "SON"])
    out = xr.Dataset(
        {
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "monthly_abs_shap": month_abs,
            "monthly_signed_shap": month_signed,
            "seasonal_abs_shap": season_abs,
            "seasonal_signed_shap": season_signed,
        },
        attrs={
            **spatial_ds.attrs,
            "description": "Mean, monthly, and seasonal summaries of per-sample CNN spatial SHAP.",
        },
    )
    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def save_fig(fig, path: Path, dpi: int = 180, tight: bool = True):
    ensure_dir(path.parent)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _spatial_subplots(nrows, ncols, figsize):
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree(central_longitude=180)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            squeeze=False,
            subplot_kw={"projection": proj}
        )
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    return fig, axes


def _draw_coastlines(ax):
    if HAS_CARTOPY:
        try:
            ax.add_feature(
                cfeature.COASTLINE.with_scale("110m"),
                edgecolor="white",
                linewidth=1.2,
                zorder=4
            )
            ax.add_feature(
                cfeature.COASTLINE.with_scale("110m"),
                edgecolor="black",
                linewidth=0.45,
                zorder=5
            )
        except Exception:
            ax.coastlines(color="white", linewidth=1.2, zorder=4)
            ax.coastlines(color="black", linewidth=0.45, zorder=5)


def _pcolor_map(ax, lon, lat, data, cmap, vmin, vmax):
    if HAS_CARTOPY:
        im = ax.pcolormesh(
            lon, lat, data,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            transform=ccrs.PlateCarree()
        )
        ax.gridlines(draw_labels=False, linewidth=0.25,
                     color="gray", alpha=0.35, linestyle="--")
    else:
        im = ax.pcolormesh(lon, lat, data,
                            shading="auto",
                            cmap=cmap,
                            vmin=vmin, vmax=vmax)

    _draw_coastlines(ax)
    return im


def _finalize_spatial_figure(fig, axes, im, cbar_label: str, suptitle: str, out_path: Path):
    # Reserve consistent room for colorbar and title; avoid tight_layout
    # because it can move the colorbar / legend into odd places.
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.08, top=0.92, wspace=0.16, hspace=0.28)
    cax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    fig.suptitle(suptitle, y=0.97)
    save_fig(fig, out_path, tight=False)


def plot_top_channel_bar(channel_ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int = 20):
    imp = channel_ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(imp))))
    ax.barh(np.arange(len(imp)), imp.values[::-1])
    ax.set_yticks(np.arange(len(imp)))
    ax.set_yticklabels(imp.index[::-1])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Top {top_n} CNN channels by mean |SHAP|")
    save_fig(fig, fig_dir / f"{prefix}_top{top_n}_channel_bar.png")


def plot_beeswarm(channel_ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int = 20, seed: int = 0):
    vals = channel_ds["shap_values"].to_pandas()
    abs_mean = channel_ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False)
    feats = list(abs_mean.head(top_n).index)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(feats))))
    for i, feat in enumerate(feats[::-1]):
        x = vals[feat].values
        if len(x) > 600:
            idx = rng.choice(len(x), size=600, replace=False)
            x = x[idx]
        y = np.full_like(x, i, dtype=float) + rng.normal(0, 0.08, size=len(x))
        ax.scatter(x, y, s=12, alpha=0.55, linewidths=0)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels(feats[::-1])
    ax.set_xlabel("Signed SHAP value")
    ax.set_title("CNN SHAP beeswarm-style distribution")
    save_fig(fig, fig_dir / f"{prefix}_shap_beeswarm.png")


def plot_violin(channel_ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int = 15):
    vals = channel_ds["shap_values"].to_pandas()
    abs_mean = channel_ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False)
    feats = list(abs_mean.head(top_n).index)
    data = [vals[f].dropna().values for f in feats[::-1]]
    fig, ax = plt.subplots(figsize=(8, max(5, 0.32 * len(feats))))
    parts = ax.violinplot(data, vert=False, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.65)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(np.arange(1, len(feats) + 1))
    ax.set_yticklabels(feats[::-1])
    ax.set_xlabel("Signed SHAP value")
    ax.set_title("Signed SHAP distributions")
    save_fig(fig, fig_dir / f"{prefix}_shap_violins.png")


def plot_monthly_heatmap(channel_ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int = 20):
    abs_mean = channel_ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False)
    feats = list(abs_mean.head(top_n).index)
    df = channel_ds["abs_shap"].sel(feature=feats).to_pandas()
    df["month"] = pd.DatetimeIndex(df.index).month
    mat = df.groupby("month")[feats].mean().reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(feats)), 4.5))
    im = ax.imshow(mat.values, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right")
    ax.set_title("Monthly mean |SHAP| by CNN channel")
    ax.set_ylabel("Initialisation month")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |SHAP|")
    save_fig(fig, fig_dir / f"{prefix}_monthly_channel_shap_heatmap.png")


def plot_hex_general(channel_ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int = 6):
    pred = channel_ds["prediction"].values
    total_abs = channel_ds["abs_shap"].sum("feature").values
    fig, ax = plt.subplots(figsize=(6, 4.5))
    hb = ax.hexbin(pred, total_abs, gridsize=28, mincnt=1)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Total |SHAP| across channels")
    ax.set_title("Prediction confidence cloud: prediction vs total attribution")
    fig.colorbar(hb, ax=ax, label="Count")
    save_fig(fig, fig_dir / f"{prefix}_hex_prediction_total_abs_shap.png")

    # Small scatter atlas for top predictors.
    imp = channel_ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False).head(top_n)
    ncols = 3
    nrows = int(np.ceil(len(imp) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows), squeeze=False)
    vals = channel_ds["shap_values"].to_pandas()
    for ax, feat in zip(axes.ravel(), imp.index):
        ax.scatter(pred, vals[feat].values, s=16, alpha=0.6, linewidths=0)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.axvline(0, color="k", lw=0.7, ls=":")
        ax.set_title(feat)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("SHAP")
    for ax in axes.ravel()[len(imp):]:
        ax.axis("off")
    fig.suptitle("Feature SHAP vs prediction scatter atlas", y=1.02)
    save_fig(fig, fig_dir / f"{prefix}_shap_vs_prediction_scatter_atlas.png")


def choose_dominant_var_lag(summary_ds: xr.Dataset) -> tuple[str, int]:
    score = summary_ds["mean_abs_shap"].mean(("lat", "lon"))
    idx = int(np.nanargmax(score.values))
    ivar, ilag = np.unravel_index(idx, score.shape)
    return str(score.coords["var"].values[ivar]), int(score.coords["lag"].values[ilag])


def plot_spatial_multi_predictor_seasons(
    summary_ds: xr.Dataset,
    fig_dir: Path,
    prefix: str,
    variables: list[str] | None = None,
    lag: int = 0,
    quantity: str = "seasonal_abs_shap",
):
    available = [str(v) for v in summary_ds.coords["var"].values]
    if variables is None or not variables:
        # choose up to four most important variables at this lag
        score = summary_ds["mean_abs_shap"].sel(lag=lag).mean(("lat", "lon")).to_pandas().sort_values(ascending=False)
        variables = list(score.index[: min(4, len(score))])
    variables = [v for v in variables if v in available]
    if not variables:
        return
    seasons = ["DJF", "MAM", "JJA", "SON"]
    nrows, ncols = len(variables), len(seasons)
    fig, axes = _spatial_subplots(nrows, ncols, figsize=(4.2 * ncols, 2.9 * nrows))
    da = summary_ds[quantity].sel(lag=lag)
    vmax = float(da.sel(var=variables, season=seasons).quantile(0.99))
    vmin = float(da.sel(var=variables, season=seasons).quantile(0.01)) if "signed" in quantity else 0.0
    cmap = "RdBu_r" if "signed" in quantity else "magma"
    if "signed" in quantity:
        lim = max(abs(vmin), abs(vmax))
        vmin, vmax = -lim, lim
    for r, var in enumerate(variables):
        for c, season in enumerate(seasons):
            ax = axes[r, c]
            data = da.sel(var=var, season=season)
            im = _pcolor_map(ax, summary_ds.lon, summary_ds.lat, data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{var} | {season}")
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
    _finalize_spatial_figure(fig, axes, im, quantity, f"Seasonal CNN spatial SHAP maps at lag {lag}", fig_dir / f"{prefix}_spatial_{quantity}_predictors_by_season_lag{lag}.png")


def plot_spatial_lag_season_grid(
    summary_ds: xr.Dataset,
    fig_dir: Path,
    prefix: str,
    var_name: str,
    quantity: str = "seasonal_abs_shap",
):
    if var_name not in summary_ds.coords["var"].values.astype(str):
        return
    seasons = ["DJF", "MAM", "JJA", "SON"]
    lags = [int(x) for x in summary_ds.coords["lag"].values]
    nrows, ncols = len(lags), len(seasons)
    fig, axes = _spatial_subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows))
    da = summary_ds[quantity].sel(var=var_name)
    vals = da.sel(season=seasons).values
    if "signed" in quantity:
        lim = float(np.nanquantile(np.abs(vals), 0.99))
        vmin, vmax, cmap = -lim, lim, "RdBu_r"
    else:
        vmin, vmax, cmap = 0.0, float(np.nanquantile(vals, 0.99)), "magma"
    for r, lag in enumerate(lags):
        for c, season in enumerate(seasons):
            ax = axes[r, c]
            data = da.sel(lag=lag, season=season)
            im = _pcolor_map(ax, summary_ds.lon, summary_ds.lat, data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"lag {lag} | {season}")
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
    _finalize_spatial_figure(fig, axes, im, quantity, f"{var_name}: lag × season CNN spatial SHAP", fig_dir / f"{prefix}_spatial_{quantity}_{var_name}_lags_by_season.png")


def plot_event_maps(
    spatial_ds: xr.Dataset,
    fig_dir: Path,
    prefix: str,
    var_name: str,
    lag: int,
):
    pred = spatial_ds["prediction"].values
    obs = spatial_ds["observed"].values if "observed" in spatial_ds else pred
    labels_idx = [
        ("El Niño-like", int(np.nanargmax(obs))),
        ("La Niña-like", int(np.nanargmin(obs))),
        ("Neutral-like", int(np.nanargmin(np.abs(obs)))),
    ]
    fig, axes = _spatial_subplots(1, 3, figsize=(12.5, 3.9))
    vals = []
    for _, idx in labels_idx:
        vals.append(spatial_ds["shap_values"].isel(time=idx).sel(var=var_name, lag=lag).values)
    lim = float(np.nanquantile(np.abs(np.stack(vals)), 0.99))
    for ax, (label, idx), data in zip(axes.ravel(), labels_idx, vals):
        im = _pcolor_map(ax, spatial_ds.lon, spatial_ds.lat, data, cmap="RdBu_r", vmin=-lim, vmax=lim)
        date = pd.to_datetime(spatial_ds.time.values[idx]).strftime("%Y-%m")
        ax.set_title(f"{label}\n{date} pred={pred[idx]:.2f}, obs={obs[idx]:.2f}")
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
    _finalize_spatial_figure(fig, axes, im, "Signed SHAP", f"Event-level signed CNN SHAP: {var_name}, lag {lag}", fig_dir / f"{prefix}_event_maps_{var_name}_lag{lag}.png")


def plot_phase_composite_maps(spatial_ds: xr.Dataset, fig_dir: Path, prefix: str, var_name: str, lag: int):
    obs = spatial_ds["observed"].values if "observed" in spatial_ds else spatial_ds["prediction"].values
    masks = {
        "ElNino": obs >= 0.5,
        "Neutral": np.abs(obs) < 0.5,
        "LaNina": obs <= -0.5,
    }
    fig, axes = _spatial_subplots(1, 3, figsize=(12.5, 3.9))
    comp_data = []
    for key, mask in masks.items():
        if mask.sum() > 0:
            comp_data.append(spatial_ds["shap_values"].sel(var=var_name, lag=lag).isel(time=mask).mean("time").values)
        else:
            comp_data.append(np.full((len(spatial_ds.lat), len(spatial_ds.lon)), np.nan))
    lim = float(np.nanquantile(np.abs(np.stack(comp_data)), 0.99)) if np.isfinite(np.stack(comp_data)).any() else 1.0
    for ax, (key, mask), data in zip(axes.ravel(), masks.items(), comp_data):
        im = _pcolor_map(ax, spatial_ds.lon, spatial_ds.lat, data, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{key} composite (n={int(mask.sum())})")
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
    _finalize_spatial_figure(fig, axes, im, "Mean signed SHAP", f"Phase-composite CNN SHAP: {var_name}, lag {lag}", fig_dir / f"{prefix}_phase_composite_maps_{var_name}_lag{lag}.png")


def make_all_figures(channel_ds: xr.Dataset, spatial_ds: xr.Dataset, summary_ds: xr.Dataset, fig_dir: Path, prefix: str, map_variables: list[str] | None):
    plot_top_channel_bar(channel_ds, fig_dir, prefix, top_n=20)
    plot_beeswarm(channel_ds, fig_dir, prefix, top_n=20)
    plot_violin(channel_ds, fig_dir, prefix, top_n=15)
    plot_monthly_heatmap(channel_ds, fig_dir, prefix, top_n=20)
    plot_hex_general(channel_ds, fig_dir, prefix, top_n=6)

    dominant_var, dominant_lag = choose_dominant_var_lag(summary_ds)
    selected_vars = map_variables or None
    plot_spatial_multi_predictor_seasons(summary_ds, fig_dir, prefix, variables=selected_vars, lag=0, quantity="seasonal_abs_shap")
    plot_spatial_multi_predictor_seasons(summary_ds, fig_dir, prefix, variables=selected_vars, lag=0, quantity="seasonal_signed_shap")
    plot_spatial_lag_season_grid(summary_ds, fig_dir, prefix, dominant_var, quantity="seasonal_abs_shap")
    plot_spatial_lag_season_grid(summary_ds, fig_dir, prefix, dominant_var, quantity="seasonal_signed_shap")
    plot_event_maps(spatial_ds, fig_dir, prefix, dominant_var, dominant_lag)
    plot_phase_composite_maps(spatial_ds, fig_dir, prefix, dominant_var, dominant_lag)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run_cnn_tuned_shap(
    cfg_path: str,
    lead: int,
    task: str,
    device: str = "cpu",
    tuned_root: str | Path | None = None,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    global_shap_dir: str | Path | None = None,
    fig_dir: str | Path | None = None,
    model_module: str = "cnn_model_tuned",
    max_eval_samples: int = 0,
    background_samples: int = 100,
    deep_batch_size: int = 6,
    map_variables: list[str] | None = None,
) -> None:
    cfg = load_config(cfg_path)
    root = Path(tuned_root) if tuned_root is not None else default_tuned_root(cfg)
    rd = run_dir(root, lead, task)
    model_dir = Path(model_dir) if model_dir is not None else rd / "models"
    output_dir = ensure_dir(Path(output_dir) if output_dir is not None else rd / "shap")
    global_shap_dir = ensure_dir(Path(global_shap_dir) if global_shap_dir is not None else default_global_shap_root(cfg))
    fig_dir = ensure_dir(Path(fig_dir) if fig_dir is not None else default_fig_root(cfg) / f"lead{lead:02d}_{task}")

    log.info("Loading tuned CNN model from %s", model_dir)
    model = load_tuned_cnn_model(cfg, lead, task, model_dir, model_module, device=device)
    device = model.device
    log.info("Loaded tuned CNN device=%s meta=%s", device, getattr(model, "_meta", {}))

    (X_tr, y_tr_fit, t_tr,
     X_val, y_val_fit, t_val,
     X_te, y_te_eval, t_te,
     meta) = load_cnn_arrays(cfg, lead, task)

    log.info("Rebuilt CNN arrays: train=%d val=%d test=%d channels=%d grid=%s",
             len(X_tr), len(X_val), len(X_te), X_te.shape[1], X_te.shape[2:])

    X_eval, y_eval, times_eval, eval_idx = subset_eval(
        X_te, y_te_eval, t_te, max_eval_samples=max_eval_samples, seed=int(cfg["experiment"]["seed"])
    )
    y_observed_raw = np.asarray(meta["raw_target_test"])[eval_idx]

    n_bg = min(int(background_samples), len(X_tr))
    X_bg_np = select_background(X_tr, n_bg, seed=int(cfg["experiment"]["seed"]))
    X_bg_t = torch.tensor(X_bg_np, dtype=torch.float32).to(device)

    net_for_shap = _Unsqueeze(model.net) if task == "regression" else model.net
    explainer = get_deep_explainer(net_for_shap, X_bg_t)

    log.info("Computing CNN DeepSHAP: eval=%d background=%d batch=%d", len(X_eval), n_bg, deep_batch_size)
    shap_raw, base_value = compute_deep_shap(explainer, X_eval, int(deep_batch_size), device)
    shap_spatial = normalise_shap_array(shap_raw)
    log.info("Spatial SHAP shape=%s", shap_spatial.shape)

    preds = predict_eval(model, X_eval, task=task)

    prefix = f"cnn_lead{lead:02d}_{task}"

    channel_ds = make_channel_dataset(
        shap_spatial, preds, y_observed_raw, meta["ch_names"], times_eval, lead, task, base_value
    )
    spatial_ds = make_spatial_sample_dataset(
        shap_spatial, preds, y_observed_raw, times_eval, meta, lead, task
    )
    summary_ds = make_spatial_summary_dataset(spatial_ds)

    channel_path = output_dir / f"{prefix}_shap.zarr"
    spatial_path = output_dir / f"{prefix}_spatial_samples.zarr"
    summary_path = output_dir / f"{prefix}_spatial_summary.zarr"
    write_zarr_safe(channel_ds, channel_path)
    write_zarr_safe(spatial_ds, spatial_path)
    write_zarr_safe(summary_ds, summary_path)
    log.info("Saved tuned CNN SHAP stores -> %s", output_dir)

    # Copy/overwrite to global easy-to-compile location.
    for src in [channel_path, spatial_path, summary_path]:
        dst = global_shap_dir / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    log.info("Copied SHAP stores -> %s", global_shap_dir)

    # CSV summaries.
    imp = channel_ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(output_dir / f"{prefix}_channel_importance.csv")
    monthly = channel_ds["abs_shap"].to_pandas()
    monthly["month"] = pd.DatetimeIndex(monthly.index).month
    monthly.groupby("month").mean().to_csv(output_dir / f"{prefix}_monthly_channel_importance.csv")

    make_all_figures(channel_ds, spatial_ds, summary_ds, fig_dir, prefix, map_variables=map_variables)
    log.info("Saved tuned CNN SHAP figures -> %s", fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SHAP for tuned CNN ENSO model and make spatial figures")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lead", type=int, required=True, choices=[3, 6, 12])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--tuned-root", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--global-shap-dir", default=None)
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--model-module", default="cnn_model_tuned")
    parser.add_argument("--max-eval-samples", type=int, default=0,
                        help="0 means use the full test period.")
    parser.add_argument("--background-samples", type=int, default=100)
    parser.add_argument("--deep-batch-size", type=int, default=6)
    parser.add_argument("--map-variables", nargs="*", default=None,
                        help="Optional variables to include in multi-predictor seasonal maps, e.g. sst d20 tauu.")
    args = parser.parse_args()

    run_cnn_tuned_shap(
        cfg_path=args.config,
        lead=args.lead,
        task=args.task,
        device=args.device,
        tuned_root=args.tuned_root,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        global_shap_dir=args.global_shap_dir,
        fig_dir=args.fig_dir,
        model_module=args.model_module,
        max_eval_samples=args.max_eval_samples,
        background_samples=args.background_samples,
        deep_batch_size=args.deep_batch_size,
        map_variables=args.map_variables,
    )
