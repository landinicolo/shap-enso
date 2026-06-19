"""Compile tuned CNN metrics and create an extended CNN XAI figure gallery.

This script assumes tuned CNN runs live under:

    data/tuned_wide/cnn/lead03_regression/
    data/tuned_wide/cnn/lead06_regression/
    data/tuned_wide/cnn/lead12_regression/

and that compute_shap_tuned_cnn.py has already produced SHAP stores in each
lead folder. It compiles skill metrics, persistence baselines, monthly/seasonal
metrics, SHAP cross-lead summaries, and multi-panel spatial SHAP figures.

Examples
--------
python scripts/compile_metrics_tuned_cnn.py \
    --config configs/default.yaml \
    --task regression \
    --tuned-root data/tuned_wide/cnn \
    --fig-dir figures/tuned_wide/cnn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

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

from src.utils.config import load_config
from src.utils.io_utils import load_zarr
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


LEADS_DEFAULT = [3, 6, 12]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_tuned_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "cnn"


def default_output_dir(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "cnn" / "compiled_analysis"


def default_fig_dir(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "figures" / "tuned_wide" / "cnn"


def run_dir(root: Path, lead: int, task: str) -> Path:
    return root / f"lead{lead:02d}_{task}"


def read_predictions(root: Path, lead: int, task: str) -> pd.DataFrame | None:
    pred_dir = run_dir(root, lead, task) / "predictions"
    candidates = [
        pred_dir / f"cnn_lead{lead:02d}_{task}_test_predictions.csv",
        *pred_dir.glob(f"*lead{lead:02d}*{task}*prediction*.csv"),
        *pred_dir.glob("*prediction*.csv"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            for col in ["init_time", "target_time"]:
                if col in df:
                    df[col] = pd.to_datetime(df[col])
            return df
    log.warning("No prediction CSV found for CNN lead=%02d task=%s in %s", lead, task, pred_dir)
    return None


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 2:
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan, "r2": np.nan, "bias": np.nan}
    y = y[ok]
    pred = pred[ok]
    err = pred - y
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(y) > 0 and np.std(pred) > 0 else np.nan
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else np.nan
    bias = float(np.mean(err))
    return {"rmse": rmse, "mae": mae, "corr": corr, "r2": r2, "bias": bias}


def target_series(cfg: dict) -> pd.Series | None:
    try:
        proc = Path(cfg["data"]["processed_dir"])
        ds = load_zarr(proc / "target_nino34.zarr").compute()
        da = ds["nino34"] if "nino34" in ds else list(ds.data_vars.values())[0]
        s = da.to_series()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as exc:
        log.warning("Could not load target_nino34.zarr for persistence baseline: %s", exc)
        return None


def add_persistence(df: pd.DataFrame, target: pd.Series | None) -> pd.DataFrame:
    df = df.copy()
    if target is None or "init_time" not in df:
        df["persistence"] = np.nan
        return df
    # Reindex exactly, then try nearest within ~20 days if needed.
    init = pd.DatetimeIndex(df["init_time"])
    pers = target.reindex(init)
    if pers.isna().any():
        pers2 = target.reindex(init, method="nearest", tolerance=pd.Timedelta(days=20))
        pers = pers.fillna(pers2)
    df["persistence"] = pers.values
    return df


def season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def compile_prediction_metrics(cfg: dict, root: Path, leads: list[int], task: str):
    targ = target_series(cfg)
    rows = []
    monthly_rows = []
    seasonal_rows = []
    all_preds = []

    for lead in leads:
        df = read_predictions(root, lead, task)
        if df is None:
            continue
        df = add_persistence(df, targ)
        df["month"] = pd.DatetimeIndex(df["init_time"]).month
        df["season"] = [season_name(int(m)) for m in df["month"]]
        all_preds.append(df)

        if task == "regression" and {"observed", "prediction"} <= set(df.columns):
            m = regression_metrics(df["observed"].values, df["prediction"].values)
            p = regression_metrics(df["observed"].values, df["persistence"].values) if "persistence" in df else {}
            row = {
                "model_type": "cnn_tuned",
                "lead_months": lead,
                "task": task,
                "n_samples": int(len(df)),
                **m,
                "persistence_rmse": p.get("rmse", np.nan),
                "persistence_corr": p.get("corr", np.nan),
                "rmse_skill_vs_persistence": (
                    1.0 - m["rmse"] / p["rmse"] if p.get("rmse", np.nan) and np.isfinite(p.get("rmse", np.nan)) else np.nan
                ),
            }
            rows.append(row)

            for mon, g in df.groupby("month"):
                mm = regression_metrics(g["observed"].values, g["prediction"].values)
                pp = regression_metrics(g["observed"].values, g["persistence"].values)
                monthly_rows.append({
                    "lead_months": lead,
                    "month": int(mon),
                    **mm,
                    "persistence_rmse": pp.get("rmse", np.nan),
                    "persistence_corr": pp.get("corr", np.nan),
                    "rmse_skill_vs_persistence": 1.0 - mm["rmse"] / pp["rmse"] if pp.get("rmse", np.nan) and np.isfinite(pp.get("rmse", np.nan)) else np.nan,
                })

            for seas, g in df.groupby("season"):
                sm = regression_metrics(g["observed"].values, g["prediction"].values)
                pp = regression_metrics(g["observed"].values, g["persistence"].values)
                seasonal_rows.append({
                    "lead_months": lead,
                    "season": seas,
                    **sm,
                    "persistence_rmse": pp.get("rmse", np.nan),
                    "persistence_corr": pp.get("corr", np.nan),
                    "rmse_skill_vs_persistence": 1.0 - sm["rmse"] / pp["rmse"] if pp.get("rmse", np.nan) and np.isfinite(pp.get("rmse", np.nan)) else np.nan,
                })
        else:
            log.warning("Only regression metric compilation is fully implemented in this script.")

    return (
        pd.DataFrame(rows),
        pd.DataFrame(monthly_rows),
        pd.DataFrame(seasonal_rows),
        pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame(),
    )


def save_fig(fig, path: Path, dpi: int = 180, tight: bool = True):
    ensure_dir(path.parent)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _open_local_zarr(path: Path) -> xr.Dataset:
    """Open workflow-generated Zarr stores without probing for consolidated metadata."""
    return xr.open_zarr(str(path), consolidated=False).load()


def _spatial_subplots(nrows: int, ncols: int, figsize: tuple[float, float]):
    if HAS_CARTOPY:
        central_longitude = 180
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, squeeze=False,
            subplot_kw={"projection": ccrs.PlateCarree(central_longitude=central_longitude)}
        )
    else:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    return fig, axes


def _draw_coastlines(ax):
    if HAS_CARTOPY:
        try:
            ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="white", linewidth=1.2, zorder=4)
            ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor="black", linewidth=0.45, zorder=5)
        except Exception:
            ax.coastlines(color="white", linewidth=1.2, zorder=4)
            ax.coastlines(color="black", linewidth=0.45, zorder=5)


def _pcolor_map(ax, lon, lat, data, cmap, vmin, vmax):
    if HAS_CARTOPY:
        im = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax.gridlines(draw_labels=False, linewidth=0.25, color="gray", alpha=0.35, linestyle="--")
    else:
        im = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    _draw_coastlines(ax)
    return im


def _finalize_spatial_figure(fig, axes, im, cbar_label: str, suptitle: str, out_path: Path):
    fig.subplots_adjust(left=0.055, right=0.90, bottom=0.08, top=0.92, wspace=0.16, hspace=0.28)
    cax = fig.add_axes([0.92, 0.16, 0.018, 0.68])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)
    fig.suptitle(suptitle, y=0.97)
    save_fig(fig, out_path, tight=False)


def plot_skill_curves(metrics: pd.DataFrame, fig_dir: Path, prefix: str):
    if metrics.empty:
        return
    df = metrics.sort_values("lead_months")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["lead_months"], df["corr"], marker="o", label="CNN tuned")
    if "persistence_corr" in df:
        axes[0].plot(df["lead_months"], df["persistence_corr"], marker="o", ls="--", label="Persistence")
    axes[0].set_xlabel("Lead time (months)")
    axes[0].set_ylabel("ACC / correlation")
    axes[0].set_title("ACC vs lead")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["lead_months"], df["rmse"], marker="o", label="CNN tuned")
    if "persistence_rmse" in df:
        axes[1].plot(df["lead_months"], df["persistence_rmse"], marker="o", ls="--", label="Persistence")
    axes[1].set_xlabel("Lead time (months)")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE vs lead")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Tuned CNN skill curves")
    save_fig(fig, fig_dir / f"{prefix}_skill_curves.png")


def plot_prediction_time_series(preds: pd.DataFrame, fig_dir: Path, prefix: str):
    if preds.empty or "observed" not in preds:
        return
    leads = sorted(preds["lead_months"].unique())
    fig, axes = plt.subplots(len(leads), 1, figsize=(11, 3.1 * len(leads)), sharex=False, squeeze=False)
    for ax, lead in zip(axes.ravel(), leads):
        g = preds[preds["lead_months"] == lead].sort_values("target_time")
        ax.plot(g["target_time"], g["observed"], label="Observed", lw=1.5)
        ax.plot(g["target_time"], g["prediction"], label="CNN tuned", lw=1.2)
        if "persistence" in g:
            ax.plot(g["target_time"], g["persistence"], label="Persistence", lw=1.0, ls="--")
        ax.axhline(0, color="k", lw=0.7)
        ax.set_title(f"Lead {lead} months")
        ax.set_ylabel("Niño3.4 anomaly")
        ax.legend(loc="best", ncol=3)
        ax.grid(alpha=0.2)
    fig.suptitle("Observed vs predicted CNN forecasts", y=1.01)
    save_fig(fig, fig_dir / f"{prefix}_prediction_timeseries.png")


def plot_obs_pred_scatter(preds: pd.DataFrame, fig_dir: Path, prefix: str):
    if preds.empty or "observed" not in preds:
        return
    leads = sorted(preds["lead_months"].unique())
    fig, axes = plt.subplots(1, len(leads), figsize=(4.2 * len(leads), 4), squeeze=False)
    for ax, lead in zip(axes.ravel(), leads):
        g = preds[preds["lead_months"] == lead]
        ax.scatter(g["observed"], g["prediction"], s=24, alpha=0.65)
        lo = float(np.nanmin([g["observed"].min(), g["prediction"].min()]))
        hi = float(np.nanmax([g["observed"].max(), g["prediction"].max()]))
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_title(f"Lead {lead}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
    fig.suptitle("Observed-predicted scatter")
    save_fig(fig, fig_dir / f"{prefix}_obs_pred_scatter.png")


def plot_error_distributions(preds: pd.DataFrame, fig_dir: Path, prefix: str):
    if preds.empty or "error" not in preds:
        return
    leads = sorted(preds["lead_months"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [preds.loc[preds["lead_months"] == lead, "error"].dropna().values for lead in leads]
    parts = ax.violinplot(data, positions=np.arange(len(leads)), showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.65)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xticks(np.arange(len(leads)))
    ax.set_xticklabels([str(l) for l in leads])
    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel("Prediction error")
    ax.set_title("CNN error distributions by lead")
    save_fig(fig, fig_dir / f"{prefix}_error_distributions.png")


def plot_monthly_metric_heatmap(monthly: pd.DataFrame, fig_dir: Path, prefix: str, metric: str):
    if monthly.empty or metric not in monthly:
        return
    mat = monthly.pivot(index="lead_months", columns="month", values=metric).reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    im = ax.imshow(mat.values, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.astype(str))
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45, ha="right")
    ax.set_xlabel("Initialisation month")
    ax.set_ylabel("Lead")
    ax.set_title(f"Monthly {metric} heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    save_fig(fig, fig_dir / f"{prefix}_monthly_{metric}_heatmap.png")


def load_channel_shap(root: Path, lead: int, task: str) -> xr.Dataset | None:
    p = run_dir(root, lead, task) / "shap" / f"cnn_lead{lead:02d}_{task}_shap.zarr"
    if not p.exists():
        log.warning("Missing channel SHAP store: %s", p)
        return None
    return _open_local_zarr(p)


def load_spatial_summary(root: Path, lead: int, task: str) -> xr.Dataset | None:
    p = run_dir(root, lead, task) / "shap" / f"cnn_lead{lead:02d}_{task}_spatial_summary.zarr"
    if not p.exists():
        log.warning("Missing spatial summary store: %s", p)
        return None
    return _open_local_zarr(p)


def load_spatial_samples(root: Path, lead: int, task: str) -> xr.Dataset | None:
    p = run_dir(root, lead, task) / "shap" / f"cnn_lead{lead:02d}_{task}_spatial_samples.zarr"
    if not p.exists():
        log.warning("Missing spatial sample store: %s", p)
        return None
    return _open_local_zarr(p)


def plot_crosslead_shap_heatmap(root: Path, leads: list[int], task: str, fig_dir: Path, prefix: str, top_n: int = 30):
    series = {}
    for lead in leads:
        ds = load_channel_shap(root, lead, task)
        if ds is not None:
            series[lead] = ds["abs_shap"].mean("time").to_pandas()
    if not series:
        return
    df = pd.DataFrame(series).fillna(0.0)
    score = df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    mat = df.loc[score].T
    fig, ax = plt.subplots(figsize=(max(9, 0.35 * len(score)), 3.7))
    im = ax.imshow(mat.values, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index.astype(str))
    ax.set_xticks(np.arange(len(score)))
    ax.set_xticklabels(score, rotation=75, ha="right")
    ax.set_xlabel("CNN channel")
    ax.set_ylabel("Lead")
    ax.set_title("Cross-lead CNN channel importance")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |SHAP|")
    save_fig(fig, fig_dir / f"{prefix}_crosslead_channel_shap_heatmap.png")


def plot_crosslead_spatial_dominant(root: Path, leads: list[int], task: str, fig_dir: Path, prefix: str, var_name: str | None = None, lag: int = 0):
    summaries = {lead: load_spatial_summary(root, lead, task) for lead in leads}
    summaries = {k: v for k, v in summaries.items() if v is not None}
    if not summaries:
        return
    if var_name is None:
        # Pick variable with max average importance over all available leads.
        scores = []
        for lead, ds in summaries.items():
            s = ds["mean_abs_shap"].sel(lag=lag).mean(("lat", "lon")).to_pandas()
            scores.append(s)
        total = pd.concat(scores, axis=1).mean(axis=1).sort_values(ascending=False)
        var_name = str(total.index[0])
    seasons = ["DJF", "MAM", "JJA", "SON"]
    nrows, ncols = len(summaries), len(seasons)
    fig, axes = _spatial_subplots(nrows, ncols, figsize=(4.0 * ncols, 2.9 * nrows))
    # shared scale
    vals = []
    for ds in summaries.values():
        if var_name in ds.coords["var"].values.astype(str):
            vals.append(ds["seasonal_abs_shap"].sel(var=var_name, lag=lag, season=seasons).values)
    vmax = float(np.nanquantile(np.concatenate([v.ravel() for v in vals]), 0.99)) if vals else 1.0
    for r, (lead, ds) in enumerate(sorted(summaries.items())):
        for c, season in enumerate(seasons):
            ax = axes[r, c]
            if var_name not in ds.coords["var"].values.astype(str):
                ax.text(0.5, 0.5, f"{var_name} missing", ha="center", va="center")
                ax.axis("off")
                continue
            data = ds["seasonal_abs_shap"].sel(var=var_name, lag=lag, season=season)
            im = _pcolor_map(ax, ds.lon, ds.lat, data, cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(f"Lead {lead} | {season}")
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
    _finalize_spatial_figure(fig, axes, im, "Seasonal mean |SHAP|", f"Cross-lead seasonal spatial SHAP: {var_name}, lag {lag}", fig_dir / f"{prefix}_crosslead_seasonal_spatial_{var_name}_lag{lag}.png")


def plot_spatial_predictor_gallery(root: Path, leads: list[int], task: str, fig_dir: Path, prefix: str, variables: list[str] | None = None, lag: int = 0):
    # One figure per lead: variables rows, seasons cols.
    seasons = ["DJF", "MAM", "JJA", "SON"]
    for lead in leads:
        ds = load_spatial_summary(root, lead, task)
        if ds is None:
            continue
        available = [str(v) for v in ds.coords["var"].values]
        if variables:
            vars_use = [v for v in variables if v in available]
        else:
            score = ds["mean_abs_shap"].sel(lag=lag).mean(("lat", "lon")).to_pandas().sort_values(ascending=False)
            vars_use = list(score.index[: min(5, len(score))])
        if not vars_use:
            continue
        fig, axes = _spatial_subplots(len(vars_use), len(seasons), figsize=(4.0 * len(seasons), 2.8 * len(vars_use)))
        vals = ds["seasonal_abs_shap"].sel(var=vars_use, lag=lag, season=seasons).values
        vmax = float(np.nanquantile(vals, 0.99))
        for r, var in enumerate(vars_use):
            for c, season in enumerate(seasons):
                ax = axes[r, c]
                data = ds["seasonal_abs_shap"].sel(var=var, lag=lag, season=season)
                im = _pcolor_map(ax, ds.lon, ds.lat, data, cmap="magma", vmin=0, vmax=vmax)
                ax.set_title(f"{var} | {season}")
                ax.set_xlabel("Lon")
                ax.set_ylabel("Lat")
        _finalize_spatial_figure(fig, axes, im, "Mean |SHAP|", f"Lead {lead}: predictor × season CNN spatial SHAP", fig_dir / f"{prefix}_lead{lead:02d}_predictor_season_spatial_gallery_lag{lag}.png")




def plot_channel_lead_difference_heatmap(root: Path, leads: list[int], task: str, fig_dir: Path, prefix: str, top_n: int = 20):
    series = {}
    for lead in leads:
        ds = load_channel_shap(root, lead, task)
        if ds is not None:
            series[lead] = ds["abs_shap"].mean("time").to_pandas()
    if len(series) < 2:
        return
    df = pd.DataFrame(series).fillna(0.0).sort_index(axis=1)
    lead_pairs = [(leads[i], leads[j]) for i in range(len(leads)) for j in range(i + 1, len(leads))]
    mean_score = df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    diff_df = pd.DataFrame({f"{b}-{a}": df[b] - df[a] for a, b in lead_pairs})
    mat = diff_df.loc[mean_score]
    lim = float(np.nanquantile(np.abs(mat.values), 0.98)) if np.isfinite(mat.values).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(lead_pairs)), max(5, 0.28 * len(mean_score))))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_yticks(np.arange(len(mean_score)))
    ax.set_yticklabels(mean_score)
    ax.set_xticks(np.arange(len(lead_pairs)))
    ax.set_xticklabels([f"{b}-{a}" for a, b in lead_pairs])
    ax.set_xlabel("Lead difference")
    ax.set_title("Difference in mean |SHAP| across leads (top channels)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Δ mean |SHAP|")
    save_fig(fig, fig_dir / f"{prefix}_lead_difference_channel_heatmap.png")
    mat.reset_index().rename(columns={"index": "feature"}).to_csv(fig_dir.parent / f"{prefix}_lead_difference_channel_heatmap.csv", index=False)


def plot_spatial_lead_difference_gallery(root: Path, leads: list[int], task: str, fig_dir: Path, prefix: str, variables: list[str] | None = None, lag: int = 0):
    summaries = {lead: load_spatial_summary(root, lead, task) for lead in leads}
    summaries = {lead: ds for lead, ds in summaries.items() if ds is not None}
    if len(summaries) < 2:
        return
    leads_sorted = sorted(summaries)
    if variables:
        vars_use = list(variables)
    else:
        scores = []
        for ds in summaries.values():
            s = ds["mean_abs_shap"].sel(lag=lag).mean(("lat", "lon")).to_pandas()
            scores.append(s)
        vars_use = list(pd.concat(scores, axis=1).mean(axis=1).sort_values(ascending=False).head(3).index)
    lead_pairs = [(leads_sorted[i], leads_sorted[j]) for i in range(len(leads_sorted)) for j in range(i + 1, len(leads_sorted))]
    fig, axes = _spatial_subplots(len(vars_use), len(lead_pairs), figsize=(4.2 * len(lead_pairs), 2.9 * len(vars_use)))
    # annual mean abs SHAP difference, later minus earlier
    vals = []
    for var in vars_use:
        for a, b in lead_pairs:
            da = summaries[b]["mean_abs_shap"].sel(var=var, lag=lag) - summaries[a]["mean_abs_shap"].sel(var=var, lag=lag)
            vals.append(da.values)
    lim = float(np.nanquantile(np.abs(np.stack(vals)), 0.99)) if vals else 1.0
    for r, var in enumerate(vars_use):
        for c, (a, b) in enumerate(lead_pairs):
            ax = axes[r, c]
            data = summaries[b]["mean_abs_shap"].sel(var=var, lag=lag) - summaries[a]["mean_abs_shap"].sel(var=var, lag=lag)
            im = _pcolor_map(ax, summaries[a].lon, summaries[a].lat, data, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"{var} | lead {b} - lead {a}")
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
    _finalize_spatial_figure(fig, axes, im, "Δ mean |SHAP|", f"Spatial SHAP lead differences at lag {lag}", fig_dir / f"{prefix}_spatial_lead_difference_gallery_lag{lag}.png")


def plot_spatial_lead_difference_by_season(root: Path, leads: list[int], task: str, fig_dir: Path, prefix: str, var_name: str | None = None, lag: int = 0):
    summaries = {lead: load_spatial_summary(root, lead, task) for lead in leads}
    summaries = {lead: ds for lead, ds in summaries.items() if ds is not None}
    if len(summaries) < 2:
        return
    leads_sorted = sorted(summaries)
    if var_name is None:
        scores = []
        for ds in summaries.values():
            s = ds["mean_abs_shap"].sel(lag=lag).mean(("lat", "lon")).to_pandas()
            scores.append(s)
        var_name = str(pd.concat(scores, axis=1).mean(axis=1).sort_values(ascending=False).index[0])
    lead_pairs = [(leads_sorted[i], leads_sorted[j]) for i in range(len(leads_sorted)) for j in range(i + 1, len(leads_sorted))]
    seasons = ["DJF", "MAM", "JJA", "SON"]
    fig, axes = _spatial_subplots(len(lead_pairs), len(seasons), figsize=(4.0 * len(seasons), 2.9 * len(lead_pairs)))
    vals = []
    for a, b in lead_pairs:
        vals.append((summaries[b]["seasonal_abs_shap"].sel(var=var_name, lag=lag, season=seasons) -
                     summaries[a]["seasonal_abs_shap"].sel(var=var_name, lag=lag, season=seasons)).values)
    lim = float(np.nanquantile(np.abs(np.stack(vals)), 0.99)) if vals else 1.0
    for r, (a, b) in enumerate(lead_pairs):
        for c, season in enumerate(seasons):
            ax = axes[r, c]
            data = summaries[b]["seasonal_abs_shap"].sel(var=var_name, lag=lag, season=season) - summaries[a]["seasonal_abs_shap"].sel(var=var_name, lag=lag, season=season)
            im = _pcolor_map(ax, summaries[a].lon, summaries[a].lat, data, cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_title(f"lead {b} - lead {a} | {season}")
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
    _finalize_spatial_figure(fig, axes, im, "Δ mean |SHAP|", f"{var_name}: seasonal spatial SHAP lead differences at lag {lag}", fig_dir / f"{prefix}_spatial_lead_difference_by_season_{var_name}_lag{lag}.png")


def compile_cnn(
    cfg_path: str,
    task: str = "regression",
    tuned_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    fig_dir: str | Path | None = None,
    leads: Iterable[int] = LEADS_DEFAULT,
    map_variables: list[str] | None = None,
    map_lag: int = 0,
):
    cfg = load_config(cfg_path)
    root = Path(tuned_root) if tuned_root is not None else default_tuned_root(cfg)
    output_dir = ensure_dir(Path(output_dir) if output_dir is not None else default_output_dir(cfg))
    fig_dir = ensure_dir(Path(fig_dir) if fig_dir is not None else default_fig_dir(cfg))
    leads = [int(x) for x in leads]
    prefix = f"cnn_tuned_{task}"

    metrics, monthly, seasonal, preds = compile_prediction_metrics(cfg, root, leads, task)

    metrics.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    monthly.to_csv(output_dir / f"{prefix}_monthly_metrics.csv", index=False)
    seasonal.to_csv(output_dir / f"{prefix}_seasonal_metrics.csv", index=False)
    preds.to_csv(output_dir / f"{prefix}_all_predictions.csv", index=False)

    plot_skill_curves(metrics, fig_dir, prefix)
    plot_prediction_time_series(preds, fig_dir, prefix)
    plot_obs_pred_scatter(preds, fig_dir, prefix)
    plot_error_distributions(preds, fig_dir, prefix)
    for metric in ["rmse", "corr", "bias", "rmse_skill_vs_persistence"]:
        plot_monthly_metric_heatmap(monthly, fig_dir, prefix, metric)

    plot_crosslead_shap_heatmap(root, leads, task, fig_dir, prefix, top_n=30)
    plot_crosslead_spatial_dominant(root, leads, task, fig_dir, prefix, var_name=None, lag=map_lag)
    plot_spatial_predictor_gallery(root, leads, task, fig_dir, prefix, variables=map_variables, lag=map_lag)
    plot_channel_lead_difference_heatmap(root, leads, task, fig_dir, prefix, top_n=20)
    plot_spatial_lead_difference_gallery(root, leads, task, fig_dir, prefix, variables=map_variables, lag=map_lag)
    plot_spatial_lead_difference_by_season(root, leads, task, fig_dir, prefix, var_name=(map_variables[0] if map_variables else None), lag=map_lag)

    # Gather SHAP importance into CSV.
    rows = []
    for lead in leads:
        ds = load_channel_shap(root, lead, task)
        if ds is None:
            continue
        imp = ds["abs_shap"].mean("time").to_pandas()
        for feat, val in imp.items():
            rows.append({"lead_months": lead, "feature": feat, "mean_abs_shap": float(val)})
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}_channel_shap_importance.csv", index=False)

    log.info("Saved tuned CNN compiled analysis -> %s", output_dir)
    log.info("Saved tuned CNN figures -> %s", fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile tuned CNN metrics and XAI figures")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--tuned-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--leads", nargs="+", type=int, default=LEADS_DEFAULT)
    parser.add_argument("--map-variables", nargs="*", default=None,
                        help="Variables for spatial predictor galleries, e.g. sst d20 tauu olr slp")
    parser.add_argument("--map-lag", type=int, default=0)
    args = parser.parse_args()

    compile_cnn(
        cfg_path=args.config,
        task=args.task,
        tuned_root=args.tuned_root,
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
        leads=args.leads,
        map_variables=args.map_variables,
        map_lag=args.map_lag,
    )
