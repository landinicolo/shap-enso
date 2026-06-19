"""Compile tuned XGB metrics and create an extended SHAP/skill figure gallery."""

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

from src.utils.config import load_config
from src.utils.io_utils import load_zarr
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
LEADS_DEFAULT = [3, 6, 12]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_tuned_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "xgb"


def default_output_dir(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "xgb" / "compiled_analysis"


def default_fig_dir(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "figures" / "tuned_wide" / "xgb"


def run_dir(root: Path, lead: int, task: str) -> Path:
    return root / f"lead{lead:02d}_{task}"


def _open_local_zarr(path: Path) -> xr.Dataset:
    return xr.open_zarr(str(path), consolidated=False).load()


def read_predictions(root: Path, lead: int, task: str) -> pd.DataFrame | None:
    pred_dir = run_dir(root, lead, task) / "predictions"
    candidates = [
        pred_dir / f"xgb_lead{lead:02d}_{task}_test_predictions.csv",
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
    log.warning("No prediction CSV found for XGB lead=%02d task=%s in %s", lead, task, pred_dir)
    return None


def regression_metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 2:
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan, "r2": np.nan, "bias": np.nan}
    y = y[ok]; pred = pred[ok]
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
        log.warning("Could not load target for persistence baseline: %s", exc)
        return None


def add_persistence(df: pd.DataFrame, target: pd.Series | None) -> pd.DataFrame:
    df = df.copy()
    if target is None or "init_time" not in df:
        df["persistence"] = np.nan
        return df
    init = pd.DatetimeIndex(df["init_time"])
    pers = target.reindex(init)
    if pers.isna().any():
        pers2 = target.reindex(init, method="nearest", tolerance=pd.Timedelta(days=20))
        pers = pers.fillna(pers2)
    df["persistence"] = pers.values
    return df


def season_name(month: int) -> str:
    if month in (12, 1, 2): return "DJF"
    if month in (3, 4, 5): return "MAM"
    if month in (6, 7, 8): return "JJA"
    return "SON"


def compile_prediction_metrics(cfg, root: Path, leads: list[int], task: str):
    target = target_series(cfg)
    rows, monthly_rows, seasonal_rows, all_preds = [], [], [], []
    for lead in leads:
        df = read_predictions(root, lead, task)
        if df is None:
            continue
        df = add_persistence(df, target)
        df["month"] = pd.DatetimeIndex(df["init_time"]).month
        df["season"] = [season_name(int(m)) for m in df["month"]]
        all_preds.append(df)
        if task == "regression" and {"observed", "prediction"} <= set(df.columns):
            m = regression_metrics(df["observed"], df["prediction"])
            p = regression_metrics(df["observed"], df["persistence"])
            rows.append({
                "model_type": "xgb_tuned", "lead_months": lead, "task": task, "n_samples": len(df),
                **m,
                "persistence_rmse": p.get("rmse", np.nan),
                "persistence_corr": p.get("corr", np.nan),
                "rmse_skill_vs_persistence": 1.0 - m["rmse"] / p["rmse"] if p.get("rmse", np.nan) and np.isfinite(p.get("rmse", np.nan)) else np.nan,
            })
            for mon, g in df.groupby("month"):
                mm = regression_metrics(g["observed"], g["prediction"])
                pp = regression_metrics(g["observed"], g["persistence"])
                monthly_rows.append({"lead_months": lead, "month": int(mon), **mm,
                    "persistence_rmse": pp.get("rmse", np.nan),
                    "persistence_corr": pp.get("corr", np.nan),
                    "rmse_skill_vs_persistence": 1.0 - mm["rmse"] / pp["rmse"] if pp.get("rmse", np.nan) and np.isfinite(pp.get("rmse", np.nan)) else np.nan})
            for seas, g in df.groupby("season"):
                sm = regression_metrics(g["observed"], g["prediction"])
                pp = regression_metrics(g["observed"], g["persistence"])
                seasonal_rows.append({"lead_months": lead, "season": seas, **sm,
                    "persistence_rmse": pp.get("rmse", np.nan),
                    "persistence_corr": pp.get("corr", np.nan),
                    "rmse_skill_vs_persistence": 1.0 - sm["rmse"] / pp["rmse"] if pp.get("rmse", np.nan) and np.isfinite(pp.get("rmse", np.nan)) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(monthly_rows), pd.DataFrame(seasonal_rows), pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()


def save_fig(fig, path: Path, dpi: int = 180):
    ensure_dir(path.parent)
    try: fig.tight_layout()
    except Exception: pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_skill_curves(metrics: pd.DataFrame, fig_dir: Path, prefix: str):
    if metrics.empty: return
    df = metrics.sort_values("lead_months")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["lead_months"], df["corr"], marker="o", label="XGB tuned")
    if "persistence_corr" in df: axes[0].plot(df["lead_months"], df["persistence_corr"], marker="o", ls="--", label="Persistence")
    axes[0].set_xlabel("Lead time (months)"); axes[0].set_ylabel("ACC / correlation"); axes[0].set_title("ACC vs lead"); axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(df["lead_months"], df["rmse"], marker="o", label="XGB tuned")
    if "persistence_rmse" in df: axes[1].plot(df["lead_months"], df["persistence_rmse"], marker="o", ls="--", label="Persistence")
    axes[1].set_xlabel("Lead time (months)"); axes[1].set_ylabel("RMSE"); axes[1].set_title("RMSE vs lead"); axes[1].legend(); axes[1].grid(alpha=0.25)
    fig.suptitle("Tuned XGB skill curves")
    save_fig(fig, fig_dir / f"{prefix}_skill_curves.png")


def plot_prediction_time_series(preds: pd.DataFrame, fig_dir: Path, prefix: str):
    if preds.empty or "observed" not in preds: return
    leads = sorted(preds["lead_months"].unique())
    fig, axes = plt.subplots(len(leads), 1, figsize=(11, 3.1 * len(leads)), squeeze=False)
    for ax, lead in zip(axes.ravel(), leads):
        g = preds[preds["lead_months"] == lead].sort_values("target_time")
        ax.plot(g["target_time"], g["observed"], label="Observed", lw=1.5)
        ax.plot(g["target_time"], g["prediction"], label="XGB tuned", lw=1.2)
        if "persistence" in g: ax.plot(g["target_time"], g["persistence"], label="Persistence", lw=1.0, ls="--")
        ax.axhline(0, color="k", lw=0.7); ax.set_title(f"Lead {lead} months"); ax.set_ylabel("Niño3.4 anomaly"); ax.legend(ncol=3); ax.grid(alpha=0.2)
    save_fig(fig, fig_dir / f"{prefix}_prediction_timeseries.png")


def plot_obs_pred_scatter(preds: pd.DataFrame, fig_dir: Path, prefix: str):
    if preds.empty or "observed" not in preds: return
    leads = sorted(preds["lead_months"].unique())
    fig, axes = plt.subplots(1, len(leads), figsize=(4.2 * len(leads), 4), squeeze=False)
    for ax, lead in zip(axes.ravel(), leads):
        g = preds[preds["lead_months"] == lead]
        ax.scatter(g["observed"], g["prediction"], s=24, alpha=0.65)
        lo = float(np.nanmin([g["observed"].min(), g["prediction"].min()])); hi = float(np.nanmax([g["observed"].max(), g["prediction"].max()]))
        ax.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls="--")
        ax.set_title(f"Lead {lead}"); ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    save_fig(fig, fig_dir / f"{prefix}_obs_pred_scatter.png")


def plot_error_distributions(preds: pd.DataFrame, fig_dir: Path, prefix: str):
    if preds.empty or "error" not in preds: return
    leads = sorted(preds["lead_months"].unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [preds.loc[preds["lead_months"] == lead, "error"].dropna().values for lead in leads]
    parts = ax.violinplot(data, positions=np.arange(len(leads)), showmeans=True, showextrema=False)
    for body in parts["bodies"]: body.set_alpha(0.65)
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.set_xticks(np.arange(len(leads))); ax.set_xticklabels([str(l) for l in leads])
    ax.set_xlabel("Lead time (months)"); ax.set_ylabel("Prediction error"); ax.set_title("XGB error distributions by lead")
    save_fig(fig, fig_dir / f"{prefix}_error_distributions.png")


def plot_monthly_metric_heatmap(monthly: pd.DataFrame, fig_dir: Path, prefix: str, metric: str):
    if monthly.empty or metric not in monthly: return
    mat = monthly.pivot(index="lead_months", columns="month", values=metric).reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    im = ax.imshow(mat.values, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(len(mat.index))); ax.set_yticklabels(mat.index.astype(str))
    ax.set_xticks(np.arange(12)); ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45, ha="right")
    ax.set_xlabel("Initialisation month"); ax.set_ylabel("Lead"); ax.set_title(f"Monthly {metric} heatmap")
    fig.colorbar(im, ax=ax, label=metric)
    save_fig(fig, fig_dir / f"{prefix}_monthly_{metric}_heatmap.png")


def load_shap(root: Path, lead: int, task: str) -> xr.Dataset | None:
    p = run_dir(root, lead, task) / "shap" / f"xgb_lead{lead:02d}_{task}_shap.zarr"
    if not p.exists():
        log.warning("Missing XGB SHAP store: %s", p)
        return None
    return _open_local_zarr(p)


def plot_crosslead_shap_heatmap(root: Path, leads: list[int], task: str, fig_dir: Path, prefix: str, top_n: int):
    series = {}
    for lead in leads:
        ds = load_shap(root, lead, task)
        if ds is not None:
            series[lead] = ds["abs_shap"].mean("time").to_pandas()
    if not series: return
    df = pd.DataFrame(series).fillna(0.0)
    feats = df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    mat = df.loc[feats].T
    fig, ax = plt.subplots(figsize=(max(9, 0.35 * len(feats)), 3.7))
    im = ax.imshow(mat.values, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(len(mat.index))); ax.set_yticklabels(mat.index.astype(str))
    ax.set_xticks(np.arange(len(feats))); ax.set_xticklabels(feats, rotation=75, ha="right")
    ax.set_xlabel("Feature"); ax.set_ylabel("Lead"); ax.set_title("Cross-lead XGB SHAP importance")
    fig.colorbar(im, ax=ax, label="Mean |SHAP|")
    save_fig(fig, fig_dir / f"{prefix}_crosslead_shap_importance_heatmap.png")


def plot_lead_difference_heatmap(root: Path, leads: list[int], task: str, fig_dir: Path, prefix: str, top_n: int):
    series = {}
    for lead in leads:
        ds = load_shap(root, lead, task)
        if ds is not None:
            series[lead] = ds["abs_shap"].mean("time").to_pandas()
    if len(series) < 2: return
    df = pd.DataFrame(series).fillna(0.0).sort_index(axis=1)
    lead_list = sorted(series)
    pairs = [(lead_list[i], lead_list[j]) for i in range(len(lead_list)) for j in range(i + 1, len(lead_list))]
    feats = df.mean(axis=1).sort_values(ascending=False).head(top_n).index
    diff = pd.DataFrame({f"{b}-{a}": df[b] - df[a] for a, b in pairs}).loc[feats]
    lim = float(np.nanquantile(np.abs(diff.values), 0.98)) if np.isfinite(diff.values).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(pairs)), max(5, 0.30 * len(feats))))
    im = ax.imshow(diff.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_yticks(np.arange(len(feats))); ax.set_yticklabels(feats)
    ax.set_xticks(np.arange(len(pairs))); ax.set_xticklabels([f"{b}-{a}" for a, b in pairs])
    ax.set_title("XGB SHAP lead differences"); ax.set_xlabel("Lead difference")
    fig.colorbar(im, ax=ax, label="Δ mean |SHAP|")
    save_fig(fig, fig_dir / f"{prefix}_lead_difference_shap_heatmap.png")
    diff.reset_index().rename(columns={"index": "feature"}).to_csv(fig_dir.parent / f"{prefix}_lead_difference_shap.csv", index=False)


def plot_shap_per_lead_figures(root: Path, leads: list[int], task: str, fig_dir: Path, prefix: str, top_n: int):
    for lead in leads:
        ds = load_shap(root, lead, task)
        if ds is None: continue
        subdir = ensure_dir(fig_dir / f"lead{lead:02d}_{task}")
        # A compact set, in case compute step was skipped but shap exists.
        imp = ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(7, max(4, 0.30 * len(imp))))
        ax.barh(np.arange(len(imp)), imp.values[::-1]); ax.set_yticks(np.arange(len(imp))); ax.set_yticklabels(imp.index[::-1])
        ax.set_xlabel("Mean |SHAP|"); ax.set_title(f"XGB lead {lead}: top features")
        save_fig(fig, subdir / f"xgb_lead{lead:02d}_{task}_top{top_n}_features.png")


def compile_xgb(cfg_path: str, task: str, tuned_root: str | Path | None, output_dir: str | Path | None, fig_dir: str | Path | None, leads: Iterable[int], top_n: int):
    cfg = load_config(cfg_path)
    root = Path(tuned_root) if tuned_root is not None else default_tuned_root(cfg)
    out = ensure_dir(Path(output_dir) if output_dir is not None else default_output_dir(cfg))
    figs = ensure_dir(Path(fig_dir) if fig_dir is not None else default_fig_dir(cfg))
    leads = [int(x) for x in leads]
    prefix = f"xgb_tuned_{task}"

    metrics, monthly, seasonal, preds = compile_prediction_metrics(cfg, root, leads, task)
    metrics.to_csv(out / f"{prefix}_metrics.csv", index=False)
    monthly.to_csv(out / f"{prefix}_monthly_metrics.csv", index=False)
    seasonal.to_csv(out / f"{prefix}_seasonal_metrics.csv", index=False)
    preds.to_csv(out / f"{prefix}_all_predictions.csv", index=False)

    plot_skill_curves(metrics, figs, prefix)
    plot_prediction_time_series(preds, figs, prefix)
    plot_obs_pred_scatter(preds, figs, prefix)
    plot_error_distributions(preds, figs, prefix)
    for metric in ["rmse", "corr", "bias", "rmse_skill_vs_persistence"]:
        plot_monthly_metric_heatmap(monthly, figs, prefix, metric)

    plot_crosslead_shap_heatmap(root, leads, task, figs, prefix, top_n=top_n)
    plot_lead_difference_heatmap(root, leads, task, figs, prefix, top_n=top_n)
    plot_shap_per_lead_figures(root, leads, task, figs, prefix, top_n=top_n)

    rows = []
    for lead in leads:
        ds = load_shap(root, lead, task)
        if ds is None: continue
        imp = ds["abs_shap"].mean("time").to_pandas()
        for feat, val in imp.items():
            rows.append({"lead_months": lead, "feature": feat, "mean_abs_shap": float(val)})
    pd.DataFrame(rows).to_csv(out / f"{prefix}_shap_importance.csv", index=False)

    manifest = {"task": task, "leads": leads, "tuned_root": str(root), "output_dir": str(out), "figure_dir": str(figs)}
    with open(out / f"{prefix}_analysis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("Saved tuned XGB compiled analysis -> %s", out)
    log.info("Saved tuned XGB figures -> %s", figs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile tuned XGB metrics and SHAP figure gallery")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--tuned-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--leads", nargs="+", type=int, default=LEADS_DEFAULT)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()
    compile_xgb(args.config, args.task, args.tuned_root, args.output_dir, args.fig_dir, args.leads, args.top_n)
