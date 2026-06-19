"""Compile tuned LSTM metrics and create an extended diagnostics gallery.

This script is tailored to the tuned-wide LSTM workflow where each lead has its
own run folder:

    data/tuned_wide/lstm/lead03_regression/
        metrics/
        predictions/
        tuning/
        models/
        shap/              # produced by compute_shap_tuned_lstm.py

It compiles final-test prediction metrics for leads 3/6/12, adds a persistence
baseline, creates monthly/seasonal skill summaries, and makes a figure gallery.
If tuned SHAP stores are present, it also creates SHAP beeswarm-style plots,
violin distributions, top-feature bars, monthly SHAP heatmaps, and a lightweight
proxy interaction matrix.

Examples
--------
python scripts/compile_metrics_tuned_lstm.py \
    --config configs/default.yaml \
    --task regression \
    --tuned-root data/tuned_wide/lstm

python scripts/compile_metrics_tuned_lstm.py \
    --config configs/default.yaml \
    --task regression \
    --tuned-root data/tuned_wide/lstm \
    --output-dir data/tuned_wide/lstm/compiled_analysis \
    --fig-dir figures/tuned_wide/lstm
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

from src.utils.config import load_config
from src.utils.io_utils import load_zarr
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_tuned_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "lstm"


def run_dir(tuned_root: Path, lead: int, task: str) -> Path:
    return tuned_root / f"lead{lead:02d}_{task}"


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def regression_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    if len(obs) == 0:
        return {k: np.nan for k in ["rmse", "mae", "corr", "r2", "bias", "mape_like", "n"]}
    err = pred - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    corr = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 1 and np.std(obs) > 0 and np.std(pred) > 0 else np.nan
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else np.nan
    # ENSO anomalies cross zero, so classic MAPE is not meaningful. This is a
    # scale-normalised absolute error using the mean absolute observed anomaly.
    scale = float(np.mean(np.abs(obs)))
    mape_like = float(mae / scale) if scale > 1e-12 else np.nan
    return {"rmse": rmse, "mae": mae, "corr": corr, "r2": r2, "bias": bias, "mape_like": mape_like, "n": float(len(obs))}


def read_target_series(cfg: dict) -> pd.Series | None:
    try:
        proc_dir = Path(cfg["data"]["processed_dir"])
        target_ds = load_zarr(proc_dir / "target_nino34.zarr").compute()
        da = target_ds["nino34"] if "nino34" in target_ds else list(target_ds.data_vars.values())[0]
        return pd.Series(da.values, index=pd.DatetimeIndex(da.time.values), name="nino34")
    except Exception as exc:
        log.warning("Could not load target_nino34.zarr for persistence baseline: %s", exc)
        return None


def add_persistence_prediction(df: pd.DataFrame, target: pd.Series | None) -> pd.DataFrame:
    df = df.copy()
    if target is None:
        df["persistence_nino34"] = np.nan
        df["persistence_error"] = np.nan
        return df
    init_time = pd.DatetimeIndex(pd.to_datetime(df["init_time"]))
    vals = []
    for t in init_time:
        if t in target.index:
            vals.append(float(target.loc[t]))
        else:
            # Fallback to nearest month if there is a harmless timestamp mismatch.
            try:
                vals.append(float(target.reindex([t], method="nearest").iloc[0]))
            except Exception:
                vals.append(np.nan)
    df["persistence_nino34"] = vals
    df["persistence_error"] = df["persistence_nino34"] - df["observed_nino34"]
    return df


def find_prediction_file(rd: Path, lead: int, task: str) -> Path | None:
    candidates = [
        rd / "predictions" / f"lstm_lead{lead:02d}_{task}_test_predictions.csv",
        rd / "models" / f"lstm_lead{lead:02d}_{task}_test_predictions.csv",
        rd / f"lstm_lead{lead:02d}_{task}_test_predictions.csv",
    ]
    candidates += list(rd.rglob(f"lstm_lead{lead:02d}_{task}_test_predictions.csv"))
    for p in candidates:
        if p.exists():
            return p
    return None


def read_predictions(tuned_root: Path, lead: int, task: str, target: pd.Series | None) -> pd.DataFrame | None:
    rd = run_dir(tuned_root, lead, task)
    pred_path = find_prediction_file(rd, lead, task)
    if pred_path is None:
        log.warning("Prediction CSV missing for lead=%02d task=%s under %s", lead, task, rd)
        return None
    df = pd.read_csv(pred_path)
    df["init_time"] = pd.to_datetime(df["init_time"])
    if "target_time" in df.columns:
        df["target_time"] = pd.to_datetime(df["target_time"])
    else:
        df["target_time"] = df["init_time"] + pd.DateOffset(months=lead)
    df["lead_months"] = lead
    df["task"] = task
    df["model_type"] = "lstm_tuned_wide"
    df["run_dir"] = str(rd)
    df["prediction_file"] = str(pred_path)
    if task == "regression":
        # Normalise common column names.
        if "prediction" in df.columns and "prediction_nino34" not in df.columns:
            df["prediction_nino34"] = df["prediction"]
        if "observed" in df.columns and "observed_nino34" not in df.columns:
            df["observed_nino34"] = df["observed"]
        if "error" not in df.columns and {"prediction_nino34", "observed_nino34"}.issubset(df.columns):
            df["error"] = df["prediction_nino34"] - df["observed_nino34"]
        df = add_persistence_prediction(df, target)
        df["init_month"] = df["init_time"].dt.month
        df["target_month"] = df["target_time"].dt.month
        df["init_season"] = df["init_month"].map(season_from_month)
        df["target_season"] = df["target_month"].map(season_from_month)
    return df


def compile_prediction_metrics(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    monthly_rows = []
    seasonal_rows = []
    for lead, g in pred_df.groupby("lead_months"):
        m = regression_metrics(g["observed_nino34"], g["prediction_nino34"])
        mp = regression_metrics(g["observed_nino34"], g["persistence_nino34"])
        row = {
            "model_type": "lstm_tuned_wide",
            "lead_months": int(lead),
            "task": "regression",
            "n_samples": int(m["n"]),
            "rmse": m["rmse"],
            "mae": m["mae"],
            "corr": m["corr"],
            "r2": m["r2"],
            "bias": m["bias"],
            "mape_like": m["mape_like"],
            "persistence_rmse": mp["rmse"],
            "persistence_mae": mp["mae"],
            "persistence_corr": mp["corr"],
            "persistence_r2": mp["r2"],
            "rmse_skill_vs_persistence": 1.0 - m["rmse"] / mp["rmse"] if np.isfinite(mp["rmse"]) and mp["rmse"] > 0 else np.nan,
        }
        rows.append(row)

        for month, mg in g.groupby("init_month"):
            mm = regression_metrics(mg["observed_nino34"], mg["prediction_nino34"])
            mmp = regression_metrics(mg["observed_nino34"], mg["persistence_nino34"])
            monthly_rows.append({
                "lead_months": int(lead),
                "init_month": int(month),
                "n_samples": int(mm["n"]),
                "rmse": mm["rmse"],
                "corr": mm["corr"],
                "r2": mm["r2"],
                "bias": mm["bias"],
                "persistence_rmse": mmp["rmse"],
                "persistence_corr": mmp["corr"],
                "rmse_skill_vs_persistence": 1.0 - mm["rmse"] / mmp["rmse"] if np.isfinite(mmp["rmse"]) and mmp["rmse"] > 0 else np.nan,
            })

        for season, sg in g.groupby("init_season"):
            sm = regression_metrics(sg["observed_nino34"], sg["prediction_nino34"])
            smp = regression_metrics(sg["observed_nino34"], sg["persistence_nino34"])
            seasonal_rows.append({
                "lead_months": int(lead),
                "init_season": season,
                "n_samples": int(sm["n"]),
                "rmse": sm["rmse"],
                "corr": sm["corr"],
                "r2": sm["r2"],
                "bias": sm["bias"],
                "persistence_rmse": smp["rmse"],
                "persistence_corr": smp["corr"],
                "rmse_skill_vs_persistence": 1.0 - sm["rmse"] / smp["rmse"] if np.isfinite(smp["rmse"]) and smp["rmse"] > 0 else np.nan,
            })

    return pd.DataFrame(rows), pd.DataFrame(monthly_rows), pd.DataFrame(seasonal_rows)


# ---------------------------------------------------------------------------
# SHAP loading and summaries
# ---------------------------------------------------------------------------


def find_shap_store(tuned_root: Path, lead: int, task: str, shap_root: Path | None = None) -> Path | None:
    rd = run_dir(tuned_root, lead, task)
    candidates: list[Path] = []
    candidates += list((rd / "shap").glob(f"lstm*lead{lead:02d}*{task}*.zarr")) if (rd / "shap").exists() else []
    candidates += list(rd.rglob(f"lstm*lead{lead:02d}*{task}*.zarr"))
    if shap_root is not None and shap_root.exists():
        candidates += list(shap_root.glob(f"lstm*lead{lead:02d}*{task}*.zarr"))
        candidates += list(shap_root.rglob(f"lstm*lead{lead:02d}*{task}*.zarr"))
    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for p in candidates:
        if p not in seen and p.exists():
            unique.append(p)
            seen.add(p)
    return unique[0] if unique else None


def load_shap_dataset(tuned_root: Path, lead: int, task: str, shap_root: Path | None = None) -> xr.Dataset | None:
    p = find_shap_store(tuned_root, lead, task, shap_root)
    if p is None:
        return None
    try:
        ds = xr.open_zarr(str(p)).load()
        ds.attrs["source_store"] = str(p)
        return ds
    except Exception as exc:
        log.warning("Could not open SHAP store %s: %s", p, exc)
        return None


def feature_importance(ds: xr.Dataset) -> pd.Series:
    return ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False)


def save_shap_tables(ds_by_lead: dict[int, xr.Dataset], out_dir: Path, task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pair_rows = []
    for lead, ds in ds_by_lead.items():
        imp = feature_importance(ds)
        for rank, (feat, val) in enumerate(imp.items(), start=1):
            rows.append({"lead_months": lead, "feature": feat, "rank": rank, "mean_abs_shap": float(val)})
        top_features = list(imp.head(20).index.astype(str))
        s = ds["shap_values"].sel(feature=top_features).to_pandas().astype(float)
        abs_arr = np.abs(s.values)
        co = abs_arr.T @ abs_arr / max(1, abs_arr.shape[0])
        for i in range(len(top_features)):
            for j in range(i + 1, len(top_features)):
                pair_rows.append({
                    "lead_months": lead,
                    "feature_i": top_features[i],
                    "feature_j": top_features[j],
                    "mean_abs_shap_product": float(co[i, j]),
                    "signed_shap_corr": float(np.corrcoef(s.iloc[:, i], s.iloc[:, j])[0, 1]),
                })
    imp_df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows).sort_values(["lead_months", "mean_abs_shap_product"], ascending=[True, False]) if pair_rows else pd.DataFrame()
    imp_df.to_csv(out_dir / f"lstm_tuned_{task}_shap_feature_importance.csv", index=False)
    pair_df.to_csv(out_dir / f"lstm_tuned_{task}_proxy_shap_interactions.csv", index=False)
    return imp_df, pair_df


# ---------------------------------------------------------------------------
# Figure helpers: prediction metrics
# ---------------------------------------------------------------------------


def plot_skill_curves(metrics: pd.DataFrame, fig_dir: Path, task: str) -> None:
    m = metrics.sort_values("lead_months")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(m["lead_months"], m["corr"], marker="o", label="Tuned LSTM")
    axes[0].plot(m["lead_months"], m["persistence_corr"], marker="o", linestyle="--", label="Persistence")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("Lead time (months)")
    axes[0].set_ylabel("ACC / correlation")
    axes[0].set_title("ACC vs lead")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(m["lead_months"], m["rmse"], marker="o", label="Tuned LSTM")
    axes[1].plot(m["lead_months"], m["persistence_rmse"], marker="o", linestyle="--", label="Persistence")
    axes[1].set_xlabel("Lead time (months)")
    axes[1].set_ylabel("RMSE (°C)")
    axes[1].set_title("RMSE vs lead")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle("Tuned LSTM skill curves with persistence baseline")
    fig.tight_layout()
    fig.savefig(fig_dir / f"fig01_lstm_tuned_{task}_skill_curves.png", dpi=200)
    plt.close(fig)


def plot_prediction_timeseries(pred_df: pd.DataFrame, fig_dir: Path, task: str) -> None:
    leads = sorted(pred_df["lead_months"].unique())
    fig, axes = plt.subplots(len(leads), 1, figsize=(12, 3.2 * len(leads)), sharex=False)
    if len(leads) == 1:
        axes = [axes]
    for ax, lead in zip(axes, leads):
        g = pred_df[pred_df["lead_months"] == lead].sort_values("target_time")
        ax.plot(g["target_time"], g["observed_nino34"], label="Observed", lw=1.5)
        ax.plot(g["target_time"], g["prediction_nino34"], label="Tuned LSTM", lw=1.2)
        ax.plot(g["target_time"], g["persistence_nino34"], label="Persistence", lw=1.0, linestyle="--", alpha=0.8)
        ax.axhline(0.5, color="red", lw=0.8, alpha=0.4)
        ax.axhline(-0.5, color="blue", lw=0.8, alpha=0.4)
        ax.set_title(f"Lead {lead} months")
        ax.set_ylabel("Niño3.4 anomaly")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", ncol=3)
    fig.suptitle("Observed vs predicted ENSO evolution")
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_prediction_timeseries.png", dpi=180)
    plt.close(fig)


def plot_obs_pred_scatter(pred_df: pd.DataFrame, fig_dir: Path, task: str) -> None:
    leads = sorted(pred_df["lead_months"].unique())
    fig, axes = plt.subplots(1, len(leads), figsize=(4.2 * len(leads), 4), squeeze=False)
    vmin = float(np.nanmin([pred_df["observed_nino34"].min(), pred_df["prediction_nino34"].min()]))
    vmax = float(np.nanmax([pred_df["observed_nino34"].max(), pred_df["prediction_nino34"].max()]))
    pad = 0.2 * max(1e-9, vmax - vmin)
    for ax, lead in zip(axes.ravel(), leads):
        g = pred_df[pred_df["lead_months"] == lead]
        ax.scatter(g["observed_nino34"], g["prediction_nino34"], s=28, alpha=0.7)
        ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], color="black", linestyle="--", lw=1)
        m = regression_metrics(g["observed_nino34"], g["prediction_nino34"])
        ax.set_title(f"Lead {lead} mo\nACC={m['corr']:.2f}, RMSE={m['rmse']:.2f}")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.25)
        ax.set_xlim(vmin - pad, vmax + pad)
        ax.set_ylim(vmin - pad, vmax + pad)
    fig.suptitle("Observed-predicted scatter")
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_obs_pred_scatter.png", dpi=180)
    plt.close(fig)


def plot_error_distributions(pred_df: pd.DataFrame, fig_dir: Path, task: str) -> None:
    leads = sorted(pred_df["lead_months"].unique())
    data = [pred_df[pred_df["lead_months"] == l]["error"].dropna().values for l in leads]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.violinplot(data, positions=np.arange(len(leads)), showmeans=True, showextrema=True)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(leads)))
    ax.set_xticklabels([f"{l} mo" for l in leads])
    ax.set_ylabel("Prediction error (pred - obs)")
    ax.set_title("Error distributions by lead")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_error_distributions.png", dpi=180)
    plt.close(fig)


def plot_monthly_metric_heatmaps(monthly: pd.DataFrame, fig_dir: Path, task: str) -> None:
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for metric in ["rmse", "corr", "rmse_skill_vs_persistence", "bias"]:
        pivot = monthly.pivot(index="lead_months", columns="init_month", values=metric).reindex(columns=range(1, 13))
        fig, ax = plt.subplots(figsize=(10, 3.4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm" if metric in {"corr", "rmse_skill_vs_persistence", "bias"} else "magma")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{int(l)} mo" for l in pivot.index])
        ax.set_xticks(range(12))
        ax.set_xticklabels(month_labels, rotation=45)
        ax.set_xlabel("Initialisation month")
        ax.set_ylabel("Lead")
        ax.set_title(f"Monthly {metric.replace('_', ' ')}")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(metric)
        fig.tight_layout()
        fig.savefig(fig_dir / f"lstm_tuned_{task}_monthly_{metric}_heatmap.png", dpi=180)
        plt.close(fig)


def plot_seasonal_metric_bars(seasonal: pd.DataFrame, fig_dir: Path, task: str) -> None:
    seasons = ["DJF", "MAM", "JJA", "SON"]
    leads = sorted(seasonal["lead_months"].unique())
    width = 0.22
    x = np.arange(len(seasons))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, metric in enumerate(["rmse", "corr"]):
        ax = axes[k]
        for i, lead in enumerate(leads):
            vals = seasonal[seasonal["lead_months"] == lead].set_index("init_season").reindex(seasons)[metric]
            ax.bar(x + (i - (len(leads)-1)/2) * width, vals.values, width=width, label=f"{lead} mo")
        ax.set_xticks(x)
        ax.set_xticklabels(seasons)
        ax.set_title(f"Seasonal {metric.upper()}")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("Seasonal skill by initialisation season")
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_seasonal_skill_bars.png", dpi=180)
    plt.close(fig)


def plot_best_params_heatmap(tuned_root: Path, leads: Iterable[int], task: str, fig_dir: Path) -> None:
    rows = []
    for lead in leads:
        rd = run_dir(tuned_root, lead, task)
        candidates = list((rd / "tuning").glob("*best_params*.json")) + list(rd.rglob("*best_params*.json"))
        if not candidates:
            # Sometimes best_params lives inside metrics payload.
            candidates = list((rd / "metrics").glob("*metrics.json")) + list(rd.rglob("*metrics.json"))
        if not candidates:
            continue
        try:
            payload = json.loads(candidates[0].read_text())
            params = payload.get("best_params", payload)
            row = {"lead_months": lead}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    row[k] = v
            rows.append(row)
        except Exception:
            continue
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("lead_months")
    df.to_csv(fig_dir.parent / f"lstm_tuned_{task}_best_params_by_lead.csv")
    # Numeric-only quicklook.
    num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if num.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * num.shape[1]), 3.2))
    # Normalise columns for display so different units can share a heatmap.
    arr = num.copy()
    arr = (arr - arr.min()) / (arr.max() - arr.min()).replace(0, 1)
    im = ax.imshow(arr.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(num.shape[1]))
    ax.set_xticklabels(num.columns, rotation=45, ha="right")
    ax.set_yticks(range(num.shape[0]))
    ax.set_yticklabels([f"{int(l)} mo" for l in num.index])
    ax.set_title("Best tuned parameters by lead, normalised scale")
    fig.colorbar(im, ax=ax, label="column-normalised value")
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_best_params_heatmap.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure helpers: SHAP
# ---------------------------------------------------------------------------


def plot_shap_top_bars(ds_by_lead: dict[int, xr.Dataset], fig_dir: Path, task: str, top_n: int) -> None:
    for lead, ds in ds_by_lead.items():
        imp = feature_importance(ds).head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(imp))))
        ax.barh(imp.index.astype(str), imp.values)
        ax.set_xlabel("Mean |SHAP|")
        ax.set_title(f"LSTM lead {lead} mo: top {top_n} predictors")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / f"lstm_tuned_{task}_lead{lead:02d}_shap_top{top_n}.png", dpi=180)
        plt.close(fig)


def plot_crosslead_importance_heatmap(ds_by_lead: dict[int, xr.Dataset], fig_dir: Path, task: str, top_n: int) -> None:
    if not ds_by_lead:
        return
    all_imp = []
    for lead, ds in ds_by_lead.items():
        s = feature_importance(ds).rename(lead)
        all_imp.append(s)
    imp_df = pd.concat(all_imp, axis=1).fillna(0.0)
    top = imp_df.max(axis=1).sort_values(ascending=False).head(top_n).index
    data = imp_df.loc[top, sorted(imp_df.columns)]
    fig, ax = plt.subplots(figsize=(6.5, max(5, 0.28 * len(top))))
    im = ax.imshow(data.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels([f"{int(c)} mo" for c in data.columns])
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index.astype(str))
    ax.set_title("Mean |SHAP| across leads")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_crosslead_shap_importance_heatmap.png", dpi=180)
    plt.close(fig)


def plot_shap_beeswarm(ds: xr.Dataset, fig_dir: Path, lead: int, task: str, top_n: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    features = list(feature_importance(ds).head(top_n).index.astype(str))
    s = ds["shap_values"].sel(feature=features).to_pandas().astype(float)
    pred = ds["prediction"].to_pandas().astype(float)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(features))))
    for j, feat in enumerate(reversed(features)):
        vals = s[feat].values
        y = rng.normal(j, 0.08, size=len(vals))
        sc = ax.scatter(vals, y, c=pred.values, cmap="coolwarm", alpha=0.65, s=16, linewidths=0)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(list(reversed(features)))
    ax.set_xlabel("Signed SHAP value")
    ax.set_title(f"Lead {lead} mo SHAP beeswarm-style plot")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Prediction")
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_lead{lead:02d}_shap_beeswarm.png", dpi=180)
    plt.close(fig)


def plot_shap_violins(ds: xr.Dataset, fig_dir: Path, lead: int, task: str, top_n: int) -> None:
    features = list(feature_importance(ds).head(top_n).index.astype(str))
    data = [ds["shap_values"].sel(feature=f).values.astype(float) for f in features]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.3 * len(features))))
    ax.violinplot(data[::-1], vert=False, showmeans=True, showextrema=True)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set_yticks(range(1, len(features) + 1))
    ax.set_yticklabels(features[::-1])
    ax.set_xlabel("Signed SHAP")
    ax.set_title(f"Lead {lead} mo SHAP distributions")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_lead{lead:02d}_shap_violins.png", dpi=180)
    plt.close(fig)


def plot_shap_monthly(ds: xr.Dataset, fig_dir: Path, lead: int, task: str, top_n: int) -> None:
    features = list(feature_importance(ds).head(top_n).index.astype(str))
    monthly = ds["abs_shap"].sel(feature=features).groupby("time.month").mean("time").to_pandas().reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(features))))
    im = ax.imshow(monthly[features].T.values, aspect="auto", cmap="magma")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_title(f"Lead {lead} mo monthly SHAP importance")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_lead{lead:02d}_monthly_shap_heatmap.png", dpi=180)
    plt.close(fig)


def plot_shap_proxy_interaction(ds: xr.Dataset, fig_dir: Path, lead: int, task: str, top_n: int) -> None:
    features = list(feature_importance(ds).head(top_n).index.astype(str))
    s = ds["shap_values"].sel(feature=features).to_pandas().astype(float)
    abs_arr = np.abs(s.values)
    co = abs_arr.T @ abs_arr / max(1, abs_arr.shape[0])
    np.fill_diagonal(co, np.nan)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(co, cmap="plasma")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(features)
    ax.set_title(f"Lead {lead} mo proxy SHAP interactions")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean(|SHAP_i| × |SHAP_j|)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_tuned_{task}_lead{lead:02d}_proxy_interaction_heatmap.png", dpi=180)
    plt.close(fig)


def make_shap_figures(ds_by_lead: dict[int, xr.Dataset], fig_dir: Path, task: str, top_n: int) -> None:
    if not ds_by_lead:
        log.warning("No SHAP stores found. Skipping SHAP figures.")
        return
    plot_shap_top_bars(ds_by_lead, fig_dir, task, top_n=15)
    plot_crosslead_importance_heatmap(ds_by_lead, fig_dir, task, top_n=top_n)
    for lead, ds in ds_by_lead.items():
        plot_shap_beeswarm(ds, fig_dir, lead, task, top_n=min(top_n, 20))
        plot_shap_violins(ds, fig_dir, lead, task, top_n=min(top_n, 15))
        plot_shap_monthly(ds, fig_dir, lead, task, top_n=min(top_n, 18))
        plot_shap_proxy_interaction(ds, fig_dir, lead, task, top_n=min(top_n, 20))


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def compile_tuned_lstm_analysis(
    cfg_path: str,
    task: str,
    leads: list[int],
    tuned_root: str | None,
    shap_root: str | None,
    output_dir: str | None,
    fig_dir: str | None,
    top_n: int,
    skip_shap: bool,
) -> None:
    cfg = load_config(cfg_path)
    root = Path(tuned_root) if tuned_root is not None else default_tuned_root(cfg)
    out_dir = ensure_dir(Path(output_dir) if output_dir else root / "compiled_analysis")
    figures = ensure_dir(Path(fig_dir) if fig_dir else root / "figures" / "compiled_analysis")
    shap_root_path = Path(shap_root) if shap_root is not None else Path(cfg["experiment"]["output_dir"]) / "data" / "shap_tuned_wide" / "lstm"

    target = read_target_series(cfg)
    pred_frames = []
    for lead in leads:
        df = read_predictions(root, lead, task, target)
        if df is not None:
            pred_frames.append(df)
    if not pred_frames:
        raise FileNotFoundError(f"No prediction CSV files found under {root}")
    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_df.to_csv(out_dir / f"lstm_tuned_{task}_all_predictions.csv", index=False)

    if task != "regression":
        log.warning("This extended compiler currently focuses on regression diagnostics. Saved combined predictions only.")
        return

    metrics, monthly, seasonal = compile_prediction_metrics(pred_df)
    metrics.to_csv(out_dir / f"lstm_tuned_{task}_metrics.csv", index=False)
    monthly.to_csv(out_dir / f"lstm_tuned_{task}_monthly_metrics.csv", index=False)
    seasonal.to_csv(out_dir / f"lstm_tuned_{task}_seasonal_metrics.csv", index=False)
    log.info("Saved compiled metrics -> %s", out_dir)
    print(metrics.round(4).to_string(index=False))

    plot_skill_curves(metrics, figures, task)
    plot_prediction_timeseries(pred_df, figures, task)
    plot_obs_pred_scatter(pred_df, figures, task)
    plot_error_distributions(pred_df, figures, task)
    plot_monthly_metric_heatmaps(monthly, figures, task)
    plot_seasonal_metric_bars(seasonal, figures, task)
    plot_best_params_heatmap(root, leads, task, figures)

    if not skip_shap:
        ds_by_lead = {}
        for lead in leads:
            ds = load_shap_dataset(root, lead, task, shap_root_path)
            if ds is not None:
                ds_by_lead[lead] = ds
                log.info("Loaded SHAP lead=%02d from %s", lead, ds.attrs.get("source_store", "unknown"))
            else:
                log.warning("No SHAP store found for lead=%02d", lead)
        if ds_by_lead:
            save_shap_tables(ds_by_lead, out_dir, task)
            make_shap_figures(ds_by_lead, figures, task, top_n=top_n)

    manifest = {
        "task": task,
        "leads": leads,
        "tuned_root": str(root),
        "shap_root": str(shap_root_path),
        "output_dir": str(out_dir),
        "figure_dir": str(figures),
        "figures": sorted([p.name for p in figures.glob("*.png")]),
        "tables": sorted([p.name for p in out_dir.glob("*.csv")]),
    }
    with open(out_dir / f"lstm_tuned_{task}_analysis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Saved figure gallery -> %s", figures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile tuned LSTM metrics and build extended figure gallery")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--leads", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--tuned-root", default=None, help="Root with leadXX_task tuned folders. Default: output/data/tuned_wide/lstm")
    parser.add_argument("--shap-root", default=None, help="Optional central SHAP root. Default: output/data/shap_tuned_wide/lstm")
    parser.add_argument("--output-dir", default=None, help="Directory for compiled CSV/JSON outputs")
    parser.add_argument("--fig-dir", default=None, help="Directory for generated PNG figures")
    parser.add_argument("--top-n", type=int, default=20, help="Top features for SHAP plots")
    parser.add_argument("--skip-shap", action="store_true", help="Only compile prediction metrics; skip SHAP plots/tables")
    args = parser.parse_args()

    compile_tuned_lstm_analysis(
        cfg_path=args.config,
        task=args.task,
        leads=args.leads,
        tuned_root=args.tuned_root,
        shap_root=args.shap_root,
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
        top_n=args.top_n,
        skip_shap=args.skip_shap,
    )
