"""Compute SHAP for tuned XGBoost ENSO models and create rich SHAP figures.

Expected tuned directory layout:

    data/tuned_wide/xgb/lead03_regression/
        models/
        metrics/
        predictions/
        tuning/

This script loads the tuned XGB model from the per-lead run folder, rebuilds
the tabular feature matrix, computes TreeSHAP on the test period, and saves
channel/feature-level SHAP stores plus a figure gallery.

Outputs per lead
----------------
data/tuned_wide/xgb/lead06_regression/shap/
    xgb_lead06_regression_shap.zarr
    xgb_lead06_regression_feature_importance.csv
    xgb_lead06_regression_monthly_feature_importance.csv
    xgb_lead06_regression_proxy_interactions.csv

figures/tuned_wide/xgb/lead06_regression/
    top bars, beeswarm, violins, monthly heatmap, hex plot,
    SHAP-vs-prediction scatter atlas, proxy interaction heatmap

Notes
-----
For XGBoost, true SHAP interaction values can be computed separately, but this
script also saves a light proxy interaction matrix:
    mean(|SHAP_i| * |SHAP_j|)
which is cheap and useful for screening paired predictor behaviour.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.utils.io_utils import load_feature_matrix
from src.utils.logging_utils import get_logger
from src.utils.preprocessing import build_class_labels, train_val_test_split_temporal

log = get_logger(__name__)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_tuned_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "xgb"


def default_global_shap_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "shap_tuned_wide" / "xgb"


def default_fig_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "figures" / "tuned_wide" / "xgb"


def run_dir(tuned_root: Path, lead: int, task: str) -> Path:
    return tuned_root / f"lead{lead:02d}_{task}"


def write_zarr_safe(ds: xr.Dataset, path: Path) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ds.to_zarr(str(path), mode="w", consolidated=False, zarr_format=2)
    except TypeError:
        ds.to_zarr(str(path), mode="w", consolidated=False, zarr_version=2)


def load_tuned_xgb_model(cfg: dict, lead: int, task: str, model_dir: Path, model_module: str):
    module = importlib.import_module(f"src.models.{model_module}")
    cls = getattr(module, "ENSOXGBModel")
    model = cls(cfg, lead, task)
    model.load(model_dir)
    return model


def load_xgb_arrays(cfg: dict, lead: int, task: str):
    feat_dir = Path(cfg["data"]["processed_dir"]) / "features"
    feat_path = feat_dir / f"features_lead{lead:02d}.npz"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing feature matrix: {feat_path}")

    X, y_reg, feat_names, times = load_feature_matrix(feat_path)
    times = pd.DatetimeIndex(times)

    (X_tr, y_tr, t_tr,
     X_val, y_val, t_val,
     X_te, y_te, t_te) = train_val_test_split_temporal(
        X, y_reg, times,
        train_years=tuple(cfg["data"]["train_years"]),
        val_years=tuple(cfg["data"]["val_years"]),
        test_years=tuple(cfg["data"]["test_years"]),
    )

    if task == "classification":
        y_tr_fit = build_class_labels(y_tr)
        y_val_fit = build_class_labels(y_val)
        y_te_eval = build_class_labels(y_te)
    else:
        y_tr_fit, y_val_fit, y_te_eval = y_tr, y_val, y_te

    return X_tr, y_tr_fit, t_tr, X_val, y_val_fit, t_val, X_te, y_te_eval, t_te, list(map(str, feat_names)), y_te


def subset_eval(X_te: np.ndarray, y_te: np.ndarray, t_te, max_eval_samples: int, seed: int):
    n = len(X_te)
    if max_eval_samples is None or max_eval_samples <= 0 or max_eval_samples >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=int(max_eval_samples), replace=False))
    return X_te[idx], np.asarray(y_te)[idx], pd.DatetimeIndex(t_te)[idx], idx


def compute_tree_shap_values(model_wrapper, X_eval: np.ndarray):
    import shap

    xgb_model = getattr(model_wrapper, "model", model_wrapper)
    explainer = shap.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_eval)

    if isinstance(shap_vals, list):
        shap_arr = np.stack(shap_vals, axis=-1)
    else:
        shap_arr = np.asarray(shap_vals)

    # Some shap versions return (N, F, C) directly for multiclass.
    base_value = getattr(explainer, "expected_value", np.nan)
    return shap_arr.astype(np.float32), np.asarray(base_value).tolist()


def predict_eval(model, X_eval: np.ndarray, task: str) -> np.ndarray:
    pred = model.predict(X_eval)
    if task == "classification":
        pred = np.asarray(pred)
        if pred.ndim == 2:
            return np.argmax(pred, axis=1).astype(np.float32)
        return pred.astype(np.float32)
    return np.asarray(pred, dtype=np.float32).ravel()


def make_shap_dataset(shap_vals, preds, observed, feat_names, times_eval, lead, task, base_value):
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:
        signed_2d = arr.mean(axis=-1)
        abs_2d = np.abs(arr).mean(axis=-1)
    else:
        signed_2d = arr
        abs_2d = np.abs(arr)

    ds = xr.Dataset(
        {
            "shap_values": (("time", "feature"), signed_2d.astype(np.float32)),
            "abs_shap": (("time", "feature"), abs_2d.astype(np.float32)),
            "prediction": ("time", np.asarray(preds, dtype=np.float32)),
            "observed": ("time", np.asarray(observed, dtype=np.float32)),
        },
        coords={
            "time": pd.DatetimeIndex(times_eval),
            "feature": np.asarray(feat_names, dtype=str),
        },
        attrs={
            "model_type": "xgb_tuned",
            "lead_months": int(lead),
            "task": str(task),
            "base_value": base_value,
            "description": "Feature-level TreeSHAP values from tuned XGBoost.",
        },
    )
    return ds


def save_fig(fig, path: Path, dpi: int = 180):
    ensure_dir(path.parent)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_top_bar(ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int):
    imp = ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7.4, max(4, 0.30 * len(imp))))
    ax.barh(np.arange(len(imp)), imp.values[::-1])
    ax.set_yticks(np.arange(len(imp)))
    ax.set_yticklabels(imp.index[::-1])
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Top {top_n} XGB features by mean |SHAP|")
    save_fig(fig, fig_dir / f"{prefix}_top{top_n}_feature_bar.png")


def plot_beeswarm(ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int, seed: int = 0):
    vals = ds["shap_values"].to_pandas()
    feats = list(ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False).head(top_n).index)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8, max(5, 0.30 * len(feats))))
    for i, feat in enumerate(feats[::-1]):
        x = vals[feat].values
        if len(x) > 800:
            idx = rng.choice(len(x), size=800, replace=False)
            x = x[idx]
        y = np.full_like(x, i, dtype=float) + rng.normal(0, 0.08, len(x))
        ax.scatter(x, y, s=13, alpha=0.60, linewidths=0)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels(feats[::-1])
    ax.set_xlabel("Signed SHAP value")
    ax.set_title("XGB SHAP beeswarm-style distribution")
    save_fig(fig, fig_dir / f"{prefix}_shap_beeswarm.png")


def plot_violins(ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int):
    vals = ds["shap_values"].to_pandas()
    feats = list(ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False).head(top_n).index)
    data = [vals[f].dropna().values for f in feats[::-1]]
    fig, ax = plt.subplots(figsize=(8, max(5, 0.33 * len(feats))))
    parts = ax.violinplot(data, vert=False, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.65)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(np.arange(1, len(feats) + 1))
    ax.set_yticklabels(feats[::-1])
    ax.set_xlabel("Signed SHAP value")
    ax.set_title("Signed XGB SHAP distributions")
    save_fig(fig, fig_dir / f"{prefix}_shap_violins.png")


def plot_monthly_heatmap(ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int):
    feats = list(ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False).head(top_n).index)
    df = ds["abs_shap"].sel(feature=feats).to_pandas()
    df["month"] = pd.DatetimeIndex(df.index).month
    mat = df.groupby("month")[feats].mean().reindex(range(1, 13))
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(feats)), 4.5))
    im = ax.imshow(mat.values, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right")
    ax.set_title("Monthly mean |SHAP| by XGB feature")
    ax.set_ylabel("Initialisation month")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |SHAP|")
    save_fig(fig, fig_dir / f"{prefix}_monthly_feature_shap_heatmap.png")


def plot_hex_and_scatter(ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int):
    pred = ds["prediction"].values
    total_abs = ds["abs_shap"].sum("feature").values
    fig, ax = plt.subplots(figsize=(6, 4.5))
    hb = ax.hexbin(pred, total_abs, gridsize=28, mincnt=1)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Total |SHAP| across features")
    ax.set_title("Prediction vs total XGB attribution")
    fig.colorbar(hb, ax=ax, label="Count")
    save_fig(fig, fig_dir / f"{prefix}_hex_prediction_total_abs_shap.png")

    feats = list(ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False).head(min(top_n, 9)).index)
    ncols = 3
    nrows = int(np.ceil(len(feats) / ncols))
    vals = ds["shap_values"].to_pandas()
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax, feat in zip(axes.ravel(), feats):
        ax.scatter(pred, vals[feat].values, s=16, alpha=0.60, linewidths=0)
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.axvline(0, color="k", lw=0.7, ls=":")
        ax.set_title(feat)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("SHAP")
    for ax in axes.ravel()[len(feats):]:
        ax.axis("off")
    fig.suptitle("XGB feature SHAP vs prediction scatter atlas", y=1.02)
    save_fig(fig, fig_dir / f"{prefix}_shap_vs_prediction_scatter_atlas.png")


def plot_phase_composites(ds: xr.Dataset, fig_dir: Path, prefix: str, top_n: int):
    obs = ds["observed"].values if "observed" in ds else ds["prediction"].values
    masks = {
        "ElNino": obs >= 0.5,
        "Neutral": np.abs(obs) < 0.5,
        "LaNina": obs <= -0.5,
    }
    feats = list(ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False).head(top_n).index)
    rows = {}
    for label, mask in masks.items():
        if mask.sum() > 0:
            rows[label] = ds["shap_values"].sel(feature=feats).isel(time=mask).mean("time").to_pandas()
        else:
            rows[label] = pd.Series(index=feats, dtype=float)
    mat = pd.DataFrame(rows).T
    lim = np.nanquantile(np.abs(mat.values), 0.98) if np.isfinite(mat.values).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(feats)), 3.8))
    im = ax.imshow(mat.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right")
    ax.set_title("Phase-composite signed SHAP")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean signed SHAP")
    save_fig(fig, fig_dir / f"{prefix}_phase_composite_shap_heatmap.png")


def save_proxy_interactions(ds: xr.Dataset, out_csv: Path, fig_dir: Path, prefix: str, top_n: int):
    abs_df = ds["abs_shap"].to_pandas()
    feats = list(abs_df.mean().sort_values(ascending=False).head(top_n).index)
    A = abs_df[feats].values
    inter = (A[:, :, None] * A[:, None, :]).mean(axis=0)
    inter_df = pd.DataFrame(inter, index=feats, columns=feats)
    inter_df.to_csv(out_csv)
    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(feats)), max(5, 0.35 * len(feats))))
    im = ax.imshow(inter_df.values, origin="lower")
    ax.set_xticks(np.arange(len(feats)))
    ax.set_yticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right")
    ax.set_yticklabels(feats)
    ax.set_title("Proxy XGB SHAP co-importance")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean(|SHAP_i| × |SHAP_j|)")
    save_fig(fig, fig_dir / f"{prefix}_proxy_interaction_heatmap.png")


def make_figures(ds: xr.Dataset, out_dir: Path, fig_dir: Path, prefix: str, top_n: int):
    plot_top_bar(ds, fig_dir, prefix, top_n)
    plot_beeswarm(ds, fig_dir, prefix, top_n)
    plot_violins(ds, fig_dir, prefix, min(top_n, 20))
    plot_monthly_heatmap(ds, fig_dir, prefix, min(top_n, 25))
    plot_hex_and_scatter(ds, fig_dir, prefix, min(top_n, 9))
    plot_phase_composites(ds, fig_dir, prefix, min(top_n, 20))
    save_proxy_interactions(ds, out_dir / f"{prefix}_proxy_interactions.csv", fig_dir, prefix, min(top_n, 25))


def run_xgb_tuned_shap(
    cfg_path: str,
    lead: int,
    task: str = "regression",
    tuned_root: str | Path | None = None,
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    global_shap_dir: str | Path | None = None,
    fig_dir: str | Path | None = None,
    model_module: str = "xgb_model_tuned",
    max_eval_samples: int = 0,
    top_n: int = 25,
):
    cfg = load_config(cfg_path)
    root = Path(tuned_root) if tuned_root is not None else default_tuned_root(cfg)
    rd = run_dir(root, lead, task)
    model_dir = Path(model_dir) if model_dir is not None else rd / "models"
    output_dir = ensure_dir(Path(output_dir) if output_dir is not None else rd / "shap")
    global_shap_dir = ensure_dir(Path(global_shap_dir) if global_shap_dir is not None else default_global_shap_root(cfg))
    fig_dir = ensure_dir(Path(fig_dir) if fig_dir is not None else default_fig_root(cfg) / f"lead{lead:02d}_{task}")

    model = load_tuned_xgb_model(cfg, lead, task, model_dir, model_module)
    (X_tr, y_tr, t_tr, X_val, y_val, t_val, X_te, y_te, t_te, feat_names, y_test_reg) = load_xgb_arrays(cfg, lead, task)
    X_eval, y_eval, times_eval, idx = subset_eval(X_te, y_te, t_te, max_eval_samples=max_eval_samples, seed=int(cfg["experiment"]["seed"]))
    observed = np.asarray(y_test_reg)[idx]

    log.info("Computing TreeSHAP tuned XGB: lead=%02d eval=%d features=%d", lead, len(X_eval), X_eval.shape[1])
    shap_vals, base_value = compute_tree_shap_values(model, X_eval)
    preds = predict_eval(model, X_eval, task=task)

    prefix = f"xgb_lead{lead:02d}_{task}"
    ds = make_shap_dataset(shap_vals, preds, observed, feat_names, times_eval, lead, task, base_value)

    store = output_dir / f"{prefix}_shap.zarr"
    write_zarr_safe(ds, store)
    dst = global_shap_dir / store.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(store, dst)

    imp = ds["abs_shap"].mean("time").to_pandas().sort_values(ascending=False)
    imp.rename("mean_abs_shap").to_csv(output_dir / f"{prefix}_feature_importance.csv")
    monthly = ds["abs_shap"].to_pandas()
    monthly["month"] = pd.DatetimeIndex(monthly.index).month
    monthly.groupby("month").mean().to_csv(output_dir / f"{prefix}_monthly_feature_importance.csv")
    make_figures(ds, output_dir, fig_dir, prefix, top_n=top_n)

    log.info("Saved tuned XGB SHAP -> %s", output_dir)
    log.info("Saved tuned XGB figures -> %s", fig_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SHAP for tuned XGB ENSO model and make figures")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lead", type=int, required=True, choices=[3, 6, 12])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--tuned-root", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--global-shap-dir", default=None)
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--model-module", default="xgb_model_tuned")
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    run_xgb_tuned_shap(
        cfg_path=args.config,
        lead=args.lead,
        task=args.task,
        tuned_root=args.tuned_root,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        global_shap_dir=args.global_shap_dir,
        fig_dir=args.fig_dir,
        model_module=args.model_module,
        max_eval_samples=args.max_eval_samples,
        top_n=args.top_n,
    )
