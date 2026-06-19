"""Compute SHAP for tuned LSTM ENSO models and create richer SHAP diagnostics.

This script is designed for the tuned LSTM directory layout created by the
wide tuning workflow:

    data/tuned_wide/lstm/lead03_regression/
        models/
        metrics/
        predictions/
        tuning/

It loads the tuned LSTM model from the per-lead run folder, rebuilds the LSTM
input sequences with the sequence length stored in the model checkpoint, applies
train-only standardisation, computes Gradient SHAP on the test period, and saves
both a Zarr store and a small gallery of diagnostic plots.

Examples
--------
python scripts/compute_shap_tuned_lstm.py \
    --config configs/default.yaml \
    --lead 6 \
    --task regression \
    --device cuda \
    --tuned-root data/tuned_wide/lstm

python scripts/compute_shap_tuned_lstm.py \
    --config configs/default.yaml \
    --lead 6 \
    --task regression \
    --model-dir data/tuned_wide/lstm/lead06_regression/models \
    --output-dir data/tuned_wide/lstm/lead06_regression/shap
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

import torch
import torch.nn as nn

from src.utils.config import load_config
from src.utils.io_utils import load_zarr
from src.utils.logging_utils import get_logger
from src.utils.preprocessing import (
    build_class_labels,
    build_lstm_sequences,
    train_val_test_split_temporal,
)
from src.shap_analysis.compute_shap import (
    select_background,
    get_gradient_explainer,
    compute_deep_shap,
    aggregate_lstm_shap,
    save_shap_dataset,
)

log = get_logger(__name__)


class _Unsqueeze(nn.Module):
    """Wrap a regression net so SHAP sees output shape (B, 1)."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out.unsqueeze(-1) if out.dim() == 1 else out


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def default_tuned_root(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "lstm"


def lead_task_dir(tuned_root: Path, lead: int, task: str) -> Path:
    return tuned_root / f"lead{lead:02d}_{task}"


def infer_model_dir(cfg: dict, tuned_root: str | None, lead: int, task: str, model_dir: str | None) -> Path:
    if model_dir is not None:
        return Path(model_dir)
    root = Path(tuned_root) if tuned_root is not None else default_tuned_root(cfg)
    return lead_task_dir(root, lead, task) / "models"


def infer_output_dir(cfg: dict, tuned_root: str | None, lead: int, task: str, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    root = Path(tuned_root) if tuned_root is not None else default_tuned_root(cfg)
    return lead_task_dir(root, lead, task) / "shap"


def infer_global_shap_dir(cfg: dict, global_shap_dir: str | None) -> Path | None:
    if global_shap_dir is None:
        return Path(cfg["experiment"]["output_dir"]) / "data" / "shap_tuned_wide" / "lstm"
    if str(global_shap_dir).lower() in {"none", "no", "false", "0"}:
        return None
    return Path(global_shap_dir)


# ---------------------------------------------------------------------------
# Data/model loading
# ---------------------------------------------------------------------------


def import_lstm_model(model_module: str):
    """Import the tuned LSTM model class.

    ``model_module`` may be ``lstm_model_tuned`` or a full module path such as
    ``src.models.lstm_model_tuned``.
    """
    module_path = model_module if "." in model_module else f"src.models.{model_module}"
    module = importlib.import_module(module_path)
    return getattr(module, "ENSOLSTMModel")


def load_tuned_model(cfg: dict, lead: int, task: str, model_dir: Path, model_module: str, device: str | None):
    ENSOLSTMModel = import_lstm_model(model_module)
    model = ENSOLSTMModel(cfg, lead, task, device=device)
    model.load(model_dir)
    model.net.eval()

    return model


def load_basin_and_target(cfg: dict) -> tuple[pd.DataFrame, xr.DataArray]:
    processed = Path(cfg["data"]["processed_dir"])
    ds_basin = load_zarr(processed / "predictors_basin.zarr").compute()
    df_basin = ds_basin.to_dataframe().dropna()
    ds_nino = load_zarr(processed / "target_nino34.zarr").compute()
    nino34 = ds_nino["nino34"] if "nino34" in ds_nino else list(ds_nino.data_vars.values())[0]
    return df_basin, nino34


def standardize_train_only(X_tr: np.ndarray, X_val: np.ndarray, X_te: np.ndarray):
    mean = np.nanmean(X_tr, axis=(0, 1), keepdims=True)
    std = np.nanstd(X_tr, axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (X_tr - mean) / std, (X_val - mean) / std, (X_te - mean) / std, mean, std


def build_lstm_test_data(cfg: dict, lead: int, seq_len: int):
    df_basin, nino34 = load_basin_and_target(cfg)
    X, y_reg, var_names, times = build_lstm_sequences(df_basin, nino34, lead, seq_len)
    (
        X_tr, y_tr, t_tr,
        X_val, y_val, t_val,
        X_te, y_te, t_te,
    ) = train_val_test_split_temporal(
        X,
        y_reg,
        times,
        train_years=tuple(cfg["data"]["train_years"]),
        val_years=tuple(cfg["data"]["val_years"]),
        test_years=tuple(cfg["data"]["test_years"]),
    )
    X_tr_s, X_val_s, X_te_s, mean, std = standardize_train_only(X_tr, X_val, X_te)
    return X_tr_s, y_tr, t_tr, X_val_s, y_val, t_val, X_te_s, y_te, t_te, list(var_names), mean, std


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _feature_importance(ds: xr.Dataset) -> pd.Series:
    imp = ds["abs_shap"].mean("time").to_pandas()
    return imp.sort_values(ascending=False)


def plot_top_bar(ds: xr.Dataset, out: Path, top_n: int = 20) -> None:
    imp = _feature_importance(ds).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(imp))))
    ax.barh(imp.index.astype(str), imp.values)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Top {top_n} LSTM predictors by mean |SHAP|")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_beeswarm_like(ds: xr.Dataset, out: Path, top_n: int = 20, max_points: int = 800, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    imp = _feature_importance(ds).head(top_n)
    features = list(imp.index.astype(str))
    shap_df = ds["shap_values"].sel(feature=features).to_pandas()
    pred = ds["prediction"].to_pandas()
    if len(shap_df) > max_points:
        keep = rng.choice(len(shap_df), size=max_points, replace=False)
        shap_df = shap_df.iloc[keep]
        pred = pred.iloc[keep]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(features))))
    ytick_labels = []
    for j, feat in enumerate(reversed(features)):
        vals = shap_df[feat].values.astype(float)
        jitter = rng.normal(loc=j, scale=0.08, size=len(vals))
        sc = ax.scatter(vals, jitter, c=pred.values, s=16, alpha=0.65, cmap="coolwarm", linewidths=0)
        ytick_labels.append(feat)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(list(reversed(features)))
    ax.set_xlabel("Signed SHAP value")
    ax.set_title("SHAP beeswarm-style distribution, colored by prediction")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Predicted Niño3.4")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_violin(ds: xr.Dataset, out: Path, top_n: int = 15) -> None:
    imp = _feature_importance(ds).head(top_n)
    features = list(imp.index.astype(str))
    data = [ds["shap_values"].sel(feature=f).values.astype(float) for f in features]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.3 * len(features))))
    parts = ax.violinplot(data[::-1], vert=False, showmeans=True, showextrema=True)
    for body in parts.get("bodies", []):
        body.set_alpha(0.65)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_yticks(range(1, len(features) + 1))
    ax.set_yticklabels(features[::-1])
    ax.set_xlabel("Signed SHAP value")
    ax.set_title("Distribution of signed SHAP values")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_monthly_heatmap(ds: xr.Dataset, out: Path, top_n: int = 18) -> None:
    imp = _feature_importance(ds).head(top_n)
    features = list(imp.index.astype(str))
    monthly = ds["abs_shap"].sel(feature=features).groupby("time.month").mean("time").to_pandas()
    monthly = monthly.reindex(range(1, 13))
    arr = monthly[features].T.values
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(features))))
    im = ax.imshow(arr, aspect="auto", cmap="magma")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=45)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_title("Monthly mean |SHAP| by predictor")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_prediction_shap_scatter(ds: xr.Dataset, out: Path, top_n: int = 6) -> None:
    imp = _feature_importance(ds).head(top_n)
    features = list(imp.index.astype(str))
    ncols = 2
    nrows = int(np.ceil(len(features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, max(4, 3.2 * nrows)), squeeze=False)
    pred = ds["prediction"].values.astype(float)
    for ax, feat in zip(axes.ravel(), features):
        vals = ds["shap_values"].sel(feature=feat).values.astype(float)
        ax.scatter(pred, vals, s=20, alpha=0.65)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(feat)
        ax.set_xlabel("Predicted Niño3.4")
        ax.set_ylabel("SHAP")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(features):]:
        ax.axis("off")
    fig.suptitle("SHAP response curves against model prediction")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_phase_composite(ds: xr.Dataset, out: Path, threshold: float = 0.5, top_n: int = 15) -> None:
    imp = _feature_importance(ds).head(top_n)
    features = list(imp.index.astype(str))
    pred = ds["prediction"].values.astype(float)
    abs_shap = ds["abs_shap"].sel(feature=features)
    groups = {
        "La Niña": pred <= -threshold,
        "Neutral": np.abs(pred) < threshold,
        "El Niño": pred >= threshold,
    }
    rows = []
    for name, mask in groups.items():
        if int(mask.sum()) == 0:
            rows.append(pd.Series(np.nan, index=features, name=name))
        else:
            rows.append(abs_shap.isel(time=mask).mean("time").to_pandas().rename(name))
    df = pd.concat(rows, axis=1).T
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(df.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=75, ha="right")
    ax.set_title(f"Phase-composite |SHAP|, threshold ±{threshold}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean |SHAP|")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_proxy_interactions(ds: xr.Dataset, fig_out: Path, csv_out: Path, top_n: int = 20) -> None:
    imp = _feature_importance(ds).head(top_n)
    features = list(imp.index.astype(str))
    s = ds["shap_values"].sel(feature=features).to_pandas().astype(float)
    # This is not a true Shapley interaction. It is a lightweight diagnostic:
    # mean co-importance of two features across samples.
    abs_arr = np.abs(s.values)
    co = abs_arr.T @ abs_arr / max(1, abs_arr.shape[0])
    np.fill_diagonal(co, np.nan)
    pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            pairs.append({
                "feature_i": features[i],
                "feature_j": features[j],
                "mean_abs_shap_product": float(co[i, j]),
                "signed_shap_corr": float(np.corrcoef(s.iloc[:, i], s.iloc[:, j])[0, 1]),
            })
    pair_df = pd.DataFrame(pairs).sort_values("mean_abs_shap_product", ascending=False)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(csv_out, index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(co, cmap="plasma")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(features)
    ax.set_title("Proxy SHAP interaction map\nmean(|SHAP_i| × |SHAP_j|)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean co-importance")
    fig.tight_layout()
    fig.savefig(fig_out, dpi=180)
    plt.close(fig)


def plot_event_waterfalls(ds: xr.Dataset, out: Path, top_n: int = 12, threshold: float = 0.5) -> None:
    pred = ds["prediction"].values.astype(float)
    event_indices = [int(np.argmax(pred)), int(np.argmin(pred)), int(np.argmin(np.abs(pred)))]
    titles = ["Warmest prediction", "Coldest prediction", "Most neutral prediction"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    for ax, idx, title in zip(axes.ravel(), event_indices, titles):
        sv = ds["shap_values"].isel(time=idx).to_pandas().astype(float)
        top = pd.concat([sv.sort_values().head(top_n // 2), sv.sort_values().tail(top_n // 2)])
        top = top.sort_values()
        colors = ["tab:blue" if v < 0 else "tab:red" for v in top.values]
        ax.barh(top.index.astype(str), top.values, color=colors, alpha=0.8)
        ax.axvline(0, color="black", lw=0.8)
        time_label = pd.to_datetime(ds.time.values[idx]).strftime("%Y-%m")
        ax.set_title(f"{title}\ninit={time_label}, pred={pred[idx]:.2f}")
        ax.set_xlabel("Signed SHAP")
    fig.suptitle("Event-level attribution cards")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_all_shap_plots(ds: xr.Dataset, fig_dir: Path, lead: int, task: str, top_n: int, seed: int) -> None:
    fig_dir = _ensure_dir(fig_dir)
    prefix = f"lstm_lead{lead:02d}_{task}"
    plot_top_bar(ds, fig_dir / f"{prefix}_top{top_n}_bar.png", top_n=top_n)
    plot_beeswarm_like(ds, fig_dir / f"{prefix}_beeswarm.png", top_n=top_n, seed=seed)
    plot_violin(ds, fig_dir / f"{prefix}_violin_distributions.png", top_n=min(top_n, 15))
    plot_monthly_heatmap(ds, fig_dir / f"{prefix}_monthly_shap_heatmap.png", top_n=min(top_n, 18))
    plot_prediction_shap_scatter(ds, fig_dir / f"{prefix}_shap_vs_prediction.png", top_n=6)
    plot_phase_composite(ds, fig_dir / f"{prefix}_phase_composite.png", top_n=min(top_n, 15))
    plot_event_waterfalls(ds, fig_dir / f"{prefix}_event_waterfalls.png", top_n=12)
    save_proxy_interactions(
        ds,
        fig_dir / f"{prefix}_proxy_interaction_heatmap.png",
        fig_dir.parent / f"{prefix}_proxy_interactions.csv",
        top_n=min(top_n, 20),
    )


# ---------------------------------------------------------------------------
# Main SHAP routine
# ---------------------------------------------------------------------------


def run_lstm_tuned_shap(
    cfg_path: str,
    lead: int,
    task: str,
    device: str | None,
    tuned_root: str | None,
    model_dir: str | None,
    model_module: str,
    output_dir: str | None,
    global_shap_dir: str | None,
    max_eval_samples: int | None,
    background_samples: int | None,
    batch_size: int | None,
    random_eval: bool,
    make_plots: bool,
    top_n: int,
) -> Path:
    cfg = load_config(cfg_path)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    model_dir_path = infer_model_dir(cfg, tuned_root, lead, task, model_dir)
    out_dir = _ensure_dir(infer_output_dir(cfg, tuned_root, lead, task, output_dir))
    fig_dir = _ensure_dir(out_dir / "figures")
    global_dir = infer_global_shap_dir(cfg, global_shap_dir)
    if global_dir is not None:
        global_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading tuned LSTM model from %s", model_dir_path)
    model = load_tuned_model(cfg, lead, task, model_dir_path, model_module, device)
    actual_device = model.device
    seq_len = int(model._meta.get("seq_len", cfg["model"]["lstm"].get("sequence_length", 12)))
    log.info("Loaded tuned LSTM: seq_len=%d device=%s meta=%s", seq_len, actual_device, model._meta)

    X_tr, y_tr, t_tr, X_val, y_val, t_val, X_te, y_te, t_te, var_names, X_mean, X_std = build_lstm_test_data(cfg, lead, seq_len)
    log.info("Rebuilt LSTM arrays: train=%d val=%d test=%d features=%d", len(t_tr), len(t_val), len(t_te), len(var_names))

    if task == "classification":
        y_eval_truth = build_class_labels(y_te)
    else:
        y_eval_truth = y_te

    max_eval = max_eval_samples if max_eval_samples is not None else int(cfg.get("shap", {}).get("max_eval_samples", len(X_te)))
    if max_eval <= 0 or max_eval >= len(X_te):
        eval_idx = np.arange(len(X_te))
    else:
        rng = np.random.default_rng(seed)
        if random_eval:
            eval_idx = rng.choice(len(X_te), size=max_eval, replace=False)
            eval_idx = np.sort(eval_idx)
        else:
            eval_idx = np.arange(max_eval)
    X_eval = X_te[eval_idx]
    y_eval = y_eval_truth[eval_idx]
    times_eval = pd.DatetimeIndex(t_te)[eval_idx]

    n_bg = background_samples if background_samples is not None else int(cfg.get("shap", {}).get("background_samples", 100))
    n_bg = min(n_bg, len(X_tr))
    X_bg = select_background(X_tr, n_bg, seed=seed)
    X_bg_t = torch.tensor(X_bg, dtype=torch.float32).to(actual_device)

    net_for_shap = _Unsqueeze(model.net) if task == "regression" else model.net


    torch.backends.cudnn.enabled = False
    
    explainer = get_gradient_explainer(net_for_shap, X_bg_t)
    deep_batch = batch_size if batch_size is not None else int(cfg.get("shap", {}).get("deep_batch_size", 50))

    log.info("Computing Gradient SHAP: eval=%d background=%d batch=%d", len(X_eval), len(X_bg), deep_batch)
    shap_3d, base_val = compute_deep_shap(explainer, X_eval, deep_batch, actual_device)
    shap_2d = aggregate_lstm_shap(shap_3d)
    log.info("SHAP raw shape=%s aggregated shape=%s", np.shape(shap_3d), np.shape(shap_2d))

    preds = model.predict(X_eval)
    if task == "classification":
        pred_for_store = preds.argmax(axis=1).astype(np.float32)
    else:
        pred_for_store = preds.astype(np.float32)

    store_path = save_shap_dataset(
        shap_2d,
        pred_for_store,
        base_val,
        var_names,
        times_eval,
        "lstm",
        lead,
        task,
        out_dir,
    )

    ds = xr.open_zarr(str(store_path)).load()
    ds.attrs["tuned_model_dir"] = str(model_dir_path)
    ds.attrs["tuned_run_dir"] = str(model_dir_path.parent)
    ds.attrs["sequence_length"] = seq_len
    ds.attrs["n_background_samples"] = int(n_bg)
    ds.attrs["n_eval_samples"] = int(len(X_eval))
    ds.attrs["random_eval"] = bool(random_eval)
    # Rewrite with extended attrs.
    shutil.rmtree(store_path)
    ds.to_zarr(str(store_path), mode="w")

    # Save a small CSV that is easy to inspect without opening Zarr.
    pred_df = pd.DataFrame({
        "init_time": times_eval,
        "target_time": times_eval + pd.DateOffset(months=lead),
        "observed_nino34": y_te[eval_idx],
    })
    if task == "regression":
        pred_df["prediction_nino34"] = pred_for_store
        pred_df["error"] = pred_df["prediction_nino34"] - pred_df["observed_nino34"]
    else:
        pred_df["predicted_class"] = pred_for_store
        pred_df["observed_class"] = y_eval
    pred_df.to_csv(out_dir / f"lstm_lead{lead:02d}_{task}_shap_eval_predictions.csv", index=False)

    meta = {
        "lead": lead,
        "task": task,
        "model_dir": str(model_dir_path),
        "output_dir": str(out_dir),
        "store_path": str(store_path),
        "sequence_length": seq_len,
        "var_names": var_names,
        "n_background_samples": int(n_bg),
        "n_eval_samples": int(len(X_eval)),
        "base_value": np.asarray(base_val).tolist(),
    }
    with open(out_dir / f"lstm_lead{lead:02d}_{task}_shap_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if make_plots:
        ds = xr.open_zarr(str(store_path)).load()
        make_all_shap_plots(ds, fig_dir, lead, task, top_n=top_n, seed=seed)
        log.info("Saved SHAP figures -> %s", fig_dir)

    # Optional central copy for compile script convenience.
    if global_dir is not None:
        central_path = global_dir / store_path.name
        if central_path.exists():
            shutil.rmtree(central_path)
        shutil.copytree(store_path, central_path)
        log.info("Copied SHAP store -> %s", central_path)

    log.info("Saved tuned LSTM SHAP store -> %s", store_path)
    return store_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SHAP for tuned LSTM models and save rich diagnostics")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lead", type=int, required=True, choices=[3, 6, 12])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--tuned-root", default=None, help="Root folder containing leadXX_task tuned LSTM folders. Default: output/data/tuned_wide/lstm")
    parser.add_argument("--model-dir", default=None, help="Explicit models/ directory for this lead/task")
    parser.add_argument("--model-module", default="lstm_model_tuned", help="src.models module containing ENSOLSTMModel")
    parser.add_argument("--output-dir", default=None, help="Per-run SHAP output dir. Default: tuned_root/leadXX_task/shap")
    parser.add_argument("--global-shap-dir", default=None, help="Central copy dir. Default: output/data/shap_tuned_wide/lstm. Use 'none' to disable.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="0 or negative means full test period")
    parser.add_argument("--background-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="SHAP eval batch size")
    parser.add_argument("--random-eval", action="store_true", help="Randomly sample test samples instead of first N")
    parser.add_argument("--no-plots", action="store_true", help="Compute SHAP only, no figure gallery")
    parser.add_argument("--top-n", type=int, default=20, help="Number of features in SHAP plots")
    args = parser.parse_args()

    run_lstm_tuned_shap(
        cfg_path=args.config,
        lead=args.lead,
        task=args.task,
        device=args.device,
        tuned_root=args.tuned_root,
        model_dir=args.model_dir,
        model_module=args.model_module,
        output_dir=args.output_dir,
        global_shap_dir=args.global_shap_dir,
        max_eval_samples=args.max_eval_samples,
        background_samples=args.background_samples,
        batch_size=args.batch_size,
        random_eval=args.random_eval,
        make_plots=not args.no_plots,
        top_n=args.top_n,
    )
