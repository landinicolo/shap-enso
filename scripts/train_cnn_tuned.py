"""Tune and train CNN ENSO forecast models.

This is the tuned counterpart of ``scripts/train_cnn.py``. It imports
``src.models.cnn_model_tuned`` and writes all outputs under a tuned experiment
folder, leaving baseline ``data/models/cnn`` untouched.

Example
-------
python scripts/train_cnn_tuned.py \
    --config configs/default.yaml \
    --lead 6 \
    --task regression \
    --grid-preset wide \
    --cv-folds 3 \
    --max-trials 128 \
    --tuning-max-epochs 60 \
    --selection-metric r2 \
    --run-output-dir data/tuned_wide/cnn/lead06_regression
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_model_tuned import ENSOCNNModel
from src.models.metrics import classification_metrics, regression_metrics
from src.utils.config import load_config
from src.utils.io_utils import load_zarr
from src.utils.logging_utils import get_logger
from src.utils.preprocessing import (
    build_class_labels,
    build_cnn_tensors,
    train_val_test_split_temporal,
)

log = get_logger(__name__)


TUNING_GRID_PRESETS: dict[str, dict[str, list[Any]]] = {
    "tiny": {
        "channels": [[16, 32], [32, 64]],
        "kernel_size": [3],
        "dropout": [0.2],
        "head_hidden": [64, 128],
        "n_head_layers": [1],
        "lr": [1e-3],
        "weight_decay": [0.0],
        "batch_size": [16],
    },
    "small": {
        "channels": [[32, 64], [32, 64, 128], [64, 128]],
        "kernel_size": [3, 5],
        "dropout": [0.1, 0.3],
        "head_hidden": [128, 256],
        "n_head_layers": [1, 2],
        "lr": [1e-3, 3e-4],
        "weight_decay": [0.0, 1e-4],
        "batch_size": [16, 32],
    },
    "wide": {
        "channels": [[32, 64, 128], [64, 128, 256], [64, 128, 256, 512], [128, 256, 512]],
        "kernel_size": [3, 5],
        "dropout": [0.0, 0.2, 0.4],
        "head_hidden": [128, 256, 512],
        "n_head_layers": [1, 2],
        "lr": [1e-3, 5e-4, 3e-4],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "batch_size": [8, 16, 32],
        "patience": [12, 25],
        "grad_clip": [0.5, 1.0],
    },
}


def main(
    cfg_path: str,
    lead: int,
    task: str,
    device: str | None = None,
    grid_preset: str = "wide",
    tuning_grid_json: str | None = None,
    tuning_grid_inline: str | None = None,
    cv_folds: int = 3,
    max_trials: int = 128,
    selection_metric: str | None = None,
    tuning_max_epochs: int | None = 60,
    run_output_dir: str | None = None,
) -> None:
    cfg = load_config(cfg_path)
    seed = int(cfg["experiment"].get("seed", 42))

    processed = Path(cfg["data"]["processed_dir"])
    if run_output_dir is None:
        run_dir = Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "cnn" / f"lead{lead:02d}_{task}"
    else:
        run_dir = Path(run_output_dir)
    dirs = prepare_run_dirs(run_dir)

    grid = load_tuning_grid(grid_preset, tuning_grid_json, tuning_grid_inline)
    if tuning_max_epochs is not None:
        grid = copy.deepcopy(grid)
        grid.setdefault("max_epochs", [int(tuning_max_epochs)])

    grid_path = processed / "predictors.zarr"
    nino_path = processed / "target_nino34.zarr"
    for p in [grid_path, nino_path]:
        if not p.exists():
            raise FileNotFoundError(f"Zarr store not found: {p}\nRun scripts/run_preprocess.py first.")

    log.info("Loading gridded predictors ...")
    ds_anom = load_zarr(grid_path).compute()
    ds_nino = load_zarr(nino_path).compute()
    nino34 = ds_nino["nino34"] if "nino34" in ds_nino else list(ds_nino.data_vars.values())[0]

    n_lags = int(cfg["data"]["lag_months"]) + 1
    era5_vars = cfg["data"]["era5_variables"] + (["d20"] if "d20" in ds_anom else [])
    log.info("Building CNN tensors lead=%02d n_lags=%d vars=%s", lead, n_lags, era5_vars)
    X, y_reg, ch_names, times = build_cnn_tensors(ds_anom, nino34, lead, n_lags, era5_vars)
    times = pd.DatetimeIndex(times)
    log.info("Tensors X=%s n_channels=%d", X.shape, X.shape[1])

    (X_tr, y_tr, t_tr,
     X_val, y_val, t_val,
     X_te, y_te, t_te) = train_val_test_split_temporal(
        X, y_reg, times,
        train_years=tuple(cfg["data"]["train_years"]),
        val_years=tuple(cfg["data"]["val_years"]),
        test_years=tuple(cfg["data"]["test_years"]),
    )
    log.info("Split train=%d val=%d test=%d", len(t_tr), len(t_val), len(t_te))

    if task == "classification":
        y_tr_fit = build_class_labels(y_tr)
        y_val_fit = build_class_labels(y_val)
        y_te_eval = build_class_labels(y_te)
        default_metric = "bss"
    else:
        y_tr_fit, y_val_fit, y_te_eval = y_tr, y_val, y_te
        default_metric = "r2"
    selection_metric = selection_metric or default_metric

    X_dev = np.concatenate([X_tr, X_val], axis=0)
    y_dev_fit = np.concatenate([y_tr_fit, y_val_fit], axis=0)
    t_dev = pd.DatetimeIndex(np.concatenate([np.asarray(t_tr), np.asarray(t_val)]))

    tuning_result = tune_cnn(
        cfg=cfg,
        X_dev=X_dev,
        y_dev=y_dev_fit,
        times_dev=t_dev,
        grid=grid,
        lead=lead,
        task=task,
        cv_folds=cv_folds,
        max_trials=max_trials,
        selection_metric=selection_metric,
        seed=seed,
        tuning_dir=dirs["tuning"],
        device=device,
    )

    best_params = tuning_result["best_params"]
    log.info("Best params: %s", best_params)

    # Final train on original train split, early-stop on original validation split.
    final_cfg = patch_cnn_cfg(cfg, best_params)
    X_tr_s, X_val_s, X_te_s, X_mean, X_std = standardize_cnn_train_val_test(X_tr, X_val, X_te)

    model = ENSOCNNModel(final_cfg, lead, task, device=device)
    val_metrics = model.fit(X_tr_s, y_tr_fit, X_val_s, y_val_fit)

    y_te_pred = model.predict(X_te_s)
    if task == "regression":
        test_metrics = regression_metrics(y_te_eval, y_te_pred)
        log.info("Test rmse=%.4f corr=%.4f r2=%.4f", test_metrics["rmse"], test_metrics["corr"], test_metrics["r2"])
    else:
        test_metrics = classification_metrics(y_te_eval, y_te_pred)
        log.info("Test acc=%.4f f1=%.4f bss=%.4f", test_metrics["accuracy"], test_metrics["f1_macro"], test_metrics["bss"])

    model_path = model.save(dirs["models"])
    save_training_history(model, dirs["models"] / f"cnn_lead{lead:02d}_{task}_training_history.csv")
    save_predictions(dirs["predictions"], lead, task, t_te, y_te_eval, y_te_pred)

    metrics_path = dirs["metrics"] / f"cnn_lead{lead:02d}_{task}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model_type": "cnn_tuned",
            "lead_months": lead,
            "task": task,
            "model_path": str(model_path),
            "selection_metric": selection_metric,
            "run_output_dir": str(run_dir),
            "best_params": make_jsonable(best_params),
            "best_cv_summary": make_jsonable(tuning_result["best_summary"]),
            "n_channels": int(X.shape[1]),
            "channel_names": list(ch_names),
            "norm_mean": np.asarray(X_mean).squeeze().tolist(),
            "norm_std": np.asarray(X_std).squeeze().tolist(),
            **val_metrics,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }, f, indent=2)
    log.info("Metrics saved -> %s", metrics_path)


def prepare_run_dirs(run_dir: Path) -> dict[str, Path]:
    dirs = {
        "run": run_dir,
        "models": run_dir / "models",
        "metrics": run_dir / "metrics",
        "predictions": run_dir / "predictions",
        "tuning": run_dir / "tuning",
        "figures": run_dir / "tuning" / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def load_tuning_grid(preset: str, json_path: str | None, inline: str | None) -> dict[str, list[Any]]:
    if inline:
        return json.loads(inline)
    if json_path:
        with open(json_path) as f:
            return json.load(f)
    if preset not in TUNING_GRID_PRESETS:
        raise ValueError(f"Unknown grid preset {preset!r}. Available: {sorted(TUNING_GRID_PRESETS)}")
    return copy.deepcopy(TUNING_GRID_PRESETS[preset])


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def sample_candidates(candidates: list[dict[str, Any]], max_trials: int, seed: int) -> list[dict[str, Any]]:
    if max_trials is None or max_trials <= 0 or max_trials >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=max_trials, replace=False)
    return [candidates[int(i)] for i in idx]


def make_expanding_folds(n_samples: int, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds < 1:
        raise ValueError("cv_folds must be >= 1")
    blocks = np.array_split(np.arange(n_samples), n_folds + 1)
    folds = []
    for k in range(n_folds):
        tr = np.concatenate(blocks[: k + 1])
        va = blocks[k + 1]
        if len(tr) == 0 or len(va) == 0:
            continue
        folds.append((tr, va))
    if not folds:
        raise ValueError("Could not create non-empty expanding-window folds")
    return folds


def patch_cnn_cfg(cfg: dict, params: dict[str, Any]) -> dict:
    cfg_t = copy.deepcopy(cfg)
    cp = cfg_t["model"].setdefault("cnn", {})
    for k, v in params.items():
        cp[k] = v
    return cfg_t


def standardize_cnn_train_val(X_train: np.ndarray, X_val: np.ndarray):
    X_mean = np.nanmean(X_train, axis=(0, 2, 3), keepdims=True)
    X_std = np.nanstd(X_train, axis=(0, 2, 3), keepdims=True)
    X_std = np.where(X_std < 1e-8, 1.0, X_std)
    X_train_s = np.nan_to_num((X_train - X_mean) / X_std, nan=0.0)
    X_val_s = np.nan_to_num((X_val - X_mean) / X_std, nan=0.0)
    return X_train_s, X_val_s, X_mean, X_std


def standardize_cnn_train_val_test(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    X_train_s, X_val_s, X_mean, X_std = standardize_cnn_train_val(X_train, X_val)
    X_test_s = np.nan_to_num((X_test - X_mean) / X_std, nan=0.0)
    return X_train_s, X_val_s, X_test_s, X_mean, X_std


def metric_column(selection_metric: str) -> tuple[str, bool]:
    metric = selection_metric if selection_metric.startswith("val_") else f"val_{selection_metric}"
    lower = metric.lower()
    maximize = not any(x in lower for x in ["rmse", "mae", "loss", "logloss"])
    return metric, maximize


def tune_cnn(
    cfg: dict,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    times_dev: pd.DatetimeIndex,
    grid: dict[str, list[Any]],
    lead: int,
    task: str,
    cv_folds: int,
    max_trials: int,
    selection_metric: str,
    seed: int,
    tuning_dir: Path,
    device: str | None,
) -> dict[str, Any]:
    tuning_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = tuning_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(tuning_dir / f"cnn_lead{lead:02d}_{task}_tuning_grid.json", "w") as f:
        json.dump(make_jsonable(grid), f, indent=2)

    order = np.argsort(times_dev.values)
    X_dev = X_dev[order]
    y_dev = y_dev[order]
    times_dev = pd.DatetimeIndex(times_dev.values[order])

    candidates = sample_candidates(expand_grid(grid), max_trials=max_trials, seed=seed)
    folds = make_expanding_folds(len(X_dev), cv_folds)
    metric_col, maximize = metric_column(selection_metric)
    log.info("CNN tuning: %d candidates, %d CV folds, selection=%s", len(candidates), len(folds), metric_col)

    fold_rows: list[dict[str, Any]] = []
    for trial_id, params in enumerate(candidates, start=1):
        log.info("Trial %d/%d params=%s", trial_id, len(candidates), params)
        for fold_id, (tr_idx, va_idx) in enumerate(folds, start=1):
            X_tr_f, X_va_f, _, _ = standardize_cnn_train_val(X_dev[tr_idx], X_dev[va_idx])
            y_tr_f, y_va_f = y_dev[tr_idx], y_dev[va_idx]
            cfg_t = patch_cnn_cfg(cfg, params)
            model = ENSOCNNModel(cfg_t, lead, task, device=device)
            try:
                metrics = model.fit(X_tr_f, y_tr_f, X_va_f, y_va_f)
                row = {
                    "trial_id": trial_id,
                    "fold_id": fold_id,
                    "train_start": str(times_dev[tr_idx][0].date()),
                    "train_end": str(times_dev[tr_idx][-1].date()),
                    "val_start": str(times_dev[va_idx][0].date()),
                    "val_end": str(times_dev[va_idx][-1].date()),
                    **make_jsonable(params),
                    **make_jsonable(metrics),
                }
            except Exception as exc:
                log.exception("Trial %d fold %d failed", trial_id, fold_id)
                row = {
                    "trial_id": trial_id,
                    "fold_id": fold_id,
                    "error": repr(exc),
                    **make_jsonable(params),
                    metric_col: np.nan,
                }
            fold_rows.append(row)
            pd.DataFrame(fold_rows).to_csv(tuning_dir / f"cnn_lead{lead:02d}_{task}_cv_folds.csv", index=False)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize_trials(fold_df, metric_col=metric_col, maximize=maximize)
    if summary_df.empty:
        raise RuntimeError("No valid CNN tuning trials completed.")

    fold_csv = tuning_dir / f"cnn_lead{lead:02d}_{task}_cv_folds.csv"
    summary_csv = tuning_dir / f"cnn_lead{lead:02d}_{task}_tuning_summary.csv"
    best_json = tuning_dir / f"cnn_lead{lead:02d}_{task}_best_params.json"
    fold_df.to_csv(fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    best_summary = summary_df.iloc[0].to_dict()
    param_cols = [c for c in grid.keys() if c in summary_df.columns]
    best_params = {k: best_summary[k] for k in param_cols}
    with open(best_json, "w") as f:
        json.dump({
            "selection_metric": selection_metric,
            "metric_column": metric_col,
            "maximize": maximize,
            "best_params": make_jsonable(best_params),
            "best_summary": make_jsonable(best_summary),
        }, f, indent=2)

    make_tuning_plots(summary_df, fold_df, metric_col, figures_dir, prefix=f"cnn_lead{lead:02d}_{task}")
    return {"best_params": best_params, "best_summary": best_summary, "summary_df": summary_df, "fold_df": fold_df}


def summarize_trials(fold_df: pd.DataFrame, metric_col: str, maximize: bool) -> pd.DataFrame:
    if metric_col not in fold_df.columns:
        raise ValueError(f"Selection metric {metric_col!r} not found. Available columns: {list(fold_df.columns)}")
    param_cols = [c for c in fold_df.columns if c not in {
        "trial_id", "fold_id", "train_start", "train_end", "val_start", "val_end", "error",
    } and not c.startswith("val_") and c not in {"best_val_loss"}]
    metric_cols = [c for c in fold_df.columns if c.startswith("val_") or c in {"best_val_loss"}]
    rows = []
    for trial_id, g in fold_df.groupby("trial_id"):
        row: dict[str, Any] = {"trial_id": trial_id, "n_folds": int(g[metric_col].notna().sum())}
        first = g.iloc[0]
        for p in param_cols:
            row[p] = first[p]
        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"mean_{m}"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"std_{m}"] = float(vals.std(ddof=0)) if vals.notna().any() else np.nan
        row["mean_selection_score"] = row.get(f"mean_{metric_col}", np.nan)
        row["std_selection_score"] = row.get(f"std_{metric_col}", np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.sort_values("mean_selection_score", ascending=not maximize, na_position="last").reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def make_tuning_plots(summary_df: pd.DataFrame, fold_df: pd.DataFrame, metric_col: str, figures_dir: Path, prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    score_col = "mean_selection_score"

    plt.figure(figsize=(9, 5))
    top = summary_df.head(40)
    plt.plot(top["rank"], top[score_col], marker="o")
    plt.xlabel("Trial rank")
    plt.ylabel(score_col)
    plt.title("CNN tuning: ranked trials")
    plt.tight_layout()
    plt.savefig(figures_dir / f"{prefix}_ranked_trials.png", dpi=180)
    plt.close()

    if metric_col in fold_df.columns:
        plt.figure(figsize=(10, 5))
        ok = fold_df.dropna(subset=[metric_col])
        plt.scatter(ok["trial_id"], ok[metric_col], alpha=0.7)
        plt.xlabel("Trial id")
        plt.ylabel(metric_col)
        plt.title("CNN tuning: fold scores")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_fold_scores.png", dpi=180)
        plt.close()

    param_cols = [c for c in ["hidden_size", "head_hidden", "dropout", "lr", "weight_decay", "batch_size", "kernel_size", "n_head_layers"] if c in summary_df.columns]
    if param_cols:
        n = min(len(param_cols), 6)
        fig, axes = plt.subplots(n, 1, figsize=(9, 3 * n))
        if n == 1:
            axes = [axes]
        for ax, p in zip(axes, param_cols[:n]):
            grouped = summary_df.groupby(p, dropna=False)[score_col].mean().sort_values()
            grouped.plot(kind="barh", ax=ax)
            ax.set_xlabel(score_col)
            ax.set_ylabel(p)
            ax.set_title(f"Mean score by {p}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_param_effects.png", dpi=180)
        plt.close()

    if {"head_hidden", "dropout"}.issubset(summary_df.columns):
        pivot = summary_df.pivot_table(index="head_hidden", columns="dropout", values=score_col, aggfunc="mean")
        if not pivot.empty:
            plt.figure(figsize=(8, 5))
            plt.imshow(pivot.values, aspect="auto")
            plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
            plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
            plt.xlabel("dropout")
            plt.ylabel("head_hidden")
            plt.title("CNN tuning: head size vs dropout")
            plt.colorbar(label=score_col)
            plt.tight_layout()
            plt.savefig(figures_dir / f"{prefix}_head_dropout_heatmap.png", dpi=180)
            plt.close()

    if {"mean_val_r2", "mean_val_rmse"}.issubset(summary_df.columns):
        plt.figure(figsize=(7, 5))
        plt.scatter(summary_df["mean_val_rmse"], summary_df["mean_val_r2"], alpha=0.8)
        plt.xlabel("Mean CV RMSE")
        plt.ylabel("Mean CV R2")
        plt.title("CNN tuning: RMSE vs R2")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_rmse_vs_r2.png", dpi=180)
        plt.close()


def save_training_history(model: ENSOCNNModel, path: Path) -> None:
    hist = getattr(model, "history", [])
    if hist:
        pd.DataFrame(hist).to_csv(path, index=False)


def save_predictions(pred_dir: Path, lead: int, task: str, init_times, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    init_times = pd.DatetimeIndex(init_times)
    df = pd.DataFrame({
        "init_time": init_times,
        "target_time": init_times + pd.DateOffset(months=int(lead)),
        "model_type": "cnn_tuned",
        "lead_months": int(lead),
        "task": task,
    })
    if task == "regression":
        df["observed"] = y_true
        df["prediction"] = y_pred
        df["error"] = df["prediction"] - df["observed"]
    else:
        df["observed_class"] = y_true
        proba = np.asarray(y_pred)
        df["p_la_nina"] = proba[:, 0]
        df["p_neutral"] = proba[:, 1]
        df["p_el_nino"] = proba[:, 2]
        df["predicted_class"] = np.argmax(proba, axis=1)
    out = pred_dir / f"cnn_lead{lead:02d}_{task}_test_predictions.csv"
    df.to_csv(out, index=False)
    log.info("Predictions saved -> %s", out)


def make_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune and train CNN ENSO model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lead", type=int, required=True, choices=[3, 6, 12])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--grid-preset", default="wide", choices=sorted(TUNING_GRID_PRESETS))
    parser.add_argument("--tuning-grid-json", default=None)
    parser.add_argument("--tuning-grid-inline", default=None)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=128, help="0 or negative means full grid")
    parser.add_argument("--selection-metric", default=None, help="Regression: r2/rmse/corr/mae. Classification: bss/accuracy/f1_macro.")
    parser.add_argument("--tuning-max-epochs", type=int, default=60)
    parser.add_argument("--run-output-dir", default=None)
    args = parser.parse_args()
    main(
        cfg_path=args.config,
        lead=args.lead,
        task=args.task,
        device=args.device,
        grid_preset=args.grid_preset,
        tuning_grid_json=args.tuning_grid_json,
        tuning_grid_inline=args.tuning_grid_inline,
        cv_folds=args.cv_folds,
        max_trials=args.max_trials,
        selection_metric=args.selection_metric,
        tuning_max_epochs=args.tuning_max_epochs,
        run_output_dir=args.run_output_dir,
    )
