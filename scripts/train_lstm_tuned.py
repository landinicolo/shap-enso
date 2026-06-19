"""Train and tune an LSTM ENSO forecast model for one lead time.

Drop-in replacement for ``scripts/train_lstm.py``.

Main changes compared with the original script
---------------------------------------------
* performs a basic expanding-window time-series CV search over LSTM
  hyperparameters before final training;
* keeps the final test years untouched until the very end;
* saves fold-level tuning results, trial summaries, best hyperparameters,
  final metrics, final predictions, and several diagnostic plots;
* writes tuned-run outputs to a separate directory by default so baseline
  models in data/models/lstm are not overwritten.

Examples
--------
    # Basic tuning with the default small grid
    python scripts/train_lstm.py --config configs/default.yaml --lead 6 --task regression

    # Limit the search to 12 sampled trials and 3 CV folds
    python scripts/train_lstm.py --lead 6 --task regression --max-trials 12 --cv-folds 3

    # Use a built-in grid preset defined inside this file
    python scripts/train_lstm.py --lead 6 --task regression --grid-preset regular

    # Use an inline grid without creating a separate JSON file
    python scripts/train_lstm.py --lead 6 --task regression \
      --tuning-grid-inline '{"hidden_size":[16,32,64],"num_layers":[1,2],"dropout":[0.1,0.3],"lr":[0.001,0.0003],"batch_size":[32]}'

Grid definition
---------------
The easiest workflow is to edit ``TUNING_GRID_PRESETS`` in this file and
run with ``--grid-preset tiny|small|regular|wide``. A separate JSON file is
still supported through ``--tuning-grid-json`` when desired.

Notes
-----
``sequence_length`` is special: if included in the grid, the script rebuilds
LSTM sequences for each candidate value. Other keys are passed into
``cfg['model']['lstm']`` for model training.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model_tuned import ENSOLSTMModel
from src.models.metrics import classification_metrics, regression_metrics
from src.utils.config import load_config
from src.utils.io_utils import load_zarr
from src.utils.logging_utils import get_logger
from src.utils.preprocessing import (
    build_class_labels,
    build_lstm_sequences,
    train_val_test_split_temporal,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Built-in tuning grids
# ---------------------------------------------------------------------------

# Edit these presets directly if you want custom tuning without creating a
# separate JSON file. The script keeps --tuning-grid-json for reproducibility,
# but for quick runs these presets are usually faster and less fussy.
#
# Valid keys are the LSTM config keys plus the special key "sequence_length".
# If "sequence_length" is included, the script rebuilds LSTM sequences for
# each candidate sequence length.
TUNING_GRID_PRESETS: dict[str, dict[str, list[Any]]] = {
    # Smoke-test sized. Useful when checking the script on a new machine.
    "tiny": {
        "hidden_size": [16, 32],
        "num_layers": [1],
        "dropout": [0.2],
        "lr": [1e-3],
        "weight_decay": [0.0],
        "batch_size": [32],
    },

    # Default: small enough for ENSO-scale monthly data, but not a toy.
    "small": {
        "hidden_size": [16, 32, 64],
        "num_layers": [1, 2],
        "dropout": [0.0, 0.2, 0.4],
        "lr": [1e-3, 3e-4],
        "weight_decay": [0.0, 1e-4],
        "batch_size": [16, 32],
    },

    # Adds sequence length as a tuned hyperparameter. More expensive because
    # sequence tensors must be rebuilt for each sequence length.
    "regular": {
        "hidden_size": [32, 64,128],
        "num_layers": [1, 2],
        "dropout": [0.0, 0.2, 0.4],
        "lr": [1e-3, 3e-4],
        "weight_decay": [0.0, 1e-4],
        "batch_size": [16, 32],
        "sequence_length": [6, 9, 12],
    },

    # Wider search for longer batch/PBS runs. Use with --max-trials unless
    # you really want the full grid.
    "wide": {
        "hidden_size": [32, 64, 128, 256],
        "num_layers": [1, 2],
        "dropout": [0.0, 0.1, 0.3, 0.5],
        "lr": [1e-3, 5e-4, 3e-4, 1e-4],
        "weight_decay": [0.0, 1e-5, 1e-4],
        "batch_size": [16, 32, 64],
        "sequence_length": [6, 9, 12],
    },
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def set_global_seed(seed: int) -> None:
    """Best-effort reproducibility for numpy, random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def apply_lstm_overrides(cfg: dict, params: dict[str, Any]) -> dict:
    """Return a deep-copied config with LSTM hyperparameter overrides."""
    new_cfg = copy.deepcopy(cfg)
    new_cfg.setdefault("model", {}).setdefault("lstm", {})
    for key, value in params.items():
        if key == "sequence_length":
            new_cfg["model"]["lstm"]["sequence_length"] = int(value)
        else:
            new_cfg["model"]["lstm"][key] = value
    return new_cfg


def standardize_train_only(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Standardize LSTM inputs using only the training partition."""
    X_mean = X_train.mean(axis=(0, 1), keepdims=True)
    X_std = X_train.std(axis=(0, 1), keepdims=True)
    X_std = np.where(X_std < 1e-8, 1.0, X_std)

    X_train_s = (X_train - X_mean) / X_std
    X_val_s = (X_val - X_mean) / X_std
    X_test_s = None if X_test is None else (X_test - X_mean) / X_std
    return X_train_s, X_val_s, X_test_s, X_mean, X_std


def validate_tuning_grid(grid: dict[str, Any]) -> dict[str, list[Any]]:
    """Validate and normalize a tuning grid loaded from any source."""
    if not isinstance(grid, dict):
        raise ValueError("Tuning grid must be an object mapping parameter names to lists.")
    out: dict[str, list[Any]] = {}
    for key, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid entry {key!r} must be a non-empty list.")
        out[key] = values
    return out


def default_tuning_grid(cfg: dict, preset: str = "small") -> dict[str, list[Any]]:
    """Return one of the built-in grids defined inside this file.

    If a preset is unknown, raise a helpful error listing available presets.
    """
    if preset not in TUNING_GRID_PRESETS:
        available = ", ".join(sorted(TUNING_GRID_PRESETS))
        raise ValueError(f"Unknown grid preset {preset!r}. Available presets: {available}")

    # Deep-ish copy to keep accidental mutations from leaking between runs.
    grid = copy.deepcopy(TUNING_GRID_PRESETS[preset])

    # If the config has a different batch size and the selected preset did not
    # explicitly include one, inherit it. Current presets do include batch_size,
    # but this keeps custom edits forgiving.
    lp = cfg.get("model", {}).get("lstm", {})
    grid.setdefault("batch_size", [int(lp.get("batch_size", 32))])
    return validate_tuning_grid(grid)


def load_tuning_grid(
    path: str | None,
    cfg: dict,
    preset: str = "small",
    inline_json: str | None = None,
) -> dict[str, list[Any]]:
    """Load a tuning grid from inline JSON, a JSON file, or a built-in preset.

    Priority order:
        1. --tuning-grid-inline
        2. --tuning-grid-json
        3. --grid-preset
    """
    if inline_json:
        try:
            grid = json.loads(inline_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse --tuning-grid-inline as JSON: {exc}") from exc
        return validate_tuning_grid(grid)

    if path is not None:
        with open(path, "r") as f:
            grid = json.load(f)
        return validate_tuning_grid(grid)

    return default_tuning_grid(cfg, preset=preset)

def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def sample_trials(trials: list[dict[str, Any]], max_trials: int | None, seed: int) -> list[dict[str, Any]]:
    if max_trials is None or max_trials <= 0 or max_trials >= len(trials):
        return trials
    rng = random.Random(seed)
    return rng.sample(trials, max_trials)


def selection_metric_name(metric: str) -> str:
    """Accept both 'corr' and 'val_corr' style metric names."""
    if metric in {"best_val_loss", "val_loss"}:
        return "best_val_loss"
    if metric.startswith("val_"):
        return metric
    return f"val_{metric}"


def higher_is_better(metric: str) -> bool:
    metric = metric.replace("val_", "")
    return metric not in {"rmse", "mae", "loss", "best_val_loss"}


def safe_metric_value(metrics: dict[str, Any], metric_col: str) -> float:
    val = metrics.get(metric_col, np.nan)
    try:
        return float(val)
    except Exception:
        return float("nan")


def make_expanding_cv_indices(n_samples: int, n_folds: int, min_train_fraction: float = 0.45):
    """Yield expanding-window train/validation indices.

    Example with 3 folds:
        train [0:45%], validate next block;
        train expands, validate next block;
        train expands, validate final block.
    """
    if n_folds < 2:
        raise ValueError("Use at least 2 CV folds.")
    min_train = max(24, int(math.floor(n_samples * min_train_fraction)))
    remaining = n_samples - min_train
    if remaining < n_folds:
        raise ValueError(
            f"Not enough samples for {n_folds} CV folds after min_train={min_train}. "
            f"Got n_samples={n_samples}."
        )
    val_size = remaining // n_folds
    folds = []
    for fold in range(n_folds):
        train_end = min_train + fold * val_size
        val_start = train_end
        val_end = n_samples if fold == n_folds - 1 else val_start + val_size
        if val_end <= val_start:
            continue
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((fold + 1, train_idx, val_idx))
    return folds


# ---------------------------------------------------------------------------
# Data loading and sequence construction
# ---------------------------------------------------------------------------

def load_basin_and_target(cfg: dict):
    processed = Path(cfg["data"]["processed_dir"])
    basin_path = processed / "predictors_basin.zarr"
    nino_path = processed / "target_nino34.zarr"
    for p in [basin_path, nino_path]:
        if not p.exists():
            raise FileNotFoundError(f"Zarr store not found: {p}\nRun scripts/run_preprocess.py first.")

    ds_basin = load_zarr(basin_path).compute()
    df_basin = ds_basin.to_dataframe().dropna()

    ds_nino = load_zarr(nino_path).compute()
    nino34 = ds_nino["nino34"] if "nino34" in ds_nino else list(ds_nino.data_vars.values())[0]
    return df_basin, nino34


def build_and_split(
    cfg: dict,
    df_basin: pd.DataFrame,
    nino34,
    lead: int,
    sequence_length: int,
):
    X, y_reg, var_names, times = build_lstm_sequences(df_basin, nino34, lead, sequence_length)
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
    return X_tr, y_tr, t_tr, X_val, y_val, t_val, X_te, y_te, t_te, var_names


def train_dev_arrays(X_tr, y_tr, t_tr, X_val, y_val, t_val):
    """Combine original train and validation periods for inner CV tuning."""
    X_dev = np.concatenate([X_tr, X_val], axis=0)
    y_dev = np.concatenate([y_tr, y_val], axis=0)
    t_dev = pd.DatetimeIndex(np.concatenate([np.asarray(t_tr), np.asarray(t_val)]))
    order = np.argsort(t_dev.values)
    return X_dev[order], y_dev[order], t_dev[order]


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def run_single_cv_fold(
    cfg: dict,
    lead: int,
    task: str,
    device: str | None,
    X_train: np.ndarray,
    y_train_reg: np.ndarray,
    X_val: np.ndarray,
    y_val_reg: np.ndarray,
    seed: int,
) -> dict[str, float]:
    set_global_seed(seed)
    X_train_s, X_val_s, _, _, _ = standardize_train_only(X_train, X_val)

    if task == "classification":
        y_train = build_class_labels(y_train_reg)
        y_val = build_class_labels(y_val_reg)
    else:
        y_train = y_train_reg
        y_val = y_val_reg

    model = ENSOLSTMModel(cfg, lead, task, device=device)
    return model.fit(X_train_s, y_train, X_val_s, y_val)


def tune_lstm(
    cfg: dict,
    df_basin: pd.DataFrame,
    nino34,
    lead: int,
    task: str,
    device: str | None,
    tuning_grid: dict[str, list[Any]],
    cv_folds: int,
    max_trials: int | None,
    selection_metric: str,
    out_dir: Path,
    seed: int,
    tuning_max_epochs: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run time-series CV tuning and save fold/trial results."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trials = expand_grid(tuning_grid)
    trials = sample_trials(all_trials, max_trials, seed)
    metric_col = selection_metric_name(selection_metric)
    maximize = higher_is_better(metric_col)

    log.info(
        "LSTM tuning: %d candidate trials selected from %d grid combinations; metric=%s (%s)",
        len(trials), len(all_trials), metric_col, "max" if maximize else "min",
    )

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    # Cache sequence tensors per sequence length so including sequence_length in
    # the grid does not rebuild the same arrays repeatedly.
    seq_cache: dict[int, tuple[Any, ...]] = {}

    for trial_id, raw_params in enumerate(trials, start=1):
        params = dict(raw_params)
        if tuning_max_epochs is not None:
            params["max_epochs"] = int(tuning_max_epochs)

        seq_len = int(params.get("sequence_length", cfg["model"]["lstm"]["sequence_length"]))
        if seq_len not in seq_cache:
            seq_cache[seq_len] = build_and_split(cfg, df_basin, nino34, lead, seq_len)

        X_tr, y_tr, t_tr, X_val, y_val, t_val, _, _, _, _ = seq_cache[seq_len]
        X_dev, y_dev, t_dev = train_dev_arrays(X_tr, y_tr, t_tr, X_val, y_val, t_val)
        folds = make_expanding_cv_indices(len(X_dev), cv_folds)

        trial_cfg = apply_lstm_overrides(cfg, params)
        fold_scores: list[float] = []

        for fold_id, train_idx, val_idx in folds:
            log.info("Trial %03d/%03d fold %d/%d params=%s", trial_id, len(trials), fold_id, cv_folds, params)
            metrics = run_single_cv_fold(
                trial_cfg,
                lead,
                task,
                device,
                X_dev[train_idx],
                y_dev[train_idx],
                X_dev[val_idx],
                y_dev[val_idx],
                seed=seed + trial_id * 100 + fold_id,
            )
            score = safe_metric_value(metrics, metric_col)
            fold_scores.append(score)

            fold_rows.append({
                "trial_id": trial_id,
                "fold": fold_id,
                "train_start": str(t_dev[train_idx[0]].date()),
                "train_end": str(t_dev[train_idx[-1]].date()),
                "val_start": str(t_dev[val_idx[0]].date()),
                "val_end": str(t_dev[val_idx[-1]].date()),
                "selection_metric": metric_col,
                "selection_score": score,
                **params,
                **metrics,
            })

        mean_score = float(np.nanmean(fold_scores))
        std_score = float(np.nanstd(fold_scores))
        summary_rows.append({
            "trial_id": trial_id,
            "selection_metric": metric_col,
            "mean_selection_score": mean_score,
            "std_selection_score": std_score,
            "n_folds": len(fold_scores),
            **params,
        })

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        "mean_selection_score",
        ascending=not maximize,
        na_position="last",
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))

    best_row = summary_df.iloc[0].to_dict()
    param_cols = list(tuning_grid.keys()) + (["max_epochs"] if tuning_max_epochs is not None else [])
    best_params = {k: best_row[k] for k in param_cols if k in best_row and pd.notna(best_row[k])}
    best_params = coerce_param_types(best_params)

    fold_path = out_dir / f"lstm_lead{lead:02d}_{task}_cv_folds.csv"
    summary_path = out_dir / f"lstm_lead{lead:02d}_{task}_tuning_summary.csv"
    best_path = out_dir / f"lstm_lead{lead:02d}_{task}_best_params.json"
    grid_path = out_dir / f"lstm_lead{lead:02d}_{task}_tuning_grid.json"

    fold_df.to_csv(fold_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    with open(grid_path, "w") as f:
        json.dump(tuning_grid, f, indent=2)

    log.info("Saved fold-level tuning results → %s", fold_path)
    log.info("Saved trial summary → %s", summary_path)
    log.info("Saved best params → %s", best_path)

    make_tuning_plots(fold_df, summary_df, metric_col, maximize, out_dir, lead, task, list(tuning_grid.keys()))
    return best_params, fold_df, summary_df


def coerce_param_types(params: dict[str, Any]) -> dict[str, Any]:
    """Make JSON/pandas values safe for config use."""
    int_keys = {"hidden_size", "num_layers", "batch_size", "max_epochs", "sequence_length", "patience"}
    float_keys = {"dropout", "lr", "weight_decay", "grad_clip", "min_lr", "scheduler_factor"}
    out: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            value = value.item()
        if key in int_keys:
            out[key] = int(value)
        elif key in float_keys:
            out[key] = float(value)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Plotting for tuning results
# ---------------------------------------------------------------------------

def _metric_label(metric_col: str) -> str:
    return metric_col.replace("val_", "validation ").replace("_", " ")


def make_tuning_plots(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metric_col: str,
    maximize: bool,
    out_dir: Path,
    lead: int,
    task: str,
    tuned_params: list[str],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    y_label = _metric_label(metric_col)

    # 1) Ranked top trials.
    top = summary_df.sort_values("rank").head(20).copy()
    top = top.sort_values("mean_selection_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(top))))
    labels = [f"trial {int(t)}" for t in top["trial_id"]]
    ax.barh(labels, top["mean_selection_score"])
    ax.set_xlabel(f"Mean {y_label}")
    ax.set_ylabel("Candidate")
    ax.set_title(f"LSTM tuning ranked trials, lead {lead:02d}, {task}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_lead{lead:02d}_{task}_ranked_trials.png", dpi=180)
    plt.close(fig)

    # 2) Fold scores by trial, with mean overlay.
    fig, ax = plt.subplots(figsize=(10, 5))
    for fold_id, part in fold_df.groupby("fold"):
        ax.scatter(part["trial_id"], part["selection_score"], label=f"fold {fold_id}", s=24, alpha=0.75)
    ax.plot(summary_df["trial_id"], summary_df["mean_selection_score"], marker="o", linewidth=1.5, label="CV mean")
    ax.set_xlabel("Trial id")
    ax.set_ylabel(y_label)
    ax.set_title(f"Fold-level CV scores, LSTM lead {lead:02d}, {task}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / f"lstm_lead{lead:02d}_{task}_fold_scores.png", dpi=180)
    plt.close(fig)

    # 3) Parameter boxplots. This is a compact way to see which knobs matter.
    params_available = [p for p in tuned_params if p in fold_df.columns]
    if params_available:
        n = len(params_available)
        ncols = min(3, n)
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for ax, param in zip(axes.ravel(), params_available):
            groups = []
            labels = []
            for value, part in fold_df.groupby(param, dropna=False):
                groups.append(part["selection_score"].dropna().values)
                labels.append(str(value))
            if groups:
                ax.boxplot(groups, labels=labels, showmeans=True)
            ax.set_title(param)
            ax.set_xlabel(param)
            ax.set_ylabel(y_label)
            ax.tick_params(axis="x", rotation=35)
            ax.grid(axis="y", alpha=0.3)
        for ax in axes.ravel()[len(params_available):]:
            ax.axis("off")
        fig.suptitle(f"Hyperparameter effects, LSTM lead {lead:02d}, {task}", y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / f"lstm_lead{lead:02d}_{task}_param_boxplots.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # 4) Heatmap for hidden_size × dropout if both are tuned.
    if {"hidden_size", "dropout"}.issubset(summary_df.columns):
        pivot = summary_df.pivot_table(
            index="hidden_size",
            columns="dropout",
            values="mean_selection_score",
            aggfunc="mean",
        ).sort_index()
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_xlabel("dropout")
        ax.set_ylabel("hidden_size")
        ax.set_title(f"Mean {y_label}: hidden size × dropout")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.3g}", ha="center", va="center", fontsize=8)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(y_label)
        fig.tight_layout()
        fig.savefig(fig_dir / f"lstm_lead{lead:02d}_{task}_hidden_dropout_heatmap.png", dpi=180)
        plt.close(fig)

    # 5) Learning-rate comparison if lr was tuned.
    if "lr" in summary_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped = summary_df.groupby("lr")["mean_selection_score"]
        xs = []
        means = []
        stds = []
        for lr, values in grouped:
            xs.append(float(lr))
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=0)))
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        means = np.asarray(means)[order]
        stds = np.asarray(stds)[order]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel(f"Mean {y_label}")
        ax.set_title(f"Learning-rate sensitivity, LSTM lead {lead:02d}, {task}")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"lstm_lead{lead:02d}_{task}_lr_sensitivity.png", dpi=180)
        plt.close(fig)

    log.info("Saved tuning plots → %s", fig_dir)


# ---------------------------------------------------------------------------
# Final training and evaluation
# ---------------------------------------------------------------------------

def train_final_model(
    cfg: dict,
    df_basin: pd.DataFrame,
    nino34,
    lead: int,
    task: str,
    device: str | None,
    model_dir: Path,
    prediction_dir: Path,
    seed: int,
) -> dict[str, Any]:
    seq_len = int(cfg["model"]["lstm"]["sequence_length"])
    log.info("Building final LSTM sequences lead=%02d seq_len=%d", lead, seq_len)
    (
        X_tr, y_tr, t_tr,
        X_val, y_val, t_val,
        X_te, y_te, t_te,
        var_names,
    ) = build_and_split(cfg, df_basin, nino34, lead, seq_len)

    log.info("Final split train=%d val=%d test=%d", len(t_tr), len(t_val), len(t_te))

    if task == "classification":
        y_tr_fit = build_class_labels(y_tr)
        y_val_fit = build_class_labels(y_val)
        y_te_eval = build_class_labels(y_te)
    else:
        y_tr_fit, y_val_fit, y_te_eval = y_tr, y_val, y_te

    X_tr_s, X_val_s, X_te_s, X_mean, X_std = standardize_train_only(X_tr, X_val, X_te)

    set_global_seed(seed)
    model = ENSOLSTMModel(cfg, lead, task, device=device)
    log.info("Training final LSTM on %s", model.device)
    val_metrics = model.fit(X_tr_s, y_tr_fit, X_val_s, y_val_fit)

    y_te_pred = model.predict(X_te_s)
    if task == "regression":
        test_metrics = regression_metrics(y_te_eval, y_te_pred)
        log.info("Final test rmse=%.4f corr=%.4f", test_metrics["rmse"], test_metrics["corr"])
    else:
        test_metrics = classification_metrics(y_te_eval, y_te_pred)
        log.info("Final test acc=%.4f bss=%.4f", test_metrics["accuracy"], test_metrics["bss"])

    model.save(model_dir)

    # Save predictions explicitly; this also fixes the earlier gap where only
    # metrics were persisted after training.
    prediction_dir.mkdir(parents=True, exist_ok=True)
    pred_path = prediction_dir / f"lstm_lead{lead:02d}_{task}_test_predictions.csv"
    pred_df = pd.DataFrame({
        "init_time": pd.DatetimeIndex(t_te),
        "target_time": pd.DatetimeIndex(t_te) + pd.DateOffset(months=lead),
        "observed_nino34": y_te,
    })
    if task == "regression":
        pred_df["prediction_nino34"] = y_te_pred
        pred_df["error"] = pred_df["prediction_nino34"] - pred_df["observed_nino34"]
    else:
        pred_df["p_la_nina"] = y_te_pred[:, 0]
        pred_df["p_neutral"] = y_te_pred[:, 1]
        pred_df["p_el_nino"] = y_te_pred[:, 2]
        pred_df["predicted_class"] = y_te_pred.argmax(axis=1)
        pred_df["observed_class"] = y_te_eval
    pred_df.to_csv(pred_path, index=False)
    log.info("Saved final test predictions → %s", pred_path)

    history_path = model_dir / f"lstm_lead{lead:02d}_{task}_training_history.csv"
    pd.DataFrame(model.training_history_).to_csv(history_path, index=False)
    log.info("Saved final training history → %s", history_path)

    return {
        "model_type": "lstm",
        "lead_months": lead,
        "task": task,
        "sequence_length": seq_len,
        "var_names": list(var_names),
        "norm_mean": X_mean.squeeze().tolist(),
        "norm_std": X_std.squeeze().tolist(),
        "prediction_file": str(pred_path),
        "training_history_file": str(history_path),
        **val_metrics,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    cfg_path: str,
    lead: int,
    task: str,
    device: str | None = None,
    no_tune: bool = False,
    cv_folds: int = 3,
    max_trials: int | None = 16,
    tuning_grid_json: str | None = None,
    tuning_grid_inline: str | None = None,
    grid_preset: str = "small",
    selection_metric: str | None = None,
    tuning_output_dir: str | None = None,
    tuning_max_epochs: int | None = None,
    run_output_dir: str | None = None,
) -> None:
    cfg = load_config(cfg_path)
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_global_seed(seed)

    out_root = Path(cfg["experiment"]["output_dir"])

    # Keep tuned runs away from the baseline outputs created by the original
    # train_lstm.py script.  The baseline script writes to:
    #   data/models/lstm/
    #   data/predictions/
    # This tuned script writes to a self-contained run directory by default:
    #   data/tuned/lstm/lead06_regression/
    #       models/
    #       predictions/
    #       metrics/
    #       tuning/
    if run_output_dir is None:
        run_dir = out_root / "data" / "tuned" / "lstm" / f"lead{lead:02d}_{task}"
    else:
        run_dir = Path(run_output_dir)
    model_dir = run_dir / "models"
    prediction_dir = run_dir / "predictions"
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if tuning_output_dir is None:
        tune_dir = run_dir / "tuning"
    else:
        tune_dir = Path(tuning_output_dir)

    df_basin, nino34 = load_basin_and_target(cfg)

    best_params: dict[str, Any] | None = None
    if not no_tune:
        grid = load_tuning_grid(
            tuning_grid_json,
            cfg,
            preset=grid_preset,
            inline_json=tuning_grid_inline,
        )
        if selection_metric is None:
            selection_metric = "corr" if task == "regression" else "bss"

        best_params, _, _ = tune_lstm(
            cfg=cfg,
            df_basin=df_basin,
            nino34=nino34,
            lead=lead,
            task=task,
            device=device,
            tuning_grid=grid,
            cv_folds=cv_folds,
            max_trials=max_trials,
            selection_metric=selection_metric,
            out_dir=tune_dir,
            seed=seed,
            tuning_max_epochs=tuning_max_epochs,
        )
        log.info("Best LSTM params selected by CV: %s", best_params)
        final_cfg = apply_lstm_overrides(cfg, best_params)
    else:
        log.info("Skipping tuning; using fixed LSTM config from %s", cfg_path)
        final_cfg = cfg

    final_metrics = train_final_model(
        final_cfg,
        df_basin,
        nino34,
        lead,
        task,
        device,
        model_dir,
        prediction_dir,
        seed=seed + 999,
    )

    metrics_path = metrics_dir / f"lstm_lead{lead:02d}_{task}_metrics.json"
    metrics_payload: dict[str, Any] = {
        **final_metrics,
        "tuned": not no_tune,
        "best_params": best_params,
        "run_output_dir": str(run_dir),
        "model_dir": str(model_dir),
        "prediction_dir": str(prediction_dir),
        "metrics_dir": str(metrics_dir),
        "tuning_output_dir": str(tune_dir) if not no_tune else None,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM ENSO model with optional basic CV tuning")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lead", type=int, required=True, choices=[3, 6, 12])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])

    parser.add_argument("--no-tune", action="store_true", help="Skip CV tuning and use config hyperparameters directly.")
    parser.add_argument("--cv-folds", type=int, default=3, help="Number of expanding-window CV folds.")
    parser.add_argument(
        "--max-trials",
        type=int,
        default=16,
        help="Maximum candidate hyperparameter combinations to evaluate. Use 0 to run the full grid.",
    )
    parser.add_argument("--tuning-grid-json", default=None, help="Optional JSON file defining the tuning grid.")
    parser.add_argument(
        "--tuning-grid-inline",
        default=None,
        help=(
            "Inline JSON object defining the tuning grid. "
            "Example: '{\"hidden_size\":[16,32],\"dropout\":[0.1,0.3],\"lr\":[0.001]}'"
        ),
    )
    parser.add_argument(
        "--grid-preset",
        default="small",
        choices=sorted(TUNING_GRID_PRESETS),
        help="Built-in tuning grid preset used when no inline/file grid is supplied.",
    )
    parser.add_argument(
        "--selection-metric",
        default=None,
        help="Metric used to select the best trial, e.g. corr, rmse, r2, bss, accuracy, best_val_loss.",
    )
    parser.add_argument("--tuning-output-dir", default=None, help="Optional directory for tuning CSV/JSON/figures.")
    parser.add_argument(
        "--run-output-dir",
        default=None,
        help=(
            "Optional root directory for all tuned-run outputs. "
            "Default: data/tuned/lstm/leadXX_task, separate from baseline data/models/lstm."
        ),
    )
    parser.add_argument(
        "--tuning-max-epochs",
        type=int,
        default=None,
        help="Optional max_epochs override used only during CV tuning to keep searches fast.",
    )

    args = parser.parse_args()
    main(
        cfg_path=args.config,
        lead=args.lead,
        task=args.task,
        device=args.device,
        no_tune=args.no_tune,
        cv_folds=args.cv_folds,
        max_trials=None if args.max_trials == 0 else args.max_trials,
        tuning_grid_json=args.tuning_grid_json,
        tuning_grid_inline=args.tuning_grid_inline,
        grid_preset=args.grid_preset,
        selection_metric=args.selection_metric,
        tuning_output_dir=args.tuning_output_dir,
        tuning_max_epochs=args.tuning_max_epochs,
        run_output_dir=args.run_output_dir,
    )
