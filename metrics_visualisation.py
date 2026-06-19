#!/usr/bin/env python3
"""
Visualize model metrics for CNN, LSTM, and XGB across lead times.

Input directory:

    /work/ext/st12/shap-enso/data/models/
        cnn/
        lstm/
        xgb/

Expected files:

    cnn_lead03_regression_metrics.json
    cnn_lead06_regression_metrics.json
    cnn_lead12_regression_metrics.json
    cnn_lead03_classification_metrics.json
    ...

Outputs are saved by default into the CURRENT JUPYTER WORKING DIRECTORY,
not into the model results directory.

The script produces:

    model_metrics_wide.csv
    model_metrics_long.csv

    regression_metrics_by_lead.png
    regression_metrics_by_lead.pdf

    classification_metrics_by_lead.png
    classification_metrics_by_lead.pdf

Optional overview heatmaps:

    regression_metrics_overview_heatmap.png
    classification_metrics_overview_heatmap.png

Run in Jupyter:

    %run visualize_model_metrics.py

Run with explicit package installation:

    %run visualize_model_metrics.py --install

Run from terminal:

    python visualize_model_metrics.py --install
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ================================================================
# Package handling
# ================================================================

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
}


def package_is_available(import_name: str) -> bool:
    return importlib.util.find_spec(import_name) is not None


def install_missing_packages() -> None:
    """
    Install missing packages.

    First tries a normal pip install.
    If that fails, tries pip install --user, which is often useful
    on shared Jupyter/HPC systems.
    """
    missing = [
        pip_name
        for import_name, pip_name in REQUIRED_PACKAGES.items()
        if not package_is_available(import_name)
    ]

    if not missing:
        return

    print(f"Missing packages detected: {', '.join(missing)}")
    print("Attempting to install them with pip...")

    base_cmd = [sys.executable, "-m", "pip", "install", *missing]

    try:
        subprocess.check_call(base_cmd)
    except subprocess.CalledProcessError:
        print("Normal pip install failed. Trying pip install --user...")
        user_cmd = [sys.executable, "-m", "pip", "install", "--user", *missing]
        subprocess.check_call(user_cmd)


def import_scientific_stack(install: bool = False) -> None:
    if install:
        install_missing_packages()

    still_missing = [
        name for name in REQUIRED_PACKAGES
        if not package_is_available(name)
    ]

    if still_missing:
        missing_text = ", ".join(still_missing)
        raise ImportError(
            f"Missing required packages: {missing_text}\n\n"
            f"Run this in Jupyter:\n\n"
            f"    %run visualize_model_metrics.py --install\n\n"
            f"or install manually:\n\n"
            f"    python -m pip install numpy pandas matplotlib"
        )

    global np, pd, plt
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


# ================================================================
# Parsing JSON metric files
# ================================================================

FILE_RE = re.compile(
    r"(?P<model>[A-Za-z0-9_]+)_lead(?P<lead>\d+)_"
    r"(?P<task>classification|regression)_metrics\.json$"
)

NON_METRIC_KEYS = {
    "model_type",
    "lead_months",
    "task",
    "n_channels",
    "channel_names",
    "norm_mean",
    "norm_std",
    "normalization_mean",
    "normalization_std",
    "mean",
    "std",
    "seed",
    "random_seed",
    "created_at",
    "timestamp",
    "train_years",
    "val_years",
    "test_years",
    "train_size",
    "val_size",
    "test_size",
    "n_train",
    "n_val",
    "n_test",
    "best_epoch",
    "epoch",
    "epochs",
}

SKIP_METRIC_ENDINGS = (
    "_support",
    "_count",
    "_counts",
    "_n",
)

SKIP_METRIC_CONTAINS = (
    "confusion_matrix",
    "channel_names",
    "norm_mean",
    "norm_std",
)


def is_scalar_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False

    if isinstance(value, (int, float)):
        return math.isfinite(float(value))

    return False


def normalize_metric_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    name = name.replace("/", "_")
    name = name.replace(".", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = re.sub(r"__+", "_", name)
    return name.strip("_")


def should_skip_metric_name(metric_name: str) -> bool:
    m = metric_name.lower()

    if any(fragment in m for fragment in SKIP_METRIC_CONTAINS):
        return True

    if any(m.endswith(ending) for ending in SKIP_METRIC_ENDINGS):
        return True

    return False


def flatten_scalar_metrics(
    obj: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Recursively flatten scalar numeric metrics.

    Big arrays are deliberately skipped, so things like norm_mean,
    norm_std, channel_names, and confusion matrices do not become plot data.
    """
    metrics: Dict[str, float] = {}

    for key, value in obj.items():
        if key in NON_METRIC_KEYS:
            continue

        full_key = f"{prefix}_{key}" if prefix else key
        metric_name = normalize_metric_name(full_key)

        if should_skip_metric_name(metric_name):
            continue

        if is_scalar_number(value):
            metrics[metric_name] = float(value)

        elif isinstance(value, dict):
            nested_metrics = flatten_scalar_metrics(value, prefix=metric_name)
            metrics.update(nested_metrics)

        else:
            continue

    return metrics


def parse_metrics_file(path: Path) -> Optional[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    match = FILE_RE.match(path.name)

    if match:
        model_from_name = match.group("model")
        lead_from_name = int(match.group("lead"))
        task_from_name = match.group("task")
    else:
        model_from_name = path.parent.name
        lead_from_name = raw.get("lead_months")
        task_from_name = raw.get("task")

    model = raw.get("model_type", model_from_name)
    task = raw.get("task", task_from_name)
    lead = raw.get("lead_months", lead_from_name)

    if model is None or task is None or lead is None:
        print(f"Skipping file with missing model/task/lead metadata: {path}")
        return None

    metrics = flatten_scalar_metrics(raw)

    if not metrics:
        print(f"No scalar metrics found in: {path}")
        return None

    record: Dict[str, Any] = {
        "model": str(model).lower(),
        "task": str(task).lower(),
        "lead_months": int(lead),
        "source_file": str(path),
    }

    record.update(metrics)

    return record


def collect_metrics(
    root: Path,
    models: Iterable[str],
    leads: Iterable[int],
) -> "pd.DataFrame":
    records: List[Dict[str, Any]] = []

    allowed_models = {m.lower() for m in models}
    allowed_leads = {int(x) for x in leads}

    if not root.exists():
        raise FileNotFoundError(f"Input root directory does not exist: {root}")

    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue

        if model_dir.name.lower() not in allowed_models:
            continue

        json_files = sorted(model_dir.glob("*_metrics.json"))

        for json_path in json_files:
            parsed = parse_metrics_file(json_path)

            if parsed is None:
                continue

            if parsed["lead_months"] not in allowed_leads:
                continue

            records.append(parsed)

    if not records:
        raise FileNotFoundError(
            f"No metric JSON files found under {root} for models "
            f"{sorted(allowed_models)} and leads {sorted(allowed_leads)}."
        )

    df = pd.DataFrame(records)
    df = df.sort_values(["task", "model", "lead_months"]).reset_index(drop=True)

    return df


def wide_to_long(df: "pd.DataFrame") -> "pd.DataFrame":
    id_cols = ["model", "task", "lead_months", "source_file"]
    metric_cols = [c for c in df.columns if c not in id_cols]

    long_df = df.melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )

    long_df = long_df.dropna(subset=["value"]).copy()
    long_df["value"] = long_df["value"].astype(float)

    return long_df


# ================================================================
# Plot appearance
# ================================================================

MODEL_ORDER = ["cnn", "lstm", "xgb"]


def pretty_model_name(model: str) -> str:
    mapping = {
        "cnn": "CNN",
        "lstm": "LSTM",
        "xgb": "XGB",
        "xgboost": "XGBoost",
    }

    return mapping.get(model.lower(), model.upper())


def pretty_metric_name(metric: str) -> str:
    replacements = {
        "rmse": "RMSE",
        "mae": "MAE",
        "mse": "MSE",
        "corr": "Correlation",
        "r2": "R²",
        "f1": "F1",
        "auc": "AUC",
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "roc": "ROC",
        "pr": "PR",
        "val": "Validation",
        "test": "Test",
        "train": "Train",
        "best": "Best",
        "acc": "Accuracy",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "macro": "Macro",
        "weighted": "Weighted",
        "avg": "Average",
    }

    parts = metric.split("_")
    pretty_parts = []

    for part in parts:
        lower = part.lower()
        pretty_parts.append(replacements.get(lower, part.capitalize()))

    label = " ".join(pretty_parts)
    label = label.replace("R² Score", "R²")

    return label


def metric_sort_key(metric: str) -> Tuple[int, int, str]:
    m = metric.lower()

    if m.startswith("best_val_loss"):
        stage_order = 0
    elif m.startswith("train"):
        stage_order = 1
    elif m.startswith("val"):
        stage_order = 2
    elif m.startswith("test"):
        stage_order = 3
    else:
        stage_order = 4

    metric_keywords = [
        ("loss", 0),
        ("rmse", 1),
        ("mae", 2),
        ("mse", 3),
        ("corr", 4),
        ("r2", 5),
        ("accuracy", 6),
        ("acc", 7),
        ("precision", 8),
        ("recall", 9),
        ("f1", 10),
        ("auc", 11),
        ("auroc", 12),
        ("auprc", 13),
        ("brier", 14),
    ]

    metric_order = 99

    for keyword, order in metric_keywords:
        if keyword in m:
            metric_order = order
            break

    return stage_order, metric_order, m


def metric_direction(metric: str) -> Optional[str]:
    m = metric.lower()

    lower_is_better = [
        "loss",
        "rmse",
        "mae",
        "mse",
        "error",
        "brier",
        "logloss",
        "cross_entropy",
    ]

    higher_is_better = [
        "corr",
        "r2",
        "accuracy",
        "acc",
        "precision",
        "recall",
        "f1",
        "auc",
        "auroc",
        "auprc",
        "balanced_accuracy",
    ]

    if any(k in m for k in lower_is_better):
        return "lower is better"

    if any(k in m for k in higher_is_better):
        return "higher is better"

    return None


def apply_metric_axis(ax, metric: str, values: "pd.Series") -> None:
    m = metric.lower()

    finite_values = values[np.isfinite(values)]

    ax.set_ylabel(pretty_metric_name(metric))

    if finite_values.empty:
        return

    ymin = float(finite_values.min())
    ymax = float(finite_values.max())

    if "corr" in m:
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0.0, linewidth=1.0, alpha=0.35)
        return

    if "r2" in m:
        lower = min(-0.05, ymin - 0.05 * max(abs(ymin), 1.0))
        upper = 1.05
        ax.set_ylim(lower, upper)
        ax.axhline(0.0, linewidth=1.0, alpha=0.35)
        return

    probability_like = [
        "accuracy",
        "acc",
        "precision",
        "recall",
        "f1",
        "auc",
        "auroc",
        "auprc",
        "balanced_accuracy",
    ]

    if any(k in m for k in probability_like):
        if ymin >= 0.0 and ymax <= 1.1:
            ax.set_ylim(-0.03, 1.03)
        return

    nonnegative = [
        "loss",
        "rmse",
        "mae",
        "mse",
        "error",
        "brier",
        "logloss",
        "cross_entropy",
    ]

    if any(k in m for k in nonnegative):
        ax.set_ylim(bottom=0.0)
        return


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.3,
            "lines.markersize": 7,
        }
    )


# ================================================================
# Plotting
# ================================================================

def ordered_models(models: Iterable[str]) -> List[str]:
    models = list(models)

    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted([m for m in models if m not in MODEL_ORDER])

    return known + unknown


def plot_task_metrics(
    long_df: "pd.DataFrame",
    task: str,
    outdir: Path,
    formats: Iterable[str],
    leads: Iterable[int],
    dpi: int,
    show: bool = False,
) -> None:
    task_df = long_df[long_df["task"] == task].copy()

    if task_df.empty:
        print(f"No metrics found for task: {task}")
        return

    metrics = sorted(task_df["metric"].unique(), key=metric_sort_key)

    n_metrics = len(metrics)
    ncols = min(3, n_metrics)
    nrows = int(math.ceil(n_metrics / ncols))

    fig_width = 5.3 * ncols
    fig_height = 3.9 * nrows

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    model_order = ordered_models(task_df["model"].unique())
    lead_ticks = sorted({int(x) for x in leads})

    for idx, metric in enumerate(metrics):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        metric_df = task_df[task_df["metric"] == metric].copy()

        metric_df = (
            metric_df
            .groupby(["model", "lead_months"], as_index=False)["value"]
            .mean()
            .sort_values(["model", "lead_months"])
        )

        for model in model_order:
            model_df = metric_df[metric_df["model"] == model].copy()

            if model_df.empty:
                continue

            model_df = model_df.sort_values("lead_months")

            ax.plot(
                model_df["lead_months"],
                model_df["value"],
                marker="o",
                label=pretty_model_name(model),
            )

        ax.set_title(pretty_metric_name(metric), pad=10)
        ax.set_xlabel("Lead time, months")
        ax.set_xticks(lead_ticks)

        apply_metric_axis(ax, metric, metric_df["value"])

        direction = metric_direction(metric)

        if direction:
            ax.text(
                0.98,
                0.04,
                direction,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                alpha=0.72,
            )

    for idx in range(n_metrics, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    handles, labels = [], []

    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            break

    fig.suptitle(
        f"{task.capitalize()} metrics by lead time",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(1, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, -0.004),
        )

    fig.tight_layout(rect=[0.0, 0.045, 1.0, 0.965])

    for fmt in formats:
        output_path = outdir / f"{task}_metrics_by_lead.{fmt}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved plot: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_metric_overview_heatmap(
    long_df: "pd.DataFrame",
    task: str,
    outdir: Path,
    formats: Iterable[str],
    dpi: int,
    show: bool = False,
) -> None:
    """
    A compact overview plot.

    Values are z-scored per metric, so this heatmap is not for reading
    absolute metric values. It is for seeing broad patterns quickly.
    """
    task_df = long_df[long_df["task"] == task].copy()

    if task_df.empty:
        return

    pivot = task_df.pivot_table(
        index=["model", "lead_months"],
        columns="metric",
        values="value",
        aggfunc="mean",
    )

    if pivot.empty or pivot.shape[1] < 2:
        return

    sorted_columns = sorted(pivot.columns, key=metric_sort_key)
    pivot = pivot.reindex(sorted_columns, axis=1)

    sorted_index = sorted(
        pivot.index,
        key=lambda x: (
            MODEL_ORDER.index(x[0]) if x[0] in MODEL_ORDER else 999,
            x[1],
        ),
    )

    pivot = pivot.reindex(sorted_index)

    z = pivot.copy()

    for col in z.columns:
        std = z[col].std(skipna=True)
        mean = z[col].mean(skipna=True)

        if std is not None and np.isfinite(std) and std > 0:
            z[col] = (z[col] - mean) / std
        else:
            z[col] = 0.0

    fig_height = max(3.2, 0.46 * len(z.index) + 1.5)
    fig_width = max(8.5, 0.58 * len(z.columns) + 3.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(z.values, aspect="auto")

    ax.set_title(
        f"{task.capitalize()} metric overview, z-scored per metric",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    ax.set_xticks(np.arange(len(z.columns)))
    ax.set_xticklabels(
        [pretty_metric_name(c) for c in z.columns],
        rotation=45,
        ha="right",
    )

    y_labels = [
        f"{pretty_model_name(model)}, lead {lead}"
        for model, lead in z.index
    ]

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Relative value within metric, z-score")

    fig.tight_layout()

    for fmt in formats:
        output_path = outdir / f"{task}_metrics_overview_heatmap.{fmt}"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved overview plot: {output_path}")

    if show:
        plt.show()

    plt.close(fig)


# ================================================================
# Command-line / Jupyter arguments
# ================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize CNN, LSTM, and XGB metric JSON files by lead time."
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/work/ext/st12/shap-enso/data/models"),
        help="Root directory containing cnn, lstm, and xgb folders.",
    )

    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Output directory for plots and CSV files. "
            "Default: current Jupyter working directory."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["cnn", "lstm", "xgb"],
        help="Model folders to include.",
    )

    parser.add_argument(
        "--leads",
        nargs="+",
        type=int,
        default=[3, 6, 12],
        help="Lead months to include.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to save.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved PNG figures.",
    )

    parser.add_argument(
        "--install",
        action="store_true",
        help="Install missing Python packages automatically with pip.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )

    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip the z-scored overview heatmaps.",
    )

    # Jupyter injects arguments such as:
    # -f /path/to/kernel.json
    # parse_known_args keeps the script from crashing on those.
    args, unknown = parser.parse_known_args()

    if unknown:
        print(f"Ignoring unknown Jupyter/IPython arguments: {unknown}")

    return args


# ================================================================
# Main
# ================================================================

def main() -> None:
    args = parse_args()

    import_scientific_stack(install=args.install)
    configure_matplotlib()

    root = args.root.expanduser().resolve()

    if args.outdir is None:
        outdir = Path.cwd().resolve()
    else:
        outdir = args.outdir.expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Model metric visualization")
    print("=" * 72)
    print(f"Input metrics directory: {root}")
    print(f"Output directory:        {outdir}")
    print(f"Current working dir:     {Path.cwd().resolve()}")
    print(f"Models:                  {', '.join(args.models)}")
    print(f"Leads:                   {', '.join(str(x) for x in args.leads)}")
    print("=" * 72)

    wide_df = collect_metrics(
        root=root,
        models=args.models,
        leads=args.leads,
    )

    long_df = wide_to_long(wide_df)

    wide_csv = outdir / "model_metrics_wide.csv"
    long_csv = outdir / "model_metrics_long.csv"

    wide_df.to_csv(wide_csv, index=False)
    long_df.to_csv(long_csv, index=False)

    print(f"Saved CSV: {wide_csv}")
    print(f"Saved CSV: {long_csv}")

    print("\nDiscovered metric files:")
    discovered = (
        wide_df[["model", "task", "lead_months", "source_file"]])
 