"""Intercompare tuned CNN, LSTM, and XGB performance and SHAP structure.

This script reads compiled tuned-model metrics and SHAP stores from:

    data/tuned_wide/{cnn,lstm,xgb}/

and creates a final model-comparison gallery:

Performance
-----------
- RMSE vs lead for all models, with persistence if available
- ACC/correlation vs lead for all models
- R² vs lead for all models
- model × lead heatmaps for RMSE, ACC, R²

SHAP comparison
---------------
Because CNN, LSTM, and XGB do not necessarily share identical feature names,
the script collapses features into broad predictor groups using simple name
parsing, e.g. sst_lag0 -> sst, d20_lag2 -> d20. It then compares normalized
mean |SHAP| shares across models and leads.

Outputs
-------
data/tuned_wide/model_intercomparison/
figures/tuned_wide/model_intercomparison/
"""

from __future__ import annotations

import argparse
import json
import re
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
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
MODELS_DEFAULT = ["cnn", "lstm", "xgb"]
LEADS_DEFAULT = [3, 6, 12]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_tuned_base(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide"


def default_output_dir(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "data" / "tuned_wide" / "model_intercomparison"


def default_fig_dir(cfg: dict) -> Path:
    return Path(cfg["experiment"]["output_dir"]) / "figures" / "tuned_wide" / "model_intercomparison"


def save_fig(fig, path: Path, dpi: int = 180):
    ensure_dir(path.parent)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compiled_metrics_path(base: Path, model: str, task: str) -> Path:
    return base / model / "compiled_analysis" / f"{model}_tuned_{task}_metrics.csv"


def compiled_predictions_path(base: Path, model: str, task: str) -> Path:
    return base / model / "compiled_analysis" / f"{model}_tuned_{task}_all_predictions.csv"


def shap_store_path(base: Path, model: str, lead: int, task: str) -> Path:
    prefix = {"cnn": "cnn", "lstm": "lstm", "xgb": "xgb"}[model]
    return base / model / f"lead{lead:02d}_{task}" / "shap" / f"{prefix}_lead{lead:02d}_{task}_shap.zarr"


def load_metrics(base: Path, models: list[str], task: str) -> pd.DataFrame:
    rows = []
    for model in models:
        p = compiled_metrics_path(base, model, task)
        if p.exists():
            df = pd.read_csv(p)
            df["model"] = model
            rows.append(df)
        else:
            log.warning("Missing compiled metrics: %s", p)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    if "model_type" in df:
        df["model_type_original"] = df["model_type"]
    df["model_type"] = df["model"]
    return df


def load_all_predictions(base: Path, models: list[str], task: str) -> pd.DataFrame:
    rows = []
    for model in models:
        p = compiled_predictions_path(base, model, task)
        if p.exists():
            df = pd.read_csv(p)
            df["model"] = model
            for col in ["init_time", "target_time"]:
                if col in df:
                    df[col] = pd.to_datetime(df[col])
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def open_shap(path: Path) -> xr.Dataset | None:
    if not path.exists():
        log.warning("Missing SHAP store: %s", path)
        return None
    return xr.open_zarr(str(path), consolidated=False).load()


def predictor_group(feature: str) -> str:
    f = str(feature).lower()
    # High-priority physical predictors.
    known = ["nino34", "nino", "sst", "d20", "z500", "tauu", "taux", "uwind", "u10", "olr", "slp", "msl", "ssh", "pr", "precip", "t2m"]
    for k in known:
        if re.search(rf"(^|[_\-/\s]){re.escape(k)}($|[_\-/\s0-9])", f):
            if k in {"taux"}:
                return "tauu"
            if k in {"msl"}:
                return "slp"
            if k in {"precip"}:
                return "pr"
            return k
    # Strip lag suffix and spatial hints if possible.
    f = re.sub(r"_lag\d+", "", f)
    f = re.sub(r"lag\d+", "", f)
    # Use first token as fallback.
    token = re.split(r"[_\-/\s:]+", f)[0]
    return token or "other"


def load_shap_group_importance(base: Path, models: list[str], leads: list[int], task: str) -> pd.DataFrame:
    rows = []
    for model in models:
        for lead in leads:
            ds = open_shap(shap_store_path(base, model, lead, task))
            if ds is None or "abs_shap" not in ds:
                continue
            imp = ds["abs_shap"].mean("time").to_pandas()
            tmp = pd.DataFrame({"feature": imp.index.astype(str), "mean_abs_shap": imp.values})
            tmp["predictor_group"] = tmp["feature"].map(predictor_group)
            grouped = tmp.groupby("predictor_group", as_index=False)["mean_abs_shap"].sum()
            total = grouped["mean_abs_shap"].sum()
            grouped["shap_share"] = grouped["mean_abs_shap"] / total if total > 0 else np.nan
            grouped["model"] = model
            grouped["lead_months"] = lead
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot_skill_curves(metrics: pd.DataFrame, fig_dir: Path, prefix: str):
    if metrics.empty:
        return
    df = metrics.sort_values(["model", "lead_months"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for model, g in df.groupby("model"):
        axes[0].plot(g["lead_months"], g["rmse"], marker="o", label=model.upper())
        if model == sorted(df["model"].unique())[0] and "persistence_rmse" in g:
            axes[0].plot(g["lead_months"], g["persistence_rmse"], marker="o", ls="--", color="gray", label="Persistence")
        axes[1].plot(g["lead_months"], g["corr"], marker="o", label=model.upper())
        if model == sorted(df["model"].unique())[0] and "persistence_corr" in g:
            axes[1].plot(g["lead_months"], g["persistence_corr"], marker="o", ls="--", color="gray", label="Persistence")
        if "r2" in g:
            axes[2].plot(g["lead_months"], g["r2"], marker="o", label=model.upper())
    axes[0].set_title("RMSE vs lead"); axes[0].set_ylabel("RMSE")
    axes[1].set_title("ACC/correlation vs lead"); axes[1].set_ylabel("ACC")
    axes[2].set_title("R² vs lead"); axes[2].set_ylabel("R²")
    for ax in axes:
        ax.set_xlabel("Lead time (months)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Tuned model performance comparison", y=1.02)
    save_fig(fig, fig_dir / f"{prefix}_skill_curves_all_models.png")


def plot_metric_heatmaps(metrics: pd.DataFrame, fig_dir: Path, prefix: str):
    if metrics.empty:
        return
    for metric in ["rmse", "corr", "r2", "rmse_skill_vs_persistence"]:
        if metric not in metrics:
            continue
        mat = metrics.pivot_table(index="model", columns="lead_months", values=metric, aggfunc="mean")
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        im = ax.imshow(mat.values, aspect="auto")
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels([m.upper() for m in mat.index])
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns.astype(str))
        ax.set_xlabel("Lead time (months)")
        ax.set_title(f"Model × lead {metric}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label=metric)
        save_fig(fig, fig_dir / f"{prefix}_model_lead_{metric}_heatmap.png")


def plot_shap_share_heatmap(shap_df: pd.DataFrame, fig_dir: Path, prefix: str, top_n: int):
    if shap_df.empty:
        return
    top = shap_df.groupby("predictor_group")["mean_abs_shap"].mean().sort_values(ascending=False).head(top_n).index
    work = shap_df[shap_df["predictor_group"].isin(top)].copy()
    work["row"] = work["model"].str.upper() + " lead " + work["lead_months"].astype(str)
    mat = work.pivot_table(index="row", columns="predictor_group", values="shap_share", aggfunc="sum").fillna(0.0)
    # Sort rows by model then lead.
    row_order = sorted(mat.index, key=lambda x: (x.split()[0], int(x.split()[-1])))
    mat = mat.loc[row_order]
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(mat.columns)), max(5, 0.34 * len(mat.index))))
    im = ax.imshow(mat.values, aspect="auto", origin="lower")
    ax.set_yticks(np.arange(len(mat.index))); ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_title("Normalized SHAP share by predictor group")
    fig.colorbar(im, ax=ax, label="Share of total mean |SHAP|")
    save_fig(fig, fig_dir / f"{prefix}_shap_predictor_share_heatmap.png")


def plot_model_diff_vs_reference(shap_df: pd.DataFrame, fig_dir: Path, prefix: str, reference_model: str, top_n: int):
    if shap_df.empty or reference_model not in set(shap_df["model"]):
        return
    top = shap_df.groupby("predictor_group")["mean_abs_shap"].mean().sort_values(ascending=False).head(top_n).index
    work = shap_df[shap_df["predictor_group"].isin(top)].copy()
    rows = []
    for lead, g in work.groupby("lead_months"):
        ref = g[g["model"] == reference_model].set_index("predictor_group")["shap_share"]
        for model, gm in g.groupby("model"):
            if model == reference_model:
                continue
            s = gm.set_index("predictor_group")["shap_share"]
            diff = s.reindex(top, fill_value=0.0) - ref.reindex(top, fill_value=0.0)
            for pred, val in diff.items():
                rows.append({"row": f"{model.upper()} - {reference_model.upper()} lead {lead}", "predictor_group": pred, "difference": val})
    if not rows:
        return
    ddf = pd.DataFrame(rows)
    mat = ddf.pivot_table(index="row", columns="predictor_group", values="difference", aggfunc="mean").fillna(0.0)
    lim = float(np.nanquantile(np.abs(mat.values), 0.98)) if np.isfinite(mat.values).any() else 1.0
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(mat.columns)), max(4, 0.34 * len(mat.index))))
    im = ax.imshow(mat.values, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_yticks(np.arange(len(mat.index))); ax.set_yticklabels(mat.index)
    ax.set_xticks(np.arange(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_title(f"SHAP predictor share difference vs {reference_model.upper()}")
    fig.colorbar(im, ax=ax, label="Δ normalized SHAP share")
    save_fig(fig, fig_dir / f"{prefix}_shap_difference_vs_{reference_model}.png")


def plot_rank_correlation_by_lead(shap_df: pd.DataFrame, fig_dir: Path, prefix: str):
    if shap_df.empty:
        return
    for lead, g in shap_df.groupby("lead_months"):
        mat = g.pivot_table(index="model", columns="predictor_group", values="shap_share", aggfunc="sum").fillna(0.0)
        if mat.shape[0] < 2:
            continue
        corr = mat.T.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(corr.columns))); ax.set_xticklabels([m.upper() for m in corr.columns], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(corr.index))); ax.set_yticklabels([m.upper() for m in corr.index])
        ax.set_title(f"SHAP predictor-rank agreement, lead {lead}")
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, label="Spearman r")
        save_fig(fig, fig_dir / f"{prefix}_shap_rank_correlation_lead{int(lead):02d}.png")


def plot_best_model_by_lead(metrics: pd.DataFrame, fig_dir: Path, prefix: str):
    if metrics.empty or "rmse" not in metrics:
        return
    rows = []
    for lead, g in metrics.groupby("lead_months"):
        best_rmse = g.sort_values("rmse").iloc[0]
        best_corr = g.sort_values("corr", ascending=False).iloc[0] if "corr" in g else best_rmse
        rows.append({"lead_months": lead, "best_rmse_model": best_rmse["model"], "best_rmse": best_rmse["rmse"], "best_corr_model": best_corr["model"], "best_corr": best_corr.get("corr", np.nan)})
    df = pd.DataFrame(rows).sort_values("lead_months")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["best_rmse"], width=0.36, label="Best RMSE")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, df["best_corr"], width=0.36, alpha=0.55, label="Best ACC")
    ax.set_xticks(x); ax.set_xticklabels([f"lead {l}" for l in df["lead_months"]])
    ax.set_ylabel("RMSE"); ax2.set_ylabel("ACC")
    labels = [f"RMSE: {m}\nACC: {c}" for m, c in zip(df["best_rmse_model"].str.upper(), df["best_corr_model"].str.upper())]
    for xi, label in zip(x, labels):
        ax.text(xi, ax.get_ylim()[1]*0.95, label, ha="center", va="top", fontsize=9)
    ax.set_title("Best tuned model by lead")
    save_fig(fig, fig_dir / f"{prefix}_best_model_by_lead.png")
    df.to_csv(fig_dir.parent / f"{prefix}_best_model_by_lead.csv", index=False)


def compare_models(cfg_path: str, task: str, tuned_base: str | Path | None, output_dir: str | Path | None, fig_dir: str | Path | None, models: Iterable[str], leads: Iterable[int], reference_model: str, top_n: int):
    cfg = load_config(cfg_path)
    base = Path(tuned_base) if tuned_base is not None else default_tuned_base(cfg)
    out = ensure_dir(Path(output_dir) if output_dir is not None else default_output_dir(cfg))
    figs = ensure_dir(Path(fig_dir) if fig_dir is not None else default_fig_dir(cfg))
    models = [m.lower() for m in models]
    leads = [int(x) for x in leads]
    prefix = f"tuned_{task}_all_models"

    metrics = load_metrics(base, models, task)
    preds = load_all_predictions(base, models, task)
    shap_df = load_shap_group_importance(base, models, leads, task)

    metrics.to_csv(out / f"{prefix}_metrics.csv", index=False)
    preds.to_csv(out / f"{prefix}_all_predictions.csv", index=False)
    shap_df.to_csv(out / f"{prefix}_shap_group_importance.csv", index=False)

    plot_skill_curves(metrics, figs, prefix)
    plot_metric_heatmaps(metrics, figs, prefix)
    plot_best_model_by_lead(metrics, figs, prefix)
    plot_shap_share_heatmap(shap_df, figs, prefix, top_n=top_n)
    plot_model_diff_vs_reference(shap_df, figs, prefix, reference_model=reference_model, top_n=top_n)
    plot_rank_correlation_by_lead(shap_df, figs, prefix)

    manifest = {
        "task": task,
        "models": models,
        "leads": leads,
        "tuned_base": str(base),
        "output_dir": str(out),
        "figure_dir": str(figs),
        "reference_model": reference_model,
        "figures": sorted(p.name for p in figs.glob("*.png")),
        "tables": sorted(p.name for p in out.glob("*.csv")),
    }
    with open(out / f"{prefix}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Saved model intercomparison -> %s", out)
    log.info("Saved figures -> %s", figs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intercompare tuned model performance and SHAP differences")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--tuned-base", default=None, help="Default: experiment output/data/tuned_wide")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--models", nargs="+", default=MODELS_DEFAULT, choices=["cnn", "lstm", "xgb"])
    parser.add_argument("--leads", nargs="+", type=int, default=LEADS_DEFAULT)
    parser.add_argument("--reference-model", default="cnn", choices=["cnn", "lstm", "xgb"])
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    compare_models(
        cfg_path=args.config,
        task=args.task,
        tuned_base=args.tuned_base,
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
        models=args.models,
        leads=args.leads,
        reference_model=args.reference_model,
        top_n=args.top_n,
    )
