"""Publication-quality matplotlib/cartopy figures for SHAP analysis.

All functions return (fig, axes). figsize width is capped at 11 inches
(marimo WASM viewport constraint, Rule 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr

CMAP_DIV = "RdBu_r"
CMAP_SEQ = "YlOrRd"
MAX_W    = 11.0   # marimo WASM max figsize width


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

_BASIN_LABELS = {
    "sst_nino34": "SST 3.4",
    "sst_nino4":  "SST 4",
    "sst_nino3":  "SST 3",
    "sst_nino12": "SST 1+2",
    "d20_nino34": "D20 3.4",
    "d20_nino4":  "D20 4",
    "tauu_eq":    "τx eq",
    "olr_eq":     "OLR eq",
    "slp_eq":     "SLP eq",
}


def _shorten(name: str) -> str:
    """Shorten basin-index feature names like sst_nino34_lag0 → SST 3.4 −0mo."""
    parts = str(name).rsplit("_lag", 1)
    base  = _BASIN_LABELS.get(parts[0], parts[0])
    lag   = f" −{parts[1]}mo" if len(parts) > 1 else ""
    return f"{base}{lag}"


def clean_labels(names: list[str] | np.ndarray) -> list[str]:
    return [_shorten(n) for n in names]


# ---------------------------------------------------------------------------
# Feature importance bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance_bar(
    importance: pd.Series,
    title: str = "Feature importance (mean |SHAP|)",
    top_n: int = 15,
    color: str = "steelblue",
) -> tuple:
    """Horizontal bar chart of top-N feature importances."""
    s      = importance.head(top_n).iloc[::-1]
    labels = clean_labels(s.index.tolist())

    fig, ax = plt.subplots(figsize=(8, max(3.0, top_n * 0.33)))
    ax.barh(labels, s.values, color=color, height=0.7)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# SHAP–prediction scatter
# ---------------------------------------------------------------------------

def plot_shap_scatter(
    ds: xr.Dataset,
    feature: str,
    title: str | None = None,
) -> tuple:
    """SHAP value vs. model prediction scatter, coloured by prediction."""
    features = list(ds.coords["feature"].values.astype(str))
    if feature not in features:
        raise ValueError(f"Feature '{feature}' not in store. Available: {features[:8]}")
    idx = features.index(feature)

    if "class" in ds["shap_values"].dims:
        sv = ds["shap_values"].isel(feature=idx, **{"class": 2}).values
    else:
        sv = ds["shap_values"].isel(feature=idx).values

    preds = ds["prediction"].values
    norm  = mcolors.TwoSlopeNorm(vcenter=0,
                                  vmin=min(preds.min(), -0.5),
                                  vmax=max(preds.max(),  0.5))

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(preds, sv, c=preds, cmap=CMAP_DIV, norm=norm, s=18, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Prediction (Niño3.4, °C)")
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("Model prediction (°C)")
    ax.set_ylabel(f"SHAP  ({_shorten(feature)})")
    ax.set_title(title or f"SHAP vs. prediction — {_shorten(feature)}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Seasonal heatmap
# ---------------------------------------------------------------------------

def plot_seasonal_heatmap(
    seasonal_df: pd.DataFrame,
    title: str = "Seasonal SHAP importance",
    top_n: int = 10,
) -> tuple:
    """Heatmap: top features (rows) × calendar month (cols).

    seasonal_df: output of seasonal_shap_mean() — shape (12, n_features).
    """
    annual   = seasonal_df.mean(axis=0).sort_values(ascending=False)
    top_cols = annual.head(top_n).index.tolist()
    mat      = seasonal_df[top_cols].values.T   # (top_n, 12)
    labels   = clean_labels(top_cols)
    months   = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    fig, ax = plt.subplots(figsize=(MAX_W, max(3.0, top_n * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap=CMAP_SEQ,
                   vmin=0, vmax=np.nanmax(mat))
    ax.set_xticks(range(12))
    ax.set_xticklabels(months)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Calendar month (init month)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Mean |SHAP|", fraction=0.025)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Spring predictability barrier line plot
# ---------------------------------------------------------------------------

def plot_spring_barrier(
    seasonal_df: pd.DataFrame,
    features: list[str] | None = None,
    title: str = "Spring predictability barrier in SHAP",
) -> tuple:
    """Seasonal SHAP cycle for selected features, with spring highlighted."""
    if features is None:
        annual   = seasonal_df.mean(axis=0).sort_values(ascending=False)
        features = annual.head(5).index.tolist()

    labels = clean_labels(features)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvspan(2, 4, alpha=0.12, color="orange", label="MAM (spring)")
    for feat, lbl in zip(features, labels):
        ax.plot(range(12), seasonal_df[feat].values, marker="o", ms=4, label=lbl)
    ax.set_xticks(range(12))
    ax.set_xticklabels(months, rotation=40, ha="right")
    ax.set_xlabel("Initialization month")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# ENSO asymmetry bar chart
# ---------------------------------------------------------------------------

def plot_enso_asymmetry(
    composite: dict[str, pd.Series],
    title: str = "El Niño vs. La Niña feature importance",
    top_n: int = 10,
) -> tuple:
    """Side-by-side bars: El Niño (red) and La Niña (blue) SHAP importance."""
    en = composite["elnino"]
    ln = composite["lanina"]

    top_en = set(en.nlargest(top_n).index)
    top_ln = set(ln.nlargest(top_n).index)
    feats  = sorted(top_en | top_ln, key=lambda f: -(en.get(f, 0) + ln.get(f, 0)))[:top_n]
    labels = clean_labels(feats)

    x, w = np.arange(len(feats)), 0.38
    fig, ax = plt.subplots(figsize=(MAX_W, 4))
    ax.bar(x - w / 2, [en.get(f, 0) for f in feats], width=w,
           color="#d62728", alpha=0.85, label="El Niño")
    ax.bar(x + w / 2, [ln.get(f, 0) for f in feats], width=w,
           color="#1f77b4", alpha=0.85, label="La Niña")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Lead-time importance heatmap
# ---------------------------------------------------------------------------

def plot_lead_importance_heatmap(
    lead_df: pd.DataFrame,
    title: str = "Feature importance vs. lead time",
    top_n: int = 12,
) -> tuple:
    """Heatmap: top features (rows) × lead months (cols), row-normalized."""
    annual    = lead_df.mean(axis=1).sort_values(ascending=False)
    top_feats = annual.head(top_n).index.tolist()
    mat       = lead_df.loc[top_feats].values        # (top_n, n_leads)
    labels    = clean_labels(top_feats)
    lead_labs = [f"{c} mo" for c in lead_df.columns]

    row_max = mat.max(axis=1, keepdims=True)
    normed  = np.where(row_max > 0, mat / row_max, 0.0)

    fig, ax = plt.subplots(figsize=(6, max(3.0, top_n * 0.42)))
    im = ax.imshow(normed, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(lead_labs)))
    ax.set_xticklabels(lead_labs)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Relative importance", fraction=0.06)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Spatial SHAP map (CNN)
# ---------------------------------------------------------------------------

def plot_spatial_shap(
    spatial_ds: xr.Dataset,
    var_name: str = "sst",
    lag: int = 0,
    title: str | None = None,
    vmax: float | None = None,
) -> tuple:
    """Cartopy map of mean |SHAP| for (var_name, lag). Falls back to imshow."""
    da   = spatial_ds["mean_abs_shap"].sel(var=var_name, lag=lag)
    lat  = da.coords["lat"].values
    lon  = da.coords["lon"].values
    data = da.values
    _title = title or f"Spatial SHAP — {var_name}  lag {lag}"

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        proj = ccrs.PlateCarree(central_longitude=200)
        fig, ax = plt.subplots(figsize=(MAX_W, 4.5), subplot_kw={"projection": proj})
        im = ax.pcolormesh(lon, lat, data, cmap=CMAP_SEQ, vmin=0,
                           vmax=vmax or data.max(),
                           transform=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.4)
        ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5)
        plt.colorbar(im, ax=ax, label="Mean |SHAP|", fraction=0.03, pad=0.07)
        ax.set_title(_title)
    except ImportError:
        fig, ax = plt.subplots(figsize=(MAX_W, 4))
        im = ax.pcolormesh(lon, lat, data, cmap=CMAP_SEQ, vmin=0,
                           vmax=vmax or data.max())
        plt.colorbar(im, ax=ax, label="Mean |SHAP|")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(_title)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Multi-model skill comparison (uses metrics CSV)
# ---------------------------------------------------------------------------

def plot_skill_vs_lead(
    metrics_df: pd.DataFrame,
    metric: str = "corr",
    title: str | None = None,
) -> tuple:
    """Line plot of skill metric vs. lead for each model type.

    metrics_df must have columns: model_type, lead_months, task, plus metric cols.
    """
    models = metrics_df["model_type"].unique()
    colors = {"xgboost": "#e377c2", "lstm": "#17becf", "cnn": "#bcbd22"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for mdl in models:
        sub = metrics_df[metrics_df["model_type"] == mdl].sort_values("lead_months")
        ax.plot(sub["lead_months"], sub[metric], marker="o", ms=5,
                label=mdl, color=colors.get(mdl))

    ax.set_xlabel("Lead time (months)")
    ax.set_ylabel(metric.upper())
    ax.set_xticks([3, 6, 12])
    ax.set_title(title or f"Skill ({metric}) vs. lead time")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax
