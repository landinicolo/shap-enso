"""Evaluation metrics for regression and classification ENSO forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression skill scores.

    Returns:
        Dict with keys: rmse, mae, corr (Pearson), r2.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "corr": corr, "r2": r2}


def brier_skill_score(
    y_true_onehot: np.ndarray,
    y_pred_proba: np.ndarray,
) -> float:
    """Brier Skill Score relative to climatological forecast.

    BSS = 1 – BS_model / BS_climatology.  BSS > 0 means better than climatology.

    Args:
        y_true_onehot: One-hot encoded truth (n_samples, n_classes).
        y_pred_proba:  Predicted probabilities  (n_samples, n_classes).
    """
    clim      = y_true_onehot.mean(axis=0)           # climatological class frequencies
    ref_proba = np.broadcast_to(clim, y_true_onehot.shape)
    bs_model  = float(np.mean(np.sum((y_pred_proba - y_true_onehot) ** 2, axis=1)))
    bs_ref    = float(np.mean(np.sum((ref_proba    - y_true_onehot) ** 2, axis=1)))
    return 0.0 if bs_ref == 0 else 1.0 - bs_model / bs_ref


def classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """Compute classification skill scores for 3-class ENSO prediction.

    Args:
        y_true:       Integer class labels  (n_samples,)  — 0 La Niña, 1 Neutral, 2 El Niño.
        y_pred_proba: Predicted probabilities (n_samples, 3).

    Returns:
        Dict with keys: accuracy, f1_macro, auc_macro, bss.
    """
    y_pred  = y_pred_proba.argmax(axis=1)
    acc     = float(accuracy_score(y_true, y_pred))
    f1      = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    y_bin   = label_binarize(y_true, classes=[0, 1, 2])

    try:
        auc = float(roc_auc_score(y_bin, y_pred_proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    bss = brier_skill_score(y_bin, y_pred_proba)
    return {"accuracy": acc, "f1_macro": f1, "auc_macro": auc, "bss": bss}


def skill_vs_lead(results: dict[int, dict]) -> pd.DataFrame:
    """Assemble a lead-vs-skill DataFrame from per-lead metric dicts.

    Args:
        results: {lead_months: metrics_dict, ...}

    Returns:
        DataFrame indexed by lead_months with one column per metric.
    """
    rows = [{"lead_months": lead, **metrics} for lead, metrics in results.items()]
    return pd.DataFrame(rows).set_index("lead_months").sort_index()
