"""SHAP computation for XGBoost (TreeExplainer), LSTM (GradientExplainer),
and CNN (DeepExplainer) ENSO forecast models.

Explainer selection rationale
------------------------------
- XGBoost  → shap.TreeExplainer   — exact Shapley values, no approximation needed
- LSTM     → shap.GradientExplainer — more stable than DeepExplainer for stateful
                                     LSTM layers (avoids gradient-shattering issues
                                     documented in Mamalakis et al. 2022 AIES)
- CNN      → shap.DeepExplainer   — stateless forward pass makes DeepLIFT reliable;
                                     returns spatial (lat × lon) attribution maps
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Background sample selection
# ---------------------------------------------------------------------------

def select_background(
    X: np.ndarray,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Randomly subsample *n_samples* rows from X for use as SHAP background.

    Background must come from the training split only (enforced by the caller).
    Uses a fixed seed for reproducibility across runs.
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx]


# ---------------------------------------------------------------------------
# Explainer factories
# ---------------------------------------------------------------------------

def get_tree_explainer(xgb_model_obj: Any) -> "shap.TreeExplainer":
    """Return a TreeExplainer wrapping the underlying XGBoost Booster.

    Args:
        xgb_model_obj: Fitted ENSOXGBModel (wrapper class) or raw XGBoost model.

    Returns:
        shap.TreeExplainer ready to call .shap_values() on numpy arrays.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed. Run: pip install 'shap>=0.44'")

    # Unwrap our model wrapper if needed
    inner = getattr(xgb_model_obj, "model", xgb_model_obj)
    return shap.TreeExplainer(inner)


def get_gradient_explainer(
    net: "torch.nn.Module",
    background: "torch.Tensor",
) -> "shap.GradientExplainer":
    """Return a GradientExplainer for a PyTorch LSTM.

    The model must be in eval mode before passing here.

    Args:
        net:        PyTorch nn.Module in eval mode.
        background: Tensor of shape (n_bg, seq_len, n_features).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed. Run: pip install 'shap>=0.44'")
    return shap.GradientExplainer(net, background)


def get_deep_explainer(
    net: "torch.nn.Module",
    background: "torch.Tensor",
) -> "shap.DeepExplainer":
    """Return a DeepExplainer for a PyTorch CNN.

    Args:
        net:        PyTorch nn.Module in eval mode.
        background: Tensor of shape (n_bg, n_channels, lat, lon).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed. Run: pip install 'shap>=0.44'")
    return shap.DeepExplainer(net, background)


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_tree_shap(
    explainer: "shap.TreeExplainer",
    X: np.ndarray,
    batch_size: int = 500,
) -> tuple[np.ndarray, float]:
    """Compute SHAP values for an XGBoost model in batches.

    Args:
        explainer:  TreeExplainer instance.
        X:          Feature matrix (n_samples, n_features).
        batch_size: Rows per batch (TreeExplainer is fast; 500 is fine).

    Returns:
        shap_values: (n, n_feat) for regression;
                     (n, n_feat, n_classes) for multi-class classification.
        base_value:  Scalar E[f(X)] from the explainer.
    """
    batches = []
    for start in range(0, len(X), batch_size):
        sv = explainer.shap_values(X[start : start + batch_size])
        batches.append(sv)
        log.debug("TreeSHAP batch %d/%d", min(start + batch_size, len(X)), len(X))

    shap_values = _concat_batches(batches)

    base = explainer.expected_value
    base_value = float(np.mean(base)) if hasattr(base, "__len__") else float(base)
    return shap_values, base_value


def compute_deep_shap(
    explainer: "shap.GradientExplainer | shap.DeepExplainer",
    X_np: np.ndarray,
    batch_size: int = 50,
    device: str = "cpu",
) -> tuple[np.ndarray, float]:
    """Compute SHAP values for a PyTorch model (LSTM or CNN) in batches.

    Converts numpy input to torch tensors internally; returns numpy output.

    Args:
        explainer:  GradientExplainer or DeepExplainer.
        X_np:       Input array (n_samples, ...).
        batch_size: Samples per batch — reduce if GPU OOM.
        device:     Torch device string.

    Returns:
        shap_values: Same spatial/sequence shape as X_np.
        base_value:  Scalar 0.0 (DeepExplainer / GradientExplainer do not
                     expose a single global base value).
    """
    import torch

    batches = []
    for start in range(0, len(X_np), batch_size):
        batch_np = X_np[start : start + batch_size]
        batch_t  = torch.tensor(batch_np, dtype=torch.float32).to(device)
        sv = explainer.shap_values(batch_t)
        sv_np = _to_numpy(sv)
        batches.append(sv_np)
        log.info(
            "DeepSHAP batch %d/%d",
            min(start + batch_size, len(X_np)), len(X_np),
        )

    shap_values = _concat_batches(batches)
    return shap_values, 0.0


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_lstm_shap(
    shap_vals: np.ndarray,
) -> np.ndarray:
    """Collapse LSTM SHAP from (N, seq_len, n_feat) → (N, n_feat).

    Sums the absolute SHAP values over the sequence axis so each feature's
    total contribution across all lag steps is captured.
    """
    if shap_vals.ndim == 3:
        return np.abs(shap_vals).sum(axis=1)   # sum |SHAP| over seq axis
    return shap_vals                            # already 2-D


def aggregate_cnn_shap_spatial(
    shap_vals: np.ndarray,
) -> np.ndarray:
    """Collapse CNN spatial SHAP (N, C, lat, lon) → (N, C) by mean |SHAP|.

    Used to obtain a per-sample, per-channel feature importance score that
    can be compared with basin-index SHAP values.
    """
    if shap_vals.ndim == 4:
        return np.abs(shap_vals).mean(axis=(-2, -1))   # mean over lat, lon
    return shap_vals


def project_shap_to_grid(
    shap_vals: np.ndarray,
    variables: list[str],
    n_lags: int,
    lat: np.ndarray,
    lon: np.ndarray,
) -> xr.DataArray:
    """Compute mean |SHAP| spatial map and reshape to (variable, lag, lat, lon).

    Args:
        shap_vals: CNN SHAP array (N, n_vars*n_lags, lat, lon).
        variables: List of variable names (length n_vars).
        n_lags:    Number of lag steps.
        lat, lon:  Coordinate arrays.

    Returns:
        DataArray with dims ['variable', 'lag', 'lat', 'lon'].
    """
    # Mean |SHAP| over samples → (n_channels, lat, lon)
    mean_abs = np.abs(shap_vals).mean(axis=0)

    n_vars = len(variables)
    assert mean_abs.shape[0] == n_vars * n_lags, (
        f"Expected {n_vars * n_lags} channels, got {mean_abs.shape[0]}"
    )

    # Reshape to (n_vars, n_lags, lat, lon)
    mean_abs_4d = mean_abs.reshape(n_vars, n_lags, len(lat), len(lon))

    return xr.DataArray(
        mean_abs_4d,
        coords={
            "var": variables,    # named 'var' to avoid conflict with xarray's .variable property
            "lag": np.arange(n_lags),
            "lat": lat,
            "lon": lon,
        },
        dims=["var", "lag", "lat", "lon"],
        name="mean_abs_shap",
        attrs={"long_name": "Mean absolute SHAP value", "units": "index units"},
    )


# ---------------------------------------------------------------------------
# Save SHAP output to Zarr
# ---------------------------------------------------------------------------

def save_shap_dataset(
    shap_vals: np.ndarray,
    predictions: np.ndarray,
    base_value: float,
    feature_names: list[str],
    times: pd.DatetimeIndex,
    model_type: str,
    lead: int,
    task: str,
    output_dir: str | Path,
) -> Path:
    """Save SHAP values and metadata to a Zarr store.

    Schema
    ------
    Regression:
        shap_values (time, feature)   — signed SHAP values
        abs_shap    (time, feature)   — |SHAP| for easy ranking
        prediction  (time,)           — model output for each sample

    Classification (extra class dimension):
        shap_values (time, feature, class)
        abs_shap    (time, feature)        — mean |SHAP| across classes

    Global attributes: model_type, lead_months, task, base_value,
                       explainer_type, created.

    Args:
        shap_vals:     (n, n_feat) or (n, n_feat, n_cls) float32 array.
        predictions:   (n,) model predictions.
        base_value:    SHAP expected value E[f(X)].
        feature_names: List of feature name strings.
        times:         DatetimeIndex of initialisation times.
        model_type:    'xgboost' | 'lstm' | 'cnn'.
        lead:          Lead time in months.
        task:          'regression' | 'classification'.
        output_dir:    Directory for SHAP Zarr stores.

    Returns:
        Path to the saved Zarr store.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store_path = output_dir / f"{model_type}_lead{lead:02d}_{task}_shap.zarr"

    coords: dict[str, Any] = {
        "time":    times,
        "feature": np.array(feature_names, dtype=str),
    }

    if shap_vals.ndim == 3:
        # Classification: (n, n_feat, n_cls)
        n_cls = shap_vals.shape[2]
        coords["class"] = np.arange(n_cls)
        sv_da   = xr.DataArray(
            shap_vals.astype(np.float32),
            coords={k: coords[k] for k in ["time", "feature", "class"]},
            dims=["time", "feature", "class"],
            name="shap_values",
        )
        abs_da  = xr.DataArray(
            np.abs(shap_vals).mean(axis=-1).astype(np.float32),
            coords={k: coords[k] for k in ["time", "feature"]},
            dims=["time", "feature"],
            name="abs_shap",
        )
    else:
        # Regression: (n, n_feat)
        sv_da  = xr.DataArray(
            shap_vals.astype(np.float32),
            coords={k: coords[k] for k in ["time", "feature"]},
            dims=["time", "feature"],
            name="shap_values",
        )
        abs_da = xr.DataArray(
            np.abs(shap_vals).astype(np.float32),
            coords={k: coords[k] for k in ["time", "feature"]},
            dims=["time", "feature"],
            name="abs_shap",
        )

    pred_da = xr.DataArray(
        predictions.astype(np.float32),
        coords={"time": times},
        dims=["time"],
        name="prediction",
    )

    ds = xr.Dataset({"shap_values": sv_da, "abs_shap": abs_da, "prediction": pred_da})
    ds.attrs.update({
        "model_type":     model_type,
        "lead_months":    lead,
        "task":           task,
        "base_value":     base_value,
        "n_features":     len(feature_names),
        "n_samples":      len(times),
        "created":        str(date.today()),
    })

    from src.utils.io_utils import save_zarr
    save_zarr(ds, store_path, chunks={"time": 120, "feature": len(feature_names)})
    log.info("Saved SHAP dataset → %s", store_path)
    return store_path


def save_spatial_shap(
    spatial_da: xr.DataArray,
    model_type: str,
    lead: int,
    task: str,
    output_dir: str | Path,
) -> Path:
    """Save the spatial SHAP map (variable, lag, lat, lon) to Zarr.

    Args:
        spatial_da: DataArray from project_shap_to_grid().
        model_type: Model identifier string.
        lead:       Lead time in months.
        task:       'regression' | 'classification'.
        output_dir: SHAP output directory.

    Returns:
        Path to the saved Zarr store.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / f"{model_type}_lead{lead:02d}_{task}_spatial_shap.zarr"
    ds = spatial_da.to_dataset()
    ds.attrs.update({"model_type": model_type, "lead_months": lead, "task": task,
                     "created": str(date.today())})
    from src.utils.io_utils import save_zarr
    save_zarr(ds, store_path, chunks={"lat": 31, "lon": 86})
    log.info("Saved spatial SHAP map → %s", store_path)
    return store_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(sv: Any) -> Any:
    """Convert SHAP output to numpy (handles tensors and lists of tensors)."""
    if isinstance(sv, list):
        return [_to_numpy(x) for x in sv]
    if hasattr(sv, "cpu"):          # torch tensor
        return sv.detach().cpu().numpy()
    return np.asarray(sv)


def _concat_batches(batches: list[Any]) -> np.ndarray:
    """Concatenate a list of (possibly list-of-array) SHAP batch outputs."""
    first = batches[0]

    if isinstance(first, list):
        # Multi-output: list[class] of arrays → (..., n_cls)
        n_out = len(first)
        stacked = [np.concatenate([b[c] for b in batches], axis=0) for c in range(n_out)]
        result = np.stack(stacked, axis=-1)
        if n_out == 1:
            # Regression wrapped with _Unsqueeze: singleton class dim — drop it
            return result[..., 0]
        return result

    return np.concatenate(batches, axis=0)
