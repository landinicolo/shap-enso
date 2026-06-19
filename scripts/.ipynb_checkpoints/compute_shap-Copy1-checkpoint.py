"""Compute and save SHAP values for a trained ENSO model.

Dispatches to the appropriate explainer (TreeExplainer for XGBoost,
GradientExplainer for LSTM, DeepExplainer for CNN) based on --model.

Background samples are drawn exclusively from the training split to
prevent data leakage into the SHAP reference distribution.

Usage
-----
    python scripts/compute_shap.py --config configs/default.yaml \\
        --model xgboost --lead 6 --task regression

    python scripts/compute_shap.py --config configs/default.yaml \\
        --model lstm --lead 6 --task regression --device cuda

    python scripts/compute_shap.py --config configs/default.yaml \\
        --model cnn --lead 6 --task regression --device cuda --save-spatial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.utils.config import load_config
from src.utils.logging_utils import get_logger


class _Unsqueeze(nn.Module):
    """Wrap a regression net so output is (B, 1) instead of (B,).

    GradientExplainer and DeepExplainer index outputs as outputs[:, idx],
    which fails for 1-D tensors produced by regression models that squeeze
    their final dimension.
    """
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out.unsqueeze(-1) if out.dim() == 1 else out
from src.shap_analysis.compute_shap-Copy1.py import (   
    select_background,
    get_tree_explainer,
    get_gradient_explainer,
    get_deep_explainer,
    compute_tree_shap,
    compute_deep_shap,
    aggregate_lstm_shap,
    aggregate_cnn_shap_spatial,
    project_shap_to_grid,
    save_shap_dataset,
    save_spatial_shap,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def run_xgb_shap(cfg: dict, lead: int, task: str) -> None:
    """Full SHAP pipeline for the XGBoost model."""
    import pandas as pd
    from src.models.xgb_model import ENSOXGBModel
    from src.utils.io_utils import load_feature_matrix
    from src.utils.preprocessing import build_class_labels, train_val_test_split_temporal

    feat_dir  = Path(cfg["data"]["processed_dir"]) / "features"
    model_dir = Path(cfg["experiment"]["output_dir"]) / "data" / "models" / "xgb"
    shap_dir  = Path(cfg["shap"]["output_dir"])

    # Load feature matrix
    feat_path = feat_dir / f"features_lead{lead:02d}.npz"
    X, y_reg, feat_names, times = load_feature_matrix(feat_path)
    times_idx = pd.DatetimeIndex(times)

    # Temporal split
    (X_tr, y_tr, t_tr,
     X_val, y_val, t_val,
     X_te, y_te, t_te) = train_val_test_split_temporal(
        X, y_reg, times_idx,
        train_years=tuple(cfg["data"]["train_years"]),
        val_years=tuple(cfg["data"]["val_years"]),
        test_years=tuple(cfg["data"]["test_years"]),
    )

    # Subsample eval set
    n_eval = min(cfg["shap"]["max_eval_samples"], len(X_te))
    rng = __import__("numpy").random.default_rng(cfg["experiment"]["seed"])
    eval_idx = rng.choice(len(X_te), size=n_eval, replace=False)
    X_eval, times_eval = X_te[eval_idx], t_te[eval_idx]

    if task == "classification":
        y_tr_fit = build_class_labels(y_tr)
    else:
        y_tr_fit = y_tr

    # Load model
    model = ENSOXGBModel(cfg, lead, task)
    model.load(model_dir)
    log.info("Loaded XGB model  lead=%02d  task=%s", lead, task)

    # Background from training split
    X_bg = select_background(X_tr, cfg["shap"]["background_samples"],
                             seed=cfg["experiment"]["seed"])
    log.info("Background n=%d  Eval n=%d", len(X_bg), n_eval)

    # Explainer
    explainer = get_tree_explainer(model)

    # SHAP values
    log.info("Computing TreeSHAP ...")
    shap_vals, base_val = compute_tree_shap(explainer, X_eval)
    log.info("SHAP shape: %s  base_value: %.4f", shap_vals.shape, base_val)

    # Predictions for eval set
    preds = model.predict(X_eval)
    if task == "classification":
        preds = preds.argmax(axis=1).astype(__import__("numpy").float32)

    save_shap_dataset(
        shap_vals, preds, base_val, feat_names, pd.DatetimeIndex(times_eval),
        "xgboost", lead, task, shap_dir,
    )


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def run_lstm_shap(cfg: dict, lead: int, task: str, device: str = "cpu") -> None:
    """Full SHAP pipeline for the LSTM model."""
    import pandas as pd
    from src.models.lstm_model import ENSOLSTMModel
    from src.utils.io_utils import load_zarr
    from src.utils.preprocessing import (
        build_class_labels, build_lstm_sequences, train_val_test_split_temporal,
    )

    processed = Path(cfg["data"]["processed_dir"])
    model_dir = Path(cfg["experiment"]["output_dir"]) / "data" / "models" / "lstm"
    shap_dir  = Path(cfg["shap"]["output_dir"])

    # Load data
    ds_basin = load_zarr(processed / "predictors_basin.zarr").compute()
    df_basin = ds_basin.to_dataframe().dropna()
    ds_nino  = load_zarr(processed / "target_nino34.zarr").compute()
    import xarray as xr
    nino34 = list(ds_nino.data_vars.values())[0]

    seq_len = cfg["model"]["lstm"]["sequence_length"]
    X, y_reg, var_names, times = build_lstm_sequences(df_basin, nino34, lead, seq_len)

    (X_tr, y_tr, t_tr,
     X_val, y_val, t_val,
     X_te, y_te, t_te) = train_val_test_split_temporal(
        X, y_reg, times,
        train_years=tuple(cfg["data"]["train_years"]),
        val_years=tuple(cfg["data"]["val_years"]),
        test_years=tuple(cfg["data"]["test_years"]),
    )

    # Per-feature standardization — must match train_lstm.py exactly
    import numpy as np
    X_mean = np.nanmean(X_tr, axis=(0, 1), keepdims=True)
    X_std  = np.nanstd(X_tr,  axis=(0, 1), keepdims=True)
    X_std  = np.where(X_std < 1e-8, 1.0, X_std)
    X_tr   = (X_tr - X_mean) / X_std
    X_te   = (X_te - X_mean) / X_std

    # Subsample
    n_eval = min(cfg["shap"]["max_eval_samples"], len(X_te))
    rng    = __import__("numpy").random.default_rng(cfg["experiment"]["seed"])
    eval_idx = rng.choice(len(X_te), size=n_eval, replace=False)
    X_eval, times_eval = X_te[eval_idx], t_te[eval_idx]

    # Load model
    model = ENSOLSTMModel(cfg, lead, task, device=device)
    model.load(model_dir)
    model.net.eval()
    device = model.device   # use actual device after CUDA fallback
    log.info("Loaded LSTM model  lead=%02d  task=%s  device=%s", lead, task, device)

    # Background tensor
    X_bg_np = select_background(X_tr, cfg["shap"]["background_samples"],
                                seed=cfg["experiment"]["seed"])
    X_bg_t  = torch.tensor(X_bg_np, dtype=torch.float32).to(device)

    # Explainer — wrap regression net so output is (B,1) not (B,)
    net_for_shap = _Unsqueeze(model.net) if task == "regression" else model.net
    explainer = get_gradient_explainer(net_for_shap, X_bg_t)

    # SHAP values (seq_len × n_vars → aggregated to n_vars)
    log.info("Computing GradientSHAP ...")
    batch_size = cfg["shap"].get("deep_batch_size", 50)
    shap_3d, base_val = compute_deep_shap(explainer, X_eval, batch_size, device)

    # Aggregate: (N, seq_len, n_vars) → (N, n_vars)
    shap_2d = aggregate_lstm_shap(shap_3d)
    log.info("SHAP shape after aggregation: %s", shap_2d.shape)

    # Predictions
    preds = model.predict(X_eval)
    if task == "classification":
        preds = preds.argmax(axis=1).astype(__import__("numpy").float32)

    save_shap_dataset(
        shap_2d, preds, base_val, var_names, pd.DatetimeIndex(times_eval),
        "lstm", lead, task, shap_dir,
    )


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

def run_cnn_shap(
    cfg: dict,
    lead: int,
    task: str,
    device: str = "cpu",
    save_spatial: bool = False,
) -> None:
    """Full SHAP pipeline for the CNN model."""
    import pandas as pd
    import numpy as np
    from src.models.cnn_model import ENSOCNNModel
    from src.utils.io_utils import load_zarr
    from src.utils.preprocessing import (
        build_class_labels, build_cnn_tensors, train_val_test_split_temporal,
    )

    processed = Path(cfg["data"]["processed_dir"])
    model_dir = Path(cfg["experiment"]["output_dir"]) / "data" / "models" / "cnn"
    shap_dir  = Path(cfg["shap"]["output_dir"])

    # Load gridded data
    ds_anom = load_zarr(processed / "predictors.zarr").compute()
    ds_nino = load_zarr(processed / "target_nino34.zarr").compute()
    nino34  = list(ds_nino.data_vars.values())[0]

    n_lags  = cfg["data"]["lag_months"] + 1
    era5_vars = cfg["data"]["era5_variables"] + (["d20"] if "d20" in ds_anom else [])
    ch_names: list[str] = [f"{v}_lag{l}" for v in era5_vars for l in range(n_lags)]

    X, y_reg, _, times = build_cnn_tensors(ds_anom, nino34, lead, n_lags, era5_vars)

    (X_tr, y_tr, t_tr,
     X_val, y_val, t_val,
     X_te, y_te, t_te) = train_val_test_split_temporal(
        X, y_reg, times,
        train_years=tuple(cfg["data"]["train_years"]),
        val_years=tuple(cfg["data"]["val_years"]),
        test_years=tuple(cfg["data"]["test_years"]),
    )

    # Per-channel standardization + land fill — must match train_cnn.py exactly
    X_mean = np.nanmean(X_tr, axis=(0, 2, 3), keepdims=True)
    X_std  = np.nanstd(X_tr,  axis=(0, 2, 3), keepdims=True)
    X_std  = np.where(X_std < 1e-8, 1.0, X_std)
    X_tr   = np.nan_to_num((X_tr - X_mean) / X_std, nan=0.0)
    X_te   = np.nan_to_num((X_te - X_mean) / X_std, nan=0.0)

    n_eval = min(cfg["shap"]["max_eval_samples"], len(X_te))
    rng    = np.random.default_rng(cfg["experiment"]["seed"])
    eval_idx = rng.choice(len(X_te), size=n_eval, replace=False)
    X_eval, times_eval = X_te[eval_idx], t_te[eval_idx]

    # Load model
    model = ENSOCNNModel(cfg, lead, task, device=device)
    model.load(model_dir)
    model.net.eval()
    device = model.device   # use actual device after CUDA fallback
    log.info("Loaded CNN model  lead=%02d  task=%s  device=%s", lead, task, device)

    # DeepExplainer uses backward hooks that are incompatible with inplace ReLU.
    # Disable inplace on all ReLU layers (inplace flag is not in the state dict).
    for m in model.net.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = False

    # Background
    n_bg   = min(cfg["shap"]["background_samples"], len(X_tr))
    X_bg_np = select_background(X_tr, n_bg, seed=cfg["experiment"]["seed"])
    X_bg_t  = torch.tensor(X_bg_np, dtype=torch.float32).to(device)

    # Explainer — wrap regression net so output is (B,1) not (B,)
    net_for_shap = _Unsqueeze(model.net) if task == "regression" else model.net
    explainer = get_deep_explainer(net_for_shap, X_bg_t)

    # SHAP — use small batches; CNN spatial SHAP is memory-heavy
    batch_size = cfg["shap"].get("cnn_batch_size", 10)
    log.info("Computing DeepSHAP (batch=%d) ...", batch_size)
    shap_spatial, base_val = compute_deep_shap(explainer, X_eval, batch_size, device)
    log.info("Spatial SHAP shape: %s", shap_spatial.shape)

    # Aggregate spatially: (N, C, lat, lon) → (N, C) feature importance
    shap_2d = aggregate_cnn_shap_spatial(shap_spatial)

    # Predictions
    preds = model.predict(X_eval)
    if task == "classification":
        preds = preds.argmax(axis=1).astype(np.float32)

    save_shap_dataset(
        shap_2d, preds, base_val, ch_names, pd.DatetimeIndex(times_eval),
        "cnn", lead, task, shap_dir,
    )

    # Optional full spatial SHAP map
    if save_spatial and shap_spatial.ndim == 4:
        lat = ds_anom.lat.values
        lon = ds_anom.lon.values
        spatial_da = project_shap_to_grid(shap_spatial, era5_vars, n_lags, lat, lon)
        save_spatial_shap(spatial_da, "cnn", lead, task, shap_dir)


# ---------------------------------------------------------------------------
# CLI dispatcher
# ---------------------------------------------------------------------------

def main(
    cfg_path: str,
    model_type: str,
    lead: int,
    task: str,
    device: str = "cpu",
    save_spatial: bool = False,
) -> None:
    cfg = load_config(cfg_path)
    log.info("Computing SHAP  model=%s  lead=%02d  task=%s", model_type, lead, task)

    if model_type == "xgboost":
        run_xgb_shap(cfg, lead, task)
    elif model_type == "lstm":
        run_lstm_shap(cfg, lead, task, device)
    elif model_type == "cnn":
        run_cnn_shap(cfg, lead, task, device, save_spatial)
    else:
        raise ValueError(f"Unknown model type '{model_type}'")

    log.info("Done — SHAP values saved to %s", cfg["shap"]["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SHAP values for ENSO models")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--model",   required=True, choices=["xgboost", "lstm", "cnn"])
    parser.add_argument("--lead",    required=True, type=int, choices=[3, 6, 12])
    parser.add_argument("--task",    default="regression", choices=["regression", "classification"])
    parser.add_argument("--device",  default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--save-spatial", action="store_true",
                        help="Also save per-variable spatial SHAP map for CNN")
    args = parser.parse_args()
    main(args.config, args.model, args.lead, args.task, args.device, args.save_spatial)
