"""Train LSTM ENSO forecast model for a given lead time and task.

Loads basin-index Zarr, builds overlapping sequences, trains an LSTM with
early stopping, and saves the model and metrics JSON.

Usage
-----
    python scripts/train_lstm.py --config configs/default.yaml --lead 6 --task regression
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm_model import ENSOLSTMModel
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


def main(cfg_path: str, lead: int, task: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    processed = Path(cfg["data"]["processed_dir"])
    model_dir = Path(cfg["experiment"]["output_dir"]) / "data" / "models" / "lstm"

    # ------------------------------------------------------------------
    # Load basin indices and target
    # ------------------------------------------------------------------
    basin_path = processed / "predictors_basin.zarr"
    nino_path  = processed / "target_nino34.zarr"
    for p in [basin_path, nino_path]:
        if not p.exists():
            raise FileNotFoundError(f"Zarr store not found: {p}\nRun scripts/run_preprocess.py first.")

    import pandas as pd
    ds_basin = load_zarr(basin_path).compute()
    df_basin = ds_basin.to_dataframe().dropna()

    ds_nino  = load_zarr(nino_path).compute()
    import xarray as xr
    nino34 = ds_nino["nino34"] if "nino34" in ds_nino else list(ds_nino.data_vars.values())[0]

    # ------------------------------------------------------------------
    # Build LSTM sequences
    # ------------------------------------------------------------------
    seq_len = cfg["model"]["lstm"]["sequence_length"]
    log.info("Building LSTM sequences  lead=%02d  seq_len=%d ...", lead, seq_len)
    X, y_reg, var_names, times = build_lstm_sequences(df_basin, nino34, lead, seq_len)
    log.info("Sequences  X=%s  vars=%s", X.shape, var_names)

    # ------------------------------------------------------------------
    # Temporal split
    # ------------------------------------------------------------------
    (X_tr, y_tr, t_tr,
     X_val, y_val, t_val,
     X_te,  y_te,  t_te) = train_val_test_split_temporal(
        X, y_reg, times,
        train_years=tuple(cfg["data"]["train_years"]),
        val_years=tuple(cfg["data"]["val_years"]),
        test_years=tuple(cfg["data"]["test_years"]),
    )
    log.info("Split  train=%d  val=%d  test=%d", len(t_tr), len(t_val), len(t_te))

    if task == "classification":
        y_tr_fit  = build_class_labels(y_tr)
        y_val_fit = build_class_labels(y_val)
        y_te_eval = build_class_labels(y_te)
    else:
        y_tr_fit, y_val_fit, y_te_eval = y_tr, y_val, y_te

    # ------------------------------------------------------------------
    # Per-feature standardization (fit on training set only)
    # ------------------------------------------------------------------
    X_mean = X_tr.mean(axis=(0, 1), keepdims=True)   # (1, 1, n_features)
    X_std  = X_tr.std(axis=(0, 1), keepdims=True)
    X_std  = np.where(X_std < 1e-8, 1.0, X_std)
    X_tr   = (X_tr  - X_mean) / X_std
    X_val  = (X_val - X_mean) / X_std
    X_te   = (X_te  - X_mean) / X_std

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model = ENSOLSTMModel(cfg, lead, task, device=device)
    log.info("Training LSTM on %s ...", model.device)
    val_metrics = model.fit(X_tr, y_tr_fit, X_val, y_val_fit)

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    y_te_pred = model.predict(X_te)
    if task == "regression":
        test_metrics = regression_metrics(y_te_eval, y_te_pred)
        log.info("Test  rmse=%.4f  corr=%.4f", test_metrics["rmse"], test_metrics["corr"])
    else:
        test_metrics = classification_metrics(y_te_eval, y_te_pred)
        log.info("Test  acc=%.4f  bss=%.4f", test_metrics["accuracy"], test_metrics["bss"])

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    model.save(model_dir)
    metrics_path = model_dir / f"lstm_lead{lead:02d}_{task}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model_type": "lstm",
            "lead_months": lead,
            "task": task,
            "norm_mean": X_mean.squeeze().tolist(),
            "norm_std":  X_std.squeeze().tolist(),
            **val_metrics,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM ENSO model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lead", type=int, required=True, choices=[3, 6, 12])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args.config, args.lead, args.task, args.device)
