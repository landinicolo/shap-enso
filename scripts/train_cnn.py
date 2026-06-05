"""Train CNN ENSO forecast model on gridded fields for a given lead time.

Loads the gridded predictor Zarr, builds multi-channel 2-D tensors, trains
a CNN with early stopping, and saves the model and metrics JSON.

Usage
-----
    python scripts/train_cnn.py --config configs/default.yaml --lead 6 --task regression
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn_model import ENSOCNNModel
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


def main(cfg_path: str, lead: int, task: str, device: str | None = None) -> None:
    cfg = load_config(cfg_path)
    processed = Path(cfg["data"]["processed_dir"])
    model_dir = Path(cfg["experiment"]["output_dir"]) / "data" / "models" / "cnn"

    # ------------------------------------------------------------------
    # Load gridded predictors and target
    # ------------------------------------------------------------------
    grid_path = processed / "predictors.zarr"
    nino_path = processed / "target_nino34.zarr"
    for p in [grid_path, nino_path]:
        if not p.exists():
            raise FileNotFoundError(f"Zarr store not found: {p}\nRun scripts/run_preprocess.py first.")

    log.info("Loading gridded predictors ...")
    ds_anom = load_zarr(grid_path).compute()
    ds_nino = load_zarr(nino_path).compute()
    nino34  = ds_nino["nino34"] if "nino34" in ds_nino else list(ds_nino.data_vars.values())[0]

    # ------------------------------------------------------------------
    # Build CNN tensors
    # ------------------------------------------------------------------
    n_lags = cfg["data"]["lag_months"] + 1   # e.g. lag_months=3 → 4 lags (0,1,2,3)
    era5_vars = cfg["data"]["era5_variables"] + (["d20"] if "d20" in ds_anom else [])
    log.info("Building CNN tensors  lead=%02d  n_lags=%d  vars=%s ...", lead, n_lags, era5_vars)
    X, y_reg, ch_names, times = build_cnn_tensors(ds_anom, nino34, lead, n_lags, era5_vars)
    log.info("Tensors  X=%s  n_channels=%d", X.shape, X.shape[1])

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
    # Train
    # ------------------------------------------------------------------
    model = ENSOCNNModel(cfg, lead, task, device=device)
    log.info("Training CNN on %s ...", model.device)
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
    metrics_path = model_dir / f"cnn_lead{lead:02d}_{task}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model_type": "cnn",
            "lead_months": lead,
            "task": task,
            "n_channels": int(X.shape[1]),
            "channel_names": ch_names,
            **val_metrics,
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN ENSO model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lead", type=int, required=True, choices=[3, 6, 12])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    args = parser.parse_args()
    main(args.config, args.lead, args.task, args.device)
