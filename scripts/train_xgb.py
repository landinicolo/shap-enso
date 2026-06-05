"""Train XGBoost ENSO forecast models for a given lead time and task.

Usage
-----
    python scripts/train_xgb.py --config configs/default.yaml --lead 6 --task regression
    python scripts/train_xgb.py --config configs/default.yaml --lead 6 --task classification
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.metrics import classification_metrics, regression_metrics
from src.models.xgb_model import ENSOXGBModel
from src.utils.config import load_config
from src.utils.io_utils import load_feature_matrix
from src.utils.logging_utils import get_logger
from src.utils.preprocessing import build_class_labels, train_val_test_split_temporal

log = get_logger(__name__)


def main(cfg_path: str, lead: int, task: str) -> None:
    cfg = load_config(cfg_path)
    feat_dir = Path(cfg["data"]["processed_dir"]) / "features"
    model_dir = Path(cfg["experiment"]["output_dir"]) / "data" / "models" / "xgb"

    # ------------------------------------------------------------------
    # Load feature matrix
    # ------------------------------------------------------------------
    feat_path = feat_dir / f"features_lead{lead:02d}.npz"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {feat_path}\n"
            "Run scripts/run_preprocess.py first."
        )
    X, y_reg, feat_names, times = load_feature_matrix(feat_path)
    log.info("Loaded features  shape=%s  lead=%02d", X.shape, lead)

    # ------------------------------------------------------------------
    # Temporal split
    # ------------------------------------------------------------------
    import pandas as pd
    times_idx = pd.DatetimeIndex(times)
    (X_tr, y_tr, t_tr,
     X_val, y_val, t_val,
     X_te,  y_te,  t_te) = train_val_test_split_temporal(
        X, y_reg, times_idx,
        train_years=tuple(cfg["data"]["train_years"]),
        val_years=tuple(cfg["data"]["val_years"]),
        test_years=tuple(cfg["data"]["test_years"]),
    )
    log.info("Split  train=%d  val=%d  test=%d", len(t_tr), len(t_val), len(t_te))

    # ------------------------------------------------------------------
    # Class labels (classification only)
    # ------------------------------------------------------------------
    if task == "classification":
        y_tr_fit  = build_class_labels(y_tr)
        y_val_fit = build_class_labels(y_val)
        y_te_eval = build_class_labels(y_te)
    else:
        y_tr_fit = y_tr
        y_val_fit = y_val
        y_te_eval = y_te

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model = ENSOXGBModel(cfg, lead, task)
    val_metrics = model.fit(X_tr, y_tr_fit, X_val, y_val_fit, feat_names)

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    y_te_pred = model.predict(X_te)
    if task == "regression":
        test_metrics = regression_metrics(y_te_eval, y_te_pred)
        log.info("Test  rmse=%.4f  corr=%.4f  r2=%.4f",
                 test_metrics["rmse"], test_metrics["corr"], test_metrics["r2"])
    else:
        test_metrics = classification_metrics(y_te_eval, y_te_pred)
        log.info("Test  acc=%.4f  f1=%.4f  bss=%.4f",
                 test_metrics["accuracy"], test_metrics["f1_macro"], test_metrics["bss"])

    # ------------------------------------------------------------------
    # Save model and metrics
    # ------------------------------------------------------------------
    model.save(model_dir)

    metrics_path = model_dir / f"xgb_lead{lead:02d}_{task}_metrics.json"
    all_metrics = {
        "model_type": "xgboost",
        "lead_months": lead,
        "task": task,
        **val_metrics,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost ENSO model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--lead", type=int, required=True, choices=[3, 6, 12])
    parser.add_argument("--task", default="regression", choices=["regression", "classification"])
    args = parser.parse_args()
    main(args.config, args.lead, args.task)
