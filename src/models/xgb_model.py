"""XGBoost model wrapper for ENSO regression and classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


class ENSOXGBModel:
    """Wrapper around XGBRegressor / XGBClassifier with reproducible training.

    Attributes:
        cfg:   Full experiment config dict.
        lead:  Forecast lead time in months.
        task:  'regression' | 'classification'.
        model: Fitted xgb model (None before fit).
    """

    def __init__(self, cfg: dict, lead: int, task: str | None = None) -> None:
        self.cfg  = cfg
        self.lead = lead
        self.task = task or cfg["model"]["task"]
        self.model: Any = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Train with early stopping; returns val metrics dict.

        Args:
            X_train, y_train: Training features and targets.
            X_val,   y_val:   Validation features and targets.
            feature_names:    Optional list used to name features inside XGBoost.

        Returns:
            Dict with val_rmse (regression) or val_accuracy (classification)
            and best_iteration.
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install 'xgboost>=2.0'")

        xp = self.cfg["model"]["xgb"]
        seed = self.cfg["experiment"]["seed"]

        common_kw = dict(
            n_estimators=xp["n_estimators"],
            max_depth=xp["max_depth"],
            learning_rate=xp["learning_rate"],
            subsample=xp["subsample"],
            colsample_bytree=xp["colsample_bytree"],
            early_stopping_rounds=xp["early_stopping_rounds"],
            random_state=seed,
            n_jobs=-1,
        )

        if self.task == "regression":
            self.model = xgb.XGBRegressor(**common_kw)
        else:
            n_classes = len(np.unique(np.concatenate([y_train, y_val])))
            self.model = xgb.XGBClassifier(
                **common_kw,
                num_class=n_classes,
                objective="multi:softprob",
                eval_metric="mlogloss",
            )

        fit_kw: dict[str, Any] = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }
        if feature_names is not None:
            fit_kw["feature_names"] = feature_names

        self.model.fit(X_train, y_train, **fit_kw)

        best_iter = int(self.model.best_iteration)
        log.info("XGB fit done — lead=%02d  task=%s  best_iter=%d", self.lead, self.task, best_iter)

        # Evaluate on val set
        y_val_pred = self.predict(X_val)
        if self.task == "regression":
            from src.models.metrics import regression_metrics
            val_m = regression_metrics(y_val, y_val_pred)
            log.info("  val  rmse=%.4f  corr=%.4f", val_m["rmse"], val_m["corr"])
        else:
            from src.models.metrics import classification_metrics
            val_m = classification_metrics(y_val, y_val_pred)
            log.info("  val  acc=%.4f  bss=%.4f", val_m["accuracy"], val_m["bss"])

        return {"best_iteration": best_iter, **{f"val_{k}": v for k, v in val_m.items()}}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions.

        Regression → (n_samples,) float array.
        Classification → (n_samples, 3) probability array.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self.task == "regression":
            return self.model.predict(X)
        return self.model.predict_proba(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Save model to directory/xgb_lead{L:02d}_{task}.ubj.

        Returns the path to the saved file.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"xgb_lead{self.lead:02d}_{self.task}.ubj"
        self.model.save_model(str(path))
        log.info("Saved XGB model → %s", path)
        return path

    def load(self, directory: str | Path) -> None:
        """Load model from directory/xgb_lead{L:02d}_{task}.ubj."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed.")
        directory = Path(directory)
        path = directory / f"xgb_lead{self.lead:02d}_{self.task}.ubj"
        if self.task == "regression":
            self.model = xgb.XGBRegressor()
        else:
            self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        log.info("Loaded XGB model ← %s", path)
