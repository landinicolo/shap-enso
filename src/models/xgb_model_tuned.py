"""Tuned XGBoost model wrapper for ENSO regression and classification.

This module is separate from ``src/models/xgb_model.py`` so tuned experiments
stay isolated from baseline models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


class ENSOXGBModel:
    """Wrapper around XGBRegressor / XGBClassifier with tunable parameters."""

    def __init__(self, cfg: dict, lead: int, task: str | None = None) -> None:
        self.cfg = cfg
        self.lead = lead
        self.task = task or cfg["model"].get("task", "regression")
        self.model: Any = None
        self.feature_names: list[str] | None = None
        self._meta: dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Train with early stopping and return validation metrics."""
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost not installed. Run: pip install 'xgboost>=2.0'") from exc

        xp = self.cfg["model"]["xgb"]
        seed = int(self.cfg["experiment"].get("seed", 42))
        self.feature_names = list(feature_names) if feature_names is not None else None

        common_kw = dict(
            n_estimators=int(xp.get("n_estimators", 500)),
            max_depth=int(xp.get("max_depth", 3)),
            learning_rate=float(xp.get("learning_rate", 0.05)),
            subsample=float(xp.get("subsample", 0.8)),
            colsample_bytree=float(xp.get("colsample_bytree", 0.8)),
            min_child_weight=float(xp.get("min_child_weight", 1.0)),
            reg_alpha=float(xp.get("reg_alpha", 0.0)),
            reg_lambda=float(xp.get("reg_lambda", 1.0)),
            gamma=float(xp.get("gamma", 0.0)),
            early_stopping_rounds=int(xp.get("early_stopping_rounds", 50)),
            random_state=seed,
            n_jobs=int(xp.get("n_jobs", -1)),
        )
        if "max_delta_step" in xp:
            common_kw["max_delta_step"] = float(xp["max_delta_step"])

        if self.task == "regression":
            self.model = xgb.XGBRegressor(
                **common_kw,
                objective=xp.get("objective", "reg:squarederror"),
                eval_metric=xp.get("eval_metric", "rmse"),
            )
        else:
            self.model = xgb.XGBClassifier(
                **common_kw,
                objective=xp.get("objective", "multi:softprob"),
                eval_metric=xp.get("eval_metric", "mlogloss"),
                num_class=int(xp.get("num_class", 3)),
            )

        X_train_fit = self._as_frame(X_train)
        X_val_fit = self._as_frame(X_val)

        self.model.fit(
            X_train_fit,
            y_train,
            eval_set=[(X_val_fit, y_val)],
            verbose=False,
        )

        best_iter = getattr(self.model, "best_iteration", None)
        best_iter = int(best_iter) if best_iter is not None else int(common_kw["n_estimators"])
        log.info("XGB fit done lead=%02d task=%s best_iter=%d", self.lead, self.task, best_iter)

        y_val_pred = self.predict(X_val)
        if self.task == "regression":
            from src.models.metrics import regression_metrics
            val_m = regression_metrics(y_val, y_val_pred)
            log.info("XGB lead=%02d val rmse=%.4f corr=%.4f r2=%.4f",
                     self.lead, val_m["rmse"], val_m["corr"], val_m["r2"])
        else:
            from src.models.metrics import classification_metrics
            val_m = classification_metrics(y_val, y_val_pred)
            log.info("XGB lead=%02d val acc=%.4f bss=%.4f",
                     self.lead, val_m["accuracy"], val_m["bss"])

        self._meta = {
            "lead": self.lead,
            "task": self.task,
            "feature_names": self.feature_names,
            "params": {k: _jsonable(v) for k, v in common_kw.items()},
            "best_iteration": best_iter,
        }
        return {"best_iteration": best_iter, **{f"val_{k}": v for k, v in val_m.items()}}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions: regression values or classification probabilities."""
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_pred = self._as_frame(X)
        if self.task == "regression":
            return self.model.predict(X_pred)
        proba = self.model.predict_proba(X_pred)
        if proba.shape[1] == 3:
            return proba
        # Defensive padding if a fold/model somehow emitted fewer classes.
        out = np.zeros((len(proba), 3), dtype=float)
        classes = getattr(self.model, "classes_", np.arange(proba.shape[1]))
        for j, cls in enumerate(classes):
            if int(cls) < 3:
                out[:, int(cls)] = proba[:, j]
        return out

    def save(self, directory: str | Path) -> Path:
        """Save model to directory/xgb_lead{L:02d}_{task}.ubj plus metadata JSON."""
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"xgb_lead{self.lead:02d}_{self.task}.ubj"
        self.model.get_booster().save_model(str(path))
        meta_path = directory / f"xgb_lead{self.lead:02d}_{self.task}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(self._meta, f, indent=2)
        log.info("Saved XGB model -> %s", path)
        return path

    def load(self, directory: str | Path) -> None:
        """Load model from directory/xgb_lead{L:02d}_{task}.ubj."""
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("xgboost not installed.") from exc
        directory = Path(directory)
        path = directory / f"xgb_lead{self.lead:02d}_{self.task}.ubj"
        if self.task == "regression":
            self.model = xgb.XGBRegressor()
        else:
            self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        meta_path = directory / f"xgb_lead{self.lead:02d}_{self.task}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
            self.feature_names = self._meta.get("feature_names")
        log.info("Loaded XGB model <- %s", path)

    def _as_frame(self, X):
        if self.feature_names is None:
            return X
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.feature_names)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
