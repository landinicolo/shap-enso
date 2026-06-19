"""LSTM model for sequential ENSO prediction, with tuning-friendly hooks.

Drop-in replacement for ``src/models/lstm_model.py``.

What changed compared with the original version
-----------------------------------------------
* keeps the same public API: ``ENSOLSTMModel(cfg, lead, task, device)``;
* reads hyperparameters from ``cfg['model']['lstm']`` as before;
* additionally supports optional keys useful during tuning:
  ``weight_decay``, ``patience``, ``scheduler_patience``, ``scheduler_factor``,
  ``min_lr``, and ``grad_clip``;
* stores ``training_history_`` and returns ``best_epoch`` / ``epochs_ran``;
* preserves save/load compatibility with the old ``.pt`` files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------

def _import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as exc:
        raise ImportError("PyTorch not installed. Run: pip install 'torch>=2.2'") from exc


class ENSOLSTMNet:
    """Lazily defined so importing this module does not require torch at import time."""


def _build_lstm_net(
    n_features: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    task: str,
    n_classes: int = 3,
):
    """Return a torch.nn.Module implementing the LSTM.

    Architecture
    ------------
    Input ``(B, T, F)`` -> LSTM -> last hidden state -> small MLP head.

    Regression returns ``(B,)``. Classification returns logits ``(B, 3)``.
    """
    torch, nn = _import_torch()

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            out_dim = 1 if task == "regression" else n_classes
            self.head = nn.Sequential(
                nn.Linear(hidden_size, max(1, hidden_size // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(1, hidden_size // 2), out_dim),
            )
            self.task = task

        def forward(self, x):                 # x: (B, T, F)
            _, (h_n, _) = self.lstm(x)        # h_n: (num_layers, B, H)
            out = self.head(h_n[-1])          # last-layer hidden: (B, H)
            if self.task == "regression":
                return out.squeeze(-1)        # (B,)
            return out                        # (B, n_classes)

    return _Net()


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

class ENSOLSTMModel:
    """Training and inference wrapper for the ENSO LSTM.

    Args:
        cfg:    Full config dict. Hyperparameters are read from
                ``cfg['model']['lstm']``.
        lead:   Forecast lead time in months.
        task:   ``'regression'`` or ``'classification'``.
        device: ``'cuda'`` or ``'cpu'``. If ``None``, auto-detected.
    """

    def __init__(self, cfg: dict, lead: int, task: str | None = None, device: str | None = None) -> None:
        self.cfg = cfg
        self.lead = lead
        self.task = task or cfg["model"].get("task", "regression")
        self.net = None
        self._meta: dict[str, Any] = {}
        self.training_history_: list[dict[str, float]] = []

        torch, _ = _import_torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available — falling back to cpu")
            device = "cpu"
        self.device = device

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,   # (N, seq_len, n_features)
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """Train the LSTM and return best-epoch validation metrics.

        This method intentionally remains simple: it trains one model with the
        hyperparameters currently stored in ``cfg['model']['lstm']``. The CV
        / grid-search loop lives in ``scripts/train_lstm.py`` so the model
        class stays reusable for normal training and for tuning.
        """
        torch, nn = _import_torch()

        lp = dict(self.cfg["model"]["lstm"])
        n_features = X_train.shape[2]

        hidden_size = int(lp["hidden_size"])
        num_layers = int(lp["num_layers"])
        dropout = float(lp.get("dropout", 0.0))
        batch_size = int(lp.get("batch_size", 32))
        lr = float(lp.get("lr", 1e-3))
        weight_decay = float(lp.get("weight_decay", 0.0))
        max_epochs = int(lp.get("max_epochs", 100))
        patience = int(lp.get("patience", 20))
        scheduler_patience = int(lp.get("scheduler_patience", 10))
        scheduler_factor = float(lp.get("scheduler_factor", 0.5))
        min_lr = float(lp.get("min_lr", 1e-5))
        grad_clip = float(lp.get("grad_clip", 1.0))

        self.net = _build_lstm_net(
            n_features=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            task=self.task,
        ).to(self.device)

        self._meta = {
            "n_features": n_features,
            "seq_len": X_train.shape[1],
            "task": self.task,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        }

        train_dl = self._make_loader(X_train, y_train, shuffle=True, batch_size=batch_size)
        val_dl = self._make_loader(X_val, y_val, shuffle=False, batch_size=batch_size * 4)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=scheduler_patience,
            factor=scheduler_factor,
            min_lr=min_lr,
        )
        criterion = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        best_epoch = 0
        self.training_history_ = []

        for epoch in range(max_epochs):
            self.net.train()
            train_loss_sum = 0.0
            train_n = 0

            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred = self.net(xb)
                if self.task == "classification":
                    yb = yb.long()
                loss = criterion(pred, yb)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=grad_clip)
                optimizer.step()

                train_loss_sum += loss.item() * len(yb)
                train_n += len(yb)

            train_loss = train_loss_sum / train_n if train_n else float("inf")
            val_loss = self._eval_loss(val_dl, criterion)
            scheduler.step(val_loss)

            row = {
                "epoch": float(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            self.training_history_.append(row)

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                patience_counter = 0
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                log.info(
                    "  epoch %3d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                    epoch + 1,
                    train_loss,
                    val_loss,
                    optimizer.param_groups[0]["lr"],
                )

            if patience_counter >= patience:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.to(self.device)

        y_val_pred = self.predict(X_val)
        if self.task == "regression":
            from src.models.metrics import regression_metrics
            val_m = regression_metrics(y_val, y_val_pred)
            log.info("LSTM lead=%02d  val  rmse=%.4f  corr=%.4f", self.lead, val_m["rmse"], val_m["corr"])
        else:
            from src.models.metrics import classification_metrics
            val_m = classification_metrics(y_val, y_val_pred)
            log.info("LSTM lead=%02d  val  acc=%.4f  bss=%.4f", self.lead, val_m["accuracy"], val_m["bss"])

        return {
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "epochs_ran": int(len(self.training_history_)),
            **{f"val_{k}": float(v) for k, v in val_m.items()},
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions.

        Regression -> ``(n_samples,)`` float.
        Classification -> ``(n_samples, 3)`` softmax probabilities.
        """
        import torch
        if self.net is None:
            raise RuntimeError("Model not fitted.")
        self.net.eval()
        dl = self._make_loader(X, np.zeros(len(X)), shuffle=False, batch_size=256)
        preds = []
        with torch.no_grad():
            for xb, _ in dl:
                out = self.net(xb.to(self.device))
                if self.task == "classification":
                    out = torch.softmax(out, dim=-1)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Save model to ``directory/lstm_lead{L:02d}_{task}.pt``."""
        import torch
        if self.net is None:
            raise RuntimeError("Model not fitted.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"lstm_lead{self.lead:02d}_{self.task}.pt"
        torch.save({"state_dict": self.net.state_dict(), "meta": self._meta}, str(path))
        log.info("Saved LSTM model → %s", path)
        return path

    def load(self, directory: str | Path) -> None:
        """Load model from ``directory/lstm_lead{L:02d}_{task}.pt``."""
        import torch
        directory = Path(directory)
        path = directory / f"lstm_lead{self.lead:02d}_{self.task}.pt"
        ckpt = torch.load(str(path), map_location=self.device)
        meta = ckpt["meta"]
        self._meta = meta
        self.net = _build_lstm_net(
            n_features=meta["n_features"],
            hidden_size=meta["hidden_size"],
            num_layers=meta["num_layers"],
            dropout=meta["dropout"],
            task=meta["task"],
        ).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        log.info("Loaded LSTM model ← %s", path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_loader(self, X, y, shuffle, batch_size):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X_t = torch.tensor(X, dtype=torch.float32)
        if self.task == "classification":
            y_t = torch.tensor(y, dtype=torch.long)
        else:
            y_t = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

    def _eval_loss(self, loader, criterion) -> float:
        import torch
        self.net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.net(xb)
                if self.task == "classification":
                    yb = yb.long()
                total += criterion(pred, yb).item() * len(yb)
                n += len(yb)
        return total / n if n > 0 else float("inf")
