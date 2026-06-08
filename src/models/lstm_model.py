"""LSTM model for sequential ENSO prediction."""

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
    except ImportError:
        raise ImportError("PyTorch not installed. Run: pip install 'torch>=2.2'")


class ENSOLSTMNet:
    """Lazily defined so importing this module doesn't require torch at import time.

    Instantiate via ENSOLSTMModel which calls _build_net() after importing torch.
    """


def _build_lstm_net(
    n_features: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    task: str,
    n_classes: int = 3,
):
    """Return a torch.nn.Module implementing the LSTM."""
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
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, out_dim),
            )
            self.task = task

        def forward(self, x):                              # x: (B, T, F)
            _, (h_n, _) = self.lstm(x)                    # h_n: (num_layers, B, H)
            out = self.head(h_n[-1])                       # last layer hidden: (B, H)
            if self.task == "regression":
                return out.squeeze(-1)                     # (B,)
            return out                                     # (B, n_classes)

    return _Net()


# ---------------------------------------------------------------------------
# Training wrapper
# ---------------------------------------------------------------------------

class ENSOLSTMModel:
    """Training and inference wrapper for the ENSO LSTM.

    Args:
        cfg:    Full config dict.
        lead:   Forecast lead time in months.
        task:   'regression' | 'classification'.
        device: 'cuda' | 'cpu' (auto-detected if not given).
    """

    def __init__(self, cfg: dict, lead: int, task: str | None = None, device: str | None = None) -> None:
        self.cfg    = cfg
        self.lead   = lead
        self.task   = task or cfg["model"]["task"]
        self.net    = None
        self._meta: dict[str, Any] = {}

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
        """Train the LSTM; returns best-epoch val metrics."""
        torch, nn = _import_torch()

        lp = self.cfg["model"]["lstm"]
        n_features = X_train.shape[2]

        self.net = _build_lstm_net(
            n_features=n_features,
            hidden_size=lp["hidden_size"],
            num_layers=lp["num_layers"],
            dropout=lp["dropout"],
            task=self.task,
        ).to(self.device)

        self._meta = {
            "n_features": n_features,
            "seq_len":    X_train.shape[1],
            "task":       self.task,
            "hidden_size": lp["hidden_size"],
            "num_layers":  lp["num_layers"],
            "dropout":     lp["dropout"],
        }

        # DataLoaders
        train_dl = self._make_loader(X_train, y_train, shuffle=True,  batch_size=lp["batch_size"])
        val_dl   = self._make_loader(X_val,   y_val,   shuffle=False, batch_size=lp["batch_size"] * 4)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lp["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-5
        )
        criterion = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 20
        best_state = None

        for epoch in range(lp["max_epochs"]):
            # Train
            self.net.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                if self.task == "classification":
                    yb = yb.long()
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()

            # Validate
            val_loss = self._eval_loss(val_dl, criterion)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                log.info("  epoch %3d  val_loss=%.4f  lr=%.2e", epoch + 1, val_loss,
                         optimizer.param_groups[0]["lr"])

            if patience_counter >= patience:
                log.info("Early stopping at epoch %d", epoch + 1)
                break

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.to(self.device)

        # Compute val metrics on best model
        y_val_pred = self.predict(X_val)
        if self.task == "regression":
            from src.models.metrics import regression_metrics
            val_m = regression_metrics(y_val, y_val_pred)
            log.info("LSTM lead=%02d  val  rmse=%.4f  corr=%.4f", self.lead, val_m["rmse"], val_m["corr"])
        else:
            from src.models.metrics import classification_metrics
            val_m = classification_metrics(y_val, y_val_pred)
            log.info("LSTM lead=%02d  val  acc=%.4f  bss=%.4f", self.lead, val_m["accuracy"], val_m["bss"])

        return {"best_val_loss": best_val_loss, **{f"val_{k}": v for k, v in val_m.items()}}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions.

        Regression → (n_samples,) float.
        Classification → (n_samples, 3) softmax probabilities.
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
        """Save model to directory/lstm_lead{L:02d}_{task}.pt."""
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
        """Load model from directory/lstm_lead{L:02d}_{task}.pt."""
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
                n     += len(yb)
        return total / n if n > 0 else float("inf")
