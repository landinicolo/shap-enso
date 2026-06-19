"""2D CNN model for spatially-resolved ENSO prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError("PyTorch not installed. Run: pip install 'torch>=2.2'")


def _build_cnn_net(
    n_channels: int,
    channels: list[int],
    kernel_size: int,
    dropout: float,
    task: str,
    n_classes: int = 3,
):
    """Build and return a CNN nn.Module.

    Architecture:
        [Conv2d → BN → ReLU → MaxPool] × (len(channels)−1)
        Conv2d → BN → ReLU → AdaptiveAvgPool2d(1,1)
        Flatten → Linear → ReLU → Dropout → Linear(output_dim)

    The final AdaptiveAvgPool makes the output size independent of input spatial dims.
    """
    torch, nn = _import_torch()

    blocks = []
    in_ch = n_channels
    for i, out_ch in enumerate(channels):
        blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
        blocks.append(nn.BatchNorm2d(out_ch))
        blocks.append(nn.ReLU(inplace=True))
        if i < len(channels) - 1:
            blocks.append(nn.MaxPool2d(2))
        else:
            blocks.append(nn.AdaptiveAvgPool2d((1, 1)))
        in_ch = out_ch

    out_dim = 1 if task == "regression" else n_classes

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels[-1], channels[-1] // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(channels[-1] // 2, out_dim),
            )
            self.task = task

        def forward(self, x):               # x: (B, C, H, W)
            x = self.features(x)
            x = self.head(x)
            if self.task == "regression":
                return x.squeeze(-1)        # (B,)
            return x                        # (B, n_classes)

    return _Net()


class ENSOCNNModel:
    """Training and inference wrapper for the ENSO CNN.

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
        X_train: np.ndarray,    # (N, n_channels, lat, lon)
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """Train the CNN; returns best-epoch val metrics."""
        torch, nn = _import_torch()

        cp = self.cfg["model"]["cnn"]
        n_channels = X_train.shape[1]

        self.net = _build_cnn_net(
            n_channels=n_channels,
            channels=cp["channels"],
            kernel_size=cp["kernel_size"],
            dropout=cp["dropout"],
            task=self.task,
        ).to(self.device)

        self._meta = {
            "n_channels": n_channels,
            "channels":   cp["channels"],
            "kernel_size": cp["kernel_size"],
            "dropout":    cp["dropout"],
            "task":       self.task,
        }

        train_dl = self._make_loader(X_train, y_train, shuffle=True,  batch_size=cp["batch_size"])
        val_dl   = self._make_loader(X_val,   y_val,   shuffle=False, batch_size=cp["batch_size"] * 4)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=cp["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-5
        )
        criterion = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 25
        best_state = None

        for epoch in range(cp["max_epochs"]):
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

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.to(self.device)

        y_val_pred = self.predict(X_val)
        if self.task == "regression":
            from src.models.metrics import regression_metrics
            val_m = regression_metrics(y_val, y_val_pred)
            log.info("CNN lead=%02d  val  rmse=%.4f  corr=%.4f", self.lead, val_m["rmse"], val_m["corr"])
        else:
            from src.models.metrics import classification_metrics
            val_m = classification_metrics(y_val, y_val_pred)
            log.info("CNN lead=%02d  val  acc=%.4f  bss=%.4f", self.lead, val_m["accuracy"], val_m["bss"])

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
        dl = self._make_loader(X, np.zeros(len(X)), shuffle=False, batch_size=64)
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
        """Save model to directory/cnn_lead{L:02d}_{task}.pt."""
        import torch
        if self.net is None:
            raise RuntimeError("Model not fitted.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"cnn_lead{self.lead:02d}_{self.task}.pt"
        torch.save({"state_dict": self.net.state_dict(), "meta": self._meta}, str(path))
        log.info("Saved CNN model → %s", path)
        return path

    def load(self, directory: str | Path) -> None:
        """Load model from directory/cnn_lead{L:02d}_{task}.pt."""
        import torch
        directory = Path(directory)
        path = directory / f"cnn_lead{self.lead:02d}_{self.task}.pt"
        ckpt = torch.load(str(path), map_location=self.device)
        meta = ckpt["meta"]
        self._meta = meta
        self.net = _build_cnn_net(
            n_channels=meta["n_channels"],
            channels=meta["channels"],
            kernel_size=meta["kernel_size"],
            dropout=meta["dropout"],
            task=meta["task"],
        ).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        log.info("Loaded CNN model ← %s", path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_loader(self, X, y, shuffle, batch_size):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle, pin_memory=False)

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
