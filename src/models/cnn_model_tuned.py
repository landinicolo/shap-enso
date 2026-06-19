"""Tuned 2D CNN model wrapper for spatially resolved ENSO prediction.

This module is intentionally separate from ``src/models/cnn_model.py`` so tuned
experiments do not quietly change the baseline model implementation.
"""

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
    except ImportError as exc:
        raise ImportError("PyTorch not installed. Run: pip install 'torch>=2.2'") from exc


def _build_cnn_net(
    n_channels: int,
    channels: list[int],
    kernel_size: int,
    dropout: float,
    task: str,
    head_hidden: int | None = None,
    n_head_layers: int = 1,
    n_classes: int = 3,
):
    """Build and return a CNN nn.Module.

    Tunable architecture:
        [Conv2d -> BatchNorm -> ReLU -> MaxPool] x (len(channels)-1)
        Conv2d -> BatchNorm -> ReLU -> AdaptiveAvgPool2d(1,1)
        Flatten -> MLP head -> output

    ``channels`` controls convolutional width/depth, while ``head_hidden`` and
    ``n_head_layers`` control the dense head. This lets tuning test wider CNNs
    without touching the baseline model file.
    """
    torch, nn = _import_torch()

    if len(channels) < 1:
        raise ValueError("channels must contain at least one integer")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size should be odd so padding preserves shape")

    blocks: list[nn.Module] = []
    in_ch = n_channels
    for i, out_ch in enumerate(channels):
        blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2))
        blocks.append(nn.BatchNorm2d(out_ch))
        blocks.append(nn.ReLU(inplace=False))
        if i < len(channels) - 1:
            blocks.append(nn.MaxPool2d(2))
        else:
            blocks.append(nn.AdaptiveAvgPool2d((1, 1)))
        in_ch = out_ch

    out_dim = 1 if task == "regression" else n_classes
    if head_hidden is None:
        head_hidden = max(8, channels[-1] // 2)
    head_hidden = int(head_hidden)
    n_head_layers = max(1, int(n_head_layers))

    head_layers: list[nn.Module] = [nn.Flatten()]
    in_features = channels[-1]
    hidden = head_hidden
    for i in range(n_head_layers):
        head_layers.append(nn.Linear(in_features, hidden))
        head_layers.append(nn.ReLU(inplace=False))
        head_layers.append(nn.Dropout(dropout))
        in_features = hidden
        hidden = max(8, hidden // 2)
    head_layers.append(nn.Linear(in_features, out_dim))

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(*blocks)
            self.head = nn.Sequential(*head_layers)
            self.task = task

        def forward(self, x):
            x = self.features(x)
            x = self.head(x)
            if self.task == "regression":
                return x.squeeze(-1)
            return x

    return _Net()


class ENSOCNNModel:
    """Training and inference wrapper for tuned ENSO CNN experiments."""

    def __init__(self, cfg: dict, lead: int, task: str | None = None, device: str | None = None) -> None:
        self.cfg = cfg
        self.lead = lead
        self.task = task or cfg["model"].get("task", "regression")
        self.net = None
        self._meta: dict[str, Any] = {}
        self.history: list[dict[str, float]] = []

        torch, _ = _import_torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but not available; falling back to cpu")
            device = "cpu"
        self.device = device

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, float]:
        """Train the CNN and return best-validation metrics."""
        torch, nn = _import_torch()

        cp = self.cfg["model"]["cnn"]
        n_channels = int(X_train.shape[1])
        channels = [int(c) for c in cp["channels"]]
        kernel_size = int(cp.get("kernel_size", 3))
        dropout = float(cp.get("dropout", 0.2))
        head_hidden = cp.get("head_hidden", max(8, channels[-1] // 2))
        n_head_layers = int(cp.get("n_head_layers", 1))

        self.net = _build_cnn_net(
            n_channels=n_channels,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            task=self.task,
            head_hidden=int(head_hidden),
            n_head_layers=n_head_layers,
        ).to(self.device)

        self._meta = {
            "n_channels": n_channels,
            "channels": channels,
            "kernel_size": kernel_size,
            "dropout": dropout,
            "head_hidden": int(head_hidden),
            "n_head_layers": n_head_layers,
            "task": self.task,
        }

        batch_size = int(cp.get("batch_size", 16))
        train_dl = self._make_loader(X_train, y_train, shuffle=True, batch_size=batch_size)
        val_dl = self._make_loader(X_val, y_val, shuffle=False, batch_size=batch_size * 4)

        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=float(cp.get("lr", 1e-3)),
            weight_decay=float(cp.get("weight_decay", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(cp.get("scheduler_patience", 10)),
            factor=float(cp.get("scheduler_factor", 0.5)),
            min_lr=float(cp.get("min_lr", 1e-5)),
        )
        criterion = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()

        max_epochs = int(cp.get("max_epochs", 100))
        patience = int(cp.get("patience", 25))
        grad_clip = float(cp.get("grad_clip", 1.0))

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        self.history = []

        for epoch in range(max_epochs):
            self.net.train()
            train_loss_total, train_n = 0.0, 0
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
                train_loss_total += float(loss.item()) * len(yb)
                train_n += len(yb)

            val_loss = self._eval_loss(val_dl, criterion)
            scheduler.step(val_loss)
            train_loss = train_loss_total / train_n if train_n else float("nan")
            lr_now = float(optimizer.param_groups[0]["lr"])
            self.history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": float(val_loss),
                "lr": lr_now,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                log.info("  epoch %3d  train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                         epoch + 1, train_loss, val_loss, lr_now)

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
            log.info("CNN lead=%02d val rmse=%.4f corr=%.4f r2=%.4f",
                     self.lead, val_m["rmse"], val_m["corr"], val_m["r2"])
        else:
            from src.models.metrics import classification_metrics
            val_m = classification_metrics(y_val, y_val_pred)
            log.info("CNN lead=%02d val acc=%.4f bss=%.4f",
                     self.lead, val_m["accuracy"], val_m["bss"])

        return {"best_val_loss": float(best_val_loss), **{f"val_{k}": v for k, v in val_m.items()}}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions: regression values or classification probabilities."""
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

    def save(self, directory: str | Path) -> Path:
        """Save model to directory/cnn_lead{L:02d}_{task}.pt."""
        import torch
        if self.net is None:
            raise RuntimeError("Model not fitted.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"cnn_lead{self.lead:02d}_{self.task}.pt"
        torch.save({"state_dict": self.net.state_dict(), "meta": self._meta}, str(path))
        log.info("Saved CNN model -> %s", path)
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
            head_hidden=meta.get("head_hidden"),
            n_head_layers=meta.get("n_head_layers", 1),
        ).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        log.info("Loaded CNN model <- %s", path)

    def _make_loader(self, X, y, shuffle, batch_size):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            pin_memory=False,
        )

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
