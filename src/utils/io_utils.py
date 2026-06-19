"""IO utilities: Zarr, feature-matrix, and metadata persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Zarr helpers
# ---------------------------------------------------------------------------

_DEFAULT_CHUNKS: dict[str, int] = {"time": 120, "lat": 31, "lon": 86}


def save_zarr(
    ds: xr.Dataset,
    path: str | Path,
    chunks: dict[str, int] | None = None,
    mode: str = "w",
) -> None:
    """Save an xr.Dataset to a Zarr store.

    Args:
        ds: Dataset to save.
        path: Destination Zarr store path.
        chunks: Chunking dict. Defaults to (time=120, lat=31, lon=86).
        mode: "w" to overwrite, "w-" to fail if exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = chunks or {k: v for k, v in _DEFAULT_CHUNKS.items() if k in ds.dims}
    # consolidated=True is a zarr v2 feature; v3 supports it experimentally.
    # Use zarr_format=2 for broad compatibility on GLADE.
    try:
        ds.chunk(chunks).to_zarr(str(path), mode=mode, consolidated=True, zarr_format=2)
    except TypeError:
        # Older xarray / zarr without zarr_format kwarg
        ds.chunk(chunks).to_zarr(str(path), mode=mode, consolidated=True)


def load_zarr(path: str | Path) -> xr.Dataset:
    """Load a Zarr store as a dask-backed xr.Dataset."""
    try:
        return xr.open_zarr(str(path), consolidated=True)
    except Exception:
        return xr.open_zarr(str(path), consolidated=False)


# ---------------------------------------------------------------------------
# Feature matrix helpers
# ---------------------------------------------------------------------------

def save_feature_matrix(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    times: np.ndarray,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a feature matrix to a compressed .npz file.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        feature_names: List of length n_features.
        times: Array of time stamps (datetime64 or string) of length n_samples.
        path: Output .npz path (extension added if missing).
        metadata: Optional dict of scalar metadata values stored as separate arrays.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "X": X.astype(np.float32),
        "y": y.astype(np.float32),
        "feature_names": np.array(feature_names, dtype=str),
        "times": np.array(times, dtype=str),
    }
    if metadata:
        for k, v in metadata.items():
            payload[f"meta_{k}"] = np.array([v])

    np.savez_compressed(path, **payload)


def load_feature_matrix(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Load a feature matrix saved by :func:`save_feature_matrix`.

    Returns:
        (X, y, feature_names, times) tuple.
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_names = list(data["feature_names"])
    times = data["times"]
    return X, y, feature_names, times
