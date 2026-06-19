"""Compile test-set skill metrics for all trained models into data/metrics.csv.

For each (model_type, lead, task) combination, loads the SHAP Zarr store
(which contains predictions) and the preprocessed Niño3.4 target, aligns
on common times, and computes RMSE/MAE/corr/R².

Usage
-----
    python scripts/compile_metrics.py --config configs/default.yaml
    python scripts/compile_metrics.py --config configs/default.yaml --task classification
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.shap_analysis.aggregate import (
    load_shap_store,
    compute_metrics_from_shap_store,
)

log = get_logger(__name__)

MODEL_TYPES = ["xgboost", "lstm", "cnn"]
LEADS       = [3, 6, 12]


def compile_metrics(cfg: dict, tasks: list[str]) -> pd.DataFrame:
    from src.utils.io_utils import load_zarr

    shap_dir  = Path(cfg["shap"]["output_dir"])
    proc_dir  = Path(cfg["data"]["processed_dir"])
    target_ds = load_zarr(proc_dir / "target_nino34.zarr")

    rows = []
    for model_type in MODEL_TYPES:
        for lead in LEADS:
            for task in tasks:
                try:
                    ds = load_shap_store(shap_dir, model_type, lead, task)
                    m  = compute_metrics_from_shap_store(ds, target_ds)
                    row = {
                        "model_type":  model_type,
                        "lead_months": lead,
                        "task":        task,
                        "n_samples":   int(ds.sizes.get("time", 0)),
                        **{k: round(float(v), 6) for k, v in m.items()},
                    }
                    rows.append(row)
                    log.info(
                        "%s lead=%02d %-14s  corr=%.3f  rmse=%.3f",
                        model_type, lead, task, m.get("corr", np.nan), m.get("rmse", np.nan),
                    )
                except FileNotFoundError:
                    log.warning("SHAP store missing: %s lead=%d %s — skipping", model_type, lead, task)
                except Exception as exc:
                    log.error("Error for %s lead=%d %s: %s", model_type, lead, task, exc)

    df = pd.DataFrame(rows)
    return df


def main(cfg_path: str, tasks: list[str]) -> None:
    cfg = load_config(cfg_path)
    df  = compile_metrics(cfg, tasks)

    if df.empty:
        log.warning("No metrics compiled — check that SHAP stores exist.")
        return

    out_dir = Path(cfg["experiment"]["output_dir"]) / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved metrics → %s  (%d rows)", out_path, len(df))
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile test-set skill metrics")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--task", nargs="+", default=["regression"],
        choices=["regression", "classification"],
        help="Tasks to compile metrics for",
    )
    args = parser.parse_args()
    main(args.config, args.task)
