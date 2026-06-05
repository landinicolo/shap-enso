"""End-to-end preprocessing pipeline for the SHAP-ENSO project.

Runs data downloads (if files are missing) then executes the full
preprocessing pipeline defined in src/utils/preprocessing.py.

Usage
-----
    # Full pipeline
    python scripts/run_preprocess.py --config configs/default.yaml

    # Skip downloads (raw files already present)
    python scripts/run_preprocess.py --config configs/default.yaml --no-download

    # Skip ERA5 (only rerun preprocessing on existing data)
    python scripts/run_preprocess.py --config configs/default.yaml --preprocess-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.preprocessing import run_preprocessing

log = get_logger(__name__)


def _download_all(cfg: dict) -> None:
    """Run all download scripts, skipping files that already exist."""
    from data.download_era5 import main as dl_era5
    from data.download_d20 import main as dl_d20
    from data.download_noaa_indices import main as dl_noaa

    cfg_path = cfg["_config_path"]   # injected below

    log.info("=" * 60)
    log.info("Step 1/3: Downloading ERA5 predictors")
    log.info("=" * 60)
    dl_era5(cfg_path)

    log.info("=" * 60)
    log.info("Step 2/3: Downloading D20 (source: %s)", cfg["data"]["d20_source"])
    log.info("=" * 60)
    dl_d20(cfg_path)

    log.info("=" * 60)
    log.info("Step 3/3: Downloading NOAA Niño 3.4 index")
    log.info("=" * 60)
    dl_noaa(cfg_path)


def main(cfg_path: str, no_download: bool = False, preprocess_only: bool = False) -> None:
    cfg = load_config(cfg_path)
    cfg["_config_path"] = cfg_path   # pass path through for sub-calls

    if not (no_download or preprocess_only):
        _download_all(cfg)

    log.info("=" * 60)
    log.info("Preprocessing: building Zarr stores and feature matrices")
    log.info("=" * 60)
    outputs = run_preprocessing(cfg)

    log.info("=" * 60)
    log.info("Done. Summary:")
    for name, path in outputs.items():
        log.info("  %-35s %s", name, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP-ENSO preprocessing pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip download steps (raw files must already exist)",
    )
    parser.add_argument(
        "--preprocess-only", action="store_true",
        help="Only run preprocessing — same as --no-download",
    )
    args = parser.parse_args()
    main(args.config, args.no_download, args.preprocess_only)
