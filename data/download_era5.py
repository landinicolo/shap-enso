"""Download ERA5 monthly mean predictors via the Copernicus CDS API.

Setup
-----
1. Register at https://cds.climate.copernicus.eu and accept the ERA5 licence.
2. Install your API key:  echo "url: ...\nkey: ..." > ~/.cdsapirc
3. conda activate shap-enso && python data/download_era5.py --config configs/default.yaml

Each variable is downloaded as a single multi-year NetCDF file. Existing files
are skipped (idempotent).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Map config short names → CDS API long names
ERA5_REQUEST: dict[str, str] = {
    "sst":  "sea_surface_temperature",
    "tauu": "eastward_turbulent_surface_stress",
    "olr":  "top_net_thermal_radiation",
    "slp":  "mean_sea_level_pressure",
}


def _year_range(start: str, end: str) -> list[str]:
    return [str(y) for y in range(int(start[:4]), int(end[:4]) + 1)]


def _month_list() -> list[str]:
    return [f"{m:02d}" for m in range(1, 13)]


def download_variable(var: str, cfg: dict, out_dir: Path) -> Path:
    """Download one ERA5 variable for the full time range defined in *cfg*.

    Args:
        var: Config short name (e.g. 'sst').
        cfg: Full config dict.
        out_dir: Directory for raw ERA5 files.

    Returns:
        Path to the downloaded NetCDF file.
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError("cdsapi not installed. Run: pip install cdsapi")

    if var not in ERA5_REQUEST:
        raise ValueError(f"Unknown ERA5 variable '{var}'. Valid: {list(ERA5_REQUEST)}")

    start, end = cfg["data"]["time_slice"]
    start_yr, end_yr = start[:4], end[:4]

    out_path = out_dir / f"{var}_monthly_{start_yr}_{end_yr}.nc"
    if out_path.exists():
        log.info("Already exists — skipping: %s", out_path)
        return out_path

    out_dir.mkdir(parents=True, exist_ok=True)
    lat_min, lat_max = cfg["data"]["lat_slice"]
    lon_min, lon_max = cfg["data"]["lon_slice"]
    # CDS uses -180/180 lon; convert from 0–360
    lon_min_180 = lon_min - 360 if lon_min > 180 else lon_min
    lon_max_180 = lon_max - 360 if lon_max > 180 else lon_max

    cds_name = ERA5_REQUEST[var]
    log.info("Downloading ERA5 %s (%s–%s) ...", var, start_yr, end_yr)

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": cds_name,
            "year": _year_range(start, end),
            "month": _month_list(),
            "time": "00:00",
            "area": [lat_max, lon_min_180, lat_min, lon_max_180],  # N/W/S/E
            "format": "netcdf",
        },
        str(out_path),
    )
    log.info("Saved: %s", out_path)
    return out_path


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    raw_dir = Path(cfg["data"]["raw_dir"]) / "era5"

    for var in cfg["data"]["era5_variables"]:
        try:
            download_variable(var, cfg, raw_dir)
        except Exception as exc:
            log.error("Failed to download %s: %s", var, exc)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ERA5 predictors via CDS API")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
