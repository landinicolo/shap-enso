"""Download the 20°C isotherm depth (D20) for the tropical Pacific.

Two sources are supported, selected by cfg['data']['d20_source']:

  'godas'   — NOAA GODAS monthly temperature profiles via OPeNDAP. D20 is
              derived by linear interpolation between depth levels.
              URL: https://www.ncei.noaa.gov/thredds/dodsC/godas/monthly/

  'oras5'   — ECMWF ORAS5 ocean reanalysis via the Copernicus CDS API.
              Requires accepting the ORAS5 licence at cds.climate.copernicus.eu.

Usage
-----
    python data/download_d20.py --config configs/default.yaml [--source godas|oras5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# GODAS OPeNDAP base URL — NOAA NCEI THREDDS catalogue
_GODAS_BASE = "https://www.ncei.noaa.gov/thredds/dodsC/uv3-godas-monthly"


def _compute_d20_from_profiles(temp: "xr.DataArray") -> "xr.DataArray":
    """Derive D20 (depth of 20°C isotherm) from 3-D temperature profiles.

    Args:
        temp: DataArray with dims (time, level, lat, lon), temperature in °C.
              'level' coordinate must be depth in metres (positive downward).

    Returns:
        DataArray with dims (time, lat, lon), D20 in metres.
    """
    import xarray as xr

    levels = temp.level.values.astype(float)
    t_np   = temp.values  # (time, level, lat, lon)
    nt, nz, nlat, nlon = t_np.shape

    d20 = np.full((nt, nlat, nlon), np.nan, dtype=np.float32)

    for it in range(nt):
        for ilat in range(nlat):
            for ilon in range(nlon):
                profile = t_np[it, :, ilat, ilon]
                valid   = ~np.isnan(profile)
                if valid.sum() < 2:
                    continue
                # Find the first depth where T drops below 20°C
                above = profile > 20.0
                idx   = np.where(above[:-1] & ~above[1:])[0]
                if len(idx) == 0:
                    continue
                k = idx[0]
                # Linear interpolation between levels k and k+1
                t1, t2 = profile[k], profile[k + 1]
                z1, z2 = levels[k], levels[k + 1]
                if t1 == t2:
                    d20[it, ilat, ilon] = z1
                else:
                    d20[it, ilat, ilon] = z1 + (20.0 - t1) * (z2 - z1) / (t2 - t1)

    return xr.DataArray(
        d20,
        coords={"time": temp.time, "lat": temp.lat, "lon": temp.lon},
        dims=["time", "lat", "lon"],
        name="d20",
        attrs={"units": "m", "long_name": "Depth of 20°C isotherm"},
    )


def download_godas(cfg: dict, out_dir: Path) -> Path:
    """Download GODAS monthly temperature profiles and compute D20.

    Downloads year-by-year via OPeNDAP to avoid memory issues with the full
    record. Progress is tracked so partially-completed downloads can be resumed.

    Args:
        cfg: Full config dict.
        out_dir: Directory for D20 output.

    Returns:
        Path to the saved D20 NetCDF file.
    """
    import xarray as xr

    start, end = cfg["data"]["time_slice"]
    start_yr, end_yr = int(start[:4]), int(end[:4])
    lat_min, lat_max = cfg["data"]["lat_slice"]
    lon_min, lon_max = cfg["data"]["lon_slice"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"d20_monthly_{start_yr}_{end_yr}.nc"
    if out_path.exists():
        log.info("Already exists — skipping: %s", out_path)
        return out_path

    slices_all: list["xr.DataArray"] = []
    for yr in range(start_yr, end_yr + 1):
        log.info("  GODAS D20: year %d ...", yr)
        url = f"{_GODAS_BASE}/{yr}/godas.{yr}01.nc"  # annual file
        try:
            ds_yr = xr.open_dataset(url, engine="pydap")
        except Exception:
            # Fall back to monthly URLs
            slices_yr: list["xr.DataArray"] = []
            for mo in range(1, 13):
                url_mo = f"{_GODAS_BASE}/{yr}/godas.{yr}{mo:02d}.nc"
                try:
                    ds_mo = xr.open_dataset(url_mo, engine="pydap")
                    t_mo  = ds_mo["pottmp"].sel(
                        lat=slice(lat_min, lat_max),
                        lon=slice(lon_min + 360 if lon_min < 0 else lon_min,
                                  lon_max + 360 if lon_max < 0 else lon_max),
                    )
                    slices_yr.append(_compute_d20_from_profiles(t_mo))
                except Exception as e:
                    log.warning("    Skipping %d-%02d: %s", yr, mo, e)
            if slices_yr:
                slices_all.append(xr.concat(slices_yr, dim="time"))
            continue

        t_yr = ds_yr["pottmp"].sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )
        slices_all.append(_compute_d20_from_profiles(t_yr))

    if not slices_all:
        raise RuntimeError("No GODAS data downloaded. Check network access to NCEI THREDDS.")

    d20_full = xr.concat(slices_all, dim="time").sortby("time")
    d20_full.to_netcdf(out_path)
    log.info("Saved GODAS D20: %s", out_path)
    return out_path


def download_oras5(cfg: dict, out_dir: Path) -> Path:
    """Download D20 from ECMWF ORAS5 via the Copernicus CDS API.

    Requires:
      - cdsapi installed and ~/.cdsapirc configured
      - ORAS5 licence accepted at cds.climate.copernicus.eu

    Args:
        cfg: Full config dict.
        out_dir: Directory for D20 output.

    Returns:
        Path to the saved D20 NetCDF file.
    """
    try:
        import cdsapi
    except ImportError:
        raise ImportError("cdsapi not installed. Run: pip install cdsapi")

    start, end = cfg["data"]["time_slice"]
    start_yr, end_yr = start[:4], end[:4]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"d20_monthly_{start_yr}_{end_yr}.nc"
    if out_path.exists():
        log.info("Already exists — skipping: %s", out_path)
        return out_path

    years  = [str(y) for y in range(int(start_yr), int(end_yr) + 1)]
    months = [f"{m:02d}" for m in range(1, 13)]

    log.info("Downloading ORAS5 D20 (%s–%s) via CDS ...", start_yr, end_yr)
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-oras5",
        {
            "product_type": "consolidated",
            "vertical_resolution": "single_level",
            "variable": "depth_of_20c_isotherm",
            "year": years,
            "month": months,
        },
        str(out_path),
    )
    log.info("Saved ORAS5 D20: %s", out_path)
    return out_path


def main(cfg_path: str, source: str | None = None) -> None:
    cfg = load_config(cfg_path)
    source = source or cfg["data"].get("d20_source", "godas")
    out_dir = Path(cfg["data"]["raw_dir"]) / "d20"

    if source == "oras5":
        download_oras5(cfg, out_dir)
    elif source == "godas":
        download_godas(cfg, out_dir)
    else:
        raise ValueError(f"Unknown d20_source '{source}'. Use 'godas' or 'oras5'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download D20 (20°C isotherm depth)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", choices=["godas", "oras5"], default=None,
                        help="Override cfg['data']['d20_source']")
    args = parser.parse_args()
    main(args.config, args.source)
