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
    """Download subsurface ocean heat content from ORAS5 via the Copernicus CDS API.

    Uses `ocean_heat_content_for_the_upper_300m` (OHC300) as a proxy for D20
    (depth of 20°C isotherm). OHC300 is physically related to thermocline depth
    and is a well-established ENSO indicator. `depth_of_20c_isotherm` is no longer
    available in the new CDS API for ORAS5.

    ORAS5 has two product types on CDS:
      - 'consolidated': 1958–2021 (quality-controlled, delayed)
      - 'operational': 2019–present (near-real-time)
    Years spanning both periods are split into two requests then merged.

    CDS returns a ZIP archive for this variable; extraction is handled automatically.

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
        import xarray as xr
        import tempfile
        import zipfile
        import glob as _glob
    except ImportError:
        raise ImportError("cdsapi / xarray not installed.")

    start, end = cfg["data"]["time_slice"]
    start_yr, end_yr = int(start[:4]), int(end[:4])
    lat_min, lat_max = cfg["data"]["lat_slice"]
    lon_min, lon_max = cfg["data"]["lon_slice"]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"d20_monthly_{start_yr}_{end_yr}.nc"
    if out_path.exists():
        log.info("Already exists — skipping: %s", out_path)
        return out_path

    months = [f"{m:02d}" for m in range(1, 13)]
    c = cdsapi.Client()

    # ORAS5 consolidated ends at 2021; operational covers 2019–present.
    # CDS enforces a per-request cost limit — download one year at a time to stay
    # within the limit (the full 43-year consolidated request returns 403).
    CONSOLIDATED_END = 2021
    OPERATIONAL_START = 2019

    def _fetch_one_year(product_type: str, year: int) -> "xr.Dataset":
        log.info("ORAS5 CDS request: product_type=%s  year=%d", product_type, year)
        tmpdir_yr = Path(tempfile.mkdtemp())
        zip_path = str(tmpdir_yr / f"oras5_{product_type}_{year}.zip")
        c.retrieve(
            "reanalysis-oras5",
            {
                "product_type": product_type,
                "vertical_resolution": "single_level",
                "variable": "ocean_heat_content_for_the_upper_300m",
                "year": [str(year)],
                "month": months,
            },
            zip_path,
        )
        # CDS returns a ZIP — extract it
        nc_dir = tmpdir_yr / "extracted"
        nc_dir.mkdir()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(nc_dir))
        nc_files = sorted(_glob.glob(str(nc_dir / "*.nc")))
        if not nc_files:
            raise RuntimeError(f"No .nc files found in ORAS5 zip for {product_type} {year}")
        return xr.open_mfdataset(nc_files, combine="by_coords")

    parts = []

    # Consolidated portion (start_yr–2021) — one year per request
    for yr in range(start_yr, min(end_yr, CONSOLIDATED_END) + 1):
        parts.append(_fetch_one_year("consolidated", yr))

    # Operational portion (2022–end_yr) — one year per request
    if end_yr > CONSOLIDATED_END:
        op_start = max(start_yr, OPERATIONAL_START, CONSOLIDATED_END + 1)
        for yr in range(op_start, end_yr + 1):
            parts.append(_fetch_one_year("operational", yr))

    if not parts:
        raise RuntimeError("No ORAS5 data retrieved.")

    ds = xr.concat(parts, dim="time").sortby("time") if len(parts) > 1 else parts[0]

    # Rename whatever variable is present to "d20" (proxy), select tropical Pacific
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]

    # Spatial subset — handle both [-180,180] and [0,360] lon conventions
    if float(da.lon.min()) < 0:
        da = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min - 360, lon_max - 360))
    else:
        da = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    da = da.rename("d20")
    da.attrs.update({
        "long_name": "Ocean heat content upper 300 m (proxy for D20)",
        "units": ds[var_name].attrs.get("units", "J m-2"),
        "source": "ORAS5 via ECMWF CDS",
    })

    encoding = {"d20": {"zlib": True, "complevel": 4}}
    da.to_netcdf(str(out_path), encoding=encoding)
    for p in parts:
        p.close()

    log.info("Saved ORAS5 OHC300 (D20 proxy): %s", out_path)
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
