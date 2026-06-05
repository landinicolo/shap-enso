"""Download the NOAA CPC Niño 3.4 monthly anomaly index.

The index is fetched from the NOAA CPC ASCII file (ERSSTv5 basis, 1950–present)
and saved as a CSV to data/raw/noaa/nino34_monthly.csv.

Usage
-----
    python data/download_noaa_indices.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# NOAA CPC ERSSTv5 Niño indices (columns: YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM)
_PRIMARY_URL = (
    "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
)
# PSL fallback — longer record, anomaly only for Niño 3.4
_FALLBACK_URL = (
    "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.anom.data"
)


def _parse_cpc_ascii(text: str) -> pd.DataFrame:
    """Parse the NOAA CPC multi-Niño ASCII table."""
    lines = [l for l in text.strip().splitlines() if not l.strip().startswith("YR")]
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 10:
            continue
        yr, mo = int(parts[0]), int(parts[1])
        # Column layout: YR MON N12 AN12 N3 AN3 N4 AN4 N34 AN34
        try:
            rows.append({
                "year":        yr,
                "month":       mo,
                "nino12":      float(parts[2]),
                "nino12_anom": float(parts[3]),
                "nino3":       float(parts[4]),
                "nino3_anom":  float(parts[5]),
                "nino4":       float(parts[6]),
                "nino4_anom":  float(parts[7]),
                "nino34":      float(parts[8]),
                "nino34_anom": float(parts[9]),
            })
        except (ValueError, IndexError):
            continue

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    df = df.set_index("time").drop(columns=["year", "month"])
    return df


def _parse_psl_fallback(text: str) -> pd.DataFrame:
    """Parse the PSL Niño 3.4 anomaly long-record file (year + 12 monthly values)."""
    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) < 13 or not parts[0].isdigit():
            continue
        yr = int(parts[0])
        for mo, val in enumerate(parts[1:13], start=1):
            try:
                v = float(val)
                if abs(v) > 50:    # sentinel for missing
                    v = float("nan")
                rows.append({"year": yr, "month": mo, "nino34_anom": v})
            except ValueError:
                continue
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
    return df.set_index("time")[["nino34_anom"]]


def download_nino34(cfg: dict, out_dir: Path) -> Path:
    """Fetch the Niño 3.4 monthly anomaly index and save to CSV.

    Args:
        cfg: Full config dict (used for time_slice filtering).
        out_dir: Directory for NOAA index files.

    Returns:
        Path to the saved CSV file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nino34_monthly.csv"
    if out_path.exists():
        log.info("Already exists — skipping: %s", out_path)
        return out_path

    df: pd.DataFrame | None = None

    # Try primary URL
    try:
        log.info("Fetching NOAA CPC Niño 3.4 index from %s ...", _PRIMARY_URL)
        resp = requests.get(_PRIMARY_URL, timeout=30)
        resp.raise_for_status()
        df = _parse_cpc_ascii(resp.text)
        log.info("Parsed %d monthly records from CPC ASCII file.", len(df))
    except Exception as exc:
        log.warning("Primary URL failed (%s) — trying PSL fallback ...", exc)

    if df is None:
        try:
            resp = requests.get(_FALLBACK_URL, timeout=30)
            resp.raise_for_status()
            df = _parse_psl_fallback(resp.text)
            log.info("Parsed %d monthly records from PSL fallback.", len(df))
        except Exception as exc:
            raise RuntimeError(
                f"Both NOAA CPC and PSL URLs failed.\nLast error: {exc}"
            )

    # Trim to configured time slice
    t_start, t_end = cfg["data"]["time_slice"]
    df = df.loc[t_start:t_end]

    df.to_csv(out_path)
    log.info(
        "Saved Niño 3.4 index (%d records, %s – %s) → %s",
        len(df), df.index[0].strftime("%Y-%m"), df.index[-1].strftime("%Y-%m"), out_path,
    )
    return out_path


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    out_dir = Path(cfg["data"]["raw_dir"]) / "noaa"
    download_nino34(cfg, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NOAA Niño 3.4 monthly index")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
