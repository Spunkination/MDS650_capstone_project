"""
AEMO MMSDM data collection script using NEMOSIS.

Downloads DISPATCHREGIONSUM data for NSW1 region at 5-minute resolution.

Required variables:
  - TOTALINTERMITTENTGENERATION (MW) — total intermittent generation
  - SEMISCHEDULE_CLEAREDMW (MW)     — utility-scale solar cleared MW
  - SS_SOLAR_CLEAREDMW (MW)         — solar-specific cleared MW (if available)
  - SS_SOLAR_UIGF (MW)              — solar unscheduled intermittent gen forecast (if available)
  - INTERVENTION                    — retain the normal dispatch scenario only

Period: 2019-01-01 to 2025-12-31 (inclusive).
"""

import os
import logging
from datetime import datetime

import pandas as pd
from nemosis import data_fetch_methods as dfm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR = "data/raw"
CACHE_DIR = os.path.join(RAW_DIR, "aemo_cache")

START_DATE = "2019/01/01 00:00:00"
END_DATE   = "2025/12/31 23:55:00"

OUTPUT_FILE = os.path.join(RAW_DIR, "aemo_dispatchregionsum_nsw1.csv")

SELECT_COLUMNS = [
    "SETTLEMENTDATE",
    "REGIONID",
    "TOTALINTERMITTENTGENERATION",
    "SEMISCHEDULE_CLEAREDMW",
    "SS_SOLAR_UIGF",
    "SS_SOLAR_CLEAREDMW",
    "SS_WIND_UIGF",
    "SS_WIND_CLEAREDMW",
    "INTERVENTION",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def collect_aemo_data():
    """Download and filter AEMO DISPATCHREGIONSUM data for NSW1."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    log.info("Downloading DISPATCHREGIONSUM from %s to %s ...",
             START_DATE[:10], END_DATE[:10])
    log.info("This may take a while (NEMOSIS downloads year-month zips).")

    raw = dfm.dynamic_data_compiler(
        start_time=START_DATE,
        end_time=END_DATE,
        table_name="DISPATCHREGIONSUM",
        raw_data_location=CACHE_DIR,
        select_columns="all",    # get all columns, filter below
        fformat="csv",
        keep_csv=False,          # clean up raw CSVs after feather build
    )

    log.info("Raw data: %d rows × %d columns", raw.shape[0], raw.shape[1])

    # Filter to NSW1
    nsw1 = raw[raw["REGIONID"] == "NSW1"].copy()
    log.info("NSW1 only: %d rows", len(nsw1))

    # Intervention rows can duplicate normal dispatch timestamps with different values.
    if "INTERVENTION" in nsw1.columns:
        intervention = pd.to_numeric(nsw1["INTERVENTION"], errors="coerce").fillna(0)
        removed = int(intervention.ne(0).sum())
        nsw1 = nsw1.loc[intervention.eq(0)].copy()
        log.info("Removed %d intervention-scenario rows", removed)

    # Keep only useful columns
    available = [c for c in SELECT_COLUMNS if c in nsw1.columns]
    missing = [c for c in SELECT_COLUMNS if c not in nsw1.columns]
    if missing:
        log.warning("Columns not available in this period: %s", missing)

    df = nsw1[available].copy()

    # AEMO SETTLEMENTDATE uses the fixed UTC+10 market clock. Preserve that
    # meaning explicitly rather than attaching a false UTC timezone.
    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"]).dt.tz_localize(None)
    df.sort_values("SETTLEMENTDATE", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Rename for consistency
    df.rename(columns={
        "SETTLEMENTDATE": "timestamp_aest",
        "REGIONID": "region",
    }, inplace=True)

    df.to_csv(OUTPUT_FILE, index=False)
    log.info("✓ Saved %d rows → %s", len(df), OUTPUT_FILE)

    # Quick summary
    log.info("Date range: %s → %s",
             df["timestamp_aest"].min(), df["timestamp_aest"].max())
    log.info("Columns: %s", list(df.columns))

    return df


if __name__ == "__main__":
    collect_aemo_data()
