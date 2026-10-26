"""
Data collection script for renewables.ninja API (v2).

Collects solar PV, weather, and wind data for NSW regions.

Variables:
  - PV (raw=true): electricity, irradiance_direct, irradiance_diffuse, temperature
  - Weather:       swgdn (GHI), cldtot (cloud cover)
  - Wind (raw):    wind_speed

API limits: sustained 50/hour, burst 1/second.
Strategy: 3 calls per batch (PV+weather+wind), then sleep 216s to stay under 50/hr.
Max date range per request: 1 year (P1Y).
"""

import os
import time
import logging
from datetime import datetime, timezone

import requests
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = "https://www.renewables.ninja/api/data"
API_KEY = os.getenv("RENEWABLES_NINJA_API_KEY", "")
HEADERS = {"Authorization": f"Token {API_KEY}"}

NSW_REGIONS = [
    {"name": "Sydney",        "lat": -33.8688, "lon": 151.2093},
    {"name": "Newcastle",     "lat": -32.9266, "lon": 151.7787},
    {"name": "Wollongong",    "lat": -34.4266, "lon": 150.8907},
    {"name": "Central Coast", "lat": -33.3166, "lon": 151.2093},
    {"name": "Dubbo",         "lat": -32.2466, "lon": 148.5857},
    {"name": "Orange",        "lat": -33.2866, "lon": 149.0857},
    {"name": "Bathurst",      "lat": -33.4166, "lon": 149.5857},
    {"name": "Tamworth",      "lat": -31.0866, "lon": 150.9257},
]

YEAR_START = 2020
YEAR_END   = 2025

RAW_DIR = "data/raw"

# Rate-limit compliance
CALL_GAP_S   = 2.0      # between calls within a batch (burst max 1/sec, use 2 for safety)
BATCH_SIZE   = 3         # calls per batch (PV + weather + wind)
SECS_PER_CALL = 72       # 3600/50 ≈ 72 seconds per call to sustain 50/hr
BATCH_CYCLE_S = BATCH_SIZE * SECS_PER_CALL  # 216 seconds per batch cycle

MAX_RETRIES = 5
RETRY_BASE_S = 30


def _get_retry_after(resp) -> int | None:
    """Extract Retry-After seconds from response headers or body."""
    # Header
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            return int(ra)
        except ValueError:
            pass
    # Body text
    body = resp.text
    import re
    m = re.search(r"available in (\d+) seconds", body)
    if m:
        return int(m.group(1))
    return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _fetch_json(url: str, params: dict, retries: int = MAX_RETRIES) -> dict:
    """GET JSON with retry + rate-limit backoff (respects Retry-After)."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
        except requests.RequestException as e:
            log.warning("  Request failed (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(RETRY_BASE_S * attempt)
            continue

        if resp.status_code == 429:
            wait = _get_retry_after(resp)
            if wait is None or wait < 5:
                wait = RETRY_BASE_S * (2 ** (attempt - 1))
            log.warning("  Rate-limited (429, attempt %d/%d). Waiting %ds ...",
                        attempt, retries, wait)
            time.sleep(wait + 2)
            continue

        if resp.status_code != 200:
            log.warning("  HTTP %d (attempt %d/%d): %s",
                        resp.status_code, attempt, retries, resp.text[:200])
            if attempt < retries:
                time.sleep(RETRY_BASE_S * attempt)
            continue

        return resp.json()

    raise RuntimeError(f"Failed after {retries} retries")


def _convert_ms_timestamp(ts_ms: str) -> datetime:
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Per region-year
# ---------------------------------------------------------------------------


def fetch_region_year(region: dict, year: int) -> pd.DataFrame:
    """Fetch PV, weather, and wind data for one region-year."""
    date_from = f"{year}-01-01"
    date_to   = f"{year}-12-31"
    lat, lon  = region["lat"], region["lon"]
    name      = region["name"]


    pv = _fetch_json(f"{API_BASE}/pv", {
        "lat": lat, "lon": lon,
        "capacity": 1.0, "dataset": "merra2",
        "system_loss": 0.1, "tracking": 0, "tilt": 35, "azim": 180,
        "format": "json", "raw": True,
        "date_from": date_from, "date_to": date_to,
    })
    time.sleep(CALL_GAP_S)

    wx = _fetch_json(f"{API_BASE}/weather", {
        "lat": lat, "lon": lon,
        "dataset": "merra2", "format": "json",
        "var_swgdn": True, "var_cldtot": True,
        "date_from": date_from, "date_to": date_to,
    })
    time.sleep(CALL_GAP_S)

    wn = _fetch_json(f"{API_BASE}/wind", {
        "lat": lat, "lon": lon,
        "capacity": 1.0, "dataset": "merra2",
        "format": "json", "raw": True,
        "height": 100, "turbine": "Vestas V80 2000",
        "date_from": date_from, "date_to": date_to,
    })

    # Align on shared timestamps
    ts_keys = (set(pv.get("data", {}))
               & set(wx.get("data", {}))
               & set(wn.get("data", {})))

    if not ts_keys:
        log.warning("    No overlapping timestamps for %s %s", name, year)
        return pd.DataFrame()

    rows = []
    for ts_ms in sorted(ts_keys):
        p = pv["data"][ts_ms]
        w = wx["data"][ts_ms]
        n = wn["data"][ts_ms]
        rows.append({
            "timestamp_utc":       _convert_ms_timestamp(ts_ms),
            "electricity":         p.get("electricity"),
            "irradiance_direct":   p.get("irradiance_direct"),
            "irradiance_diffuse":  p.get("irradiance_diffuse"),
            "temperature":         p.get("temperature"),
            "ghi":                 w.get("swgdn"),
            "cloud_cover":         w.get("cldtot"),
            "wind_speed":          n.get("wind_speed"),
            "region":              name,
            "year":                year,
        })

    df = pd.DataFrame(rows)
    log.info("    → %7d hourly records", len(df))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def collect_all_data():
    if not API_KEY:
        log.error("Set RENEWABLES_NINJA_API_KEY=<token> and re-run.")
        return

    os.makedirs(RAW_DIR, exist_ok=True)

    # Check what's already been collected so we can skip
    existing = set()
    for f in os.listdir(RAW_DIR):
        if f.startswith("renewables_ninja_") and f.endswith(".csv"):
            region_name = f.replace("renewables_ninja_", "").replace(".csv", "").replace("_", " ").title()
            existing.add(region_name)
            # Normalise: handle "Central Coast" -> "Central Coast"
            # Our region names: Sydney, Newcastle, Wollongong, Central Coast, Dubbo, Orange, Bathurst, Tamworth
    # More reliable: just normalise by splitting on '_' and capitalising, then rejoining
    existing = set()
    for f in os.listdir(RAW_DIR):
        if f.startswith("renewables_ninja_") and f.endswith(".csv"):
            name_part = f[len("renewables_ninja_"):-len(".csv")]
            # Convert "central_coast" -> "Central Coast" etc
            formatted = " ".join(w.capitalize() for w in name_part.split("_"))
            existing.add(formatted)

    for region in NSW_REGIONS:
        name = region["name"]
        if name in existing:
            log.info("⏭  %-20s already collected, skipping", name)
            continue

        region_dfs = []
        log.info("── %s ─────────────────", name)

        for year in range(YEAR_START, YEAR_END + 1):
            log.info("  [%s %d]", name, year)
            batch_start = time.monotonic()

            try:
                df = fetch_region_year(region, year)
            except Exception as e:
                log.error("  FAILED: %s", e)
                continue

            if not df.empty:
                region_dfs.append(df)

            # Rate-limit: sleep until the batch cycle completes
            elapsed = time.monotonic() - batch_start
            sleep_needed = BATCH_CYCLE_S - elapsed
            if year < YEAR_END and sleep_needed > 0:
                log.info("  Rate-limit pause: %.0fs (%d calls/%dh window) ...",
                         sleep_needed, BATCH_SIZE, 1)
                time.sleep(sleep_needed)

        if region_dfs:
            combined = pd.concat(region_dfs, ignore_index=True)
            combined.sort_values("timestamp_utc", inplace=True)
            fname = f"renewables_ninja_{name.lower().replace(' ', '_')}.csv"
            fpath = os.path.join(RAW_DIR, fname)
            combined.to_csv(fpath, index=False)
            log.info("✓ %-20s  %5d rows → %s", name, len(combined), fname)
        else:
            log.warning("✗ %s — no data collected", name)

    log.info("Done.")


if __name__ == "__main__":
    collect_all_data()
