"""
Pull all NYC Open Data datasets for the Social Services Intelligence Engine.
Saves to raw/

Usage:
    python pull_all.py                          # all datasets
    python pull_all.py pluto doe_schools        # specific ones
    NYC_APP_TOKEN=xxx python pull_all.py        # with token (recommended)
"""

import os, sys, time, logging
import pandas as pd
from sodapy import Socrata
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(Path(__file__).resolve().parent / "pull.log")),
    ],
)
log = logging.getLogger(__name__)

RAW = Path(__file__).resolve().parent / "raw"
RAW.mkdir(parents=True, exist_ok=True)

APP_TOKEN = os.environ.get("NYC_APP_TOKEN")
client = Socrata("data.cityofnewyork.us", APP_TOKEN, timeout=120)

# ──────────────────────────────────────────────
# Dataset registry
# (name, socrata_id, limit, where_clause, select_cols or None)
# ──────────────────────────────────────────────
DATASETS = {
    # ── Core spatial base ──────────────────────────────────────────
    "pluto": (
        "64uk-42ks", 900_000, None,
        "bbl,address,borough,zipcode,landuse,yearbuilt,numfloors,"
        "unitsres,lotarea,bldgarea,ownername,latitude,longitude",
    ),

    # ── Shelters ────────────────────────────────────────────────────
    "dhs_shelter_census": (
        "3pjg-ncn9", 10_000, None, None,   # daily census: occupancy per shelter
    ),
    "dhs_facilities": (
        "b922-gxih", 5_000, None, None,    # shelter facility list
    ),
    "homeless_dropin": (
        "bmxf-3rd4", 500, None, None,      # drop-in centers
    ),
    "domestic_violence": (
        "5ziv-wcy4", 500, None, None,      # DV resource directory
    ),

    # ── Food ────────────────────────────────────────────────────────
    "hra_resources": (
        "7btz-mnc8", 5_000, None, None,    # food banks, pantries, soup kitchens
    ),

    # ── Health ──────────────────────────────────────────────────────
    "dohmh_facilities": (
        "ji82-xba5", 10_000, None, None,   # DOHMH-licensed facilities
    ),
    "hhc_hospitals": (
        "f7b6-v6v3", 500, None, None,      # NYC Health + Hospitals (public system)
    ),

    # ── Benefits / Social Services Offices ─────────────────────────
    "hra_benefits_centers": (
        "9d9t-bmk7", 500, None, None,      # SNAP/Medicaid/cash enrollment offices
    ),
    "womens_resources": (
        "pqg4-dm6b", 2_000, None, None,    # legal aid, DV, language services
    ),

    # ── Education ───────────────────────────────────────────────────
    "doe_schools": (
        "wg9x-pef3", 2_000, None, None,
    ),

    # ── Transit ─────────────────────────────────────────────────────
    "mta_stations": (
        "kk4q-3rt2", 2_000, None, None,   # subway + bus stops
    ),

    # ── Emergency sites ─────────────────────────────────────────────
    "hurricane_evacuation_centers": (
        "ayer-cga7", 1_000, None, None,   # pre-designated evacuation/overflow sites
    ),
    "cooling_centers": (
        "h2bn-gu9k", 500, None, None,     # summer cooling / winter warming
    ),

    # ── Housing ─────────────────────────────────────────────────────
    "nycha_developments": (
        "phvi-damg", 500, None, None,     # 218 NYCHA developments, 400K+ residents
    ),

    # ── Safety / Quality signals ─────────────────────────────────────
    "nypd_complaints": (
        "qgea-i56i", 500_000,
        "cmplnt_fr_dt>'2023-01-01'",
        "cmplnt_num,cmplnt_fr_dt,ofns_desc,boro_nm,latitude,longitude",
    ),
    "311_hpd": (
        "erm2-nwe9", 300_000,
        "created_date>'2024-01-01' AND agency='HPD'",
        "unique_key,created_date,complaint_type,descriptor,"
        "incident_address,borough,latitude,longitude",
    ),
}


def pull(name, dataset_id, limit, where, select):
    out = RAW / f"{name}.parquet"
    if out.exists():
        mb = out.stat().st_size / 1e6
        log.info("SKIP  %-30s already exists (%.1f MB)", name, mb)
        return True

    log.info("START %-30s id=%s limit=%d", name, dataset_id, limit)
    t0 = time.time()
    frames = []
    chunk = 50_000
    offset = 0

    try:
        while offset < limit:
            kwargs = {
                "limit": min(chunk, limit - offset),
                "offset": offset,
            }
            if where:
                kwargs["where"] = where
            if select:
                kwargs["select"] = select

            rows = client.get(dataset_id, **kwargs)
            if not rows:
                break
            frames.append(pd.DataFrame.from_records(rows))
            offset += len(rows)
            log.info("  %-28s %d rows pulled", name, offset)
            if len(rows) < chunk:
                break
            time.sleep(0.3)

        if not frames:
            log.warning("EMPTY %-30s no rows returned", name)
            return False

        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(out, index=False)
        mb = out.stat().st_size / 1e6
        elapsed = time.time() - t0
        log.info("DONE  %-30s %d rows  %.1f MB  %.0fs", name, len(df), mb, elapsed)
        return True

    except Exception as e:
        log.error("FAIL  %-30s %s", name, e)
        return False


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(DATASETS.keys())
    unknown = [t for t in targets if t not in DATASETS]
    if unknown:
        log.error("Unknown datasets: %s. Valid: %s", unknown, list(DATASETS.keys()))
        sys.exit(1)

    log.info("Token: %s", "SET" if APP_TOKEN else "NOT SET (throttled, set NYC_APP_TOKEN)")
    log.info("Saving to: %s", RAW)
    log.info("Datasets to pull: %s", targets)

    results = {}
    for name in targets:
        results[name] = pull(name, *DATASETS[name])

    log.info("\n=== Summary ===")
    for name, ok in results.items():
        p = RAW / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            log.info("  %-30s %6d rows  %.1f MB  %s",
                     name, len(df), p.stat().st_size / 1e6,
                     "OK" if ok else "SKIP")
        else:
            log.info("  %-30s MISSING", name)


if __name__ == "__main__":
    main()
