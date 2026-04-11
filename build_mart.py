"""
build_mart.py — Build unified resource mart from cleaned stage/*.parquet files.

Combines all ~14K resource rows into one resource_mart.parquet with:
  - Consistent schema: resource_id, resource_type, name, address, borough, lat, lon
  - Safety score: NYPD complaint density within 500m
  - Quality score: 311 complaint density within 500m (inverted)
  - Nearest transit station + walk distance

Output: data/resource_mart.parquet

Run: python build_mart.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging, sys, time

# GPU acceleration
try:
    import cudf
    import cupy as cp
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    USE_GPU = True
except ImportError:
    USE_GPU = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(Path(__file__).resolve().parent / "build_mart.log")),
    ],
)
log = logging.getLogger(__name__)

STAGE = Path(__file__).resolve().parent / "stage"
DATA  = Path(__file__).resolve().parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load and combine all resource stage files
# ─────────────────────────────────────────────────────────────────────────────
RESOURCE_FILES = [
    "shelters", "food_banks", "hospitals", "dohmh_facilities",
    "schools", "domestic_violence", "benefits_centers",
    "dropin_centers", "nycha", "cooling_centers", "transit_stations",
]

# Columns every resource must have
CORE_COLS = ["resource_id", "resource_type", "name", "address",
             "borough", "latitude", "longitude"]

def load_resources():
    log.info("Step 1/5: Loading resource files…")
    frames = []
    for fname in tqdm(RESOURCE_FILES, desc="Loading stage files", unit="file"):
        path = STAGE / f"{fname}.parquet"
        if not path.exists():
            log.warning("  MISSING: %s", path)
            continue
        df = pd.read_parquet(path)
        if len(df) == 0:
            log.warning("  EMPTY: %s", fname)
            continue
        # Ensure core columns exist
        for col in CORE_COLS:
            if col not in df.columns:
                df[col] = None
        frames.append(df[CORE_COLS + [c for c in df.columns if c not in CORE_COLS]])
        log.info("  %-25s %5d rows", fname, len(df))

    df = pd.concat(frames, ignore_index=True)
    log.info("Combined: %d total resource rows", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Normalize and validate
# ─────────────────────────────────────────────────────────────────────────────
def normalize(df):
    log.info("Step 2/5: Normalizing and validating…")

    df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Drop rows missing coordinates
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(40.4, 40.95)]
    df = df[df["longitude"].between(-74.3, -73.6)]
    log.info("  Dropped %d rows with bad coordinates", before - len(df))

    # Deduplicate by (name, lat rounded to 4 decimal places)
    df["_lat4"] = df["latitude"].round(4)
    df["_lon4"] = df["longitude"].round(4)
    before = len(df)
    df = df.drop_duplicates(subset=["resource_type", "_lat4", "_lon4"])
    df = df.drop(columns=["_lat4", "_lon4"])
    log.info("  Dropped %d spatial duplicates", before - len(df))

    # Re-issue clean sequential resource_ids
    df = df.reset_index(drop=True)
    df["resource_id"] = df["resource_type"].str[:8] + "_" + df.index.astype(str)

    # Fill missing name/address
    df["name"]    = df["name"].fillna("Unknown").str.strip()
    df["address"] = df["address"].fillna("").str.strip()

    log.info("  Final resource count: %d", len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Nearest transit station + walk distance
# ─────────────────────────────────────────────────────────────────────────────
def add_transit_proximity(df):
    log.info("Step 3/5: Computing nearest transit station…")
    transit = pd.read_parquet(STAGE / "transit_stations.parquet")
    transit = transit.dropna(subset=["latitude", "longitude"])

    t_lat = transit["latitude"].values
    t_lon = transit["longitude"].values
    t_id  = transit["resource_id"].values if "resource_id" in transit.columns else transit.index.astype(str).values
    t_name = transit["name"].values if "name" in transit.columns else np.array([""] * len(transit))

    # Approximate degrees → meters (NYC latitude ~40.7)
    LAT_M = 111_320
    LON_M = 85_390  # 111_320 * cos(40.7°)

    nearest_ids   = []
    nearest_names = []
    nearest_dists = []

    resources = df[["latitude", "longitude"]].values
    for lat, lon in tqdm(resources, desc="Nearest transit", unit="resource", miniters=500):
        dlat = (t_lat - lat) * LAT_M
        dlon = (t_lon - lon) * LON_M
        dists = np.sqrt(dlat**2 + dlon**2)
        idx = np.argmin(dists)
        nearest_ids.append(t_id[idx])
        nearest_names.append(t_name[idx])
        nearest_dists.append(round(dists[idx]))

    df["nearest_transit_id"]   = nearest_ids
    df["nearest_transit_name"] = nearest_names
    df["nearest_transit_dist_m"] = nearest_dists
    # Walk time: ~80m/min walking
    df["nearest_transit_walk_min"] = (df["nearest_transit_dist_m"] / 80).round(1)

    log.info("  Median walk to transit: %.0f m", df["nearest_transit_dist_m"].median())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Safety score from NYPD complaints (count within 500m)
# ─────────────────────────────────────────────────────────────────────────────
def _count_within_radius_gpu(resource_coords, complaint_coords, radius_m=500):
    """GPU-accelerated: count complaints within radius of each resource using cuPy."""
    LAT_M, LON_M = 111_320, 85_390
    r_lat = cp.asarray(resource_coords[:, 0]) * LAT_M
    r_lon = cp.asarray(resource_coords[:, 1]) * LON_M
    c_lat = cp.asarray(complaint_coords[:, 0]) * LAT_M
    c_lon = cp.asarray(complaint_coords[:, 1]) * LON_M

    counts = cp.zeros(len(r_lat), dtype=cp.int32)
    # Process in batches to manage GPU memory
    batch_size = 500
    for i in range(0, len(r_lat), batch_size):
        end = min(i + batch_size, len(r_lat))
        # Broadcasting: (batch, 1) - (1, complaints)
        dlat = r_lat[i:end, None] - c_lat[None, :]
        dlon = r_lon[i:end, None] - c_lon[None, :]
        dist_sq = dlat**2 + dlon**2
        counts[i:end] = (dist_sq <= radius_m**2).sum(axis=1)

    return cp.asnumpy(counts)


def _count_within_radius_cpu(resource_coords, complaint_coords, radius_m=500):
    """CPU fallback: count complaints within radius of each resource."""
    LAT_M, LON_M = 111_320, 85_390
    c_lat = complaint_coords[:, 0]
    c_lon = complaint_coords[:, 1]
    dlat_deg = radius_m / LAT_M
    dlon_deg = radius_m / LON_M

    counts = []
    for lat, lon in tqdm(resource_coords, desc="Scoring", unit="resource", miniters=200):
        mask = (
            (c_lat >= lat - dlat_deg) & (c_lat <= lat + dlat_deg) &
            (c_lon >= lon - dlon_deg) & (c_lon <= lon + dlon_deg)
        )
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            counts.append(0)
            continue
        dlat = (c_lat[candidates] - lat) * LAT_M
        dlon = (c_lon[candidates] - lon) * LON_M
        within = (dlat**2 + dlon**2) <= radius_m**2
        counts.append(int(within.sum()))
    return np.array(counts)


def add_safety_score(df):
    log.info("Step 4/5: Computing safety scores from NYPD complaints (%s)…",
             "GPU" if USE_GPU else "CPU")
    nypd = pd.read_parquet(STAGE / "nypd_complaints.parquet")
    nypd = nypd.dropna(subset=["latitude", "longitude"])
    log.info("  NYPD complaints: %d", len(nypd))

    r_coords = df[["latitude", "longitude"]].values
    c_coords = nypd[["latitude", "longitude"]].values

    if USE_GPU:
        counts = _count_within_radius_gpu(r_coords, c_coords, 500)
    else:
        counts = _count_within_radius_cpu(r_coords, c_coords, 500)

    df["n_nypd_500m"] = counts
    p99 = np.percentile(counts, 99) or 1
    df["safety_score"] = (1 - (df["n_nypd_500m"] / p99)).clip(0, 1).round(3)
    log.info("  Median NYPD within 500m: %.0f", df["n_nypd_500m"].median())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Quality score from 311 complaints (count within 500m, inverted)
# ─────────────────────────────────────────────────────────────────────────────
def add_quality_score(df):
    log.info("Step 5/5: Computing quality scores from 311 complaints (%s)…",
             "GPU" if USE_GPU else "CPU")
    c311 = pd.read_parquet(STAGE / "311_complaints.parquet")
    c311 = c311.dropna(subset=["latitude", "longitude"])
    log.info("  311 complaints: %d", len(c311))

    r_coords = df[["latitude", "longitude"]].values
    c_coords = c311[["latitude", "longitude"]].values

    if USE_GPU:
        counts = _count_within_radius_gpu(r_coords, c_coords, 500)
    else:
        counts = _count_within_radius_cpu(r_coords, c_coords, 500)

    df["n_311_500m"] = counts
    p99 = np.percentile(counts, 99) or 1
    df["quality_score"] = (1 - (df["n_311_500m"] / p99)).clip(0, 1).round(3)
    log.info("  Median 311 within 500m: %.0f", df["n_311_500m"].median())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def save_pluto_layer():
    """Save PLUTO spatial layer as second-tier mart (queried at simulation time)."""
    log.info("Saving PLUTO spatial layer…")
    pluto = pd.read_parquet(STAGE / "pluto.parquet")
    # Keep only columns needed for overflow site discovery + building lookup
    keep = ["bbl", "address", "borough", "latitude", "longitude",
            "landuse", "numfloors", "yearbuilt", "ownername",
            "unitsres", "is_overflow_candidate"]
    pluto = pluto[[c for c in keep if c in pluto.columns]]
    out = DATA / "pluto_layer.parquet"
    pluto.to_parquet(out, index=False)
    overflow_count = pluto["is_overflow_candidate"].sum() if "is_overflow_candidate" in pluto.columns else "?"
    log.info("  PLUTO layer: %d lots, %s overflow candidates → %s",
             len(pluto), overflow_count, out)


if __name__ == "__main__":
    t0 = time.time()

    df = load_resources()
    df = normalize(df)
    df = add_transit_proximity(df)
    df = add_safety_score(df)
    df = add_quality_score(df)

    # Coerce any mixed-type object columns that should be numeric
    for col in ["capacity", "total_units"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = DATA / "resource_mart.parquet"
    df.to_parquet(out, index=False)

    save_pluto_layer()
    mb = out.stat().st_size / 1e6
    elapsed = time.time() - t0

    log.info("\n=== Resource Mart Built ===")
    log.info("  Rows      : %d", len(df))
    log.info("  Columns   : %d", len(df.columns))
    log.info("  Size      : %.1f MB", mb)
    log.info("  Time      : %.0f s", elapsed)
    log.info("  Saved to  : %s", out)
    log.info("\n  Resource type breakdown:")
    for rtype, cnt in df["resource_type"].value_counts().items():
        log.info("    %-30s %4d", rtype, cnt)
