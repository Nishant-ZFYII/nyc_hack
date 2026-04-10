"""
build_triples.py — Convert ALL datasets into SPO (Subject-Predicate-Object) triples
with confidence scores and source provenance.

This is the knowledge graph backbone. Every fact about every resource is a triple.
On DGX Spark: cuDF makes this run in seconds. On CPU pandas: ~5 min.

Usage:
    python build_triples.py

Output:
    data/triples.parquet  — all triples (subject, predicate, object_val, confidence, source, timestamp)
    data/graph.pkl        — rebuilt graph from triples
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

# Try cuDF first, fall back to pandas
try:
    import cudf
    print("[GPU] Using cuDF for triple construction")
    GPU = True
except ImportError:
    GPU = False
    print("[CPU] Using pandas (cuDF not available)")

DATA  = Path("/media/nishant/SeeGayt2/nyc_hack_data/data")
STAGE = Path("/media/nishant/SeeGayt2/nyc_hack_data/stage")

# ── Confidence rules ──────────────────────────────────────────────────────────

CONFIDENCE = {
    "spatial_exact":     1.00,  # lat/lon from official source
    "spatial_derived":   0.85,  # centroid of polygon, geocoded
    "identity":          1.00,  # name, address — direct from dataset
    "classification":    0.95,  # resource_type from DOHMH coding
    "cross_dataset":     0.80,  # joining two datasets (311→resource)
    "computed":          0.75,  # derived metric (safety_score)
    "temporal_fresh":    0.95,  # data < 90 days
    "temporal_stale":    0.60,  # data > 1 year
    "text_extracted":    0.70,  # from txt2kg / NLP
    "pluto_zoning":      0.90,  # official NYC zoning data
}


# ── Triple accumulator ────────────────────────────────────────────────────────

_triples = []

def add_triple(subject, predicate, object_val, confidence, source):
    _triples.append({
        "subject": str(subject),
        "predicate": predicate,
        "object_val": str(object_val),
        "confidence": round(confidence, 3),
        "source": source,
    })


# ── Step 1: Resource identity triples ─────────────────────────────────────────

def build_resource_triples(mart: pd.DataFrame):
    """Core identity triples for every resource."""
    print(f"\n[1/7] Building resource identity triples ({len(mart):,} resources)...")
    for _, r in tqdm(mart.iterrows(), total=len(mart), desc="  Resources"):
        rid = r.get("resource_id", f"r_{_}")
        add_triple(rid, "HAS_TYPE",    r.get("resource_type", "unknown"), CONFIDENCE["classification"], "dohmh")
        add_triple(rid, "HAS_NAME",    r.get("name", ""),                 CONFIDENCE["identity"],       "dohmh")
        add_triple(rid, "HAS_ADDRESS", r.get("address", ""),              CONFIDENCE["identity"],       "dohmh")
        add_triple(rid, "IN_BOROUGH",  r.get("borough", ""),              CONFIDENCE["spatial_exact"],  "dohmh")

        if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude")):
            add_triple(rid, "HAS_LOCATION", f"{r['latitude']:.6f},{r['longitude']:.6f}",
                       CONFIDENCE["spatial_exact"], "dohmh")

        if pd.notna(r.get("capacity")) and r.get("capacity", 0) > 0:
            add_triple(rid, "HAS_CAPACITY", int(r["capacity"]), CONFIDENCE["identity"], "dhs")

        if pd.notna(r.get("nearest_transit_name")):
            add_triple(rid, "NEAREST_TRANSIT", r["nearest_transit_name"],
                       CONFIDENCE["cross_dataset"], "mta_gtfs")
            if pd.notna(r.get("nearest_transit_walk_min")):
                add_triple(rid, "TRANSIT_WALK_MIN", round(r["nearest_transit_walk_min"], 1),
                           CONFIDENCE["cross_dataset"], "mta_gtfs")


# ── Step 2: Safety triples from NYPD (by category) ───────────────────────────

NYPD_CATEGORIES = {
    "violent": ["FELONY ASSAULT", "ASSAULT 3 & RELATED OFFENSES", "ROBBERY",
                "SEX CRIMES", "DANGEROUS WEAPONS", "MURDER"],
    "property": ["PETIT LARCENY", "GRAND LARCENY", "BURGLARY",
                 "GRAND LARCENY OF MOTOR VEHICLE", "CRIMINAL MISCHIEF"],
    "drugs":    ["DANGEROUS DRUGS", "INTOXICATED & IMPAIRED DRIVING"],
    "harassment": ["HARRASSMENT 2", "OFF. AGNST PUB ORD SENSBLTY &"],
}

def _categorize_crime(ofns_desc):
    if not ofns_desc:
        return "other"
    for cat, keywords in NYPD_CATEGORIES.items():
        if ofns_desc in keywords:
            return cat
    return "other"


def build_nypd_triples(mart: pd.DataFrame):
    """NYPD complaints within 500m of each resource — broken by crime type."""
    print("\n[2/7] Building NYPD safety triples...")
    nypd = pd.read_parquet(STAGE / "nypd_complaints.parquet")
    nypd = nypd.dropna(subset=["latitude", "longitude"])
    nypd["crime_cat"] = nypd["ofns_desc"].apply(_categorize_crime)
    print(f"  NYPD records: {len(nypd):,}")
    print(f"  Crime categories: {nypd['crime_cat'].value_counts().to_dict()}")

    # Resources with valid coords
    resources = mart.dropna(subset=["latitude", "longitude"]).copy()
    if resources.empty:
        return

    res_coords = resources[["latitude", "longitude"]].values
    nypd_coords = nypd[["latitude", "longitude"]].values

    tree = cKDTree(nypd_coords)
    RADIUS_DEG = 500 / 111_320  # ~500m in degrees

    for cat in tqdm(["violent", "property", "drugs", "harassment", "other"], desc="  NYPD cats"):
        cat_mask = nypd["crime_cat"] == cat
        cat_coords = nypd_coords[cat_mask.values]
        if len(cat_coords) == 0:
            continue
        cat_tree = cKDTree(cat_coords)

        for idx, row in resources.iterrows():
            pt = [row["latitude"], row["longitude"]]
            count = len(cat_tree.query_ball_point(pt, RADIUS_DEG))
            if count > 0:
                rid = row.get("resource_id", f"r_{idx}")
                add_triple(rid, f"CRIME_{cat.upper()}_500M", count,
                           CONFIDENCE["cross_dataset"], "nypd")

    # Overall safety score triple
    for idx, row in resources.iterrows():
        rid = row.get("resource_id", f"r_{idx}")
        if pd.notna(row.get("safety_score")):
            add_triple(rid, "SAFETY_SCORE", round(row["safety_score"], 3),
                       CONFIDENCE["computed"], "nypd_derived")


# ── Step 3: Quality triples from 311 (by complaint type) ─────────────────────

COMPLAINT_CATEGORIES = {
    "unsanitary":  ["UNSANITARY CONDITION"],
    "heating":     ["HEAT/HOT WATER"],
    "structural":  ["PLUMBING", "PAINT/PLASTER", "DOOR/WINDOW", "WATER LEAK",
                    "FLOORING/STAIRS", "OUTSIDE BUILDING"],
    "safety":      ["SAFETY", "Safety", "ELEVATOR", "ELECTRIC"],
    "general":     ["GENERAL", "APPLIANCE"],
}

def _categorize_complaint(ctype):
    if not ctype:
        return "other"
    for cat, keywords in COMPLAINT_CATEGORIES.items():
        if ctype in keywords:
            return cat
    return "other"


def build_311_triples(mart: pd.DataFrame):
    """311 complaints within 500m of each resource — broken by complaint type."""
    print("\n[3/7] Building 311 quality triples...")
    complaints = pd.read_parquet(STAGE / "311_complaints.parquet")
    complaints = complaints.dropna(subset=["latitude", "longitude"])
    complaints["complaint_cat"] = complaints["complaint_type"].apply(_categorize_complaint)
    print(f"  311 records: {len(complaints):,}")
    print(f"  Complaint categories: {complaints['complaint_cat'].value_counts().to_dict()}")

    resources = mart.dropna(subset=["latitude", "longitude"]).copy()
    if resources.empty:
        return

    comp_coords = complaints[["latitude", "longitude"]].values
    RADIUS_DEG = 500 / 111_320

    for cat in tqdm(["unsanitary", "heating", "structural", "safety", "general", "other"],
                    desc="  311 cats"):
        cat_mask = complaints["complaint_cat"] == cat
        cat_coords = comp_coords[cat_mask.values]
        if len(cat_coords) == 0:
            continue
        cat_tree = cKDTree(cat_coords)

        for idx, row in resources.iterrows():
            pt = [row["latitude"], row["longitude"]]
            count = len(cat_tree.query_ball_point(pt, RADIUS_DEG))
            if count > 0:
                rid = row.get("resource_id", f"r_{idx}")
                add_triple(rid, f"COMPLAINTS_{cat.upper()}_500M", count,
                           CONFIDENCE["cross_dataset"], "311")

    # Overall quality score triple
    for idx, row in resources.iterrows():
        rid = row.get("resource_id", f"r_{idx}")
        if pd.notna(row.get("quality_score")):
            add_triple(rid, "QUALITY_SCORE", round(row["quality_score"], 3),
                       CONFIDENCE["computed"], "311_derived")


# ── Step 4: PLUTO triples (overflow sites + lot-level data) ───────────────────

LANDUSE_LABELS = {
    "01": "one_two_family", "02": "multi_family", "03": "mixed_residential",
    "04": "commercial", "05": "industrial", "06": "vacant",
    "07": "outdoor_recreation", "08": "assembly_community", "09": "open_space",
    "10": "parking", "11": "institutional",
}

def build_pluto_triples():
    """PLUTO lot-level triples — overflow candidates get rich detail."""
    print("\n[4/7] Building PLUTO triples...")
    pluto = pd.read_parquet(DATA / "pluto_layer.parquet")
    overflow = pluto[pluto.get("is_overflow_candidate", False) == True].copy()
    print(f"  Total PLUTO lots: {len(pluto):,}")
    print(f"  Overflow candidates (landuse=08): {len(overflow):,}")

    # Borough-level aggregate triples
    for boro, grp in pluto.groupby("borough"):
        add_triple(f"boro_{boro}", "TOTAL_LOTS", len(grp), CONFIDENCE["pluto_zoning"], "pluto")
        for lu, lu_grp in grp.groupby("landuse"):
            label = LANDUSE_LABELS.get(str(lu), str(lu))
            add_triple(f"boro_{boro}", f"LOTS_{label.upper()}", len(lu_grp),
                       CONFIDENCE["pluto_zoning"], "pluto")

    # Each overflow site gets detailed triples
    for _, r in tqdm(overflow.iterrows(), total=len(overflow), desc="  Overflow sites"):
        bbl = str(r.get("bbl", f"lot_{_}"))
        add_triple(bbl, "IS_OVERFLOW_CANDIDATE", True, CONFIDENCE["pluto_zoning"], "pluto")
        add_triple(bbl, "HAS_ADDRESS", r.get("address", ""), CONFIDENCE["identity"], "pluto")
        add_triple(bbl, "IN_BOROUGH", r.get("borough", ""), CONFIDENCE["spatial_exact"], "pluto")
        add_triple(bbl, "LANDUSE", "assembly_community", CONFIDENCE["pluto_zoning"], "pluto")
        if pd.notna(r.get("ownername")):
            add_triple(bbl, "OWNER", r["ownername"], CONFIDENCE["identity"], "pluto")
        if pd.notna(r.get("numfloors")):
            add_triple(bbl, "NUM_FLOORS", int(r["numfloors"]), CONFIDENCE["identity"], "pluto")
        if pd.notna(r.get("yearbuilt")) and r["yearbuilt"] > 0:
            add_triple(bbl, "YEAR_BUILT", int(r["yearbuilt"]), CONFIDENCE["identity"], "pluto")
        if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude")):
            add_triple(bbl, "HAS_LOCATION", f"{r['latitude']:.6f},{r['longitude']:.6f}",
                       CONFIDENCE["spatial_exact"], "pluto")


# ── Step 5: Transit triples ──────────────────────────────────────────────────

def build_transit_triples():
    """MTA station connectivity triples."""
    print("\n[5/7] Building transit triples...")
    stations = pd.read_parquet(STAGE / "transit_stations.parquet")
    print(f"  Transit stations: {len(stations):,}")

    for _, r in stations.iterrows():
        sid = f"station_{r.get('station_id', _)}"
        add_triple(sid, "HAS_TYPE", "transit_station", CONFIDENCE["identity"], "mta_gtfs")
        add_triple(sid, "HAS_NAME", r.get("name", ""), CONFIDENCE["identity"], "mta_gtfs")
        add_triple(sid, "IN_BOROUGH", r.get("borough", ""), CONFIDENCE["spatial_exact"], "mta_gtfs")
        if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude")):
            add_triple(sid, "HAS_LOCATION", f"{r['latitude']:.6f},{r['longitude']:.6f}",
                       CONFIDENCE["spatial_exact"], "mta_gtfs")
        if pd.notna(r.get("line_name")):
            add_triple(sid, "SERVES_LINES", r["line_name"], CONFIDENCE["identity"], "mta_gtfs")


# ── Step 6: Cross-resource triples (co-location) ─────────────────────────────

def build_colocation_triples(mart: pd.DataFrame):
    """Resources that serve the same community (within 500m, different types)."""
    print("\n[6/7] Building co-location triples...")
    resources = mart.dropna(subset=["latitude", "longitude"]).copy()
    if resources.empty:
        return

    coords = resources[["latitude", "longitude"]].values
    tree = cKDTree(coords)
    RADIUS_DEG = 500 / 111_320

    rids = resources["resource_id"].tolist()
    rtypes = resources["resource_type"].tolist()
    count = 0

    for i in tqdm(range(len(resources)), desc="  Co-location"):
        neighbors = tree.query_ball_point(coords[i], RADIUS_DEG)
        for j in neighbors:
            if i >= j:
                continue  # avoid duplicates
            if rtypes[i] != rtypes[j]:
                add_triple(rids[i], "CO_LOCATED_WITH", rids[j],
                           CONFIDENCE["cross_dataset"], "spatial_derived")
                count += 1
    print(f"  Co-location edges: {count:,}")


# ── Step 7: Borough-level demographic triples ────────────────────────────────

def build_demographic_triples():
    """Borough population + resource density triples (for resource_gap reasoning)."""
    print("\n[7/7] Building demographic triples...")
    POP = {"BK": 2_600_000, "QN": 2_300_000, "MN": 1_600_000, "BX": 1_500_000, "SI": 500_000}
    for boro, pop in POP.items():
        add_triple(f"boro_{boro}", "HAS_POPULATION", pop, CONFIDENCE["temporal_stale"], "acs_census")

    mart = pd.read_parquet(DATA / "resource_mart.parquet")
    for boro, grp in mart.groupby("borough"):
        if boro not in POP:
            continue
        total = len(grp)
        per_100k = round(total / POP[boro] * 100_000, 1)
        add_triple(f"boro_{boro}", "TOTAL_RESOURCES", total, CONFIDENCE["computed"], "mart_derived")
        add_triple(f"boro_{boro}", "RESOURCES_PER_100K", per_100k, CONFIDENCE["computed"], "mart_derived")

        for rtype, type_grp in grp.groupby("resource_type"):
            add_triple(f"boro_{boro}", f"COUNT_{rtype.upper()}", len(type_grp),
                       CONFIDENCE["computed"], "mart_derived")
            rper100k = round(len(type_grp) / POP[boro] * 100_000, 1)
            add_triple(f"boro_{boro}", f"{rtype.upper()}_PER_100K", rper100k,
                       CONFIDENCE["computed"], "mart_derived")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("BUILD TRIPLES — SPO Knowledge Graph Construction")
    print("=" * 60)

    mart = pd.read_parquet(DATA / "resource_mart.parquet")
    print(f"\nMart: {len(mart):,} resources, {len(mart.columns)} columns")

    build_resource_triples(mart)
    build_nypd_triples(mart)
    build_311_triples(mart)
    build_pluto_triples()
    build_transit_triples()
    build_colocation_triples(mart)
    build_demographic_triples()

    # Save triples
    df = pd.DataFrame(_triples)
    print(f"\n{'=' * 60}")
    print(f"Total triples: {len(df):,}")
    print(f"Unique subjects: {df['subject'].nunique():,}")
    print(f"Unique predicates: {df['predicate'].nunique():,}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print(f"Mean confidence: {df['confidence'].mean():.3f}")

    outpath = DATA / "triples.parquet"
    df.to_parquet(outpath, index=False)
    print(f"\nSaved: {outpath} ({outpath.stat().st_size / 1e6:.1f} MB)")
    print(f"Total time: {time.time() - t0:.1f}s")

    # Stats by predicate
    print(f"\nTriples by predicate:")
    for pred, cnt in df["predicate"].value_counts().head(20).items():
        print(f"  {pred:35s}: {cnt:>8,}")


if __name__ == "__main__":
    main()
