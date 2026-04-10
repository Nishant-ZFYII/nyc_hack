"""
clean_all.py — Clean and normalize all raw datasets into stage/*.parquet
Each dataset gets a consistent schema with: resource_id, resource_type, name,
address, borough, latitude, longitude, and type-specific columns.

Run: python clean_all.py
Output: /media/nishant/SeeGayt2/nyc_hack_data/stage/
"""

import re
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/media/nishant/SeeGayt2/nyc_hack_data/clean.log"),
    ],
)
log = logging.getLogger(__name__)

RAW   = Path("/media/nishant/SeeGayt2/nyc_hack_data/raw")
STAGE = Path("/media/nishant/SeeGayt2/nyc_hack_data/stage")
STAGE.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

BOROUGH_MAP = {
    "manhattan": "MN", "mn": "MN", "1": "MN", "new york": "MN",
    "brooklyn":  "BK", "bk": "BK", "2": "BK",
    "queens":    "QN", "qn": "QN", "4": "QN",
    "bronx":     "BX", "bx": "BX", "the bronx": "BX", "3": "BX",
    "staten island": "SI", "si": "SI", "5": "SI",
}

def norm_borough(s):
    if pd.isna(s):
        return None
    return BOROUGH_MAP.get(str(s).strip().lower(), str(s).strip().upper()[:2])

def extract_latlon_from_point(s):
    """Extract (lat, lon) from 'POINT (-73.98 40.75)' strings."""
    if pd.isna(s):
        return None, None
    m = re.search(r'POINT\s*\(\s*([-\d.]+)\s+([-\d.]+)\s*\)', str(s))
    if m:
        lon, lat = float(m.group(1)), float(m.group(2))
        return lat, lon
    return None, None

def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def save(df, name):
    out = STAGE / f"{name}.parquet"
    df.to_parquet(out, index=False)
    log.info("SAVED %-35s %5d rows  %.1f MB → %s",
             name, len(df), out.stat().st_size / 1e6, out.name)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 1. PLUTO — base spatial layer (870K lots)
# ─────────────────────────────────────────────────────────────────────────────
def clean_pluto():
    log.info("Cleaning PLUTO…")
    df = pd.read_parquet(RAW / "pluto.parquet")

    # Keep only needed columns
    cols = ["bbl", "address", "borough", "zipcode", "landuse",
            "yearbuilt", "numfloors", "unitsres", "ownername",
            "latitude", "longitude"]
    df = df[[c for c in cols if c in df.columns]].copy()

    df["latitude"]  = pd.to_numeric(df.get("latitude"),  errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["yearbuilt"] = pd.to_numeric(df.get("yearbuilt"), errors="coerce")
    df["numfloors"] = pd.to_numeric(df.get("numfloors"), errors="coerce")
    df["unitsres"]  = pd.to_numeric(df.get("unitsres"),  errors="coerce")
    df["borough"]   = df["borough"].apply(norm_borough)
    df["landuse"]   = df["landuse"].fillna("").str.zfill(2)

    # Overflow candidates: landuse 08 = civic/community, 09 = transportation,
    # 12 = open space/parks — all potential emergency sites
    df["is_overflow_candidate"] = df["landuse"].isin(["08", "09", "12"])

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(40.4, 40.95)]
    df = df[df["longitude"].between(-74.3, -73.6)]

    return save(df, "pluto")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Shelters — derived from DOHMH facilities (HUMAN SERVICES group)
# Note: DHS shelter census is system-wide aggregate totals, not per-shelter.
# DOHMH FacDB has individual shelter/human-service locations with coordinates.
# ─────────────────────────────────────────────────────────────────────────────
def clean_shelters():
    log.info("Cleaning shelters (from DOHMH HUMAN SERVICES)…")
    df = pd.read_parquet(RAW / "dohmh_facilities.parquet")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    shelter_groups = {"human services", "adult services"}
    shelter_types = {"shelter", "homeless", "transitional", "drop-in", "drop_in",
                     "overnight", "safe haven", "crisis", "residence"}

    grp_mask = df.get("facgroup", pd.Series(dtype=str)).str.lower().isin(shelter_groups)
    typ_mask = df.get("factype", pd.Series(dtype=str)).str.lower().apply(
        lambda x: any(kw in str(x) for kw in shelter_types)
    )
    df = df[grp_mask | typ_mask].copy()
    log.info("  Shelter rows from DOHMH: %d", len(df))

    df["latitude"]  = pd.to_numeric(df.get("latitude"),  errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["borough"]   = df.get("boro", df.get("borough", pd.Series(dtype=str))).apply(norm_borough)

    for col in ["facname", "name"]:
        if col in df.columns:
            df = df.rename(columns={col: "name"})
            break

    df["resource_type"] = "shelter"
    df["capacity"]      = pd.to_numeric(df.get("capacity"), errors="coerce")
    df["address"]       = df.get("address", "")
    df["operator_name"] = df.get("opname", df.get("operator_name", ""))

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(40.4, 40.95)]
    df = df.reset_index(drop=True)
    df["resource_id"] = "shelter_" + df.index.astype(str)

    cols = ["resource_id", "resource_type", "name", "address", "borough",
            "latitude", "longitude", "capacity", "operator_name"]
    df = df[[c for c in cols if c in df.columns]]
    return save(df, "shelters")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hospitals — from manually downloaded CSV
# ─────────────────────────────────────────────────────────────────────────────
def clean_hospitals():
    log.info("Cleaning hospitals…")
    df = pd.read_csv(RAW / "hospital_20260410.csv")
    log.info("  Raw cols: %s", df.columns.tolist())

    df.columns = [c.strip() for c in df.columns]

    # Extract lat/lon from POINT geometry column
    geom_col = next((c for c in df.columns if "location" in c.lower() or "point" in c.lower()), None)
    if geom_col:
        df["latitude"], df["longitude"] = zip(*df[geom_col].apply(extract_latlon_from_point))
    else:
        df["latitude"] = pd.to_numeric(df.get("Latitude", df.get("latitude")), errors="coerce")
        df["longitude"] = pd.to_numeric(df.get("Longitude", df.get("longitude")), errors="coerce")

    rename = {
        "Facility Name": "name",
        "Facility Type": "facility_type",
        "Borough": "borough",
        "Cross Streets": "cross_streets",
        "Phone": "phone",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["resource_type"] = df.get("facility_type", "hospital").apply(
        lambda x: "hospital" if any(w in str(x).lower() for w in ["hospital", "acute", "trauma"])
        else "clinic"
    )
    df["borough"] = df.get("borough", pd.Series(dtype=str)).apply(norm_borough)
    df["address"] = df.get("cross_streets", df.get("address", ""))

    df = df.dropna(subset=["latitude", "longitude"])
    df = df.reset_index(drop=True)
    df["resource_id"] = "hospital_" + df.index.astype(str)

    return save(df, "hospitals")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DOHMH Facilities — clinics, child health centers, etc.
# ─────────────────────────────────────────────────────────────────────────────
def clean_dohmh():
    log.info("Cleaning DOHMH facilities…")
    df = pd.read_parquet(RAW / "dohmh_facilities.parquet")
    log.info("  Raw cols: %s", df.columns.tolist()[:15])

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    rename = {
        "facname": "name", "factype": "facility_type", "facsubgrp": "facility_subgroup",
        "facgroup": "facility_group", "opname": "operator_name",
        "address": "address", "city": "city", "boro": "borough",
        "zipcode": "zipcode", "latitude": "latitude", "longitude": "longitude",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["latitude"]  = pd.to_numeric(df.get("latitude"),  errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["borough"]   = df.get("borough", pd.Series(dtype=str)).apply(norm_borough)

    # Drop rows that are clearly not social services
    JUNK_GROUPS = {
        "transportation", "solid waste", "historical sites",
        "cultural institutions", "parks and plazas",
        "city agency parking, maintenance, and storage",
        "water and wastewater", "telecommunications",
        "material supplies and markets",
    }
    JUNK_TYPES = {
        "commercial garage & parking lot", "school bus depot",
        "tow truck company", "textiles", "compost", "garden",
        "triangle/plaza", "playground", "state historic place",
        "city council awards",
    }
    grp_col = "facility_group"
    typ_col = "facility_type"
    junk_mask = (
        df.get(grp_col, pd.Series(dtype=str)).str.lower().isin(JUNK_GROUPS) |
        df.get(typ_col, pd.Series(dtype=str)).str.lower().isin(JUNK_TYPES)
    )
    df = df[~junk_mask].copy()
    log.info("  After dropping junk: %d rows", len(df))

    # Classify resource_type with expanded categories
    def classify(row):
        grp = str(row.get("facility_group", "")).lower()
        typ = str(row.get("facility_type", "")).lower()
        sub = str(row.get("facility_subgroup", "")).lower()

        if any(w in typ for w in ["food pantry", "soup kitchen", "feeding site",
                                   "food bank", "emergency food", "congregate meal"]):
            return "food_bank"
        if any(w in grp for w in ["health care"]) or any(w in typ for w in [
                "hospital", "diagnostic", "treatment center", "urgent care",
                "ambulatory care", "health center"]):
            return "hospital"
        if any(w in typ for w in ["mental health", "outpatient mental",
                                   "support mental", "psychiatric", "substance"]):
            return "mental_health"
        if any(w in typ for w in ["shelter", "transitional", "safe haven",
                                   "drop-in", "overnight", "crisis residence"]):
            return "shelter"
        if any(w in grp for w in ["schools"]) or any(w in typ for w in [
                "elementary school", "high school", "middle school",
                "k-12", "public school", "non-public school"]):
            return "school"
        if any(w in typ for w in ["senior center", "aged", "elderly",
                                   "meals on wheels", "home care"]):
            return "senior_services"
        if any(w in grp for w in ["day care", "child"]) or any(w in typ for w in [
                "day care", "early education", "head start",
                "pre-k", "childcare", "afterschool"]):
            return "childcare"
        if any(w in grp for w in ["human services", "adult services"]):
            return "shelter"
        if any(w in typ for w in ["legal", "law", "court", "justice"]):
            return "legal_aid"
        if any(w in grp for w in ["youth services"]):
            return "youth_services"
        if any(w in grp for w in ["higher education", "vocational"]):
            return "education"
        if any(w in grp for w in ["libraries"]):
            return "library"
        if any(w in grp for w in ["emergency services", "public safety"]):
            return "emergency_services"
        return "community_center"

    df["resource_type"] = df.apply(classify, axis=1)

    # Log breakdown
    for rtype, cnt in df["resource_type"].value_counts().items():
        log.info("  %-25s %4d", rtype, cnt)

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(40.4, 40.95)]
    df = df.reset_index(drop=True)
    df["resource_id"] = "dohmh_" + df.index.astype(str)

    return save(df, "dohmh_facilities")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Food Banks — derived from DOHMH facilities
# Note: HRA resources (7btz-mnc8) is a healthcare provider directory, not food banks.
# DOHMH FacDB has FOOD PANTRY + SUMMER ONLY FEEDING SITE facility types.
# ─────────────────────────────────────────────────────────────────────────────
def clean_food_banks():
    log.info("Cleaning food banks (from DOHMH food facility types)…")
    df = pd.read_parquet(RAW / "dohmh_facilities.parquet")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    food_types = {
        "food pantry", "summer only feeding site", "soup kitchen",
        "food bank", "emergency food", "congregate meals", "food distribution",
        "food scrap", "meals on wheels",
    }

    factype_col = next((c for c in ["factype", "facility_type"] if c in df.columns), None)
    if factype_col:
        mask = df[factype_col].str.lower().apply(
            lambda x: any(kw in str(x) for kw in food_types)
        )
        df = df[mask].copy()
    log.info("  Food bank rows from DOHMH: %d", len(df))

    df["latitude"]  = pd.to_numeric(df.get("latitude"),  errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["borough"]   = df.get("boro", df.get("borough", pd.Series(dtype=str))).apply(norm_borough)

    for col in ["facname", "name"]:
        if col in df.columns:
            df = df.rename(columns={col: "name"})
            break

    df["resource_type"] = "food_bank"
    df["address"]       = df.get("address", "")
    df["operator_name"] = df.get("opname", df.get("operator_name", ""))
    df["facility_subtype"] = df.get(factype_col, "")

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(40.4, 40.95)]
    df = df.reset_index(drop=True)
    df["resource_id"] = "food_" + df.index.astype(str)

    cols = ["resource_id", "resource_type", "name", "address", "borough",
            "latitude", "longitude", "operator_name", "facility_subtype"]
    df = df[[c for c in cols if c in df.columns]]
    return save(df, "food_banks")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MTA Subway Stations
# ─────────────────────────────────────────────────────────────────────────────
def clean_transit():
    log.info("Cleaning MTA subway stations…")
    df = pd.read_csv(RAW / "MTA_Subway_Stations.csv")
    log.info("  Raw cols: %s", df.columns.tolist())

    df.columns = [c.strip() for c in df.columns]

    rename = {
        "Stop Name": "name", "Station ID": "station_id",
        "GTFS Stop ID": "gtfs_stop_id", "GTFS Latitude": "latitude",
        "GTFS Longitude": "longitude", "Borough": "borough",
        "Daytime Routes": "subway_lines", "ADA": "ada_accessible",
        "Line": "line_name", "Division": "division",
        "Structure": "structure",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["latitude"]     = pd.to_numeric(df.get("latitude"),  errors="coerce")
    df["longitude"]    = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["ada_accessible"] = df.get("ada_accessible", 0).apply(lambda x: x in [1, "1", True, "True"])
    df["borough"]      = df.get("borough", pd.Series(dtype=str)).apply(norm_borough)
    df["resource_type"] = "transit_station"

    df = df.dropna(subset=["latitude", "longitude"])
    df = df.reset_index(drop=True)
    df["resource_id"] = "transit_" + df.index.astype(str)

    return save(df, "transit_stations")


# ─────────────────────────────────────────────────────────────────────────────
# 7. DOE Schools — from womens_resources + dohmh or standalone if available
# ─────────────────────────────────────────────────────────────────────────────
def clean_schools():
    log.info("Cleaning schools (from DOHMH + DOE quality reports)…")
    frames = []

    # Pull school-type rows from DOHMH
    dohmh = pd.read_parquet(STAGE / "dohmh_facilities.parquet")
    schools_dohmh = dohmh[dohmh["resource_type"] == "school"].copy()
    if len(schools_dohmh):
        log.info("  Schools from DOHMH: %d", len(schools_dohmh))
        frames.append(schools_dohmh[["resource_id", "name", "address", "borough",
                                      "latitude", "longitude", "resource_type"]].copy())

    # Pull school rows from women's resource network
    womens = pd.read_parquet(RAW / "womens_resources.parquet")
    womens.columns = [c.lower().replace(" ", "_") for c in womens.columns]
    log.info("  Womens resource cols: %s", womens.columns.tolist()[:10])
    if "category" in womens.columns:
        school_mask = womens["category"].str.lower().str.contains("school|education|literacy|ged", na=False)
        sw = womens[school_mask].copy()
        log.info("  Schools from womens resources: %d", len(sw))
        sw["resource_type"] = "school"
        sw["latitude"]  = pd.to_numeric(sw.get("latitude",  sw.get("lat")), errors="coerce")
        sw["longitude"] = pd.to_numeric(sw.get("longitude", sw.get("lng")), errors="coerce")
        sw["borough"]   = sw.get("borough", pd.Series(dtype=str)).apply(norm_borough)
        for col in ["site_name", "organization_name", "name"]:
            if col in sw.columns:
                sw = sw.rename(columns={col: "name"})
                break
        for col in ["street_address", "address"]:
            if col in sw.columns:
                sw = sw.rename(columns={col: "address"})
                break
        sw = sw.dropna(subset=["latitude", "longitude"])
        sw = sw.reset_index(drop=True)
        sw["resource_id"] = "school_w_" + sw.index.astype(str)
        frames.append(sw[["resource_id", "name", "address", "borough",
                           "latitude", "longitude", "resource_type"]])

    if not frames:
        log.warning("  No school data found!")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["name", "latitude"])
    df = df.reset_index(drop=True)
    # Re-index IDs
    df["resource_id"] = "school_" + df.index.astype(str)
    return save(df, "schools")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Domestic Violence resources
# ─────────────────────────────────────────────────────────────────────────────
def clean_dv():
    log.info("Cleaning domestic violence resources…")
    df = pd.read_parquet(RAW / "domestic_violence.parquet")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    log.info("  Raw cols: %s", df.columns.tolist())

    for col in ["organization_name", "program_name", "name", "agency_name"]:
        if col in df.columns:
            df = df.rename(columns={col: "name"})
            break
    for col in ["street_address", "address", "location"]:
        if col in df.columns:
            df = df.rename(columns={col: "address"})
            break

    df["latitude"]  = pd.to_numeric(df.get("latitude",  df.get("lat")), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", df.get("lng")), errors="coerce")

    # Try extracting from geometry column
    if df["latitude"].isna().all():
        for col in df.columns:
            if "location" in col or "point" in col or "geom" in col:
                lats, lons = zip(*df[col].apply(extract_latlon_from_point))
                df["latitude"], df["longitude"] = lats, lons
                break

    df["borough"]       = df.get("borough", pd.Series(dtype=str)).apply(norm_borough)
    df["resource_type"] = "domestic_violence"
    df["ada_accessible"] = df.get("ada_accessible", False)

    df = df.dropna(subset=["latitude", "longitude"])
    df = df.reset_index(drop=True)
    df["resource_id"] = "dv_" + df.index.astype(str)

    return save(df, "domestic_violence")


# ─────────────────────────────────────────────────────────────────────────────
# 9. NYCHA developments
# ─────────────────────────────────────────────────────────────────────────────
def clean_nycha():
    log.info("Cleaning NYCHA developments…")
    df = pd.read_parquet(RAW / "nycha_developments.parquet")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    log.info("  Raw cols: %s", df.columns.tolist())

    for col in ["developmen", "development_name", "developmentname", "name"]:
        if col in df.columns:
            df = df.rename(columns={col: "name"})
            break
    for col in ["address", "street_address", "location"]:
        if col in df.columns:
            df = df.rename(columns={col: "address"})
            break

    df["latitude"]  = pd.to_numeric(df.get("latitude",  df.get("lat")), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", df.get("lng")), errors="coerce")

    # NYCHA raw has 'the_geom' as a dict with 'coordinates' containing numpy arrays
    # Extract centroid by averaging all coordinates in the MultiPolygon
    if df["latitude"].isna().all() and "the_geom" in df.columns:
        def geom_centroid(g):
            try:
                coords = g["coordinates"]
                # Flatten nested list/arrays of [lon, lat] pairs
                all_pts = []
                for poly in coords:
                    for ring in poly:
                        for pt in ring:
                            all_pts.append((float(pt[0]), float(pt[1])))
                if all_pts:
                    lon = sum(p[0] for p in all_pts) / len(all_pts)
                    lat = sum(p[1] for p in all_pts) / len(all_pts)
                    return lat, lon
            except Exception:
                pass
            return None, None
        lats, lons = zip(*df["the_geom"].apply(geom_centroid))
        df["latitude"], df["longitude"] = lats, lons

    df["borough"]       = df.get("borough", pd.Series(dtype=str)).apply(norm_borough)
    df["resource_type"] = "nycha"
    df["total_units"]   = pd.to_numeric(df.get("total_apartment_units", df.get("units")), errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"])
    df = df.reset_index(drop=True)
    df["resource_id"] = "nycha_" + df.index.astype(str)

    return save(df, "nycha")


# ─────────────────────────────────────────────────────────────────────────────
# 10. HRA Benefits Centers
# ─────────────────────────────────────────────────────────────────────────────
def clean_benefits():
    log.info("Cleaning HRA benefits centers…")
    df = pd.read_parquet(RAW / "hra_benefits_centers.parquet")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    log.info("  Raw cols: %s", df.columns.tolist())

    for col in ["center_name", "name", "facility_name"]:
        if col in df.columns:
            df = df.rename(columns={col: "name"})
            break
    for col in ["address", "street_address", "location_1"]:
        if col in df.columns:
            df = df.rename(columns={col: "address"})
            break

    df["latitude"]  = pd.to_numeric(df.get("latitude",  df.get("lat")), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", df.get("lng")), errors="coerce")

    if df["latitude"].isna().all():
        for col in df.columns:
            if "location" in col or "geom" in col:
                lats, lons = zip(*df[col].apply(extract_latlon_from_point))
                df["latitude"], df["longitude"] = lats, lons
                break

    df["borough"]       = df.get("borough", pd.Series(dtype=str)).apply(norm_borough)
    df["resource_type"] = "benefits_center"
    df["phone"]         = df.get("phone", df.get("telephone", ""))
    df["hours_open"]    = df.get("hours_of_operation", df.get("hours", ""))

    df = df.dropna(subset=["latitude", "longitude"])
    df = df.reset_index(drop=True)
    df["resource_id"] = "benefits_" + df.index.astype(str)

    return save(df, "benefits_centers")


# ─────────────────────────────────────────────────────────────────────────────
# 11. Cooling Centers
# ─────────────────────────────────────────────────────────────────────────────
def clean_cooling():
    log.info("Cleaning cooling/warming centers…")
    df = pd.read_parquet(RAW / "cooling_centers.parquet")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    log.info("  Raw cols: %s", df.columns.tolist())

    # Raw dataset uses 'propertyname' for name, 'x'=longitude, 'y'=latitude (decimal degrees)
    for col in ["propertyname", "site_name", "facility_name", "name"]:
        if col in df.columns:
            df = df.rename(columns={col: "name"})
            break
    for col in ["street_address", "address", "location"]:
        if col in df.columns:
            df = df.rename(columns={col: "address"})
            break

    # x = longitude (~-74), y = latitude (~40) — already decimal degrees
    if "x" in df.columns and "y" in df.columns:
        df["longitude"] = pd.to_numeric(df["x"], errors="coerce")
        df["latitude"]  = pd.to_numeric(df["y"], errors="coerce")
    else:
        df["latitude"]  = pd.to_numeric(df.get("latitude",  df.get("lat")), errors="coerce")
        df["longitude"] = pd.to_numeric(df.get("longitude", df.get("lng")), errors="coerce")

    df["borough"]       = df.get("borough", pd.Series(dtype=str)).apply(norm_borough)
    df["resource_type"] = "cooling_center"
    df["center_type"]   = df.get("featuretype", "")
    df["status"]        = df.get("status", "")
    df["ada_accessible"] = df.get("accessible", df.get("ada", False))

    df = df.dropna(subset=["latitude", "longitude"])
    df = df.reset_index(drop=True)
    df["resource_id"] = "cool_" + df.index.astype(str)

    return save(df, "cooling_centers")


# ─────────────────────────────────────────────────────────────────────────────
# 12. NYPD Complaints — spatial signal (NOT a resource, used for safety scores)
# ─────────────────────────────────────────────────────────────────────────────
def clean_nypd():
    log.info("Cleaning NYPD complaints…")
    df = pd.read_parquet(RAW / "nypd_complaints.parquet")
    df.columns = [c.lower() for c in df.columns]

    df["latitude"]  = pd.to_numeric(df.get("latitude"),  errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["boro_nm"]   = df.get("boro_nm", pd.Series(dtype=str)).apply(norm_borough)
    df["ofns_desc"] = df.get("ofns_desc", "").str.strip()

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(40.4, 40.95)]

    # Keep only columns needed for safety scoring
    keep = ["latitude", "longitude", "boro_nm", "ofns_desc", "cmplnt_fr_dt"]
    df = df[[c for c in keep if c in df.columns]]

    return save(df, "nypd_complaints")


# ─────────────────────────────────────────────────────────────────────────────
# 13. 311 Complaints — spatial signal for service quality
# ─────────────────────────────────────────────────────────────────────────────
def clean_311():
    log.info("Cleaning 311 HPD complaints…")
    df = pd.read_parquet(RAW / "311_hpd.parquet")
    df.columns = [c.lower() for c in df.columns]

    df["latitude"]  = pd.to_numeric(df.get("latitude"),  errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["borough"]   = df.get("borough", pd.Series(dtype=str)).apply(norm_borough)

    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(40.4, 40.95)]

    keep = ["latitude", "longitude", "borough", "complaint_type", "descriptor",
            "incident_address", "created_date"]
    df = df[[c for c in keep if c in df.columns]]

    return save(df, "311_complaints")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Homeless Drop-In Centers
# ─────────────────────────────────────────────────────────────────────────────
def clean_dropin():
    log.info("Cleaning homeless drop-in centers…")
    df = pd.read_parquet(RAW / "homeless_dropin.parquet")
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    log.info("  Raw cols: %s", df.columns.tolist())

    for col in ["center_name", "name", "facility_name", "site_name"]:
        if col in df.columns:
            df = df.rename(columns={col: "name"})
            break
    for col in ["address", "street_address", "location_1"]:
        if col in df.columns:
            df = df.rename(columns={col: "address"})
            break

    df["latitude"]  = pd.to_numeric(df.get("latitude",  df.get("lat")), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude", df.get("lng")), errors="coerce")

    if df["latitude"].isna().all():
        for col in df.columns:
            if "location" in col or "geom" in col or "point" in col:
                lats, lons = zip(*df[col].apply(extract_latlon_from_point))
                df["latitude"], df["longitude"] = lats, lons
                break

    df["borough"]       = df.get("borough", pd.Series(dtype=str)).apply(norm_borough)
    df["resource_type"] = "dropin_center"

    df = df.dropna(subset=["latitude", "longitude"])
    df = df.reset_index(drop=True)
    df["resource_id"] = "dropin_" + df.index.astype(str)

    return save(df, "dropin_centers")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    steps = {
        "pluto":      clean_pluto,
        "shelters":   clean_shelters,
        "hospitals":  clean_hospitals,
        "dohmh":      clean_dohmh,
        "food_banks": clean_food_banks,
        "transit":    clean_transit,
        "schools":    clean_schools,
        "dv":         clean_dv,
        "nycha":      clean_nycha,
        "benefits":   clean_benefits,
        "cooling":    clean_cooling,
        "nypd":       clean_nypd,
        "311":        clean_311,
        "dropin":     clean_dropin,
    }

    targets = sys.argv[1:] if len(sys.argv) > 1 else list(steps.keys())
    log.info("Running cleaning steps: %s", targets)

    results = {}
    for name in targets:
        if name not in steps:
            log.error("Unknown step: %s", name)
            continue
        try:
            df = steps[name]()
            results[name] = ("OK", len(df) if df is not None else 0)
        except Exception as e:
            log.error("FAILED %s: %s", name, e, exc_info=True)
            results[name] = ("FAIL", str(e))

    log.info("\n=== Cleaning Summary ===")
    for name, (status, info) in results.items():
        log.info("  %-20s %s  %s", name, status, info)
