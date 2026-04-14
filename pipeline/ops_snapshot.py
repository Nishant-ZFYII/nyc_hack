"""
pipeline/ops_snapshot.py — Build the admin live-ops map payload.

Joins pipeline.cases.list_cases() with the resource mart to produce the
shape the admin deck.gl map consumes:

    {
      "sites": [ {id, lat, lon, cap, load, type, name}, ... ],   # up to 1.5k
      "cases": [ {id, lat, lon, status, urgency, name}, ... ],
      "arcs":  [ {from, to, status, color}, ... ],
      "stats": { open, resolved, critical, by_borough: {...} }
    }

Uses borough centroids as fallback when a case lacks an explicit location.
Pure CPU — works on any laptop.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import cases as _cases

_DATA = Path(__file__).resolve().parent.parent / "data"
_mart: pd.DataFrame | None = None

# Borough centroids as fallback for cases without explicit lat/lon.
_BOROUGH_CENTROID = {
    "Manhattan":    (40.7831, -73.9712),
    "Brooklyn":     (40.6782, -73.9442),
    "Queens":       (40.7282, -73.7949),
    "Bronx":        (40.8448, -73.8648),
    "Staten Island":(40.5795, -74.1502),
    "MN": (40.7831, -73.9712),
    "BK": (40.6782, -73.9442),
    "QN": (40.7282, -73.7949),
    "BX": (40.8448, -73.8648),
    "SI": (40.5795, -74.1502),
}

# Rough jitter radius so overlapping case dots are visible
_JITTER = 0.008   # ~900 m


def _load_mart() -> pd.DataFrame:
    global _mart
    if _mart is None:
        df = pd.read_parquet(_DATA / "resource_mart.parquet")
        if "latitude" in df.columns and "lat" not in df.columns:
            df["lat"] = df["latitude"]
        if "longitude" in df.columns and "lon" not in df.columns:
            df["lon"] = df["longitude"]
        df = df.dropna(subset=["lat", "lon"]).copy()
        df["lat"] = df["lat"].astype(float)
        df["lon"] = df["lon"].astype(float)
        if "capacity" in df.columns:
            df["capacity"] = df["capacity"].fillna(30).astype(int).clip(lower=5, upper=500)
        else:
            df["capacity"] = 30
        _mart = df
    return _mart


def _case_location(case: dict, i: int) -> tuple[float, float]:
    """Derive lat/lon for a case. Prefer current_location, else borough centroid."""
    loc = case.get("current_location")
    if isinstance(loc, dict) and loc.get("lat") and loc.get("lon"):
        return float(loc["lat"]), float(loc["lon"])
    # Try a borough hint from needs/intents
    borough = ""
    for n in case.get("needs", []):
        if n.get("borough"):
            borough = n["borough"]; break
    for d in case.get("destination_intents", []):
        if d.get("borough"):
            borough = d["borough"]; break
    if borough in _BOROUGH_CENTROID:
        lat, lon = _BOROUGH_CENTROID[borough]
    else:
        lat, lon = _BOROUGH_CENTROID["Brooklyn"]
    # Deterministic jitter so dots don't perfectly overlap
    import math
    a = (i * 2654435761) & 0xFFFFFFFF
    dx = ((a & 0xFFFF) / 0xFFFF - 0.5) * _JITTER
    dy = (((a >> 16) & 0xFFFF) / 0xFFFF - 0.5) * _JITTER
    return lat + dy, lon + dx


def _urgency_color(urgency: str) -> list[int]:
    return {
        "critical": [255, 85, 119, 240],
        "high":     [255, 159, 67, 220],
        "medium":   [255, 204, 51, 200],
        "low":      [92, 255, 177, 180],
    }.get(urgency, [150, 170, 200, 180])


def _case_urgency(case: dict) -> str:
    """Cheap urgency heuristic from the case's open needs."""
    open_cats = [n.get("category", "") for n in case.get("needs", []) if n.get("status") != "resolved"]
    if any(c in open_cats for c in ("safety", "medical", "emergency")):
        return "critical"
    if any(c in open_cats for c in ("housing", "shelter")):
        return "high"
    if open_cats:
        return "medium"
    return "low"


def build_snapshot(site_limit: int = 1200) -> dict[str, Any]:
    mart = _load_mart()
    cases_summary = _cases.list_cases()

    # ── Sites ────────────────────────────────────────────────────────────
    site_types = {"shelter", "food_bank", "clinic", "hospital",
                  "benefits_center", "community_center", "cooling_center",
                  "domestic_violence"}
    sites_df = mart[mart["resource_type"].isin(site_types)].head(site_limit).copy()
    sites_df["used"] = 0  # computed below

    # ── Cases + arcs ─────────────────────────────────────────────────────
    cases: list[dict] = []
    arcs: list[dict] = []
    stats = {"open": 0, "resolved": 0, "critical": 0,
             "by_borough": {"MN": 0, "BK": 0, "QN": 0, "BX": 0, "SI": 0}}

    for i, cs in enumerate(cases_summary):
        case = _cases.load_case(cs["case_id"])
        if not case:
            continue
        clat, clon = _case_location(case, i)
        urgency = _case_urgency(case)
        status = "resolved" if all(n.get("status") == "resolved" for n in case.get("needs", [])) else "open"
        if status == "open":
            stats["open"] += 1
        else:
            stats["resolved"] += 1
        if urgency == "critical":
            stats["critical"] += 1

        name = case.get("name") or cs.get("case_id", "")
        cases.append({
            "id": cs["case_id"],
            "name": name,
            "lat": round(clat, 5),
            "lon": round(clon, 5),
            "status": status,
            "urgency": urgency,
            "color": _urgency_color(urgency),
            "open_needs": [n.get("category") for n in case.get("needs", []) if n.get("status") != "resolved"],
        })

        # Arc = case → first active destination intent
        for di in case.get("destination_intents", []):
            if di.get("state") in {"resolved", "cancelled"}:
                continue
            rn = di.get("resource_name", "")
            if not rn:
                continue
            match = mart[mart["name"].astype(str).str.contains(str(rn)[:30], case=False, na=False, regex=False)]
            if len(match):
                site = match.iloc[0]
                arcs.append({
                    "from": [clon, clat],
                    "to":   [float(site["lon"]), float(site["lat"])],
                    "from_id": cs["case_id"],
                    "status": di.get("state", "intent_confirmed"),
                    "color": _urgency_color(urgency),
                })
                # Bump the site load
                mask = sites_df["resource_id"] == site.get("resource_id", "")
                if mask.any():
                    sites_df.loc[mask, "used"] = sites_df.loc[mask, "used"] + 1
            break

    # Collapse sites to the frontend shape
    sites = []
    for _, r in sites_df.iterrows():
        sites.append({
            "id": str(r.get("resource_id", "")),
            "name": str(r.get("name", "")),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "cap": int(r["capacity"]),
            "load": int(r.get("used", 0)),
            "type": str(r["resource_type"]),
        })

    return {
        "sites": sites,
        "cases": cases,
        "arcs":  arcs,
        "stats": stats,
    }
