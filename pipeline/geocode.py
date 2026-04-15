"""
pipeline/geocode.py — Geocode locations from user queries + distance sorting.

Uses free Nominatim (OpenStreetMap) API — no API key needed.

Usage:
    from pipeline.geocode import geocode_location, sort_by_distance
"""
from __future__ import annotations
import math
import re
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
HEADERS = {"User-Agent": "NYC-SocialServices-Engine/1.0"}

# NYC neighborhood → approximate lat/lon for fast fallback
NYC_LANDMARKS = {
    "flatbush": (40.6501, -73.9496),
    "bedstuy": (40.6872, -73.9418), "bed-stuy": (40.6872, -73.9418),
    "bushwick": (40.6944, -73.9213),
    "williamsburg": (40.7081, -73.9571),
    "crown heights": (40.6694, -73.9422),
    "east new york": (40.6590, -73.8759),
    "brownsville": (40.6614, -73.9085),
    "sunset park": (40.6454, -73.9926),
    "bay ridge": (40.6345, -74.0283),
    "coney island": (40.5755, -73.9707),
    "harlem": (40.8116, -73.9465),
    "washington heights": (40.8417, -73.9394),
    "inwood": (40.8677, -73.9212),
    "east harlem": (40.7957, -73.9389),
    "lower east side": (40.7150, -73.9843),
    "chelsea": (40.7465, -74.0014),
    "midtown": (40.7549, -73.9840),
    "times square": (40.7580, -73.9855),
    "port authority": (40.7569, -73.9900),
    "penn station": (40.7506, -73.9935),
    "grand central": (40.7527, -73.9772),
    "jamaica": (40.7028, -73.7890),
    "flushing": (40.7580, -73.8330),
    "astoria": (40.7721, -73.9301),
    "long island city": (40.7440, -73.9565),
    "jackson heights": (40.7557, -73.8831),
    "fordham": (40.8615, -73.8905),
    "hunts point": (40.8094, -73.8803),
    "mott haven": (40.8085, -73.9230),
    "south bronx": (40.8176, -73.9182),
    "riverdale": (40.9003, -73.9068),
    "st george": (40.6433, -74.0735),
    "stapleton": (40.6266, -74.0758),
}


def geocode_location(text: str) -> dict | None:
    """
    Extract and geocode a location from user text.

    Returns {"lat": float, "lon": float, "display_name": str} or None.
    """
    # 1. Try known NYC landmarks/neighborhoods first (instant, no API call)
    text_lower = text.lower()
    for name, (lat, lon) in NYC_LANDMARKS.items():
        if name in text_lower:
            return {"lat": lat, "lon": lon, "display_name": name.title(), "source": "landmark"}

    # 2. Try to extract a street address from the text
    address = _extract_address(text)
    if not address:
        return None

    # 3. Geocode via Nominatim
    try:
        params = {
            "q": f"{address}, New York City, NY",
            "format": "json",
            "limit": "1",
            "countrycodes": "us",
        }
        resp = requests.get(f"{NOMINATIM_URL}/search", params=params,
                           headers=HEADERS, timeout=5)
        if resp.status_code == 200:
            results = resp.json()
            if results:
                r = results[0]
                return {
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "display_name": r.get("display_name", address),
                    "source": "nominatim",
                }
    except Exception:
        pass

    return None


def _extract_address(text: str) -> str | None:
    """Try to extract a street address from natural language text."""
    # Match patterns like "53rd street", "123 Main St", "66 Boerum Place"
    patterns = [
        r'(\d+\s+\w+(?:\s+\w+)?\s+(?:street|st|avenue|ave|boulevard|blvd|place|pl|road|rd|drive|dr|way|lane|ln))',
        r'(\d+(?:st|nd|rd|th)\s+(?:street|st|avenue|ave))',
        r'((?:east|west|north|south|e|w|n|s)\.?\s+\d+(?:st|nd|rd|th)\s+(?:street|st))',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def haversine_miles(lat1, lon1, lat2, lon2):
    """Distance in miles between two lat/lon points."""
    R = 3959
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.asin(math.sqrt(a))


def sort_by_distance(resources_df, user_lat, user_lon):
    """
    Sort a DataFrame of resources by distance from user location.
    Adds 'distance_miles' and 'walk_min_est' columns.
    """
    import pandas as pd
    import numpy as np

    df = resources_df.copy()
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    df["distance_miles"] = df.apply(
        lambda r: haversine_miles(user_lat, user_lon,
                                  float(r["latitude"]), float(r["longitude"]))
        if pd.notna(r.get("latitude")) and pd.notna(r.get("longitude")) else 999,
        axis=1
    )
    df["walk_min_est"] = (df["distance_miles"] * 20).round(0).astype(int)  # ~3mph walking
    df = df.sort_values("distance_miles")

    return df
