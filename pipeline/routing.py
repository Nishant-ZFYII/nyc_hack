"""
pipeline/routing.py — Multi-modal routing: walk, transit, or both.

Provides step-by-step directions from user location to a resource,
considering the user's budget (free = walk only, has MetroCard = transit).

Uses:
  - OSRM (free, no auth) for walking directions
  - Our MTA transit graph for subway routing
  - Nominatim for geocoding addresses

Usage:
    from pipeline.routing import get_directions
"""
import math
import requests

OSRM_URL = "https://router.project-osrm.org"
NOMINATIM_URL = "https://nominatim.openstreetmap.org"
HEADERS = {"User-Agent": "NYC-SocialServices-Engine/1.0"}

# MTA fare
MTA_FARE = 2.90


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3959
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def geocode(address: str) -> dict | None:
    """Geocode an address to lat/lon via Nominatim."""
    try:
        params = {"q": f"{address}, New York City, NY", "format": "json",
                  "limit": "1", "countrycodes": "us"}
        resp = requests.get(f"{NOMINATIM_URL}/search", params=params,
                           headers=HEADERS, timeout=5)
        if resp.status_code == 200 and resp.json():
            r = resp.json()[0]
            return {"lat": float(r["lat"]), "lon": float(r["lon"]),
                    "display_name": r.get("display_name", address)}
    except Exception:
        pass
    return None


def get_walking_route(from_lat, from_lon, to_lat, to_lon) -> dict | None:
    """Get walking directions via OSRM (free, no auth)."""
    try:
        url = (f"{OSRM_URL}/route/v1/foot/"
               f"{from_lon},{from_lat};{to_lon},{to_lat}"
               f"?overview=full&geometries=geojson&steps=true")
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None

        route = data["routes"][0]
        dist_miles = route["distance"] / 1609.34
        # Sanity check: walking is ~3 mph = 20 min/mile. OSRM foot profile
        # can return unrealistically fast times, so use max(osrm, distance-based).
        osrm_min = route["duration"] / 60
        distance_based_min = dist_miles * 20  # 3 mph
        duration_min = max(osrm_min, distance_based_min)

        legs = route.get("legs", [{}])
        raw_steps = []
        for leg in legs:
            for step in leg.get("steps", []):
                maneuver_type = step.get("maneuver", {}).get("type", "")
                name = step.get("name", "")
                dist_m = step.get("distance", 0)
                dur_s = step.get("duration", 0)
                raw_steps.append({
                    "type": maneuver_type,
                    "name": name,
                    "distance_m": dist_m,
                    "duration_s": dur_s,
                })

        # Consolidate steps: merge tiny steps, filter noise, humanize instructions
        steps = _consolidate_walk_steps(raw_steps)

        return {
            "mode": "walk",
            "distance_miles": round(dist_miles, 2),
            "duration_min": round(duration_min),
            "cost": 0.0,
            "steps": steps,
            "geometry": route.get("geometry"),
        }
    except Exception:
        return None


def _consolidate_walk_steps(raw_steps: list) -> list:
    """Merge small walk steps into meaningful directions. Filter highway noise."""
    SKIP_NAMES = {"", "ramp", "off ramp", "on ramp", "link"}
    SKIP_KEYWORDS = {"expressway", "interstate", "highway", "turnpike", "parkway ramp"}
    MANEUVER_MAP = {
        "depart": "Head", "turn": "Turn onto", "new name": "Continue on",
        "arrive": "Arrive at", "merge": "Continue on", "end of road": "At end of road, go to",
        "fork": "Keep on", "roundabout": "Take roundabout to",
    }

    consolidated = []
    for s in raw_steps:
        name = s["name"].strip()
        mtype = s["type"]
        dist_m = s["distance_m"]
        dur_s = s["duration_s"]

        # Skip tiny steps (<50m) unless arrive
        if dist_m < 50 and mtype != "arrive":
            continue
        # Skip highway/ramp names
        name_lower = name.lower()
        if name_lower in SKIP_NAMES or any(kw in name_lower for kw in SKIP_KEYWORDS):
            # Accumulate distance into previous step
            if consolidated:
                consolidated[-1]["distance_ft"] += round(dist_m * 3.281)
                consolidated[-1]["duration_min"] += round(dur_s / 60, 1)
            continue

        verb = MANEUVER_MAP.get(mtype, "Continue on")
        instruction = f"{verb} {name}" if name else verb
        # Recalculate duration based on 3mph walk speed
        walk_min = round((dist_m / 1609.34) * 20)

        consolidated.append({
            "instruction": instruction,
            "distance_ft": round(dist_m * 3.281),
            "duration_min": walk_min,
        })

    # Cap at 8 most significant steps (by distance)
    if len(consolidated) > 8:
        consolidated.sort(key=lambda s: s["distance_ft"], reverse=True)
        consolidated = consolidated[:8]
        consolidated.sort(key=lambda s: s["duration_min"])  # re-sort by time

    return consolidated


def get_transit_estimate(from_lat, from_lon, to_lat, to_lon) -> dict:
    """
    Estimate transit route using our MTA station data.
    Finds nearest stations to origin/destination and estimates travel time.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    try:
        from pipeline.executor import load_state
        _, payload = load_state()
        transit_df = payload.get("transit")
        if transit_df is None or len(transit_df) == 0:
            return _fallback_transit_estimate(from_lat, from_lon, to_lat, to_lon)

        import pandas as pd
        if hasattr(transit_df, 'to_pandas'):
            transit_df = transit_df.to_pandas()

        # Find nearest station to origin
        transit_df["_d_from"] = transit_df.apply(
            lambda r: haversine_miles(from_lat, from_lon,
                                      float(r["latitude"]), float(r["longitude"]))
            if pd.notna(r.get("latitude")) else 999, axis=1)
        origin_station = transit_df.nsmallest(1, "_d_from").iloc[0]

        # Find nearest station to destination
        transit_df["_d_to"] = transit_df.apply(
            lambda r: haversine_miles(to_lat, to_lon,
                                      float(r["latitude"]), float(r["longitude"]))
            if pd.notna(r.get("latitude")) else 999, axis=1)
        dest_station = transit_df.nsmallest(1, "_d_to").iloc[0]

        walk_to_station = origin_station["_d_from"] * 20  # min at 3mph
        walk_from_station = dest_station["_d_to"] * 20
        direct_dist = haversine_miles(
            float(origin_station["latitude"]), float(origin_station["longitude"]),
            float(dest_station["latitude"]), float(dest_station["longitude"]))
        # NYC subway avg ~20mph including stops (express can be 30+)
        subway_time = max(direct_dist / 20 * 60, 3)  # min 3 min

        total_time = walk_to_station + subway_time + walk_from_station
        origin_lines = str(origin_station.get("subway_lines", ""))
        dest_lines = str(dest_station.get("subway_lines", ""))
        origin_name = str(origin_station.get("name", "station"))
        dest_name = str(dest_station.get("name", "station"))

        steps = [
            {"instruction": f"Walk to {origin_name}",
             "duration_min": round(walk_to_station, 1), "mode": "walk",
             "detail": f"Lines: {origin_lines}" if origin_lines else ""},
            {"instruction": f"Take subway to {dest_name}",
             "duration_min": round(subway_time, 1), "mode": "transit",
             "detail": f"Lines at destination: {dest_lines}" if dest_lines else "",
             "lines": dest_lines},
            {"instruction": f"Walk to destination from {dest_name}",
             "duration_min": round(walk_from_station, 1), "mode": "walk"},
        ]

        # Find 2 nearest stations to give alternatives
        alt_origins = transit_df.nsmallest(3, "_d_from")
        nearby_stations = []
        for _, st in alt_origins.iterrows():
            sn = str(st.get("name", ""))
            sl = str(st.get("subway_lines", ""))
            sd = round(float(st["_d_from"]) * 20, 1)  # walk min
            if sn and sn != origin_name:
                nearby_stations.append({"name": sn, "lines": sl, "walk_min": sd})

        return {
            "mode": "transit",
            "distance_miles": round(haversine_miles(from_lat, from_lon, to_lat, to_lon), 2),
            "duration_min": round(total_time),
            "cost": MTA_FARE,
            "origin_station": origin_name,
            "dest_station": dest_name,
            "subway_lines_origin": origin_lines,
            "subway_lines_dest": dest_lines,
            "nearby_stations": nearby_stations[:2],
            "steps": steps,
        }
    except Exception:
        return _fallback_transit_estimate(from_lat, from_lon, to_lat, to_lon)


def _fallback_transit_estimate(from_lat, from_lon, to_lat, to_lon):
    """Simple distance-based estimate when station data unavailable."""
    dist = haversine_miles(from_lat, from_lon, to_lat, to_lon)
    return {
        "mode": "transit",
        "distance_miles": round(dist, 2),
        "duration_min": round(dist * 3 + 10),  # rough: 3 min/mile + 10 min overhead
        "cost": MTA_FARE,
        "steps": [
            {"instruction": "Walk to nearest subway station", "duration_min": 5, "mode": "walk"},
            {"instruction": f"Take subway ({round(dist, 1)} miles)", "duration_min": round(dist * 2), "mode": "transit"},
            {"instruction": "Walk to destination", "duration_min": 5, "mode": "walk"},
        ],
    }


def _find_nearest_hra(lat, lon) -> dict | None:
    """Find nearest HRA Benefits Center for free MetroCard / travel vouchers."""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from pipeline.executor import load_state
        import pandas as pd

        mart, _ = load_state()
        if hasattr(mart, 'to_pandas'):
            mart = mart.to_pandas()
        hra = mart[mart["resource_type"] == "benefits_center"].copy()
        if hra.empty:
            return None
        hra["_dist"] = hra.apply(
            lambda r: haversine_miles(lat, lon, float(r["latitude"]), float(r["longitude"]))
            if pd.notna(r.get("latitude")) else 999, axis=1)
        nearest = hra.nsmallest(1, "_dist").iloc[0]
        return {
            "name": str(nearest.get("name", "")),
            "address": str(nearest.get("address", "")),
            "borough": str(nearest.get("borough", "")),
            "distance_miles": round(float(nearest["_dist"]), 2),
            "lat": float(nearest["latitude"]),
            "lon": float(nearest["longitude"]),
        }
    except Exception:
        return None


def get_directions(from_lat, from_lon, to_lat, to_lon, budget: float = None) -> dict:
    """
    Get multi-modal directions from origin to destination.

    Parameters:
        from_lat, from_lon: origin coordinates
        to_lat, to_lon: destination coordinates
        budget: available money (None = show all options, 0 = walk only)

    Returns dict with options: walk, transit, and recommendation.
    """
    dist = haversine_miles(from_lat, from_lon, to_lat, to_lon)

    result = {
        "distance_miles": round(dist, 2),
        "options": [],
        "recommendation": "",
    }

    # Always get walking route
    walk = get_walking_route(from_lat, from_lon, to_lat, to_lon)
    if walk:
        result["options"].append(walk)
    else:
        # Fallback estimate
        result["options"].append({
            "mode": "walk",
            "distance_miles": round(dist, 2),
            "duration_min": round(dist * 20),  # 3 mph
            "cost": 0.0,
            "steps": [{"instruction": f"Walk {round(dist, 1)} miles to destination",
                       "duration_min": round(dist * 20)}],
        })

    # Get transit route if distance > 0.5 miles
    if dist > 0.5:
        transit = get_transit_estimate(from_lat, from_lon, to_lat, to_lon)
        result["options"].append(transit)

    # Find nearest HRA benefits center (for free MetroCard / travel assistance)
    nearest_hra = _find_nearest_hra(from_lat, from_lon)

    # Recommendation based on budget and distance
    if budget is not None and budget < MTA_FARE:
        result["recommendation"] = (
            f"Since you have ${budget:.2f}, walking is your best option "
            f"({result['options'][0]['duration_min']} min, {round(dist, 1)} miles). "
            f"MTA fare is ${MTA_FARE:.2f}."
        )
        if dist > 1.5 and nearest_hra:
            result["recommendation"] += (
                f" That's a long walk. For a free MetroCard or travel voucher, "
                f"visit {nearest_hra['name']} at {nearest_hra['address']} "
                f"({nearest_hra['distance_miles']:.1f} mi from you)."
            )
            result["free_metrocard_location"] = nearest_hra
        elif dist > 1.5:
            result["recommendation"] += (
                f" That's a long walk. Visit an HRA Benefits Center for a free MetroCard."
            )
    elif dist < 0.5:
        result["recommendation"] = f"It's a short {round(dist * 20)} minute walk ({round(dist, 2)} miles)."
    elif dist < 2:
        result["recommendation"] = (
            f"You can walk ({result['options'][0]['duration_min']} min) or "
            f"take the subway (${MTA_FARE}, ~{result['options'][-1]['duration_min']} min)."
        )
    else:
        result["recommendation"] = (
            f"Take the subway (${MTA_FARE}, ~{result['options'][-1]['duration_min']} min). "
            f"Walking would take {result['options'][0]['duration_min']} min."
        )

    return result
