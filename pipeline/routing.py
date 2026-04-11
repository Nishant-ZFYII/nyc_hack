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
        legs = route.get("legs", [{}])
        steps = []
        for leg in legs:
            for step in leg.get("steps", []):
                instruction = step.get("maneuver", {}).get("type", "")
                name = step.get("name", "")
                dist_m = step.get("distance", 0)
                dur_s = step.get("duration", 0)
                if name or instruction:
                    steps.append({
                        "instruction": f"{instruction} on {name}" if name else instruction,
                        "distance_ft": round(dist_m * 3.281),
                        "duration_min": round(dur_s / 60, 1),
                    })

        return {
            "mode": "walk",
            "distance_miles": round(route["distance"] / 1609.34, 2),
            "duration_min": round(route["duration"] / 60),
            "cost": 0.0,
            "steps": steps,
            "geometry": route.get("geometry"),
        }
    except Exception:
        return None


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
        subway_time = max(direct_dist * 2, 3)  # ~30mph avg subway, min 3 min

        total_time = walk_to_station + subway_time + walk_from_station
        origin_lines = str(origin_station.get("subway_lines", ""))
        dest_lines = str(dest_station.get("subway_lines", ""))

        steps = [
            {"instruction": f"Walk to {origin_station.get('name', 'station')} ({origin_lines})",
             "duration_min": round(walk_to_station, 1), "mode": "walk"},
            {"instruction": f"Take subway toward {dest_station.get('name', 'station')} ({dest_lines})",
             "duration_min": round(subway_time, 1), "mode": "transit"},
            {"instruction": f"Walk to destination from {dest_station.get('name', 'station')}",
             "duration_min": round(walk_from_station, 1), "mode": "walk"},
        ]

        return {
            "mode": "transit",
            "distance_miles": round(haversine_miles(from_lat, from_lon, to_lat, to_lon), 2),
            "duration_min": round(total_time),
            "cost": MTA_FARE,
            "origin_station": origin_station.get("name", ""),
            "dest_station": dest_station.get("name", ""),
            "subway_lines_origin": origin_lines,
            "subway_lines_dest": dest_lines,
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

    # Recommendation based on budget and distance
    if budget is not None and budget < MTA_FARE:
        result["recommendation"] = (
            f"Since you have ${budget:.2f}, walking is your best option "
            f"({result['options'][0]['duration_min']} min, {round(dist, 1)} miles). "
            f"MTA fare is ${MTA_FARE:.2f}."
        )
        if dist > 2:
            result["recommendation"] += (
                f" That's a long walk. For a free MetroCard, visit an HRA Benefits Center."
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
