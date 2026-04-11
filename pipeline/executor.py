"""
pipeline/executor.py — Execute a JSON plan against the resource mart + graph.

Handles:
  - lookup: filter resource_mart by type + constraints
  - needs_assessment: run multiple resource_searches, return combined results
  - simulate: placeholder stubs for cold_emergency / resource_gap
"""
import pickle
from pathlib import Path
from typing import Any

import numpy as np

# GPU / CPU backend selection
try:
    import cudf
    import cupy as cp
    import pandas as pd  # still needed for .to_dict() and interop
    USE_GPU = True
except ImportError:
    import pandas as pd
    USE_GPU = False

try:
    import cugraph
    import networkx as nx
    USE_CUGRAPH = True
except ImportError:
    import networkx as nx
    USE_CUGRAPH = False

try:
    from cuopt import routing as cuopt_routing
    import cudf as _cudf_opt
    USE_CUOPT = True
except ImportError:
    USE_CUOPT = False

DATA = Path(__file__).resolve().parent.parent / "data"

# ── Cached state ──────────────────────────────────────────────────────────────
_mart: pd.DataFrame | None = None
_graph_payload: dict | None = None

BOROUGH_MAP = {
    "manhattan": "MN", "mn": "MN", "midtown": "MN", "mta": "MN",
    "new york": "MN", "nyc": "MN",
    "brooklyn":  "BK", "bk": "BK",
    "queens":    "QN", "qn": "QN",
    "bronx":     "BX", "bx": "BX", "the bronx": "BX",
    "staten island": "SI", "si": "SI",
}


def load_state():
    global _mart, _graph_payload
    if _mart is None:
        if USE_GPU:
            _mart = cudf.read_parquet(DATA / "resource_mart.parquet")
        else:
            _mart = pd.read_parquet(DATA / "resource_mart.parquet")
    if _graph_payload is None:
        with open(DATA / "graph.pkl", "rb") as f:
            _graph_payload = pickle.load(f)
        # Rebuild cuGraph from edges if graph was saved as None (cuGraph can't be pickled)
        if _graph_payload.get("graph") is None and USE_CUGRAPH:
            import cugraph as _cg
            import cudf as _cudf
            edges = _graph_payload["edges"]
            gdf = _cudf.DataFrame({
                "src": edges["src"].values,
                "dst": edges["dst"].values,
                "weight": edges["weight"].values,
            })
            G = _cg.Graph(directed=True)
            G.from_cudf_edgelist(gdf, source="src", destination="dst",
                                 edge_attr="weight", renumber=False)
            _graph_payload["graph"] = G
        elif _graph_payload.get("graph") is None:
            G = nx.DiGraph()
            edges = _graph_payload["edges"]
            for _, row in edges.iterrows():
                G.add_edge(int(row["src"]), int(row["dst"]),
                           weight=row["weight"], edge_type=row.get("edge_type", ""))
            _graph_payload["graph"] = G
    return _mart, _graph_payload


def _norm_borough(b):
    if not b:
        return None
    return BOROUGH_MAP.get(str(b).strip().lower(), str(b).strip().upper()[:2])


def _borough_from_coords(lat, lon):
    """Detect NYC borough from lat/lon coordinates."""
    # Approximate bounding boxes for NYC boroughs
    if lat > 40.8 and lon > -73.93:
        return "BX"  # Bronx
    if lat > 40.7 and lon > -74.01 and lon < -73.9:
        return "MN"  # Manhattan
    if lat < 40.7 and lon < -73.85:
        return "BK"  # Brooklyn
    if lat > 40.7 and lon > -73.85 and lon < -73.7:
        return "QN"  # Queens
    if lat < 40.65 and lon < -74.05:
        return "SI"  # Staten Island
    # Fallback: check which borough centroid is closest
    centroids = {
        "MN": (40.7831, -73.9712), "BK": (40.6501, -73.9496),
        "QN": (40.7282, -73.7949), "BX": (40.8448, -73.8648),
        "SI": (40.5795, -74.1502),
    }
    import math
    closest = min(centroids.items(),
                  key=lambda x: math.hypot(lat - x[1][0], lon - x[1][1]))
    return closest[0]


# ── Core filter ───────────────────────────────────────────────────────────────
_excluded_resources: list[str] = []

def set_excluded_resources(excluded: list[str]):
    global _excluded_resources
    _excluded_resources = excluded


def filter_resources(
    resource_types: list[str],
    filters: dict,
    limit: int = 5,
    user_location: dict | None = None,
) -> pd.DataFrame:
    mart, _ = load_state()
    df = mart.copy()

    # Convert cuDF → pandas early for geocode compatibility
    if USE_GPU and hasattr(df, 'to_pandas'):
        df = df.to_pandas()

    # Exclude user-reported resources
    if _excluded_resources and "name" in df.columns:
        df = df[~df["name"].isin(_excluded_resources)]

    if resource_types:
        df = df[df["resource_type"].isin(resource_types)]

    # Detect borough from user location if available (more reliable than LLM)
    borough = _norm_borough(filters.get("borough"))
    if user_location and user_location.get("lat"):
        detected_borough = _borough_from_coords(user_location["lat"], user_location.get("lon", 0))
        if detected_borough:
            borough = detected_borough

    if borough and not user_location:
        df = df[df["borough"] == borough]
    elif borough and user_location:
        # Try borough filter first, but fall back to all if empty
        filtered = df[df["borough"] == borough]
        if len(filtered) > 0:
            df = filtered
        # else: skip borough filter, distance sort will handle it

    if filters.get("ada_accessible"):
        if "ada_accessible" in df.columns:
            ada_df = df[df["ada_accessible"].astype(str).isin(["True", "1", "true"])]
            if len(ada_df) > 0:
                df = ada_df

    # Location-aware sorting: if user location is known, sort by distance
    if user_location and user_location.get("lat") and user_location.get("lon"):
        try:
            from pipeline.geocode import sort_by_distance
            df = sort_by_distance(df, user_location["lat"], user_location["lon"])
        except Exception:
            # Fallback to safety score sort
            sort_cols = [c for c in ["safety_score", "quality_score"] if c in df.columns]
            if sort_cols:
                df = df.sort_values(sort_cols, ascending=False)
    else:
        # Default: sort by safety_score desc, then quality_score desc
        sort_cols = [c for c in ["safety_score", "quality_score"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False)

    result = df.head(limit).reset_index(drop=True)
    return result


# ── Graph-based nearest resource ──────────────────────────────────────────────
def graph_nearest(resource_types: list[str], from_lat: float, from_lon: float,
                  limit: int = 5) -> pd.DataFrame:
    """Find nearest resources of given types using graph proximity."""
    mart, payload = load_state()
    G = payload["graph"]
    resources = _to_pd(payload["resources"])

    # Filter to target types
    targets = resources[resources["resource_type"].isin(resource_types)].copy()
    if targets.empty:
        return filter_resources(resource_types, {}, limit)

    # Find closest by straight-line distance (graph SSSP too slow on CPU for demo)
    LAT_M, LON_M = 111_320, 85_390
    dlat = (targets["latitude"].values  - from_lat) * LAT_M
    dlon = (targets["longitude"].values - from_lon) * LON_M
    targets["_dist_m"] = np.sqrt(dlat**2 + dlon**2)
    targets["walk_min"] = (targets["_dist_m"] / 80).round(1)

    return targets.nsmallest(limit, "_dist_m").reset_index(drop=True)


# ── Simulate stubs ────────────────────────────────────────────────────────────
def _to_pd(df):
    """Convert cuDF DataFrame to pandas if needed."""
    if hasattr(df, 'to_pandas'):
        return df.to_pandas()
    return df


def _cuopt_allocate(sites_df, people, centroid_lat, centroid_lon):
    """Use cuOpt VRP to optimally allocate people across sites minimizing travel time."""
    try:
        LAT_M, LON_M = 111_320, 85_390
        n_sites = len(sites_df)
        if n_sites == 0:
            return []

        # Build cost matrix: distances between centroid (depot) and all sites
        lats = sites_df["latitude"].values
        lons = sites_df["longitude"].values
        # Node 0 = depot (crisis epicenter), nodes 1..n = shelter sites
        all_lats = np.array([centroid_lat] + list(lats))
        all_lons = np.array([centroid_lon] + list(lons))
        n = len(all_lats)

        # Pairwise distance matrix in minutes (walk speed 80m/min)
        cost = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                d = np.sqrt(((all_lats[i] - all_lats[j]) * LAT_M)**2 +
                            ((all_lons[i] - all_lons[j]) * LON_M)**2)
                cost[i][j] = d / 80.0  # minutes walking

        cost_df = _cudf_opt.DataFrame(cost)

        # Set up VRP: 1 "vehicle" per site (vehicle = transport route to that shelter)
        capacities = []
        for _, row in sites_df.iterrows():
            cap = row.get("capacity", None)
            if pd.isna(cap) or cap is None or cap == 0:
                cap = max(people // n_sites, 20)
            capacities.append(int(cap))

        # n locations, n_sites vehicles
        data_model = cuopt_routing.DataModel(n, n_sites)
        data_model.add_cost_matrix(cost_df)

        # Demands per order: depot=0, each shelter order gets proportional demand
        demands = [0]  # depot (location 0)
        per_site = people // n_sites
        remainder = people % n_sites
        for i in range(n_sites):
            demands.append(per_site + (1 if i < remainder else 0))

        # Add capacity dimension: "people" with demand per order and capacity per vehicle
        data_model.add_capacity_dimension(
            "people",
            _cudf_opt.Series(demands),
            _cudf_opt.Series(capacities),
        )

        # Vehicle start/end at depot (location 0)
        data_model.set_vehicle_locations(
            _cudf_opt.Series([0] * n_sites),  # start at depot
            _cudf_opt.Series([0] * n_sites),  # return to depot
        )

        # Order locations: order i is at location i (all n locations including depot)
        data_model.set_order_locations(
            _cudf_opt.Series(list(range(n)))
        )

        solver = cuopt_routing.SolverSettings()
        solver.set_time_limit(2.0)
        result = cuopt_routing.Solve(data_model, solver)

        status = result.get_status()
        if status == 0:
            allocation = []
            for i, (_, row) in enumerate(sites_df.iterrows()):
                allocation.append({
                    "name": row.get("name", "?"),
                    "address": row.get("address", ""),
                    "borough": row.get("borough", ""),
                    "assigned_people": demands[i + 1],
                    "travel_min": round(cost[0][i + 1], 1),
                    "optimized_by": "cuOpt VRP",
                })
            return allocation
    except Exception as e:
        import traceback
        traceback.print_exc()
    return None


def _greedy_allocate(sites_df, people):
    """Greedy fallback: allocate people proportionally across sites."""
    n_sites = max(len(sites_df), 1)
    per_site = people // n_sites
    remainder = people % n_sites
    allocation = []
    for i, (_, row) in enumerate(sites_df.iterrows()):
        allocation.append({
            "name": row.get("name", "?"),
            "address": row.get("address", ""),
            "borough": row.get("borough", ""),
            "assigned_people": per_site + (1 if i < remainder else 0),
            "optimized_by": "greedy",
        })
    return allocation


def simulate_cold_emergency(params: dict) -> dict:
    mart, payload = load_state()
    mart = _to_pd(mart)
    pluto = pd.read_parquet(DATA / "pluto_layer.parquet")

    borough = _norm_borough(params.get("borough", "BK"))
    people  = params.get("people_displaced", 200)
    temp_f  = params.get("temperature_f", 15)

    BOROUGH_CENTROIDS = {
        "BK": (40.6501, -73.9496), "MN": (40.7831, -73.9712),
        "QN": (40.7282, -73.7949), "BX": (40.8448, -73.8648), "SI": (40.5795, -74.1502),
    }
    shelters = mart[mart["resource_type"] == "shelter"].copy()
    if borough:
        shelters = shelters[shelters["borough"] == borough]

    # Distance sort: nearest to the affected borough centroid
    clat, clon = BOROUGH_CENTROIDS.get(borough, (40.7128, -74.0060))
    if "latitude" in shelters.columns and "longitude" in shelters.columns:
        LAT_M, LON_M = 111_320, 85_390
        shelters = shelters.dropna(subset=["latitude", "longitude"])
        shelters["_dist_m"] = (
            ((shelters["latitude"]  - clat) * LAT_M) ** 2 +
            ((shelters["longitude"] - clon) * LON_M) ** 2
        ) ** 0.5
        shelters = shelters.sort_values("_dist_m")

    available = shelters.head(8)

    # cuOpt VRP allocation or greedy fallback
    allocation = None
    optimizer_used = "greedy"
    if USE_CUOPT and len(available) > 0:
        allocation = _cuopt_allocate(available, people, clat, clon)
        if allocation:
            optimizer_used = "cuOpt VRP"
    if allocation is None:
        allocation = _greedy_allocate(available, people)

    # Find PLUTO overflow sites
    overflow = pluto[pluto["is_overflow_candidate"] == True].copy()
    if borough:
        overflow = overflow[overflow["borough"] == borough]
    overflow = overflow.head(5)

    # Find nearby food banks
    food = mart[mart["resource_type"] == "food_bank"].copy()
    if borough:
        food = food[food["borough"] == borough]
    food = food.head(3)

    return {
        "intent": "simulate",
        "scenario": "cold_emergency",
        "people_displaced": people,
        "temperature_f": temp_f,
        "optimizer": optimizer_used,
        "allocation": allocation,
        "available_shelters": available.assign(resource_type="shelter")[[c for c in ["name", "address", "borough", "capacity", "latitude", "longitude", "resource_type"] if c in available.columns or c == "resource_type"]].to_dict("records"),
        "overflow_sites": overflow[["address", "ownername", "landuse"]].to_dict("records"),
        "food_distribution": food[["name", "address"]].to_dict("records"),
        "recommendation": f"[{optimizer_used}] Allocate {people} people across {len(allocation)} shelter sites. "
                          f"Activate {len(overflow)} overflow sites. "
                          f"Deploy {len(food)} food distribution points nearby."
    }


def simulate_capacity_change(params: dict) -> dict:
    """What happens if we add N beds in a borough?"""
    mart, _ = load_state()
    mart = _to_pd(mart)
    borough   = _norm_borough(params.get("borough", "BK"))
    new_beds  = params.get("new_beds", 500)
    rtype     = params.get("resource_type", "shelter")

    before = mart[mart["resource_type"] == rtype].copy()
    after  = before.copy()

    # Simulate adding beds: distribute proportionally to existing sites in target borough
    in_boro = after[after["borough"] == borough]
    n_sites = max(len(in_boro), 1)
    beds_each = new_beds // n_sites

    if "capacity" in after.columns:
        after.loc[after["borough"] == borough, "capacity"] = (
            after.loc[after["borough"] == borough, "capacity"].fillna(0) + beds_each
        )

    by_boro_before = before.groupby("borough").size().reset_index(name="sites_before")
    by_boro_after  = after.groupby("borough").size().reset_index(name="sites_after")
    summary = by_boro_before.merge(by_boro_after, on="borough", how="outer").fillna(0)
    summary["pop_estimate"] = summary["borough"].map({
        "BK": 2_600_000, "QN": 2_300_000, "MN": 1_600_000, "BX": 1_500_000, "SI": 500_000,
    }).fillna(500_000)
    summary["coverage_per_100k_before"] = (summary["sites_before"] / summary["pop_estimate"] * 100_000).round(1)
    summary["coverage_per_100k_after"]  = (summary["sites_after"]  / summary["pop_estimate"] * 100_000).round(1)
    summary["improvement"] = (summary["coverage_per_100k_after"] - summary["coverage_per_100k_before"]).round(2)

    return {
        "intent": "simulate",
        "scenario": "capacity_change",
        "borough": borough,
        "new_beds": new_beds,
        "resource_type": rtype,
        "sites_added": n_sites,
        "summary": summary.to_dict("records"),
        "recommendation": (
            f"Adding {new_beds} {rtype} beds across {n_sites} sites in {borough} "
            f"improves coverage from "
            f"{summary.loc[summary['borough']==borough,'coverage_per_100k_before'].values[0] if len(summary[summary['borough']==borough]) else '?'} "
            f"to "
            f"{summary.loc[summary['borough']==borough,'coverage_per_100k_after'].values[0] if len(summary[summary['borough']==borough]) else '?'} "
            f"per 100K residents."
        ),
    }


def simulate_migrant_allocation(params: dict) -> dict:
    """Allocate newly arrived migrants to shelter+food+school clusters."""
    mart, _ = load_state()
    mart = _to_pd(mart)
    people    = params.get("people", 80)
    languages = params.get("languages", [])
    needs     = params.get("needs", ["shelter", "food_bank", "school"])
    borough   = _norm_borough(params.get("borough"))

    results_by_need = {}
    for need in needs:
        df = mart[mart["resource_type"] == need].copy()
        if borough:
            df = df[df["borough"] == borough]
        # Prefer sites with language match if field exists
        if languages and "languages_spoken" in df.columns:
            lang_lower = [l.lower() for l in languages]
            def lang_score(s):
                if pd.isna(s):
                    return 0
                s_lower = str(s).lower()
                return sum(1 for l in lang_lower if l in s_lower)
            df["lang_match"] = df["languages_spoken"].apply(lang_score)
            df = df.sort_values(["lang_match", "safety_score"], ascending=[False, False])
        elif "safety_score" in df.columns:
            df = df.sort_values("safety_score", ascending=False)
        results_by_need[need] = df.head(5).reset_index(drop=True)

    # Build allocation plan: cuOpt VRP or greedy fallback
    shelter_df = results_by_need.get("shelter", pd.DataFrame())
    allocation = None
    optimizer_used = "greedy"

    if USE_CUOPT and not shelter_df.empty and "latitude" in shelter_df.columns:
        # Use NYC midpoint as depot (arrival point)
        depot_lat = shelter_df["latitude"].mean()
        depot_lon = shelter_df["longitude"].mean()
        allocation = _cuopt_allocate(shelter_df, people, depot_lat, depot_lon)
        if allocation:
            optimizer_used = "cuOpt VRP"
            # Add language info to cuOpt results
            for i, (_, row) in enumerate(shelter_df.iterrows()):
                if i < len(allocation):
                    allocation[i]["languages"] = row.get("languages_spoken", "—")

    if allocation is None:
        allocation = []
        n = max(len(shelter_df), 1)
        per_site = people // n
        remainder = people % n
        for i, (_, row) in enumerate(shelter_df.iterrows()):
            allocation.append({
                "name": row.get("name", "?"),
                "address": row.get("address", ""),
                "borough": row.get("borough", ""),
                "assigned_people": per_site + (1 if i < remainder else 0),
                "languages": row.get("languages_spoken", "—"),
                "optimized_by": "greedy",
            })

    return {
        "intent": "simulate",
        "scenario": "migrant_allocation",
        "people": people,
        "languages": languages,
        "optimizer": optimizer_used,
        "allocation": allocation,
        "resources_by_need": {k: v[["name","address","borough"]].to_dict("records")
                               for k, v in results_by_need.items() if not v.empty},
        "recommendation": (
            f"[{optimizer_used}] Distribute {people} people across {len(allocation)} shelter sites. "
            f"Language services available at "
            f"{sum(1 for a in allocation if a.get('languages', '—') != '—')} sites."
        ),
    }


def simulate_resource_gap(params: dict) -> dict:
    mart, _ = load_state()
    mart = _to_pd(mart)
    resources = mart[mart["resource_type"].isin(["shelter", "food_bank", "hospital"])].copy()
    # Group by borough
    by_borough = resources.groupby("borough").size().reset_index(name="resource_count")
    by_borough["pop_estimate"] = by_borough["borough"].map({
        "BK": 2_600_000, "QN": 2_300_000, "MN": 1_600_000,
        "BX": 1_500_000, "SI": 500_000,
    }).fillna(500_000)
    by_borough["resources_per_100k"] = (
        by_borough["resource_count"] / by_borough["pop_estimate"] * 100_000
    ).round(1)
    by_borough = by_borough.sort_values("resources_per_100k")
    return {
        "intent": "simulate",
        "scenario": "resource_gap",
        "gaps": by_borough.to_dict("records"),
        "most_underserved": by_borough.iloc[0]["borough"],
    }


# ── Main execute entry point ──────────────────────────────────────────────────
def _get_user_location(plan: dict) -> dict | None:
    """Try to extract user location from the plan for distance-based sorting."""
    # Priority 1: explicit user location from frontend GPS/manual input
    if plan.get("_user_location"):
        return plan["_user_location"]

    # Priority 2: geocode from query text
    try:
        from pipeline.geocode import geocode_location
        profile = plan.get("client_profile", {})
        situation = profile.get("situation", "")

        query_text = plan.get("_original_query", situation)
        if query_text:
            loc = geocode_location(query_text)
            if loc:
                return loc
    except Exception:
        pass
    return None


def execute(plan: dict) -> dict[str, Any]:
    intent = plan.get("intent", "lookup")
    user_loc = _get_user_location(plan)

    if intent == "lookup":
        results = filter_resources(
            resource_types=plan.get("resource_types", []),
            filters=plan.get("filters", {}),
            limit=plan.get("limit", 5),
            user_location=user_loc,
        )
        return {"intent": "lookup", "results": results, "user_location": user_loc}

    elif intent == "needs_assessment":
        profile  = plan.get("client_profile", {})
        needs    = plan.get("identified_needs", [])
        searches = plan.get("resource_searches", [])

        all_results = {}
        for search in searches:
            rtypes  = search.get("resource_types", [])
            filters = search.get("filters", {})
            if not filters.get("borough") and profile.get("borough"):
                filters["borough"] = profile["borough"]
            key = "+".join(rtypes)
            all_results[key] = filter_resources(rtypes, filters, limit=search.get("limit", 5),
                                                 user_location=user_loc)

        return {
            "intent": "needs_assessment",
            "client_profile": profile,
            "identified_needs": needs,
            "results_by_need": all_results,
        }

    elif intent == "simulate":
        scenario = plan.get("scenario", "")
        params   = plan.get("params", {})
        if scenario == "cold_emergency":
            return simulate_cold_emergency(params)
        elif scenario == "resource_gap":
            return simulate_resource_gap(params)
        elif scenario == "capacity_change":
            return simulate_capacity_change(params)
        elif scenario == "migrant_allocation":
            return simulate_migrant_allocation(params)
        else:
            return {"intent": "simulate", "scenario": scenario, "error": "Scenario not yet implemented"}

    elif intent == "explain":
        question = plan.get("question", "")
        target   = plan.get("target", "")
        try:
            import importlib.util as _ilu
            _sp = _ilu.spec_from_file_location("kg_conf", str(Path(__file__).resolve().parent.parent / "engine" / "confidence.py"))
            _cm = _ilu.module_from_spec(_sp)
            _sp.loader.exec_module(_cm)
            explain_underserved = _cm.explain_underserved
            explain_resource_recommendation = _cm.explain_resource_recommendation
            explain_cold_emergency = _cm.explain_cold_emergency
            if question == "why_underserved":
                return {"intent": "explain", "question": question, **explain_underserved(target)}
            elif question == "why_recommend":
                return {"intent": "explain", "question": question, **explain_resource_recommendation(target)}
            elif question == "confidence_emergency":
                return {"intent": "explain", "question": question, **explain_cold_emergency(target or "BK")}
            else:
                return {"intent": "explain", "question": question, "error": "Unknown explain question"}
        except Exception as e:
            return {"intent": "explain", "error": str(e)}

    else:
        return {"intent": "unknown", "error": f"Unknown intent: {intent}"}
