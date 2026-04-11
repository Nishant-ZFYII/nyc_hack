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

DATA = Path(__file__).resolve().parent.parent / "data"

# ── Cached state ──────────────────────────────────────────────────────────────
_mart: pd.DataFrame | None = None
_graph_payload: dict | None = None

BOROUGH_MAP = {
    "manhattan": "MN", "mn": "MN",
    "brooklyn":  "BK", "bk": "BK",
    "queens":    "QN", "qn": "QN",
    "bronx":     "BX", "bx": "BX",
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


# ── Core filter ───────────────────────────────────────────────────────────────
_excluded_resources: list[str] = []

def set_excluded_resources(excluded: list[str]):
    global _excluded_resources
    _excluded_resources = excluded


def filter_resources(
    resource_types: list[str],
    filters: dict,
    limit: int = 5,
) -> pd.DataFrame:
    mart, _ = load_state()
    df = mart.copy()

    # Exclude user-reported resources
    if _excluded_resources and "name" in df.columns:
        df = df[~df["name"].isin(_excluded_resources)]

    if resource_types:
        df = df[df["resource_type"].isin(resource_types)]

    borough = _norm_borough(filters.get("borough"))
    if borough:
        df = df[df["borough"] == borough]

    if filters.get("ada_accessible"):
        if "ada_accessible" in df.columns:
            ada_df = df[df["ada_accessible"].astype(str).isin(["True", "1", "true"])]
            if len(ada_df) > 0:
                df = ada_df
            # else: ADA data unavailable, return all results rather than empty

    # Sort by safety_score desc, then quality_score desc
    sort_cols = [c for c in ["safety_score", "quality_score"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False)

    result = df.head(limit).reset_index(drop=True)
    # Convert cuDF → pandas for downstream compatibility (synth, verify, UI)
    if USE_GPU and hasattr(result, 'to_pandas'):
        result = result.to_pandas()
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


def simulate_cold_emergency(params: dict) -> dict:
    mart, payload = load_state()
    mart = _to_pd(mart)
    pluto = pd.read_parquet(DATA / "pluto_layer.parquet")

    borough = _norm_borough(params.get("borough", "BK"))
    people  = params.get("people_displaced", 200)
    temp_f  = params.get("temperature_f", 15)

    # Find available shelters — sort by distance from borough centroid (GPU-accelerated on DGX)
    BOROUGH_CENTROIDS = {
        "BK": (40.6501, -73.9496), "MN": (40.7831, -73.9712),
        "QN": (40.7282, -73.7949), "BX": (40.8448, -73.8648), "SI": (40.5795, -74.1502),
    }
    shelters = mart[mart["resource_type"] == "shelter"].copy()
    if borough:
        shelters = shelters[shelters["borough"] == borough]

    # Distance sort: nearest to the affected borough centroid
    if "latitude" in shelters.columns and "longitude" in shelters.columns and borough in BOROUGH_CENTROIDS:
        clat, clon = BOROUGH_CENTROIDS[borough]
        LAT_M, LON_M = 111_320, 85_390
        shelters = shelters.dropna(subset=["latitude", "longitude"])
        shelters["_dist_m"] = (
            ((shelters["latitude"]  - clat) * LAT_M) ** 2 +
            ((shelters["longitude"] - clon) * LON_M) ** 2
        ) ** 0.5
        shelters = shelters.sort_values("_dist_m")

    available = shelters.head(5)

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
        "available_shelters": available.assign(resource_type="shelter")[[c for c in ["name", "address", "borough", "capacity", "latitude", "longitude", "resource_type"] if c in available.columns or c == "resource_type"]].to_dict("records"),
        "overflow_sites": overflow[["address", "ownername", "landuse"]].to_dict("records"),
        "food_distribution": food[["name", "address"]].to_dict("records"),
        "recommendation": f"Activate {len(overflow)} overflow sites and {len(available)} shelters. "
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

    # Build allocation plan: assign people across shelter sites
    shelter_df = results_by_need.get("shelter", pd.DataFrame())
    allocation = []
    if not shelter_df.empty:
        per_site = people // max(len(shelter_df), 1)
        remainder = people % max(len(shelter_df), 1)
        for i, (_, row) in enumerate(shelter_df.iterrows()):
            assigned = per_site + (1 if i < remainder else 0)
            allocation.append({
                "shelter": row.get("name", "?"),
                "address": row.get("address", ""),
                "borough": row.get("borough", ""),
                "assigned_people": assigned,
                "languages": row.get("languages_spoken", "—"),
            })

    return {
        "intent": "simulate",
        "scenario": "migrant_allocation",
        "people": people,
        "languages": languages,
        "allocation": allocation,
        "resources_by_need": {k: v[["name","address","borough"]].to_dict("records")
                               for k, v in results_by_need.items() if not v.empty},
        "recommendation": (
            f"Distribute {people} people across {len(allocation)} shelter sites. "
            f"Language services available at "
            f"{sum(1 for a in allocation if a['languages'] != '—')} sites."
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
def execute(plan: dict) -> dict[str, Any]:
    intent = plan.get("intent", "lookup")

    if intent == "lookup":
        results = filter_resources(
            resource_types=plan.get("resource_types", []),
            filters=plan.get("filters", {}),
            limit=plan.get("limit", 5),
        )
        return {"intent": "lookup", "results": results}

    elif intent == "needs_assessment":
        profile  = plan.get("client_profile", {})
        needs    = plan.get("identified_needs", [])
        searches = plan.get("resource_searches", [])

        all_results = {}
        for search in searches:
            rtypes  = search.get("resource_types", [])
            filters = search.get("filters", {})
            # Inherit borough from client profile if not specified
            if not filters.get("borough") and profile.get("borough"):
                filters["borough"] = profile["borough"]
            key = "+".join(rtypes)
            all_results[key] = filter_resources(rtypes, filters, limit=search.get("limit", 5))

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
