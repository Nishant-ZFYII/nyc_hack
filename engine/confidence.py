"""
engine/confidence.py — Confidence-scored multi-hop graph traversal.

Given a question like "Why is the Bronx underserved?", traverses the triple store
and returns the reasoning path with cumulative confidence at each hop.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

_triples: pd.DataFrame | None = None


def load_triples() -> pd.DataFrame:
    global _triples
    if _triples is None:
        _triples = pd.read_parquet(DATA / "triples.parquet")
    return _triples


def query_triples(subject=None, predicate=None, object_val=None,
                  source=None, min_confidence=0.0) -> pd.DataFrame:
    """Filter triples by any combination of fields."""
    df = load_triples()
    if subject is not None:
        df = df[df["subject"] == str(subject)]
    if predicate is not None:
        df = df[df["predicate"] == predicate]
    if object_val is not None:
        df = df[df["object_val"] == str(object_val)]
    if source is not None:
        df = df[df["source"] == source]
    if min_confidence > 0:
        df = df[df["confidence"] >= min_confidence]
    return df


def traverse_path(start_subject: str, predicates: list[str],
                  min_confidence: float = 0.3) -> list[dict]:
    """
    Multi-hop traversal: follow a chain of predicates from a starting subject.
    Returns the path with cumulative confidence.

    Example:
        traverse_path("boro_BX", ["TOTAL_RESOURCES", "HAS_POPULATION"])
        → [
            {"hop": 1, "subject": "boro_BX", "predicate": "TOTAL_RESOURCES",
             "object": "1439", "confidence": 0.75, "cumulative": 0.75},
            {"hop": 2, "subject": "boro_BX", "predicate": "HAS_POPULATION",
             "object": "1500000", "confidence": 0.60, "cumulative": 0.45},
          ]
    """
    path = []
    current_subject = start_subject
    cumulative_conf = 1.0

    for i, pred in enumerate(predicates):
        triples = query_triples(subject=current_subject, predicate=pred)
        if triples.empty:
            path.append({
                "hop": i + 1, "subject": current_subject,
                "predicate": pred, "object": "NOT_FOUND",
                "confidence": 0.0, "cumulative": 0.0, "source": "—",
            })
            break

        # Take highest confidence triple if multiple
        best = triples.sort_values("confidence", ascending=False).iloc[0]
        cumulative_conf *= best["confidence"]

        path.append({
            "hop": i + 1,
            "subject": current_subject,
            "predicate": pred,
            "object": best["object_val"],
            "confidence": round(best["confidence"], 3),
            "cumulative": round(cumulative_conf, 3),
            "source": best["source"],
        })

        # The object becomes the next subject (for entity hops)
        # For value predicates, stay on same subject
        if best["object_val"].startswith(("boro_", "r_", "station_", "shelter_",
                                          "food_bank_", "hospital_", "lot_")):
            current_subject = best["object_val"]

    return path


def explain_underserved(borough: str) -> dict:
    """Multi-hop explanation: why is a borough underserved?"""
    boro_key = f"boro_{borough}"
    df = load_triples()

    # Gather all borough-level triples
    boro_triples = query_triples(subject=boro_key)
    if boro_triples.empty:
        return {"error": f"No triples found for {boro_key}"}

    # Get this borough's stats
    pop = boro_triples[boro_triples["predicate"] == "HAS_POPULATION"]
    total_res = boro_triples[boro_triples["predicate"] == "TOTAL_RESOURCES"]
    per_100k = boro_triples[boro_triples["predicate"] == "RESOURCES_PER_100K"]

    pop_val = int(pop.iloc[0]["object_val"]) if not pop.empty else 0
    res_val = int(total_res.iloc[0]["object_val"]) if not total_res.empty else 0
    per_100k_val = float(per_100k.iloc[0]["object_val"]) if not per_100k.empty else 0

    # Get per-type breakdowns
    type_triples = boro_triples[boro_triples["predicate"].str.endswith("_PER_100K")]
    type_breakdown = {}
    for _, t in type_triples.iterrows():
        rtype = t["predicate"].replace("_PER_100K", "").lower()
        type_breakdown[rtype] = float(t["object_val"])

    # Compare across all boroughs
    all_boros = {}
    for b in ["BK", "QN", "MN", "BX", "SI"]:
        bk = f"boro_{b}"
        bdf = query_triples(subject=bk, predicate="RESOURCES_PER_100K")
        if not bdf.empty:
            all_boros[b] = float(bdf.iloc[0]["object_val"])

    ranking = sorted(all_boros.items(), key=lambda x: x[1])

    # Build reasoning path
    path = [
        {"hop": 1, "fact": f"{borough} has population {pop_val:,}",
         "confidence": 0.60, "source": "acs_census"},
        {"hop": 2, "fact": f"{borough} has {res_val:,} total resources",
         "confidence": 0.75, "source": "mart_derived"},
        {"hop": 3, "fact": f"That's {per_100k_val} resources per 100K residents",
         "confidence": 0.75, "source": "mart_derived"},
        {"hop": 4, "fact": f"Borough ranking (low→high): {', '.join(f'{b}={v}' for b,v in ranking)}",
         "confidence": 0.75, "source": "mart_derived"},
    ]

    # Identify which resource types are most lacking
    weakest_types = sorted(type_breakdown.items(), key=lambda x: x[1])[:3]
    for i, (rtype, val) in enumerate(weakest_types):
        path.append({
            "hop": 5 + i,
            "fact": f"Lowest coverage: {rtype} at {val} per 100K",
            "confidence": 0.75, "source": "mart_derived",
        })

    cumulative = 1.0
    for p in path:
        cumulative *= p["confidence"]
        p["cumulative"] = round(cumulative, 3)

    return {
        "borough": borough,
        "population": pop_val,
        "total_resources": res_val,
        "resources_per_100k": per_100k_val,
        "ranking": ranking,
        "weakest_types": weakest_types,
        "reasoning_path": path,
        "overall_confidence": round(cumulative, 3),
    }


def explain_resource_recommendation(resource_id: str) -> dict:
    """Explain why a particular resource was recommended."""
    triples = query_triples(subject=resource_id)
    if triples.empty:
        return {"error": f"No triples for {resource_id}"}

    facts = {}
    for _, t in triples.iterrows():
        facts[t["predicate"]] = {
            "value": t["object_val"],
            "confidence": t["confidence"],
            "source": t["source"],
        }

    # Build explanation path
    path = []
    priority_preds = ["HAS_TYPE", "IN_BOROUGH", "SAFETY_SCORE", "QUALITY_SCORE",
                      "NEAREST_TRANSIT", "TRANSIT_WALK_MIN",
                      "CRIME_VIOLENT_500M", "COMPLAINTS_UNSANITARY_500M"]
    cumulative = 1.0
    for i, pred in enumerate(priority_preds):
        if pred in facts:
            f = facts[pred]
            cumulative *= f["confidence"]
            path.append({
                "hop": i + 1,
                "fact": f"{pred}: {f['value']}",
                "confidence": f["confidence"],
                "cumulative": round(cumulative, 3),
                "source": f["source"],
            })

    return {
        "resource_id": resource_id,
        "total_triples": len(triples),
        "reasoning_path": path,
        "overall_confidence": round(cumulative, 3),
        "all_facts": facts,
    }


def explain_cold_emergency(borough: str) -> dict:
    """Explain confidence in a cold emergency plan."""
    boro_key = f"boro_{borough}"

    # How many shelters, overflow sites, food banks?
    df = load_triples()
    shelter_count = query_triples(subject=boro_key, predicate="COUNT_SHELTER")
    food_count = query_triples(subject=boro_key, predicate="COUNT_FOOD_BANK")
    overflow_count = query_triples(subject=boro_key, predicate="LOTS_ASSEMBLY_COMMUNITY")

    path = []
    cumulative = 1.0

    def add_hop(fact, conf, source):
        nonlocal cumulative
        cumulative *= conf
        path.append({"hop": len(path)+1, "fact": fact,
                      "confidence": conf, "cumulative": round(cumulative, 3), "source": source})

    if not shelter_count.empty:
        add_hop(f"Shelters in {borough}: {shelter_count.iloc[0]['object_val']}",
                0.95, "mart_derived")
    if not overflow_count.empty:
        add_hop(f"Assembly-zoned overflow sites in {borough}: {overflow_count.iloc[0]['object_val']}",
                0.90, "pluto")
    if not food_count.empty:
        add_hop(f"Food banks in {borough}: {food_count.iloc[0]['object_val']}",
                0.95, "mart_derived")

    add_hop("Shelter data source: DOHMH facilities registry (official)", 0.95, "dohmh")
    add_hop("Overflow sites from PLUTO zoning (landuse=08, assembly)", 0.90, "pluto")
    add_hop("Distance sort: Euclidean from borough centroid", 0.85, "spatial_derived")

    return {
        "borough": borough,
        "reasoning_path": path,
        "overall_confidence": round(cumulative, 3),
    }
