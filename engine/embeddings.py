"""
engine/embeddings.py — Knowledge Graph Embeddings for resource similarity.

Computes feature vectors for each resource based on their triples,
then enables "find similar resources" queries via cosine similarity.

On DGX: replace with cuML KNN + cuGraph node2vec for richer embeddings.

Usage:
    from engine.embeddings import build_embeddings, find_similar, get_embedding
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

_embeddings: dict | None = None      # {resource_id: np.array}
_feature_names: list | None = None   # feature dimension names
_resource_meta: pd.DataFrame | None = None  # resource_id → name, type, borough


def build_embeddings(force: bool = False) -> dict:
    """
    Build feature-vector embeddings for each resource from their triples.

    Each resource gets a vector with dimensions:
    - One-hot: resource_type (19 dims)
    - One-hot: borough (5 dims)
    - Numeric: safety_score, quality_score, transit_walk_min
    - Count: crime categories (violent, property, harassment, drugs, other)
    - Count: complaint categories (unsanitary, heating, structural, safety, general)
    - Count: co-located resource types
    - Binary: has_capacity, is near transit (<10 min)

    Returns dict of {resource_id: np.array}
    """
    global _embeddings, _feature_names, _resource_meta

    if _embeddings is not None and not force:
        return _embeddings

    t0 = time.time()
    triples = pd.read_parquet(DATA / "triples.parquet")
    mart = pd.read_parquet(DATA / "resource_mart.parquet")

    # Define feature dimensions
    resource_types = sorted(mart["resource_type"].unique())
    boroughs = ["BK", "BX", "MN", "QN", "SI"]
    crime_cats = ["VIOLENT", "PROPERTY", "HARASSMENT", "DRUGS", "OTHER"]
    complaint_cats = ["UNSANITARY", "HEATING", "STRUCTURAL", "SAFETY", "GENERAL"]

    feature_names = (
        [f"type_{t}" for t in resource_types] +
        [f"boro_{b}" for b in boroughs] +
        ["safety_score", "quality_score", "transit_walk_min"] +
        [f"crime_{c}" for c in crime_cats] +
        [f"complaint_{c}" for c in complaint_cats] +
        ["has_capacity", "near_transit", "n_colocated"]
    )
    n_features = len(feature_names)

    embeddings = {}
    resource_meta = {}

    for _, row in mart.iterrows():
        rid = row.get("resource_id", "")
        if not rid:
            continue

        vec = np.zeros(n_features, dtype=np.float32)

        # Resource type one-hot
        rtype = row.get("resource_type", "")
        if rtype in resource_types:
            vec[resource_types.index(rtype)] = 1.0

        # Borough one-hot
        boro = row.get("borough", "")
        if boro in boroughs:
            vec[len(resource_types) + boroughs.index(boro)] = 1.0

        # Numeric features (normalized 0-1)
        base = len(resource_types) + len(boroughs)
        if pd.notna(row.get("safety_score")):
            vec[base] = float(row["safety_score"])
        if pd.notna(row.get("quality_score")):
            vec[base + 1] = float(row["quality_score"])
        if pd.notna(row.get("nearest_transit_walk_min")):
            # Normalize: 0 min = 1.0, 30+ min = 0.0
            vec[base + 2] = max(0, 1.0 - float(row["nearest_transit_walk_min"]) / 30)

        # Crime counts from triples (log-scaled)
        rid_triples = triples[triples["subject"] == rid]
        for i, cat in enumerate(crime_cats):
            crime_t = rid_triples[rid_triples["predicate"] == f"CRIME_{cat}_500M"]
            if not crime_t.empty:
                count = float(crime_t.iloc[0]["object_val"])
                vec[base + 3 + i] = min(1.0, np.log1p(count) / 7)  # log-normalize

        # Complaint counts (log-scaled)
        for i, cat in enumerate(complaint_cats):
            comp_t = rid_triples[rid_triples["predicate"] == f"COMPLAINTS_{cat}_500M"]
            if not comp_t.empty:
                count = float(comp_t.iloc[0]["object_val"])
                vec[base + 3 + len(crime_cats) + i] = min(1.0, np.log1p(count) / 7)

        # Binary features
        bin_base = base + 3 + len(crime_cats) + len(complaint_cats)
        if pd.notna(row.get("capacity")) and float(row.get("capacity", 0)) > 0:
            vec[bin_base] = 1.0
        if pd.notna(row.get("nearest_transit_walk_min")) and float(row["nearest_transit_walk_min"]) < 10:
            vec[bin_base + 1] = 1.0

        # Co-location count
        coloc = rid_triples[rid_triples["predicate"] == "CO_LOCATED_WITH"]
        vec[bin_base + 2] = min(1.0, len(coloc) / 20)

        embeddings[rid] = vec
        resource_meta[rid] = {
            "name": row.get("name", ""),
            "resource_type": rtype,
            "borough": boro,
            "address": row.get("address", ""),
        }

    _embeddings = embeddings
    _feature_names = feature_names
    _resource_meta = pd.DataFrame.from_dict(resource_meta, orient="index")

    print(f"[KGE] Built {len(embeddings)} embeddings, {n_features} dimensions, "
          f"in {time.time()-t0:.1f}s")
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def find_similar(resource_id: str, k: int = 5, same_type: bool = False,
                 same_borough: bool = False) -> pd.DataFrame:
    """
    Find k most similar resources to a given resource based on KGE embedding similarity.

    Parameters
    ----------
    resource_id : str
    k : int — number of similar resources to return
    same_type : bool — if True, only return same resource_type
    same_borough : bool — if True, only return same borough

    Returns DataFrame with columns: resource_id, name, type, borough, similarity
    """
    embs = build_embeddings()
    if resource_id not in embs:
        return pd.DataFrame(columns=["resource_id", "name", "type", "borough", "similarity"])

    target_vec = embs[resource_id]
    meta = _resource_meta

    target_type = meta.loc[resource_id, "resource_type"] if resource_id in meta.index else ""
    target_boro = meta.loc[resource_id, "borough"] if resource_id in meta.index else ""

    scores = []
    for rid, vec in embs.items():
        if rid == resource_id:
            continue
        if same_type and rid in meta.index and meta.loc[rid, "resource_type"] != target_type:
            continue
        if same_borough and rid in meta.index and meta.loc[rid, "borough"] != target_boro:
            continue

        sim = cosine_similarity(target_vec, vec)
        scores.append({"resource_id": rid, "similarity": round(sim, 4)})

    result = pd.DataFrame(scores).nlargest(k, "similarity")
    if not result.empty and meta is not None:
        result = result.merge(
            meta[["name", "resource_type", "borough", "address"]],
            left_on="resource_id", right_index=True, how="left"
        )
    return result.reset_index(drop=True)


def find_similar_to_query(resource_types: list[str], borough: str = None,
                          needs: list[str] = None, k: int = 5) -> pd.DataFrame:
    """
    Find resources similar to a QUERY profile (not an existing resource).
    Constructs a synthetic embedding from the query parameters.
    """
    embs = build_embeddings()
    mart = pd.read_parquet(DATA / "resource_mart.parquet")

    all_types = sorted(mart["resource_type"].unique())
    boroughs = ["BK", "BX", "MN", "QN", "SI"]

    vec = np.zeros(len(_feature_names), dtype=np.float32)

    # Set resource type preferences
    for rt in resource_types:
        if rt in all_types:
            vec[all_types.index(rt)] = 1.0

    # Set borough
    if borough in boroughs:
        vec[len(all_types) + boroughs.index(borough)] = 1.0

    # Prefer high safety + quality
    base = len(all_types) + len(boroughs)
    vec[base] = 0.9      # want high safety
    vec[base + 1] = 0.9  # want high quality
    vec[base + 2] = 0.8  # want near transit

    scores = []
    for rid, emb_vec in embs.items():
        sim = cosine_similarity(vec, emb_vec)
        scores.append({"resource_id": rid, "similarity": round(sim, 4)})

    result = pd.DataFrame(scores).nlargest(k, "similarity")
    if not result.empty and _resource_meta is not None:
        result = result.merge(
            _resource_meta[["name", "resource_type", "borough", "address"]],
            left_on="resource_id", right_index=True, how="left"
        )
    return result.reset_index(drop=True)


def get_embedding(resource_id: str) -> dict:
    """Get the embedding vector for a resource with feature names."""
    embs = build_embeddings()
    if resource_id not in embs:
        return {"error": f"No embedding for {resource_id}"}

    vec = embs[resource_id]
    nonzero = [(name, float(val)) for name, val in zip(_feature_names, vec) if val != 0]
    nonzero.sort(key=lambda x: -abs(x[1]))

    return {
        "resource_id": resource_id,
        "dimensions": len(_feature_names),
        "nonzero_features": len(nonzero),
        "features": nonzero[:15],  # top 15 features
        "vector_norm": float(np.linalg.norm(vec)),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building KGE embeddings...")
    embs = build_embeddings()
    print(f"Total embeddings: {len(embs)}")
    print(f"Dimensions: {len(_feature_names)}")

    # Test similarity
    sample_rid = list(embs.keys())[0]
    print(f"\nFinding resources similar to: {sample_rid}")
    print(f"  ({_resource_meta.loc[sample_rid].to_dict()})")
    similar = find_similar(sample_rid, k=5)
    print(similar.to_string(index=False))

    # Test query-based similarity
    print("\nQuery: shelters in Brooklyn, high safety")
    query_results = find_similar_to_query(["shelter"], "BK", k=5)
    print(query_results.to_string(index=False))
