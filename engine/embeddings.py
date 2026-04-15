"""
engine/embeddings.py — Knowledge Graph Embeddings for resource similarity.

Two modes:
  1. PyKEEN mode (preferred): Loads real TransE/RotatE embeddings trained on 328K triples.
     Run engine/train_kge.py first to generate data/kge_embeddings.pkl.
  2. Fallback mode: Hand-crafted 40-dim feature vectors from resource mart + triples.

Usage:
    from engine.embeddings import build_embeddings, find_similar, get_embedding
"""
from __future__ import annotations
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

# cuML GPU acceleration for similarity search
try:
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    import cupy as cp
    _HAS_CUML = True
except ImportError:
    _HAS_CUML = False

DATA = Path(__file__).resolve().parent.parent / "data"

_embeddings: dict | None = None      # {resource_id: np.array}
_feature_names: list | None = None   # feature dimension names
_resource_meta: pd.DataFrame | None = None  # resource_id → name, type, borough
_kge_mode: str = "unknown"           # "pykeen" or "handcrafted"

# cuML NearestNeighbors index (GPU) — built once, reused for all queries
_knn_index = None
_resource_ids_ordered: list | None = None  # order matches the index


def _load_pykeen_embeddings() -> dict | None:
    """Try to load PyKEEN-trained embeddings from data/kge_embeddings.pkl."""
    kge_path = DATA / "kge_embeddings.pkl"
    if not kge_path.exists():
        return None
    try:
        with open(kge_path, "rb") as f:
            payload = pickle.load(f)
        entity_embs = payload.get("entity_embeddings", {})
        if not entity_embs:
            return None
        print(f"[KGE] Loaded PyKEEN {payload.get('model_name', '?')} embeddings: "
              f"{len(entity_embs):,} entities, {payload.get('embedding_dim', '?')} dims")
        return entity_embs
    except Exception as e:
        print(f"[KGE] Failed to load PyKEEN embeddings: {e}")
        return None


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
    global _embeddings, _feature_names, _resource_meta, _kge_mode

    if _embeddings is not None and not force:
        return _embeddings

    t0 = time.time()
    mart = pd.read_parquet(DATA / "resource_mart.parquet")

    # Build resource metadata (needed for both modes)
    resource_meta = {}
    for _, row in mart.iterrows():
        rid = row.get("resource_id", "")
        if rid:
            resource_meta[rid] = {
                "name": row.get("name", ""),
                "resource_type": row.get("resource_type", ""),
                "borough": row.get("borough", ""),
                "address": row.get("address", ""),
            }
    _resource_meta = pd.DataFrame.from_dict(resource_meta, orient="index")

    # Try PyKEEN embeddings first
    pykeen_embs = _load_pykeen_embeddings()
    if pykeen_embs is not None:
        # Filter to only resource IDs that exist in the mart
        resource_ids = set(mart["resource_id"].values)
        embeddings = {k: v for k, v in pykeen_embs.items() if k in resource_ids}
        if embeddings:
            _embeddings = embeddings
            _feature_names = [f"kge_dim_{i}" for i in range(len(next(iter(embeddings.values()))))]
            _kge_mode = "pykeen"
            print(f"[KGE] Using PyKEEN embeddings: {len(embeddings)} resources, "
                  f"{len(_feature_names)} dims ({time.time()-t0:.1f}s)")
            return _embeddings
        print("[KGE] PyKEEN embeddings loaded but no resource IDs matched — falling back")

    # Fallback: hand-crafted feature vectors
    triples = pd.read_parquet(DATA / "triples.parquet")
    _kge_mode = "handcrafted"

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

    _embeddings = embeddings
    _feature_names = feature_names

    print(f"[KGE] Built {len(embeddings)} handcrafted embeddings, {n_features} dimensions, "
          f"in {time.time()-t0:.1f}s")
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def _build_knn_index():
    """Build cuML NearestNeighbors index on GPU — one-time setup per process."""
    global _knn_index, _resource_ids_ordered
    if _knn_index is not None:
        return _knn_index

    embs = build_embeddings()
    if not embs:
        return None

    # Fix resource order + convert to matrix
    _resource_ids_ordered = list(embs.keys())
    matrix = np.stack([embs[rid] for rid in _resource_ids_ordered]).astype(np.float32)

    # Normalize rows so L2 distance == 2*(1 - cosine_sim)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    if _HAS_CUML:
        # GPU: upload to device, build cuML kNN
        t0 = time.time()
        gpu_matrix = cp.asarray(matrix)
        _knn_index = cuNearestNeighbors(n_neighbors=min(50, len(matrix)),
                                         metric="euclidean", algorithm="brute")
        _knn_index.fit(gpu_matrix)
        print(f"[KGE] Built cuML GPU kNN index: {len(matrix):,} × {matrix.shape[1]} dims "
              f"in {time.time()-t0:.2f}s")
        _knn_index._matrix = gpu_matrix  # stash for query-from-vec
        _knn_index._backend = "cuml"
    else:
        # CPU fallback: sklearn
        from sklearn.neighbors import NearestNeighbors as skNN
        t0 = time.time()
        _knn_index = skNN(n_neighbors=min(50, len(matrix)), metric="euclidean", algorithm="brute")
        _knn_index.fit(matrix)
        _knn_index._matrix = matrix
        _knn_index._backend = "sklearn"
        print(f"[KGE] Built sklearn CPU kNN index: {len(matrix):,} × {matrix.shape[1]} dims "
              f"in {time.time()-t0:.2f}s")

    return _knn_index


def find_similar(resource_id: str, k: int = 5, same_type: bool = False,
                 same_borough: bool = False) -> pd.DataFrame:
    """
    Find k most similar resources via cuML kNN on GPU (falls back to sklearn).

    Returns DataFrame with columns: resource_id, name, resource_type, borough,
    address, similarity, backend (cuml|sklearn).
    """
    embs = build_embeddings()
    if resource_id not in embs:
        return pd.DataFrame(columns=["resource_id", "name", "resource_type", "borough", "similarity"])

    index = _build_knn_index()
    if index is None:
        return pd.DataFrame()

    meta = _resource_meta
    target_idx = _resource_ids_ordered.index(resource_id)
    target_type = meta.loc[resource_id, "resource_type"] if resource_id in meta.index else ""
    target_boro = meta.loc[resource_id, "borough"] if resource_id in meta.index else ""

    # Query: over-fetch so filtering still returns k
    query_k = min(k * 5 + 10, len(_resource_ids_ordered))

    if index._backend == "cuml":
        query_vec = index._matrix[target_idx:target_idx + 1]
        distances, indices = index.kneighbors(query_vec, n_neighbors=query_k)
        distances = cp.asnumpy(distances)[0]
        indices = cp.asnumpy(indices)[0]
    else:
        query_vec = index._matrix[target_idx:target_idx + 1]
        distances, indices = index.kneighbors(query_vec, n_neighbors=query_k)
        distances = distances[0]
        indices = indices[0]

    # Convert L2 on normalized vectors → cosine sim:  cos = 1 - d^2 / 2
    scores = []
    for idx, dist in zip(indices, distances):
        rid = _resource_ids_ordered[idx]
        if rid == resource_id:
            continue
        if same_type and rid in meta.index and meta.loc[rid, "resource_type"] != target_type:
            continue
        if same_borough and rid in meta.index and meta.loc[rid, "borough"] != target_boro:
            continue
        sim = round(1 - (float(dist) ** 2) / 2, 4)
        scores.append({"resource_id": rid, "similarity": sim, "backend": index._backend})
        if len(scores) >= k:
            break

    result = pd.DataFrame(scores)
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
