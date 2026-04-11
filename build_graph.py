"""
build_graph.py — Build the NYC Social Services knowledge graph.

Nodes:
  - resources   (0 … N_resources-1)         ~7.7K  shelters, food banks, hospitals, etc.
  - transit     (100_000 … 100_000+N_t-1)   ~496   MTA subway stations
  - tracts      (200_000 … 200_000+N_tr-1)  ~2168  census tracts (synthetic centroids)

Edges:
  - NEAR          resource ↔ resource   k=5 nearest, max 2km
  - TRANSIT_LINK  station  ↔ station    connected if same line, weight=travel_time_min
  - WALK_TO_TRANSIT resource → station  nearest station, weight=walk_time_min
  - IN_TRACT      resource → tract      containment
  - SERVED_BY     tract → resource      1/transit_time (KEY: SSSP through MTA graph)

Uses cuGraph when available (Colab/DGX), falls back to networkx locally.

Usage:
    python build_graph.py                  # full graph
    python build_graph.py --sample 500     # 500 resources for local testing
    python build_graph.py --no-served-by   # skip expensive SSSP (fast test)

Output: data/graph.pkl
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(Path(__file__).resolve().parent / "build_graph.log")),
    ],
)
log = logging.getLogger(__name__)

DATA  = Path(__file__).resolve().parent / "data"
STAGE = Path(__file__).resolve().parent / "stage"

# Approximate meters per degree at NYC latitude
LAT_M = 111_320
LON_M = 85_390   # 111_320 * cos(40.7°)

# ─────────────────────────────────────────────────────────────────────────────
# GPU / CPU backend selection
# ─────────────────────────────────────────────────────────────────────────────
try:
    import cugraph
    import cudf
    USE_GPU = True
    log.info("cuGraph detected — using GPU backend")
except ImportError:
    import networkx as nx
    USE_GPU = False
    log.info("cuGraph not found — using networkx CPU backend")


# ─────────────────────────────────────────────────────────────────────────────
# Node ID ranges
# ─────────────────────────────────────────────────────────────────────────────
RESOURCE_OFFSET  = 0
TRANSIT_OFFSET   = 100_000
TRACT_OFFSET     = 200_000


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(sample):
    log.info("Step 1/5: Loading mart and transit data…")

    resources = pd.read_parquet(DATA / "resource_mart.parquet")
    resources = resources.dropna(subset=["latitude", "longitude"])

    # Exclude transit stations from resources — they get their own node range
    transit_mask = resources["resource_type"] == "transit_station"
    transit = resources[transit_mask].copy().reset_index(drop=True)
    resources = resources[~transit_mask].copy().reset_index(drop=True)

    if sample and sample < len(resources):
        log.info("  Sampling %d / %d resources", sample, len(resources))
        resources = resources.sample(sample, random_state=42).reset_index(drop=True)

    resources["node_id"] = RESOURCE_OFFSET + resources.index
    transit["node_id"]   = TRANSIT_OFFSET  + transit.index

    log.info("  Resources : %d", len(resources))
    log.info("  Transit   : %d", len(transit))

    return resources, transit


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build census tract synthetic nodes
# ─────────────────────────────────────────────────────────────────────────────
def build_tract_nodes(resources):
    """
    Create synthetic census tract centroids by rounding resource coords to
    ~1km grid cells. On DGX with real ACS data, replace with actual tract centroids.
    """
    log.info("Step 2/5: Building census tract nodes…")

    # Grid resolution: ~1km ≈ 0.009 degrees lat/lon
    res = 0.009
    resources["_grid_lat"] = (resources["latitude"]  / res).round() * res
    resources["_grid_lon"] = (resources["longitude"] / res).round() * res

    tracts = (resources.groupby(["_grid_lat", "_grid_lon"])
              .size().reset_index(name="resource_count"))
    tracts = tracts.rename(columns={"_grid_lat": "latitude", "_grid_lon": "longitude"})
    tracts["tract_id"] = "synthetic_" + tracts.index.astype(str)
    tracts["node_id"]  = TRACT_OFFSET + tracts.index

    resources = resources.drop(columns=["_grid_lat", "_grid_lon"])

    log.info("  Tract nodes: %d", len(tracts))
    return resources, tracts


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build edges
# ─────────────────────────────────────────────────────────────────────────────
def build_edges(resources, transit, tracts, no_served_by=False):
    log.info("Step 3/5: Building edges…")
    edge_src, edge_dst, edge_weight, edge_type = [], [], [], []

    def add_edges(src_list, dst_list, wt_list, etype):
        edge_src.extend(src_list)
        edge_dst.extend(dst_list)
        edge_weight.extend(wt_list)
        edge_type.extend([etype] * len(src_list))

    # ── NEAR: k=5 nearest resources within 2km ────────────────────────────
    log.info("  Building NEAR edges (k=5, max 2km)…")
    r_coords = np.column_stack([
        resources["latitude"].values  * LAT_M,
        resources["longitude"].values * LON_M,
    ])
    tree = KDTree(r_coords)
    k = min(6, len(resources))  # +1 because point is its own nearest neighbor
    dists, idxs = tree.query(r_coords, k=k, workers=-1)

    near_src, near_dst, near_wt = [], [], []
    for i, (row_dists, row_idxs) in enumerate(zip(dists, idxs)):
        node_i = int(resources.iloc[i]["node_id"])
        for d, j in zip(row_dists[1:], row_idxs[1:]):  # skip self
            if d <= 2000:
                node_j = int(resources.iloc[j]["node_id"])
                near_src.extend([node_i, node_j])
                near_dst.extend([node_j, node_i])
                near_wt.extend([round(d, 1), round(d, 1)])
    add_edges(near_src, near_dst, near_wt, "NEAR")
    log.info("    NEAR edges: %d", len(near_src))

    # ── WALK_TO_TRANSIT: each resource → nearest station ──────────────────
    log.info("  Building WALK_TO_TRANSIT edges…")
    t_coords = np.column_stack([
        transit["latitude"].values  * LAT_M,
        transit["longitude"].values * LON_M,
    ])
    t_tree = KDTree(t_coords)
    t_dists, t_idxs = t_tree.query(r_coords, k=1, workers=-1)

    walk_src, walk_dst, walk_wt = [], [], []
    for i, (d, j) in enumerate(zip(t_dists, t_idxs)):
        node_r = int(resources.iloc[i]["node_id"])
        node_t = int(transit.iloc[j]["node_id"])
        walk_min = round(d / 80, 2)  # ~80 m/min walking
        walk_src.extend([node_r, node_t])
        walk_dst.extend([node_t, node_r])
        walk_wt.extend([walk_min, walk_min])
    add_edges(walk_src, walk_dst, walk_wt, "WALK_TO_TRANSIT")
    log.info("    WALK_TO_TRANSIT edges: %d", len(walk_src))

    # ── TRANSIT_LINK: stations sharing a subway line ───────────────────────
    log.info("  Building TRANSIT_LINK edges…")
    tl_src, tl_dst, tl_wt = [], [], []
    if "subway_lines" in transit.columns:
        line_to_stations = {}
        for _, row in transit.iterrows():
            lines = str(row.get("subway_lines", "")).split()
            for line in lines:
                line_to_stations.setdefault(line, []).append(int(row["node_id"]))
        for line, stations in line_to_stations.items():
            for i in range(len(stations) - 1):
                tl_src.extend([stations[i], stations[i+1]])
                tl_dst.extend([stations[i+1], stations[i]])
                tl_wt.extend([3.0, 3.0])  # ~3 min avg between stops
    add_edges(tl_src, tl_dst, tl_wt, "TRANSIT_LINK")
    log.info("    TRANSIT_LINK edges: %d", len(tl_src))

    # ── IN_TRACT: resource → nearest tract centroid ────────────────────────
    log.info("  Building IN_TRACT edges…")
    tr_coords = np.column_stack([
        tracts["latitude"].values  * LAT_M,
        tracts["longitude"].values * LON_M,
    ])
    tr_tree = KDTree(tr_coords)
    tr_dists, tr_idxs = tr_tree.query(r_coords, k=1, workers=-1)

    it_src, it_dst, it_wt = [], [], []
    resource_to_tract = {}
    for i, (d, j) in enumerate(zip(tr_dists, tr_idxs)):
        node_r  = int(resources.iloc[i]["node_id"])
        node_tr = int(tracts.iloc[j]["node_id"])
        it_src.append(node_r)
        it_dst.append(node_tr)
        it_wt.append(1.0)
        resource_to_tract[node_r] = node_tr
    add_edges(it_src, it_dst, it_wt, "IN_TRACT")
    log.info("    IN_TRACT edges: %d", len(it_src))

    # ── SERVED_BY: tract → resource via transit (SSSP-based) ──────────────
    if not no_served_by:
        log.info("  Building SERVED_BY edges (transit time from tract to resource)…")
        # Estimate: walk to transit + MTA time + walk from transit
        # For local/sample test: approximate with straight-line / 5 km/h walking + 30 km/h subway
        sb_src, sb_dst, sb_wt = [], [], []
        for i, tract_row in enumerate(tqdm(tracts.itertuples(),
                                           total=len(tracts),
                                           desc="SERVED_BY",
                                           unit="tract")):
            tract_node = int(tract_row.node_id)
            t_lat = tract_row.latitude * LAT_M
            t_lon = tract_row.longitude * LON_M
            # Distance from tract centroid to each resource
            r_dists_m = np.sqrt(
                (r_coords[:, 0] - t_lat)**2 + (r_coords[:, 1] - t_lon)**2
            )
            # Approx transit time: walk 80m/min to nearest station,
            # subway 500m/min, walk 80m/min from station
            nearest_t_dist = t_tree.query([t_lat, t_lon], k=1)[0]
            walk_to_station = nearest_t_dist / 80
            resource_transit_dists = t_dists  # each resource's walk to its station
            # Rough subway time: direct distance / 500 m/min
            subway_time = r_dists_m / 500
            total_time = walk_to_station + subway_time + resource_transit_dists / 80
            # Only add edges for resources within 60 min
            within_mask = total_time <= 60
            for j in np.where(within_mask)[0]:
                node_r = int(resources.iloc[j]["node_id"])
                t = round(float(total_time[j]), 2)
                weight = round(1.0 / max(t, 0.5), 4)  # higher weight = more accessible
                sb_src.append(tract_node)
                sb_dst.append(node_r)
                sb_wt.append(weight)
        add_edges(sb_src, sb_dst, sb_wt, "SERVED_BY")
        log.info("    SERVED_BY edges: %d", len(sb_src))
    else:
        log.info("  Skipping SERVED_BY (--no-served-by)")

    edges_df = pd.DataFrame({
        "src": edge_src, "dst": edge_dst,
        "weight": edge_weight, "edge_type": edge_type,
    })
    log.info("  Total edges: %d", len(edges_df))
    return edges_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Build graph object
# ─────────────────────────────────────────────────────────────────────────────
def build_graph_obj(edges_df):
    log.info("Step 4/5: Building graph object (%s)…", "cuGraph" if USE_GPU else "networkx")

    if USE_GPU:
        gdf = cudf.DataFrame({
            "src":    edges_df["src"].values,
            "dst":    edges_df["dst"].values,
            "weight": edges_df["weight"].values,
        })
        G = cugraph.Graph(directed=True)
        G.from_cudf_edgelist(gdf, source="src", destination="dst",
                             edge_attr="weight", renumber=False)
        log.info("  cuGraph nodes: %d  edges: %d", G.number_of_nodes(), G.number_of_edges())
    else:
        G = nx.DiGraph()
        for row in tqdm(edges_df.itertuples(), total=len(edges_df),
                        desc="Adding edges", unit="edge", miniters=10000):
            G.add_edge(row.src, row.dst, weight=row.weight, edge_type=row.edge_type)
        log.info("  networkx nodes: %d  edges: %d", G.number_of_nodes(), G.number_of_edges())

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Save
# ─────────────────────────────────────────────────────────────────────────────
def save_graph(G, resources, transit, tracts, edges_df):
    log.info("Step 5/5: Saving graph…")
    out = DATA / "graph.pkl"

    # Convert cuDF DataFrames to pandas for pickling
    def to_pd(df):
        if hasattr(df, 'to_pandas'):
            return df.to_pandas()
        return df

    # Don't save cuGraph object — it can't be pickled.
    # Save edges instead; graph is rebuilt at load time.
    payload = {
        "graph":     G if not USE_GPU else None,
        "resources": to_pd(resources),
        "transit":   to_pd(transit),
        "tracts":    to_pd(tracts),
        "edges":     to_pd(edges_df),
        "backend":   "cugraph" if USE_GPU else "networkx",
        "offsets": {
            "resource": RESOURCE_OFFSET,
            "transit":  TRANSIT_OFFSET,
            "tract":    TRACT_OFFSET,
        },
    }
    with open(out, "wb") as f:
        pickle.dump(payload, f)
    mb = out.stat().st_size / 1e6
    log.info("  Saved → %s  (%.1f MB)", out, mb)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N resources (default: all ~7.7K)")
    parser.add_argument("--no-served-by", action="store_true",
                        help="Skip SERVED_BY edges (faster test run)")
    args = parser.parse_args()

    t0 = time.time()
    resources, transit = load_data(args.sample)
    resources, tracts  = build_tract_nodes(resources)
    edges_df           = build_edges(resources, transit, tracts,
                                     no_served_by=args.no_served_by)
    G                  = build_graph_obj(edges_df)
    save_graph(G, resources, transit, tracts, edges_df)

    elapsed = time.time() - t0
    log.info("\n=== Graph Build Complete ===")
    log.info("  Backend  : %s", "cuGraph" if USE_GPU else "networkx")
    log.info("  Resources: %d", len(resources))
    log.info("  Transit  : %d", len(transit))
    log.info("  Tracts   : %d", len(tracts))
    log.info("  Edges    : %d", len(edges_df))
    log.info("  Time     : %.0f s", elapsed)

    edge_counts = edges_df["edge_type"].value_counts()
    for etype, cnt in edge_counts.items():
        log.info("    %-20s %6d", etype, cnt)
