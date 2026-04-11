"""
server.py — FastAPI backend for NYC Social Services Intelligence Engine.

Serves the custom HTML frontend + REST API for the pipeline.

Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import sys
import time
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.planner import generate_plan
from pipeline.executor import execute, load_state, set_excluded_resources
from pipeline.synth import answer
from llm.client import get_active_provider
import pandas as pd

app = FastAPI(title="NYC Social Services Intelligence Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    query: str
    demo_mode: bool = False


class FeedbackRequest(BaseModel):
    resource_name: str
    issue: str
    detail: str


# ── State ────────────────────────────────────────────────────────────────────
_excluded: list[str] = []


@app.get("/")
async def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/status")
async def status():
    try:
        mart, payload = load_state()
        mart_pd = mart.to_pandas() if hasattr(mart, 'to_pandas') else mart
        edges = payload.get("edges")
        edge_count = len(edges) if edges is not None else 0
        resource_types = mart_pd["resource_type"].value_counts().to_dict()
        return {
            "llm": get_active_provider(),
            "resources": len(mart_pd),
            "edges": edge_count,
            "backend": payload.get("backend", "networkx"),
            "resource_types": resource_types,
            "status": "online",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/query")
async def query(req: QueryRequest):
    t0 = time.time()
    set_excluded_resources(_excluded)

    # Plan
    t1 = time.time()
    try:
        plan = generate_plan(req.query)
    except Exception as e:
        raise HTTPException(500, f"Planner error: {e}")
    plan_time = time.time() - t1

    # Execute
    t2 = time.time()
    try:
        result = execute(plan)
    except Exception as e:
        raise HTTPException(500, f"Executor error: {e}")
    exec_time = time.time() - t2

    # Synthesize
    t3 = time.time()
    try:
        response = answer(req.query, plan, result)
    except Exception as e:
        response = f"Error: {e}"
    synth_time = time.time() - t3

    # Extract resources for map
    resources = []
    if result.get("intent") == "lookup":
        df = result.get("results")
        if isinstance(df, pd.DataFrame) and len(df):
            for _, row in df.iterrows():
                resources.append({
                    "name": str(row.get("name", "")),
                    "address": str(row.get("address", "")),
                    "type": str(row.get("resource_type", "")),
                    "borough": str(row.get("borough", "")),
                    "lat": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
                    "lon": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
                    "safety": float(row["safety_score"]) if pd.notna(row.get("safety_score")) else None,
                })
    elif result.get("intent") == "needs_assessment":
        for key, df in result.get("results_by_need", {}).items():
            if isinstance(df, pd.DataFrame) and len(df):
                for _, row in df.iterrows():
                    resources.append({
                        "name": str(row.get("name", "")),
                        "address": str(row.get("address", "")),
                        "type": str(row.get("resource_type", "")),
                        "borough": str(row.get("borough", "")),
                        "lat": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
                        "lon": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
                        "safety": float(row["safety_score"]) if pd.notna(row.get("safety_score")) else None,
                        "need": key,
                    })
    elif result.get("intent") == "simulate":
        for s in result.get("available_shelters", []):
            if s.get("latitude") and s.get("longitude"):
                resources.append({
                    "name": str(s.get("name", "")),
                    "address": str(s.get("address", "")),
                    "type": "shelter",
                    "borough": str(s.get("borough", "")),
                    "lat": float(s["latitude"]),
                    "lon": float(s["longitude"]),
                })

    total = time.time() - t0
    return {
        "answer": response,
        "plan": plan,
        "resources": resources,
        "timing": {
            "total": round(total, 1),
            "plan": round(plan_time, 1),
            "execute": round(exec_time, 2),
            "synth": round(synth_time, 1),
        },
        "llm": get_active_provider(),
    }


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    if req.resource_name and req.resource_name != "unknown":
        _excluded.append(req.resource_name)
    return {"excluded": _excluded, "count": len(_excluded)}


@app.get("/api/resources")
async def all_resources():
    """Get all resources with coordinates for initial map load."""
    mart, _ = load_state()
    mart_pd = mart.to_pandas() if hasattr(mart, 'to_pandas') else mart
    cols = ["resource_id", "resource_type", "name", "address", "borough",
            "latitude", "longitude", "safety_score"]
    df = mart_pd[[c for c in cols if c in mart_pd.columns]].dropna(subset=["latitude", "longitude"])
    return df.head(2000).to_dict("records")
