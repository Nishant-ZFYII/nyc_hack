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
from pipeline.verify import verify_answer, build_reasoning_path, summarize_reasoning
from pipeline.clarify import get_clarifying_question, merge_query
from pipeline.feedback import parse_feedback
from pipeline.cases import (load_case, create_case, add_visit, mark_resource_visited,
                             resolve_need, get_case_summary, list_cases,
                             choose_resource, checkin, get_failed_resources, get_progress)
from llm.client import get_active_provider
import pandas as pd

# KGE embeddings (optional)
try:
    from engine.embeddings import find_similar, find_similar_to_query
    _HAS_KGE = True
except Exception:
    _HAS_KGE = False

app = FastAPI(title="NYC Social Services Intelligence Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class LocationModel(BaseModel):
    lat: float
    lon: float

class QueryRequest(BaseModel):
    query: str
    demo_mode: bool = False
    case_id: str | None = None
    location: LocationModel | None = None  # user's current location for distance sorting


class FeedbackRequest(BaseModel):
    resource_name: str
    issue: str
    detail: str


class ClarifyAnswerRequest(BaseModel):
    original_query: str
    question: str
    answer: str


class SimilarRequest(BaseModel):
    resource_id: str
    k: int = 5


# ── State ────────────────────────────────────────────────────────────────────
_excluded: list[str] = []
_last_result: dict = {}  # cache last query result for verify/clarify
_last_plan: dict = {}
_last_answer: str = ""
_last_query: str = ""


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

    # Combine global exclusions + case-specific failed resources
    excluded = list(_excluded)
    if req.case_id:
        excluded.extend(get_failed_resources(req.case_id))
    set_excluded_resources(excluded)

    # Build user location dict for executor
    user_loc = None
    if req.location:
        user_loc = {"lat": req.location.lat, "lon": req.location.lon}

    # Plan
    t1 = time.time()
    try:
        plan = generate_plan(req.query)
    except Exception as e:
        raise HTTPException(500, f"Planner error: {e}")
    plan_time = time.time() - t1

    # Inject user location into plan so executor can use it for distance sorting
    if user_loc:
        plan["_user_location"] = user_loc

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
    def _row_to_resource(row, need=None):
        r = {
            "name": str(row.get("name", "")),
            "address": str(row.get("address", "")),
            "type": str(row.get("resource_type", "")),
            "borough": str(row.get("borough", "")),
            "lat": float(row["latitude"]) if pd.notna(row.get("latitude")) else None,
            "lon": float(row["longitude"]) if pd.notna(row.get("longitude")) else None,
            "safety": float(row["safety_score"]) if pd.notna(row.get("safety_score")) else None,
        }
        if need:
            r["need"] = need
        # Add distance if available (from geocode-based sorting)
        if "distance_miles" in row.index and pd.notna(row.get("distance_miles")):
            r["distance_miles"] = round(float(row["distance_miles"]), 2)
            r["walk_min"] = int(row.get("walk_min_est", 0))
        return r

    resources = []
    if result.get("intent") == "lookup":
        df = result.get("results")
        if isinstance(df, pd.DataFrame) and len(df):
            for _, row in df.iterrows():
                resources.append(_row_to_resource(row))
    elif result.get("intent") == "needs_assessment":
        for key, df in result.get("results_by_need", {}).items():
            if isinstance(df, pd.DataFrame) and len(df):
                for _, row in df.iterrows():
                    resources.append(_row_to_resource(row, need=key))
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

    # Verification (skip in demo mode)
    verification = None
    verify_time = 0
    clarify_question = ""
    if not req.demo_mode:
        t4 = time.time()
        try:
            verification = verify_answer(response, result)
        except Exception:
            verification = None
        verify_time = time.time() - t4

    # Clarification question
    try:
        clarify_question = get_clarifying_question(req.query, response, 0)
    except Exception:
        clarify_question = ""

    # Reasoning path
    reasoning = []
    reasoning_summary = ""
    try:
        reasoning = build_reasoning_path(plan, result)
        reasoning_summary = summarize_reasoning(reasoning, plan, result)
    except Exception:
        pass

    # Cache for follow-up endpoints
    global _last_result, _last_plan, _last_answer, _last_query
    _last_result = result
    _last_plan = plan
    _last_answer = response
    _last_query = req.query

    # Auto-save to case if case_id provided
    if req.case_id:
        try:
            add_visit(req.case_id, req.query, response, resources,
                      location=None, plan=plan)
        except Exception:
            pass

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
            "verify": round(verify_time, 1),
        },
        "llm": get_active_provider(),
        "verification": verification,
        "clarify_question": clarify_question,
        "reasoning": reasoning,
        "reasoning_summary": reasoning_summary,
    }


@app.post("/api/feedback")
async def feedback(req: FeedbackRequest):
    if req.resource_name and req.resource_name != "unknown":
        _excluded.append(req.resource_name)
    return {"excluded": _excluded, "count": len(_excluded)}


@app.post("/api/clarify")
async def clarify(req: ClarifyAnswerRequest):
    """User answered a clarifying question — merge and re-run pipeline."""
    enriched = merge_query(req.original_query, req.question, req.answer)
    # Re-run with enriched query
    new_req = QueryRequest(query=enriched, demo_mode=False)
    return await query(new_req)


@app.post("/api/similar")
async def similar(req: SimilarRequest):
    """Get KGE-similar resources."""
    if not _HAS_KGE:
        return {"similar": [], "error": "KGE embeddings not loaded"}
    try:
        df = find_similar(req.resource_id, k=req.k)
        if isinstance(df, pd.DataFrame) and len(df):
            cols = [c for c in ["name", "resource_type", "borough", "address", "similarity"]
                    if c in df.columns]
            return {"similar": df[cols].to_dict("records")}
        return {"similar": []}
    except Exception as e:
        return {"similar": [], "error": str(e)}


@app.get("/api/geocode")
async def geocode(q: str):
    """Proxy geocoding to avoid CORS issues with Nominatim."""
    import requests
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{q}, NYC", "format": "json", "limit": "1", "countrycodes": "us"},
            headers={"User-Agent": "NYC-SocialServices-Engine/1.0"},
            timeout=5,
        )
        return resp.json()
    except Exception as e:
        return []


class DirectionsRequest(BaseModel):
    from_lat: float
    from_lon: float
    to_lat: float
    to_lon: float
    budget: float | None = None  # None = show all, 0 = walk only


@app.post("/api/directions")
async def directions(req: DirectionsRequest):
    """Get multi-modal directions from user location to a resource."""
    try:
        from pipeline.routing import get_directions
        result = get_directions(req.from_lat, req.from_lon, req.to_lat, req.to_lon, req.budget)
        return result
    except Exception as e:
        return {"error": str(e), "options": []}


# ── Case Management ──────────────────────────────────────────────────────────

class CaseLoginRequest(BaseModel):
    case_id: str
    name: str = ""


class CaseVisitedRequest(BaseModel):
    case_id: str
    resource_name: str
    feedback: str | None = None


class CaseResolveRequest(BaseModel):
    case_id: str
    category: str


@app.post("/api/case/login")
async def case_login(req: CaseLoginRequest):
    """Login or create a case. Returns summary if returning user."""
    case = load_case(req.case_id)
    if case:
        summary = get_case_summary(req.case_id)
        return {
            "case": case,
            "summary": summary,
            "returning": True,
        }
    else:
        case = create_case(req.case_id, name=req.name)
        return {
            "case": case,
            "summary": f"Welcome, {req.name or req.case_id}. Tell me what's going on and I'll find help for you.",
            "returning": False,
        }


@app.post("/api/case/save")
async def case_save(case_id: str, query: str, answer_text: str,
                    resources: list = [], location: dict = None, plan: dict = None):
    """Save current interaction to a case."""
    case = add_visit(case_id, query, answer_text, resources, location, plan)
    return {"case_id": case["case_id"], "visits": len(case["visits"])}


@app.post("/api/case/visited")
async def case_visited(req: CaseVisitedRequest):
    """Mark that user visited a resource (with optional feedback)."""
    case = mark_resource_visited(req.case_id, req.resource_name, req.feedback)
    return case


@app.post("/api/case/resolve")
async def case_resolve(req: CaseResolveRequest):
    """Mark a need as resolved."""
    case = resolve_need(req.case_id, req.category)
    return case


class ChooseResourceRequest(BaseModel):
    case_id: str
    need_category: str
    resource_name: str
    resource_address: str = ""
    resource_type: str = ""


class CheckinRequest(BaseModel):
    case_id: str
    arrived: bool
    resource_name: str = ""
    feedback: str = ""
    location: LocationModel | None = None


@app.post("/api/case/choose")
async def case_choose(req: ChooseResourceRequest):
    """User selects a resource for a specific need."""
    case = choose_resource(req.case_id, req.need_category, req.resource_name,
                           req.resource_address, req.resource_type)
    return {"case": case, "message": f"Got it — heading to {req.resource_name} for {req.need_category}."}


@app.post("/api/case/checkin")
async def case_checkin(req: CheckinRequest):
    """User confirms arrival (or not) at a resource."""
    loc = {"lat": req.location.lat, "lon": req.location.lon} if req.location else None
    case = checkin(req.case_id, req.arrived, req.resource_name, req.feedback, loc)

    if req.arrived:
        progress = get_progress(req.case_id)
        open_needs = [n for n in progress["needs"] if n["status"] == "open"]
        if open_needs:
            next_need = open_needs[0]["category"]
            msg = (f"Great, glad you made it to {req.resource_name}! "
                   f"You still have {len(open_needs)} open need(s). "
                   f"Next up: {next_need}. Want to find resources for that?")
        else:
            msg = f"Wonderful! You've addressed all your needs. Come back anytime if you need more help."
        return {"case": case, "message": msg, "progress": progress}
    else:
        # Not arrived — resource might be full
        failed = get_failed_resources(req.case_id)
        msg = (f"I'm sorry {req.resource_name} didn't work out. "
               f"Let me find alternatives (excluding {len(failed)} resource(s) you've already tried).")
        return {"case": case, "message": msg, "failed_resources": failed}


@app.get("/api/case/progress/{case_id}")
async def case_progress(case_id: str):
    """Get structured progress report for a case."""
    return get_progress(case_id)


@app.get("/api/cases")
async def cases_list():
    """List all cases (admin view)."""
    return list_cases()


@app.get("/api/resources")
async def all_resources():
    """Get all resources with coordinates for initial map load."""
    mart, _ = load_state()
    mart_pd = mart.to_pandas() if hasattr(mart, 'to_pandas') else mart
    cols = ["resource_id", "resource_type", "name", "address", "borough",
            "latitude", "longitude", "safety_score"]
    df = mart_pd[[c for c in cols if c in mart_pd.columns]].dropna(subset=["latitude", "longitude"])
    return df.head(2000).to_dict("records")
