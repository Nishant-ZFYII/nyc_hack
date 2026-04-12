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
from pipeline.eligibility import calculate_eligibility, get_rights, get_stories
from pipeline.agent import run_autonomous_agent, generate_plan_pdf
from guardrails import check_safety, check_safety_async

# nat (NeMo Agent Toolkit) — lazy so server still starts if nat missing
try:
    import agent.register  # registers our 4 tool groups
    from nat.runtime.loader import load_workflow as _nat_load_workflow
    _NAT_AVAILABLE = True
    _NAT_CONFIG = str(Path(__file__).parent / "agent" / "config.yml")
except Exception as _nat_err:
    _NAT_AVAILABLE = False
    _NAT_IMPORT_ERROR = str(_nat_err)
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

    # ── Guardrails: two-layer safety check (regex + NeMo Guardrails) ─────────
    guard = await check_safety_async(req.query, use_llm_fallback=True)
    if not guard["allow"]:
        checked_by = guard.get("checked_by", "regex")
        return {
            "answer": guard["replacement_response"],
            "plan": {"intent": "safety_block", "_reason": guard["reason"]},
            "resources": [],
            "timing": {"total": round(time.time() - t0, 2), "plan": 0, "execute": 0, "synth": 0, "verify": 0},
            "llm": get_active_provider(),
            "verification": {"verified": True, "confidence": "SAFETY_BLOCK"},
            "clarify_question": "",
            "reasoning": [{
                "hop": 1,
                "fact": f"Safety guardrail triggered: {guard['reason']} (via {checked_by})",
                "confidence": 1.0,
                "source": "NeMo_Guardrails" if checked_by == "nemo_guardrails" else "regex_prefilter",
                "cumulative": 1.0,
            }],
            "reasoning_summary": f"Safety check: {guard['reason']} (checked by {checked_by}).",
            "safety_block": True,
            "crisis_type": guard.get("crisis_type"),
            "guardrail_layer": checked_by,
        }

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


# NYC neighborhood fallback lookup (used when Nominatim is unavailable)
_NYC_NEIGHBORHOODS = {
    "manhattan": (40.7831, -73.9712),
    "midtown": (40.7549, -73.9840),
    "midtown manhattan": (40.7549, -73.9840),
    "times square": (40.7580, -73.9855),
    "harlem": (40.8116, -73.9465),
    "upper east side": (40.7736, -73.9566),
    "upper west side": (40.7870, -73.9754),
    "lower east side": (40.7150, -73.9843),
    "east village": (40.7265, -73.9815),
    "west village": (40.7352, -74.0031),
    "soho": (40.7233, -74.0020),
    "tribeca": (40.7163, -74.0086),
    "chelsea": (40.7465, -74.0014),
    "washington heights": (40.8417, -73.9393),
    "financial district": (40.7074, -74.0113),

    "brooklyn": (40.6782, -73.9442),
    "flatbush": (40.6415, -73.9590),
    "flatbush brooklyn": (40.6415, -73.9590),
    "williamsburg": (40.7081, -73.9571),
    "park slope": (40.6710, -73.9814),
    "bed-stuy": (40.6872, -73.9418),
    "bedford stuyvesant": (40.6872, -73.9418),
    "crown heights": (40.6697, -73.9446),
    "bushwick": (40.6944, -73.9213),
    "bay ridge": (40.6266, -74.0330),
    "coney island": (40.5755, -73.9707),
    "downtown brooklyn": (40.6922, -73.9877),
    "boerum hill": (40.6864, -73.9858),
    "prospect heights": (40.6773, -73.9699),

    "queens": (40.7282, -73.7949),
    "astoria": (40.7643, -73.9236),
    "long island city": (40.7447, -73.9485),
    "jackson heights": (40.7557, -73.8831),
    "flushing": (40.7680, -73.8333),
    "jamaica": (40.7019, -73.7890),
    "forest hills": (40.7157, -73.8448),
    "elmhurst": (40.7365, -73.8821),

    "bronx": (40.8448, -73.8648),
    "the bronx": (40.8448, -73.8648),
    "south bronx": (40.8262, -73.9064),
    "riverdale": (40.8976, -73.9064),
    "fordham": (40.8621, -73.8976),
    "mott haven": (40.8107, -73.9244),

    "staten island": (40.5795, -74.1502),
    "st george": (40.6437, -74.0735),
    "tottenville": (40.5108, -74.2489),
    "new dorp": (40.5710, -74.1181),

    "nyc": (40.7128, -74.0060),
    "new york city": (40.7128, -74.0060),
    "new york": (40.7128, -74.0060),
}


@app.get("/api/geocode")
async def geocode(q: str):
    """Geocode an address — try Nominatim first, fallback to NYC neighborhood table."""
    import requests
    # Try Nominatim (external API)
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{q}, NYC", "format": "json", "limit": "1", "countrycodes": "us"},
            headers={"User-Agent": "NYC-SocialServices-Engine/1.0"},
            timeout=3,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return data
    except Exception:
        pass

    # Fallback: NYC neighborhood lookup
    q_lower = q.strip().lower()
    # Remove common suffixes
    for suffix in [", nyc", ", ny", ", new york", ", brooklyn", ", manhattan", ", queens", ", bronx"]:
        q_lower = q_lower.replace(suffix, "").strip()

    if q_lower in _NYC_NEIGHBORHOODS:
        lat, lon = _NYC_NEIGHBORHOODS[q_lower]
        return [{"lat": str(lat), "lon": str(lon), "display_name": f"{q.strip()}, New York, NY"}]

    # Partial match
    for key, (lat, lon) in _NYC_NEIGHBORHOODS.items():
        if key in q_lower or q_lower in key:
            return [{"lat": str(lat), "lon": str(lon), "display_name": f"{key.title()}, New York, NY"}]

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


# ── Eligibility / Rights / Stories ────────────────────────────────────────────

class EligibilityRequest(BaseModel):
    household_size: int = 1
    annual_income: float = 0
    has_children: bool = False
    has_pregnant: bool = False
    has_disabled: bool = False
    has_senior: bool = False
    is_veteran: bool = False
    housing_status: str = ""  # homeless, at_risk, stable
    has_id: bool = True
    immigration_status: str = "any"


@app.post("/api/eligibility")
async def eligibility(req: EligibilityRequest):
    """Calculate benefits eligibility for a household profile.

    Returns which programs they qualify for (SNAP, Medicaid, WIC, Cash Assistance,
    Fair Fares, emergency shelter, etc.) with estimated monthly amounts and
    documents needed.
    """
    return calculate_eligibility(
        household_size=req.household_size,
        annual_income=req.annual_income,
        has_children=req.has_children,
        has_pregnant=req.has_pregnant,
        has_disabled=req.has_disabled,
        has_senior=req.has_senior,
        is_veteran=req.is_veteran,
        housing_status=req.housing_status,
        has_id=req.has_id,
        immigration_status=req.immigration_status,
    )


@app.get("/api/rights")
async def rights(resource_type: str = "default"):
    """Get know-your-rights info for a resource type.

    Returns legal rights a person has at that type of resource
    (shelter, food bank, hospital, school, benefits center, etc.)
    """
    return {"resource_type": resource_type, "rights": get_rights(resource_type)}


@app.get("/api/stories")
async def stories(need: str = None, k: int = 3):
    """Get success stories, optionally filtered by need category.

    Returns 2-3 anonymized journeys of people who found help in similar
    situations. Used to build trust with new users.
    """
    return {"need": need, "stories": get_stories(need, k)}


# ── Refine: progressive disclosure checkbox handler ───────────────────────────

class RefineRequest(BaseModel):
    original_query: str
    case_id: str | None = None
    location: LocationModel | None = None
    # Checkbox profile
    has_id: bool = True
    has_children: bool = False
    has_pregnant: bool = False
    has_disabled: bool = False
    has_senior: bool = False
    has_insurance: bool = True
    is_veteran: bool = False
    is_undocumented: bool = False
    household_size: int = 1
    annual_income: float = 0
    # Optional free-text additional info
    additional_info: str = ""


@app.post("/api/refine")
async def refine(req: RefineRequest):
    """
    Refine query results with a checkbox profile.

    Each checkbox maps to a deterministic skill call:
      - has_id=False  → includes 'no ID needed' rights
      - has_pregnant  → adds WIC + emergency Medicaid
      - has_senior    → adds SCRIE + senior services
      - has_disabled  → adds disability benefits
    Free-text additional_info is appended to the query and goes through
    the normal LLM + guardrails pipeline.
    """
    t0 = time.time()

    # Build an enriched query
    situation_bits = [req.original_query]
    if not req.has_id:
        situation_bits.append("(no ID)")
    if req.has_children:
        situation_bits.append("(with children)")
    if req.has_pregnant:
        situation_bits.append("(pregnant)")
    if req.has_disabled:
        situation_bits.append("(has disability)")
    if req.has_senior:
        situation_bits.append("(senior 65+)")
    if not req.has_insurance:
        situation_bits.append("(no insurance)")
    if req.is_veteran:
        situation_bits.append("(veteran)")
    if req.is_undocumented:
        situation_bits.append("(undocumented)")
    if req.additional_info:
        situation_bits.append(req.additional_info)

    enriched_query = " ".join(situation_bits)

    # Calculate eligibility from the profile
    eligibility = calculate_eligibility(
        household_size=req.household_size,
        annual_income=req.annual_income,
        has_children=req.has_children,
        has_pregnant=req.has_pregnant,
        has_disabled=req.has_disabled,
        has_senior=req.has_senior,
        is_veteran=req.is_veteran,
        has_id=req.has_id,
        immigration_status="undocumented" if req.is_undocumented else "any",
    )

    # Build relevant rights based on profile
    applicable_rights = []
    if not req.has_id:
        applicable_rights.append({
            "right": "Apply without ID",
            "detail": "You can apply for benefits without ID using a 'Request for Proof' form at HRA. Staff will help you get documents.",
        })
    if req.is_undocumented:
        applicable_rights.append({
            "right": "Immigration status protected",
            "detail": "NYC agencies cannot share your information with ICE. You can use shelter, food, and healthcare regardless of status.",
        })
    if not req.has_insurance:
        applicable_rights.append({
            "right": "EMTALA — emergency care",
            "detail": "Hospitals must treat emergencies regardless of insurance. NYC Health + Hospitals offers sliding-scale care.",
        })
    if req.has_children:
        applicable_rights.append({
            "right": "School without address",
            "detail": "Children can enroll in NYC public school without a permanent address (McKinney-Vento Act). Free meals included.",
        })

    # Run the normal query pipeline with enriched query
    user_loc = None
    if req.location:
        user_loc = {"lat": req.location.lat, "lon": req.location.lon}

    # Optional safety check on additional_info if provided
    if req.additional_info:
        guard = await check_safety_async(req.additional_info, use_llm_fallback=False)
        if not guard["allow"]:
            return {
                "safety_block": True,
                "replacement_response": guard["replacement_response"],
                "reason": guard["reason"],
            }

    plan = generate_plan(enriched_query)
    if user_loc:
        plan["_user_location"] = user_loc

    # Exclude failed resources for this case
    if req.case_id:
        set_excluded_resources(get_failed_resources(req.case_id))

    result = execute(plan)

    # Extract resources
    resources = []
    if result.get("intent") == "lookup":
        df = result.get("results")
        if isinstance(df, pd.DataFrame) and len(df):
            resources = df.to_dict("records")
    elif result.get("intent") == "needs_assessment":
        for key, df in result.get("results_by_need", {}).items():
            if isinstance(df, pd.DataFrame) and len(df):
                for rec in df.to_dict("records"):
                    rec["need"] = key
                    resources.append(rec)

    # Clean NaN
    import math
    def _clean(obj):
        if isinstance(obj, float):
            return None if math.isnan(obj) or math.isinf(obj) else obj
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        return obj

    return _clean({
        "enriched_query": enriched_query,
        "profile": {
            "household_size": req.household_size,
            "annual_income": req.annual_income,
            "has_id": req.has_id,
            "has_children": req.has_children,
            "has_pregnant": req.has_pregnant,
            "has_disabled": req.has_disabled,
            "has_senior": req.has_senior,
            "has_insurance": req.has_insurance,
            "is_veteran": req.is_veteran,
            "is_undocumented": req.is_undocumented,
        },
        "eligibility": {
            "qualifying_programs": eligibility["qualifying_programs"],
            "monthly_benefits": eligibility["estimated_monthly_benefits"],
            "programs": eligibility["programs"],
        },
        "rights": applicable_rights,
        "resources": resources[:10],
        "timing": {"total": round(time.time() - t0, 2)},
    })


# ── Autonomous Agent ──────────────────────────────────────────────────────────

class AgentPlanRequest(BaseModel):
    query: str
    location: LocationModel | None = None
    case_id: str | None = None


@app.post("/api/agent/plan")
async def agent_plan(req: AgentPlanRequest):
    """
    Run the autonomous agent. Given a query, the agent chains multiple
    skill calls (planner → executor → eligibility → directions → stories)
    and returns a complete action plan.

    Returns:
      - summary: human-readable overview
      - steps: ordered step-by-step plan with resources, directions, rights
      - eligibility: qualifying benefits programs + monthly estimates
      - stories: relevant success stories
      - trace: log of skill calls the agent made (for judges/debugging)
    """
    loc = None
    if req.location:
        loc = {"lat": req.location.lat, "lon": req.location.lon}

    try:
        result = run_autonomous_agent(
            query=req.query,
            location=loc,
            case_id=req.case_id,
        )
        return result
    except Exception as e:
        raise HTTPException(500, f"Agent error: {e}")


@app.post("/api/agent/openclaw")
async def agent_openclaw(req: AgentPlanRequest):
    """
    Run the query through the actual OpenClaw agent framework.

    This invokes `openclaw agent --local --json --message "..."` as a
    subprocess. The OpenClaw agent runs with our nyc-caseworker skill
    registered, uses Nemotron-3-Nano via Ollama, and returns a structured
    response.

    This proves OpenClaw is actually running (not just config files).
    """
    import subprocess, json as _json

    t0 = time.time()

    # Guardrails check first
    guard = await check_safety_async(req.query, use_llm_fallback=False)
    if not guard["allow"]:
        return {
            "answer": guard["replacement_response"],
            "safety_block": True,
            "reason": guard["reason"],
            "via": "guardrails (pre-openclaw)",
        }

    # Build message with location context if provided
    message = req.query
    if req.location:
        message += f" (location: {req.location.lat:.4f}, {req.location.lon:.4f})"

    try:
        # Run: openclaw agent --agent main --local --json --message "..."
        # Merge stderr into stdout because OpenClaw may write JSON to stderr
        result = subprocess.run(
            ["openclaw", "agent", "--agent", "main", "--local", "--json",
             "--message", message, "--timeout", "60"],
            capture_output=True, text=True, timeout=90,
        )

        # Combine both streams — the JSON block is somewhere in them
        out = (result.stdout or "") + "\n" + (result.stderr or "")

        if result.returncode != 0 and "{" not in out:
            return {
                "error": "OpenClaw agent failed",
                "stderr": (result.stderr or "")[:500],
                "returncode": result.returncode,
                "via": "openclaw-subprocess",
            }
        # Look for the JSON block containing "payloads" (the actual response)
        payloads_idx = out.find('"payloads"')
        if payloads_idx == -1:
            return {"error": "No payloads in openclaw output",
                    "raw": out[:800], "via": "openclaw-subprocess"}
        start = out.rfind("{", 0, payloads_idx)
        if start == -1:
            return {"error": "No opening brace before payloads",
                    "via": "openclaw-subprocess"}

        # Find the matching closing brace (balance tracking)
        depth = 0
        end = -1
        in_string = False
        escape = False
        for i, ch in enumerate(out[start:], start):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            return {"error": "Unterminated JSON in openclaw output",
                    "raw": out[start:start+500], "via": "openclaw-subprocess"}

        data = _json.loads(out[start:end])
        total_time = time.time() - t0

        return {
            "answer": data.get("payloads", [{}])[0].get("text", ""),
            "session_id": data.get("meta", {}).get("agentMeta", {}).get("sessionId", ""),
            "model": data.get("meta", {}).get("agentMeta", {}).get("model", ""),
            "provider": data.get("meta", {}).get("agentMeta", {}).get("provider", ""),
            "tokens": data.get("meta", {}).get("agentMeta", {}).get("usage", {}),
            "duration_ms": data.get("meta", {}).get("durationMs", 0),
            "total_time_s": round(total_time, 2),
            "via": "openclaw",
            "skill_loaded": "nyc-caseworker",
        }

    except subprocess.TimeoutExpired:
        return {"error": "OpenClaw agent timed out after 90s", "via": "openclaw-subprocess"}
    except FileNotFoundError:
        return {"error": "openclaw CLI not found — is it installed?", "via": "openclaw-subprocess"}
    except Exception as e:
        return {"error": f"OpenClaw error: {e}", "via": "openclaw-subprocess"}


@app.post("/api/agent/nat")
async def agent_nat(req: AgentPlanRequest):
    """
    Run the query through the NeMo Agent Toolkit (`nat`) ReAct agent.

    The agent is configured in agent/config.yml with 4 tool groups
    (resources, eligibility, directions, cases) registered in agent/register.py.
    Nemotron-3-Nano drives the ReAct loop via Ollama.

    Proves nat is actually running, not just instruction-based.
    """
    if not _NAT_AVAILABLE:
        raise HTTPException(500, f"nat not available: {_NAT_IMPORT_ERROR}")

    t0 = time.time()

    # Guardrails first
    guard = await check_safety_async(req.query, use_llm_fallback=False)
    if not guard["allow"]:
        return {
            "answer": guard["replacement_response"],
            "safety_block": True,
            "reason": guard["reason"],
            "via": "guardrails (pre-nat)",
        }

    message = req.query
    if req.location:
        message += f" (user location: lat={req.location.lat:.4f}, lon={req.location.lon:.4f})"
    if req.case_id:
        message += f" (case_id: {req.case_id})"

    try:
        # Start per-request trace (contextvar — no cross-request leakage)
        from agent.register import start_trace
        trace = start_trace()

        async with _nat_load_workflow(_NAT_CONFIG) as workflow:
            async with workflow.run(message) as runner:
                if hasattr(runner, "result"):
                    r = runner.result
                    result = await r() if callable(r) else r
                elif hasattr(runner, "get_result"):
                    result = await runner.get_result()
                else:
                    result = runner

        return {
            "answer": str(result),
            "trace": trace,  # list of {tool, inputs, output_preview, duration_ms}
            "tool_call_count": len(trace),
            "total_time_s": round(time.time() - t0, 2),
            "via": "nat-react-agent",
            "model": "nemotron-3-nano",
            "framework": "NVIDIA NeMo Agent Toolkit",
        }
    except Exception as e:
        return {
            "error": f"nat agent error: {e}",
            "trace": trace if 'trace' in locals() else [],
            "tool_call_count": len(trace) if 'trace' in locals() else 0,
            "total_time_s": round(time.time() - t0, 2),
            "via": "nat-react-agent",
        }


@app.post("/api/agent/pdf")
async def agent_pdf(req: AgentPlanRequest):
    """
    Run the autonomous agent and return the action plan as HTML
    (suitable for printing/saving as PDF via browser Print-to-PDF).
    """
    loc = None
    if req.location:
        loc = {"lat": req.location.lat, "lon": req.location.lon}

    try:
        result = run_autonomous_agent(query=req.query, location=loc, case_id=req.case_id)
        html = generate_plan_pdf(result)
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(500, f"PDF generation error: {e}")



@app.get("/api/resources")
async def all_resources():
    """Get all resources with coordinates for initial map load."""
    mart, _ = load_state()
    mart_pd = mart.to_pandas() if hasattr(mart, 'to_pandas') else mart
    cols = ["resource_id", "resource_type", "name", "address", "borough",
            "latitude", "longitude", "safety_score"]
    df = mart_pd[[c for c in cols if c in mart_pd.columns]].dropna(subset=["latitude", "longitude"])
    return df.head(2000).to_dict("records")
