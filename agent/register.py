"""
agent/register.py — Register NYC Caseworker tool groups with NeMo Agent Toolkit.

Exposes our pipeline functions as tools the ReAct agent can invoke.
Each group follows Colin's GridWatch pattern:
  - FunctionGroupBaseConfig subclass with `name=`
  - @register_function_group(config_type=...) async generator
  - Inner async functions with docstrings (used by LLM for tool selection)
  - group.add_function(name=, fn=, description=)
"""
from __future__ import annotations

import json
import sys
import time
import functools
import inspect
from collections.abc import AsyncGenerator
from pathlib import Path

# Module-level trace. Each request resets it via start_trace() before
# invoking the workflow; tool calls append here. contextvars didn't propagate
# through langgraph's task boundaries, so we use a module-level list —
# safe because we only handle one agent request at a time per server instance.
_current_trace: list = []


def start_trace() -> list:
    """Reset and return the module-level trace list for this request."""
    global _current_trace
    _current_trace = []
    return _current_trace


def _record_call(tool_name: str, inputs: dict, output: str, duration_ms: float):
    out_preview = output if len(output) <= 800 else output[:800] + "…"
    _current_trace.append({
        "tool": tool_name,
        "inputs": inputs,
        "output_preview": out_preview,
        "duration_ms": round(duration_ms, 1),
    })


def _traced(name: str, fn):
    """Wrap an async tool fn to record (name, inputs, output, duration) in the trace.

    Preserves the wrapped function's signature so nat's pydantic schema
    introspection sees the real parameters (not generic *args/**kwargs).
    """
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            result = await fn(*args, **kwargs)
            _record_call(name, kwargs or {"_args": list(args)}, str(result),
                         (time.time() - t0) * 1000)
            return result
        except Exception as e:
            _record_call(name, kwargs or {"_args": list(args)}, f"ERROR: {e}",
                         (time.time() - t0) * 1000)
            raise
    # Copy signature explicitly — functools.wraps alone doesn't do this reliably
    # for pydantic/langchain introspection.
    wrapper.__signature__ = inspect.signature(fn)
    wrapper.__annotations__ = dict(getattr(fn, "__annotations__", {}))
    return wrapper

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

# Add project root to path so pipeline imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.executor import execute, load_state, set_excluded_resources
from pipeline.planner import generate_plan
from pipeline.routing import get_directions as pipeline_get_directions
from pipeline.eligibility import calculate_eligibility as pipeline_calc_elig, get_rights as pipeline_get_rights, get_stories as pipeline_get_stories
from pipeline.cases import (
    load_case, list_cases, get_case_summary, get_progress, get_failed_resources,
    choose_resource, checkin, add_destination_intent,
    update_need_status, update_destination_state, save_admin_notes,
)
try:
    from pipeline.briefing import generate_briefing, _estimate_urgency
    _BRIEFING_AVAILABLE = True
except Exception:
    _BRIEFING_AVAILABLE = False

import pandas as pd


def _df_to_records(df, limit=5):
    if isinstance(df, pd.DataFrame) and len(df):
        return df.head(limit).to_dict("records")
    return []


def _clean(obj):
    """Recursively replace NaN/Inf with None so json.dumps works."""
    import math
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    return obj


# ── Tool Group 1: Resource Finding ───────────────────────────────────────────

class ResourceToolConfig(FunctionGroupBaseConfig, name="nyc_resource_tools"):
    include: list[str] = Field(
        default_factory=lambda: [
            "find_resources", "find_resources_by_type",
        ],
        description="NYC social services resource search tools",
    )


@register_function_group(config_type=ResourceToolConfig)
async def nyc_resource_tools(_config, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    group = FunctionGroup(config=_config)

    async def _find_resources(query: str, lat: float = 0, lon: float = 0,
                              case_id: str = "") -> str:
        """Find NYC social services resources matching a query. ALWAYS call this before giving resource recommendations. Query examples: 'shelter in Brooklyn', 'food pantry Manhattan', 'hospital near Flatbush'. Pass lat/lon if known for distance sorting. Pass case_id to exclude resources the user previously marked as full."""
        plan = generate_plan(query)
        if lat and lon:
            plan["_user_location"] = {"lat": lat, "lon": lon}
        if case_id:
            set_excluded_resources(get_failed_resources(case_id))
        result = execute(plan)

        resources = []
        if result.get("intent") == "lookup":
            resources = _df_to_records(result.get("results"), 5)
        elif result.get("intent") == "needs_assessment":
            for key, df in result.get("results_by_need", {}).items():
                for rec in _df_to_records(df, 3):
                    rec["need"] = key
                    resources.append(rec)

        # Simplify for LLM context — drop verbose fields
        simple = []
        for r in resources[:8]:
            simple.append({
                "name": r.get("name", ""),
                "address": r.get("address", ""),
                "type": r.get("resource_type", ""),
                "borough": r.get("borough", ""),
                "distance_miles": r.get("distance_miles"),
                "walk_min": r.get("walk_min_est"),
            })
        return json.dumps(_clean({"intent": result.get("intent"), "resources": simple}),
                          indent=2, default=str)

    async def _find_resources_by_type(resource_type: str, borough: str = "",
                                       limit: int = 5) -> str:
        """Find NYC resources filtered by type and borough. Types: shelter, food_bank, hospital, clinic, school, childcare, benefits_center, domestic_violence, legal_aid, senior_services, mental_health. Boroughs: MN, BK, QN, BX, SI (or leave blank)."""
        plan = {
            "intent": "lookup",
            "resource_types": [resource_type],
            "filters": {"borough": borough} if borough else {},
            "limit": limit,
        }
        result = execute(plan)
        resources = _df_to_records(result.get("results"), limit)
        simple = [{"name": r.get("name"), "address": r.get("address"),
                   "borough": r.get("borough")} for r in resources]
        return json.dumps(_clean(simple), indent=2, default=str)

    group.add_function(name="find_resources", fn=_traced("find_resources", _find_resources),
                       description=_find_resources.__doc__)
    group.add_function(name="find_resources_by_type", fn=_traced("find_resources_by_type", _find_resources_by_type),
                       description=_find_resources_by_type.__doc__)
    yield group


# ── Tool Group 2: Eligibility / Rights / Stories ─────────────────────────────

class EligibilityToolConfig(FunctionGroupBaseConfig, name="nyc_eligibility_tools"):
    include: list[str] = Field(
        default_factory=lambda: ["calculate_eligibility", "get_rights", "get_stories"],
        description="Benefits eligibility + legal rights + success stories",
    )


@register_function_group(config_type=EligibilityToolConfig)
async def nyc_eligibility_tools(_config, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    group = FunctionGroup(config=_config)

    async def _calculate_eligibility(household_size: int = 1,
                                      annual_income: float = 0,
                                      has_children: bool = False,
                                      has_pregnant: bool = False,
                                      has_disabled: bool = False,
                                      has_senior: bool = False,
                                      housing_status: str = "") -> str:
        """Calculate which NYC/federal benefits a household qualifies for (SNAP, Medicaid, WIC, Cash Assistance, Fair Fares, Emergency Shelter). ALWAYS call when user mentions income, kids, pregnancy, or asks about benefits."""
        result = pipeline_calc_elig(
            household_size=household_size, annual_income=annual_income,
            has_children=has_children, has_pregnant=has_pregnant,
            has_disabled=has_disabled, has_senior=has_senior,
            housing_status=housing_status,
        )
        simple = {
            "qualifying_programs": result["qualifying_programs"],
            "monthly_benefits": result["estimated_monthly_benefits"],
            "details": {k: {"qualifies": v["qualifies"], "monthly_estimate": v.get("monthly_estimate"),
                             "where_to_apply": v.get("where_to_apply")}
                         for k, v in result["programs"].items()},
        }
        return json.dumps(_clean(simple), indent=2, default=str)

    async def _get_rights(resource_type: str = "default") -> str:
        """Get the legal rights a person has at a resource type. ALWAYS call when user mentions 'no ID', 'undocumented', 'no insurance', 'no address', or asks 'is X a problem?'. Types: shelter, food_bank, hospital, school, benefits_center, domestic_violence."""
        rights = pipeline_get_rights(resource_type)
        return json.dumps(rights[:5], indent=2)

    async def _get_stories(need: str = "housing") -> str:
        """Get 2-3 anonymized success stories of people in similar situations. Used to build trust. Need categories: housing, medical, benefits, safety, employment."""
        stories = pipeline_get_stories(need=need, k=2)
        simple = [{"name": s["name"], "situation": s["situation"],
                   "outcome": s["outcome"], "timeframe": s["timeframe"]}
                  for s in stories]
        return json.dumps(simple, indent=2)

    group.add_function(name="calculate_eligibility", fn=_traced("calculate_eligibility", _calculate_eligibility),
                       description=_calculate_eligibility.__doc__)
    group.add_function(name="get_rights", fn=_traced("get_rights", _get_rights),
                       description=_get_rights.__doc__)
    group.add_function(name="get_stories", fn=_traced("get_stories", _get_stories),
                       description=_get_stories.__doc__)
    yield group


# ── Tool Group 3: Directions ─────────────────────────────────────────────────

class DirectionsToolConfig(FunctionGroupBaseConfig, name="nyc_directions_tools"):
    include: list[str] = Field(
        default_factory=lambda: ["get_directions"],
        description="Walking + transit directions",
    )


@register_function_group(config_type=DirectionsToolConfig)
async def nyc_directions_tools(_config, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    group = FunctionGroup(config=_config)

    async def _get_directions(from_lat: float, from_lon: float,
                               to_lat: float, to_lon: float,
                               budget: float = -1) -> str:
        """Get walking + transit directions from origin to destination. Budget: -1=show all, 0=walk only (includes nearest free MetroCard location), >2.90=subway OK. Pass coordinates as floats."""
        b = None if budget < 0 else budget
        result = pipeline_get_directions(from_lat, from_lon, to_lat, to_lon, b)
        simple = {
            "distance_miles": result.get("distance_miles"),
            "recommendation": result.get("recommendation"),
            "options": [{"mode": o["mode"], "duration_min": o["duration_min"],
                          "cost": o["cost"]} for o in result.get("options", [])],
        }
        if result.get("free_metrocard_location"):
            simple["free_metrocard"] = result["free_metrocard_location"]
        return json.dumps(_clean(simple), indent=2)

    group.add_function(name="get_directions", fn=_traced("get_directions", _get_directions),
                       description=_get_directions.__doc__)
    yield group


# ── Tool Group 4: Case Management ────────────────────────────────────────────

class CaseToolConfig(FunctionGroupBaseConfig, name="nyc_case_tools"):
    include: list[str] = Field(
        default_factory=lambda: ["get_case_summary", "get_progress",
                                  "choose_resource", "checkin_resource"],
        description="Persistent case management",
    )


@register_function_group(config_type=CaseToolConfig)
async def nyc_case_tools(_config, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    group = FunctionGroup(config=_config)

    async def _get_case_summary(case_id: str) -> str:
        """Get a contextual summary of a returning client's case history. Includes open needs, resources visited, and last location. Use this when a returning client logs in."""
        return get_case_summary(case_id) or "No case found."

    async def _get_progress(case_id: str) -> str:
        """Get structured progress for a case: which needs are resolved, in progress, or open. Returns counts + per-need status."""
        p = get_progress(case_id)
        return json.dumps(_clean(p), indent=2)

    async def _choose_resource(case_id: str, need_category: str,
                                resource_name: str, resource_address: str = "",
                                resource_type: str = "") -> str:
        """Record that the user has chosen a specific resource for a need category (e.g. housing, medical, benefits). Marks the need as in_progress."""
        case = choose_resource(case_id, need_category, resource_name,
                               resource_address, resource_type)
        return f"Saved: {resource_name} chosen for {need_category}."

    async def _checkin_resource(case_id: str, arrived: bool,
                                  resource_name: str, feedback: str = "") -> str:
        """Record whether the user arrived at a resource. arrived=true resolves the need; arrived=false marks resource as failed and excludes it from future searches."""
        case = checkin(case_id, arrived, resource_name, feedback, None)
        status = "Need resolved" if arrived else "Resource marked failed"
        return f"Checkin saved: arrived={arrived} at {resource_name}. {status}."

    group.add_function(name="get_case_summary", fn=_traced("get_case_summary", _get_case_summary),
                       description=_get_case_summary.__doc__)
    group.add_function(name="get_progress", fn=_traced("get_progress", _get_progress),
                       description=_get_progress.__doc__)
    group.add_function(name="choose_resource", fn=_traced("choose_resource", _choose_resource),
                       description=_choose_resource.__doc__)
    group.add_function(name="checkin_resource", fn=_traced("checkin_resource", _checkin_resource),
                       description=_checkin_resource.__doc__)
    yield group


# ── Tool Group 5: Admin / Supervisor Tools ───────────────────────────────────

class AdminToolConfig(FunctionGroupBaseConfig, name="nyc_admin_tools"):
    include: list[str] = Field(
        default_factory=lambda: [
            "list_all_cases", "get_case_details", "get_city_stats",
            "find_critical_cases", "generate_case_briefing",
            "update_case_need_status", "advance_destination_state",
            "add_admin_note",
        ],
        description="Admin/supervisor tools for NYC Social Services dashboard",
    )


@register_function_group(config_type=AdminToolConfig)
async def nyc_admin_tools(_config, _builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    group = FunctionGroup(config=_config)

    def _case_row(case):
        needs = case.get("needs", [])
        failed = []
        for n in needs:
            failed.extend(n.get("failed_resources", []))
        open_needs = [n.get("category") for n in needs if n.get("status") != "resolved"]
        urgency = _estimate_urgency(needs, failed) if _BRIEFING_AVAILABLE else "unknown"
        active_dests = [i for i in case.get("destination_intents", [])
                        if i.get("state") not in {"resolved", "cancelled"}]
        return {
            "case_id": case.get("case_id"),
            "urgency": urgency,
            "open_needs": open_needs,
            "active_destinations": len(active_dests),
            "failed_resources": len(failed),
            "last_visit": case.get("last_visit"),
            "visits": len(case.get("visits", [])),
        }

    async def _list_all_cases(filter_urgency: str = "", filter_open_need: str = "",
                              limit: int = 20) -> str:
        """List NYC Social Services cases for the dashboard. Filter by urgency (critical/high/medium/low) or by a specific open need category (housing/medical/benefits/safety/employment). Default returns top 20 most recent. USE THIS FIRST for any 'show me cases' or 'who needs help' question."""
        out = []
        for c_summary in list_cases():
            case = load_case(c_summary["case_id"])
            if not case:
                continue
            row = _case_row(case)
            if filter_urgency and row["urgency"] != filter_urgency.lower():
                continue
            if filter_open_need and filter_open_need.lower() not in [n.lower() for n in row["open_needs"]]:
                continue
            out.append(row)
        out.sort(key=lambda x: (x["urgency"] != "critical", x["urgency"] != "high"))
        return json.dumps(_clean(out[:limit]), indent=2, default=str)

    async def _get_case_details(case_id: str) -> str:
        """Get full record for a single case including all needs, visits, destinations, and notes. Use after list_all_cases identifies a case of interest."""
        case = load_case(case_id)
        if not case:
            return json.dumps({"error": f"Case '{case_id}' not found"})
        # Strip verbose history fields to fit LLM context
        trimmed = {
            "case_id": case.get("case_id"),
            "needs": case.get("needs", []),
            "destination_intents": case.get("destination_intents", []),
            "admin_notes": case.get("admin_notes", ""),
            "current_location": case.get("current_location"),
            "last_visit": case.get("last_visit"),
            "visits": case.get("visits", [])[-5:],  # last 5 only
            "emergency_contact": case.get("emergency_contact", {}),
        }
        return json.dumps(_clean(trimmed), indent=2, default=str)

    async def _get_city_stats() -> str:
        """Get aggregate stats across all cases: urgency breakdown, need category counts, total visits, resolved vs open needs. Use for dashboard summaries and 'how is the city doing' questions."""
        urgency_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "unknown": 0}
        need_counts: dict = {}
        resolved, open_ = 0, 0
        total_visits = 0
        for c_summary in list_cases():
            case = load_case(c_summary["case_id"])
            if not case:
                continue
            total_visits += len(case.get("visits", []))
            for n in case.get("needs", []):
                cat = n.get("category", "other")
                need_counts[cat] = need_counts.get(cat, 0) + 1
                if n.get("status") == "resolved":
                    resolved += 1
                else:
                    open_ += 1
            row = _case_row(case)
            urgency_counts[row["urgency"]] = urgency_counts.get(row["urgency"], 0) + 1
        return json.dumps(_clean({
            "urgency_breakdown": urgency_counts,
            "need_categories": need_counts,
            "needs_resolved": resolved,
            "needs_open": open_,
            "total_visits": total_visits,
            "total_cases": len(list_cases()),
        }), indent=2, default=str)

    async def _find_critical_cases(limit: int = 10) -> str:
        """Shortcut: list only critical-urgency cases that need immediate dispatcher attention. Use for emergency response queries."""
        return await _list_all_cases(filter_urgency="critical", limit=limit)

    async def _generate_case_briefing(case_id: str, resource_name: str = "",
                                       resource_type: str = "",
                                       resource_address: str = "") -> str:
        """Generate a structured AI briefing for a specific case + resource combination. Used when a dispatcher needs talking points before a client arrives. Returns diagnosis, urgency, document gaps, edge cases, and recommended approach."""
        if not _BRIEFING_AVAILABLE:
            return json.dumps({"error": "briefing pipeline unavailable"})
        case = load_case(case_id)
        if not case:
            return json.dumps({"error": f"Case '{case_id}' not found"})
        resource = {"name": resource_name, "resource_type": resource_type,
                    "address": resource_address}
        b = generate_briefing(case, resource)
        return json.dumps(_clean(b), indent=2, default=str)

    async def _update_case_need_status(case_id: str, category: str, status: str) -> str:
        """Admin action: set a need's status to open / in_progress / resolved."""
        result = update_need_status(case_id, category, status)
        if result.get("error"):
            return json.dumps(result)
        return f"Updated case {case_id} need '{category}' to '{status}'."

    async def _advance_destination_state(case_id: str, resource_name: str,
                                          new_state: str) -> str:
        """Admin action: advance a destination intent's state (intent_confirmed / en_route / arrived / resolved / cancelled)."""
        update_destination_state(case_id, resource_name, new_state)
        return f"Advanced destination '{resource_name}' on case {case_id} to state '{new_state}'."

    async def _add_admin_note(case_id: str, notes: str) -> str:
        """Admin action: save supervisor notes on a case."""
        result = save_admin_notes(case_id, notes)
        if result.get("error"):
            return json.dumps(result)
        return f"Saved admin notes to case {case_id}."

    group.add_function(name="list_all_cases", fn=_traced("list_all_cases", _list_all_cases),
                       description=_list_all_cases.__doc__)
    group.add_function(name="get_case_details", fn=_traced("get_case_details", _get_case_details),
                       description=_get_case_details.__doc__)
    group.add_function(name="get_city_stats", fn=_traced("get_city_stats", _get_city_stats),
                       description=_get_city_stats.__doc__)
    group.add_function(name="find_critical_cases", fn=_traced("find_critical_cases", _find_critical_cases),
                       description=_find_critical_cases.__doc__)
    group.add_function(name="generate_case_briefing", fn=_traced("generate_case_briefing", _generate_case_briefing),
                       description=_generate_case_briefing.__doc__)
    group.add_function(name="update_case_need_status", fn=_traced("update_case_need_status", _update_case_need_status),
                       description=_update_case_need_status.__doc__)
    group.add_function(name="advance_destination_state", fn=_traced("advance_destination_state", _advance_destination_state),
                       description=_advance_destination_state.__doc__)
    group.add_function(name="add_admin_note", fn=_traced("add_admin_note", _add_admin_note),
                       description=_add_admin_note.__doc__)
    yield group
