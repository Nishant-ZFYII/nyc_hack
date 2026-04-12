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
from collections.abc import AsyncGenerator
from pathlib import Path

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
    load_case, get_case_summary, get_progress, get_failed_resources,
    choose_resource, checkin, add_destination_intent
)

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

    group.add_function(name="find_resources", fn=_find_resources,
                       description=_find_resources.__doc__)
    group.add_function(name="find_resources_by_type", fn=_find_resources_by_type,
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

    group.add_function(name="calculate_eligibility", fn=_calculate_eligibility,
                       description=_calculate_eligibility.__doc__)
    group.add_function(name="get_rights", fn=_get_rights,
                       description=_get_rights.__doc__)
    group.add_function(name="get_stories", fn=_get_stories,
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

    group.add_function(name="get_directions", fn=_get_directions,
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
        return (f"Checkin saved: arrived={arrived} at {resource_name}. "
                f"Resource failed' if not arrived else 'Need resolved'")

    group.add_function(name="get_case_summary", fn=_get_case_summary,
                       description=_get_case_summary.__doc__)
    group.add_function(name="get_progress", fn=_get_progress,
                       description=_get_progress.__doc__)
    group.add_function(name="choose_resource", fn=_choose_resource,
                       description=_choose_resource.__doc__)
    group.add_function(name="checkin_resource", fn=_checkin_resource,
                       description=_checkin_resource.__doc__)
    yield group
