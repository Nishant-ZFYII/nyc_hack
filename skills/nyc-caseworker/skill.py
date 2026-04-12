"""
skills/nyc-caseworker/skill.py — OpenClaw skill entry point.

Exposes the NYC caseworker functionality as callable tools for any agent
framework (OpenClaw, LangChain, etc).

Usage:
    from skills.nyc_caseworker.skill import (
        find_resources, get_directions, calculate_eligibility,
        get_rights, get_stories, case_login, case_checkin, case_choose
    )
"""
import sys
from pathlib import Path

# Add project root to path so we can import the pipeline modules
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.planner import generate_plan
from pipeline.executor import execute
from pipeline.synth import answer as synth_answer
from pipeline.routing import get_directions as _get_directions
from pipeline.cases import (
    load_case, create_case, add_visit, get_case_summary, get_progress,
    choose_resource, checkin, get_failed_resources
)
from pipeline.eligibility import (
    calculate_eligibility as _calc_eligibility,
    get_rights as _get_rights,
    get_stories as _get_stories
)


# ── Tool 1: Find resources ────────────────────────────────────────────────────
def find_resources(query: str, location: dict = None, case_id: str = None) -> dict:
    """
    Find NYC social service resources matching a query.

    Args:
        query: Natural language situation (e.g. "I need a shelter tonight")
        location: Optional {lat, lon} — sorts results by walking distance
        case_id: Optional client ID — excludes previously failed resources

    Returns:
        {
            "answer": caseworker-style response,
            "plan": LLM-generated query plan,
            "resources": list of matching resources with distance/walk time,
            "intent": "lookup" | "needs_assessment" | "simulate"
        }
    """
    plan = generate_plan(query)
    if location:
        plan["_user_location"] = location

    # Exclude failed resources for this case
    if case_id:
        from pipeline.executor import set_excluded_resources
        set_excluded_resources(get_failed_resources(case_id))

    result = execute(plan)
    answer_text = synth_answer(query, plan, result)

    # Extract resources (flatten for both lookup and needs_assessment)
    resources = []
    import pandas as pd
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

    # Auto-save to case
    if case_id:
        add_visit(case_id, query, answer_text, resources,
                  location=location, plan=plan)

    return {
        "answer": answer_text,
        "plan": plan,
        "resources": resources,
        "intent": result.get("intent", "lookup"),
    }


# ── Tool 2: Get directions ────────────────────────────────────────────────────
def get_directions(from_lat: float, from_lon: float,
                   to_lat: float, to_lon: float,
                   budget: float = None) -> dict:
    """
    Get multi-modal directions from origin to destination.

    Args:
        from_lat, from_lon: Origin coordinates
        to_lat, to_lon: Destination coordinates
        budget: Available money. None = show all options.
                0 = walking only, recommends free MetroCard location.

    Returns:
        {
            "distance_miles": float,
            "options": [{mode, duration_min, cost, steps, geometry}],
            "recommendation": natural language advice,
            "free_metrocard_location": nearest HRA center (if budget is low)
        }
    """
    return _get_directions(from_lat, from_lon, to_lat, to_lon, budget)


# ── Tool 3: Calculate eligibility ─────────────────────────────────────────────
def calculate_eligibility(
    household_size: int = 1,
    annual_income: float = 0,
    has_children: bool = False,
    has_pregnant: bool = False,
    has_disabled: bool = False,
    has_senior: bool = False,
    is_veteran: bool = False,
    housing_status: str = "",
    has_id: bool = True,
    immigration_status: str = "any",
) -> dict:
    """
    Calculate benefits eligibility for a household.

    Returns qualifying programs (SNAP, Medicaid, WIC, Cash Assistance,
    Fair Fares, Emergency Shelter, Free School Meals, Document Assistance)
    with monthly estimates and documents needed.
    """
    return _calc_eligibility(
        household_size=household_size,
        annual_income=annual_income,
        has_children=has_children,
        has_pregnant=has_pregnant,
        has_disabled=has_disabled,
        has_senior=has_senior,
        is_veteran=is_veteran,
        housing_status=housing_status,
        has_id=has_id,
        immigration_status=immigration_status,
    )


# ── Tool 4: Know your rights ──────────────────────────────────────────────────
def get_rights(resource_type: str = "default") -> list:
    """
    Get know-your-rights info for a resource type.

    Supported types: shelter, food_bank, hospital, school,
                     benefits_center, domestic_violence, default

    Returns list of {right: short description, detail: explanation}.
    """
    return _get_rights(resource_type)


# ── Tool 5: Success stories ───────────────────────────────────────────────────
def get_stories(need: str = None, k: int = 3) -> list:
    """
    Get anonymized success stories.

    Args:
        need: Filter by need category (housing, medical, benefits, etc.)
        k: Number of stories to return

    Returns list of {name, situation, outcome, timeframe, quote}.
    """
    return _get_stories(need, k)


# ── Tool 6: Case management ───────────────────────────────────────────────────
def case_login(case_id: str, name: str = "") -> dict:
    """
    Login or create a client case. Returns contextual summary if returning user.

    Returns:
        {
            "case": full case object,
            "summary": welcome message,
            "returning": true/false,
            "progress": structured progress (if existing case)
        }
    """
    existing = load_case(case_id)
    if existing:
        return {
            "case": existing,
            "summary": get_case_summary(case_id),
            "returning": True,
            "progress": get_progress(case_id),
        }
    case = create_case(case_id, name=name)
    return {
        "case": case,
        "summary": f"Welcome, {name or case_id}. Tell me what's going on and I'll find help for you.",
        "returning": False,
        "progress": {"total_needs": 0, "resolved_count": 0, "open_count": 0, "needs": []},
    }


def case_choose(case_id: str, need_category: str, resource_name: str,
                resource_address: str = "", resource_type: str = "") -> dict:
    """
    Client selects a resource for a specific need.

    Marks the need as in_progress and saves the chosen resource.
    Use this when the client taps "Choose this" on a recommended resource.
    """
    case = choose_resource(case_id, need_category, resource_name,
                           resource_address, resource_type)
    return {"case": case, "message": f"Got it — heading to {resource_name}."}


def case_checkin(case_id: str, arrived: bool, resource_name: str = "",
                 feedback: str = "", location: dict = None) -> dict:
    """
    Client confirms arrival at a resource.

    Args:
        arrived: True = client made it. Marks need as resolved.
                 False = didn't make it. Marks resource as failed,
                 excludes from future queries for this client.
        feedback: Why they didn't arrive (full, closed, unsafe, etc.)
        location: Current location after checkin

    Returns progress summary + next suggested step.
    """
    case = checkin(case_id, arrived, resource_name, feedback, location)
    progress = get_progress(case_id)

    if arrived:
        open_needs = [n for n in progress["needs"] if n["status"] == "open"]
        if open_needs:
            next_need = open_needs[0]["category"]
            msg = (f"Great, glad you made it! You still have "
                   f"{len(open_needs)} open need(s). Next: {next_need}.")
        else:
            msg = "All your needs are addressed! Come back anytime for more help."
    else:
        failed = get_failed_resources(case_id)
        msg = (f"Sorry {resource_name} didn't work out. Let me find alternatives "
               f"(excluding {len(failed)} resources you've already tried).")

    return {"case": case, "message": msg, "progress": progress}


# ── Convenience: the full agent workflow ──────────────────────────────────────
def caseworker_agent(query: str, case_id: str = None,
                     location: dict = None, budget: float = None) -> dict:
    """
    Full caseworker workflow in one call.

    Handles the complete flow:
    1. Resolve location (GPS or address)
    2. Find matching resources
    3. Calculate eligibility if income/household mentioned
    4. Show relevant success stories
    5. Include rights info for the resource type
    6. Auto-save to case if case_id provided

    Returns a complete response for the UI.
    """
    result = find_resources(query, location, case_id)

    # Detect main resource type for rights info
    resources = result.get("resources", [])
    rights = []
    if resources:
        top_type = resources[0].get("resource_type", "default")
        rights = get_rights(top_type)

    # Find relevant stories
    needs = [n.get("category") for n in result.get("plan", {}).get("identified_needs", [])]
    stories = get_stories(needs[0] if needs else None, k=2)

    return {
        "query": query,
        "answer": result["answer"],
        "resources": resources[:8],
        "rights": rights,
        "stories": stories,
        "intent": result["intent"],
    }


if __name__ == "__main__":
    # Quick test
    print("Testing NYC Caseworker skill...\n")

    # Test 1: Find shelters
    print("1. Finding shelters near Flatbush...")
    r = find_resources("I need a shelter", location={"lat": 40.65, "lon": -73.95})
    print(f"   Found {len(r['resources'])} resources")
    print(f"   First: {r['resources'][0].get('name') if r['resources'] else 'none'}\n")

    # Test 2: Eligibility
    print("2. Checking eligibility for Tina...")
    e = calculate_eligibility(household_size=5, annual_income=28000, has_children=True)
    print(f"   Qualifies for {len(e['qualifying_programs'])} programs")
    print(f"   Estimated monthly: ${e['estimated_monthly_benefits']}\n")

    # Test 3: Rights
    print("3. Rights at a shelter...")
    r = get_rights("shelter")
    print(f"   {len(r)} rights listed")
    print(f"   First: {r[0]['right']}\n")

    # Test 4: Case management
    print("4. Creating case for test client...")
    c = case_login("test-client", "Test User")
    print(f"   Case created. Returning: {c['returning']}")
    print(f"   Summary: {c['summary'][:80]}")
