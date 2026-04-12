"""
pipeline/agent.py — Autonomous agent that chains skill calls.

Given a user query, the agent autonomously:
  1. Decomposes the situation (planner)
  2. Finds resources for each need (executor)
  3. Extracts household profile and calculates eligibility
  4. Gets directions to top resource for each need
  5. Compiles a complete action plan
  6. Returns structured output + trace of steps

The frontend shows the trace so judges see the agent working.
"""
import re
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.planner import generate_plan
from pipeline.executor import execute
from pipeline.synth import answer as synth_answer
from pipeline.routing import get_directions
from pipeline.eligibility import calculate_eligibility, get_rights, get_stories
from pipeline.cases import load_case, add_visit, get_progress

import pandas as pd


# ── Extract household profile from query text ────────────────────────────────
def _extract_profile(query: str, plan: dict) -> dict:
    """
    Extract household profile from query + plan for eligibility calculation.
    Uses regex on the query, with plan profile as fallback.
    """
    q = query.lower()
    profile = plan.get("client_profile", {}) or {}

    # Household size
    household_size = profile.get("household_size") or 1
    m = re.search(r'(\d+)\s*(?:kids?|children)', q)
    if m:
        household_size = max(household_size, int(m.group(1)) + 1)  # +1 for parent

    m = re.search(r'family\s*of\s*(\d+)', q)
    if m:
        household_size = max(household_size, int(m.group(1)))

    m = re.search(r'household\s*of\s*(\d+)', q)
    if m:
        household_size = max(household_size, int(m.group(1)))

    # Annual income
    annual_income = 0
    m = re.search(r'\$?\s*(\d{1,3}(?:,\d{3})*|\d+)\s*(?:k|K)\b', q)
    if m:
        raw = m.group(1).replace(",", "")
        annual_income = int(raw) * 1000 if int(raw) < 1000 else int(raw)
    else:
        m = re.search(r'\$?\s*(\d{4,6})(?:\s*(?:per|a|/)\s*year|\s*annual|\s*income)?', q)
        if m:
            annual_income = int(m.group(1))

    if profile.get("income"):
        annual_income = max(annual_income, profile["income"])

    # Other flags
    has_children = bool(re.search(r'kids?|children|son|daughter|baby|toddler|infant', q))
    has_pregnant = bool(re.search(r'pregnan|expecting', q))
    has_disabled = bool(re.search(r'disabled|disability|wheelchair|adhd|autism', q))
    has_senior = bool(re.search(r'elderly|senior|grandm|grandp|aged', q))
    is_veteran = bool(re.search(r'veteran|military|served', q))

    # Housing status
    housing_status = ""
    if re.search(r'homeless|no\s*place\s*to\s*sleep|on\s*the\s*street', q):
        housing_status = "homeless"
    elif re.search(r'evict|kick(?:ing|ed)?\s*(?:me|us)\s*out|losing\s*(?:my|our)?\s*(?:home|apartment|place)|sister\s*(?:is\s*)?kicking', q):
        housing_status = "at_risk"

    # ID status
    has_id = not bool(re.search(r'no\s*(?:id|identification|papers|documents)|lost\s*my\s*id|without\s*id', q))

    return {
        "household_size": household_size,
        "annual_income": annual_income,
        "has_children": has_children,
        "has_pregnant": has_pregnant,
        "has_disabled": has_disabled,
        "has_senior": has_senior,
        "is_veteran": is_veteran,
        "housing_status": housing_status,
        "has_id": has_id,
    }


# ── Priority ordering for needs ──────────────────────────────────────────────
PRIORITY_ORDER = {
    "safety": 1, "domestic_violence": 1,
    "housing": 2, "shelter": 2,
    "medical": 3, "healthcare": 3, "health": 3,
    "food": 4,
    "benefits": 5, "documents": 5, "id": 5,
    "employment": 6, "job": 6,
    "school": 7, "education": 7, "childcare": 7,
    "legal": 8,
    "senior": 9,
}

TIME_FRAMES = {
    1: "URGENT — NOW",
    2: "TONIGHT",
    3: "TODAY / TOMORROW",
    4: "THIS WEEK",
    5: "THIS WEEK",
    6: "NEXT WEEK",
    7: "NEXT 2 WEEKS",
    8: "WHEN SETTLED",
    9: "ONGOING",
}


# ── Main autonomous agent ────────────────────────────────────────────────────
def run_autonomous_agent(
    query: str,
    location: dict = None,
    case_id: str = None,
    skip_directions: bool = False,
) -> dict:
    """
    Run the full autonomous agent workflow.

    Returns a complete action plan with:
      - step_by_step_plan: ordered list of steps (tonight, tomorrow, this week)
      - resources: all resources found
      - eligibility: benefits they qualify for
      - rights: legal rights for each resource type
      - stories: 2 similar success stories
      - total_monthly_benefits: estimated $$ amount
      - trace: step-by-step log of what the agent did (for UI display)
    """
    t0 = time.time()
    trace = []

    def log(step_num, description, duration):
        trace.append({
            "step": step_num,
            "action": description,
            "duration_s": round(duration, 2),
        })

    # ── Step 1: Decompose situation ──────────────────────────────────────────
    t = time.time()
    plan = generate_plan(query)
    if location:
        plan["_user_location"] = location
    log(1, "Analyzed situation and identified needs", time.time() - t)

    needs = plan.get("identified_needs", [])
    if not needs and plan.get("intent") == "lookup":
        # Simple lookup query — treat as one need
        rtypes = plan.get("resource_types", [])
        if rtypes:
            needs = [{"category": rtypes[0], "priority": 1}]

    # ── Step 2: Find resources for each need ─────────────────────────────────
    t = time.time()
    result = execute(plan)

    resources_by_need = {}
    if result.get("intent") == "needs_assessment":
        for key, df in result.get("results_by_need", {}).items():
            if isinstance(df, pd.DataFrame) and len(df):
                resources_by_need[key] = df.to_dict("records")[:5]
    elif result.get("intent") == "lookup":
        df = result.get("results")
        if isinstance(df, pd.DataFrame) and len(df):
            key = "+".join(plan.get("resource_types", ["resource"]))
            resources_by_need[key] = df.to_dict("records")[:5]

    total_resources = sum(len(v) for v in resources_by_need.values())
    log(2, f"Found {total_resources} matching resources across {len(resources_by_need)} categories",
        time.time() - t)

    # ── Step 3: Extract profile and calculate eligibility ────────────────────
    t = time.time()
    profile = _extract_profile(query, plan)
    eligibility = calculate_eligibility(**profile)
    qualifying = [p for p in eligibility["programs"].values() if p.get("qualifies")]
    log(3, f"Calculated eligibility: {len(qualifying)} programs qualify, "
           f"~${eligibility['estimated_monthly_benefits']}/month",
        time.time() - t)

    # ── Step 4: Get directions for top resource per need ─────────────────────
    t = time.time()
    directions_by_resource = {}
    if not skip_directions and location:
        for key, resource_list in resources_by_need.items():
            if resource_list:
                top = resource_list[0]
                lat = top.get("latitude")
                lon = top.get("longitude")
                if lat and lon:
                    try:
                        d = get_directions(location["lat"], location["lon"],
                                           float(lat), float(lon), budget=None)
                        directions_by_resource[top.get("name", "")] = {
                            "distance_miles": d["distance_miles"],
                            "walk_min": next((o["duration_min"] for o in d["options"]
                                              if o["mode"] == "walk"), None),
                            "recommendation": d["recommendation"],
                        }
                    except Exception:
                        pass
    log(4, f"Computed directions for {len(directions_by_resource)} top resources",
        time.time() - t)

    # ── Step 5: Build step-by-step action plan ───────────────────────────────
    t = time.time()
    steps = []

    # Add each need as a step, ordered by priority
    for need in sorted(needs, key=lambda n: n.get("priority", 99)):
        cat = need.get("category", "")
        resources = resources_by_need.get(cat, [])
        if not resources:
            # Try to find by expanded type
            for key, rlist in resources_by_need.items():
                if cat in key:
                    resources = rlist
                    break

        priority = PRIORITY_ORDER.get(cat, need.get("priority", 5))
        timeframe = TIME_FRAMES.get(priority, "SOON")

        top_resource = resources[0] if resources else None
        directions = directions_by_resource.get(top_resource.get("name", "") if top_resource else "")

        rights = get_rights(top_resource.get("resource_type", cat) if top_resource else cat)

        step = {
            "priority": priority,
            "timeframe": timeframe,
            "category": cat,
            "action": f"Address {cat.replace('_', ' ')} need",
            "top_resource": top_resource,
            "alternatives": resources[1:4] if len(resources) > 1 else [],
            "directions": directions,
            "rights": rights[:3],  # top 3 rights
        }

        # Match eligibility programs relevant to this need
        if cat in ["benefits", "documents", "id"]:
            step["eligibility"] = eligibility["qualifying_programs"]
        elif cat == "food":
            step["eligibility"] = [p for p in eligibility["qualifying_programs"]
                                   if "SNAP" in p or "WIC" in p or "meal" in p.lower()]
        elif cat == "medical":
            step["eligibility"] = [p for p in eligibility["qualifying_programs"]
                                   if "Medicaid" in p]
        elif cat == "school":
            step["eligibility"] = [p for p in eligibility["qualifying_programs"]
                                   if "School" in p]

        steps.append(step)
    log(5, f"Assembled {len(steps)} prioritized action steps", time.time() - t)

    # ── Step 6: Get stories for the top need ─────────────────────────────────
    t = time.time()
    stories = []
    if needs:
        top_cat = sorted(needs, key=lambda n: n.get("priority", 99))[0].get("category", "")
        stories = get_stories(need=top_cat, k=2)
    else:
        stories = get_stories(k=2)
    log(6, f"Retrieved {len(stories)} relevant success stories", time.time() - t)

    # ── Step 7: Save to case if case_id provided ─────────────────────────────
    if case_id:
        t = time.time()
        try:
            all_resources = []
            for r_list in resources_by_need.values():
                all_resources.extend(r_list)
            add_visit(case_id, query, f"Agent generated {len(steps)}-step plan",
                     all_resources, location=location, plan=plan)
            log(7, f"Saved plan to case {case_id}", time.time() - t)
        except Exception as e:
            log(7, f"Case save skipped: {e}", time.time() - t)

    # ── Assemble the final response ──────────────────────────────────────────
    total_time = time.time() - t0
    action_plan_summary = _generate_summary(steps, eligibility, profile)

    return {
        "query": query,
        "summary": action_plan_summary,
        "profile": profile,
        "steps": steps,
        "total_monthly_benefits": eligibility["estimated_monthly_benefits"],
        "qualifying_programs": eligibility["qualifying_programs"],
        "eligibility_details": eligibility["programs"],
        "stories": stories,
        "trace": trace,
        "total_time_s": round(total_time, 2),
        "created_at": datetime.now().isoformat(),
    }


def _generate_summary(steps: list, eligibility: dict, profile: dict) -> str:
    """Generate a human-readable summary of the action plan."""
    if not steps:
        return "We couldn't identify specific needs from your query. Try being more specific about what's going on."

    household = profile.get("household_size", 1)
    income = profile.get("annual_income", 0)
    income_str = f"${income:,.0f}/year" if income else "unknown income"

    n_steps = len(steps)
    top_priorities = [s["category"] for s in sorted(steps, key=lambda x: x["priority"])[:3]]

    benefits_str = ""
    if eligibility["estimated_monthly_benefits"] > 0:
        benefits_str = (f" You qualify for about ${eligibility['estimated_monthly_benefits']:,.0f}/month in "
                        f"benefits across {len(eligibility['qualifying_programs'])} programs.")

    return (f"Your personalized action plan has {n_steps} steps. "
            f"Top priorities: {', '.join(top_priorities)}. "
            f"Based on your situation (household of {household}, {income_str}).{benefits_str} "
            f"Start with the first item — it's the most urgent.")


# ── PDF generation (optional — falls back to plain text if reportlab missing) ─
def generate_plan_pdf(agent_result: dict, output_path: str = None) -> str:
    """
    Generate a printable plan as HTML (which can be converted to PDF via browser).
    Returns HTML string or file path if saved.
    """
    steps = agent_result.get("steps", [])
    summary = agent_result.get("summary", "")
    profile = agent_result.get("profile", {})
    programs = agent_result.get("qualifying_programs", [])
    total = agent_result.get("total_monthly_benefits", 0)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Your Action Plan</title>
<style>
body{{font-family:-apple-system,sans-serif;max-width:700px;margin:20px auto;padding:20px;color:#222}}
h1{{color:#76b900;border-bottom:3px solid #76b900;padding-bottom:8px}}
h2{{color:#333;margin-top:28px;border-left:4px solid #76b900;padding-left:10px}}
.step{{background:#f9f9f9;border-radius:8px;padding:16px;margin:12px 0}}
.timeframe{{background:#76b900;color:white;padding:4px 10px;border-radius:12px;font-size:11px;font-weight:bold;display:inline-block}}
.resource-name{{font-size:17px;font-weight:600;color:#222;margin-top:8px}}
.resource-addr{{color:#666;font-size:13px}}
.alt{{color:#888;font-size:12px;margin-left:10px}}
.rights{{background:#e8f5e9;padding:10px;border-radius:6px;margin-top:8px;font-size:12px}}
.rights-title{{font-weight:bold;color:#2e7d32}}
.benefits{{background:#fff3e0;padding:12px;border-radius:8px;margin:16px 0}}
.benefits h3{{color:#e65100;margin:0 0 6px 0}}
ul{{padding-left:20px}}
.created{{color:#999;font-size:11px;text-align:right;margin-top:30px}}
</style></head><body>
<h1>Your Action Plan</h1>
<p>{summary}</p>
"""

    if programs:
        html += f'<div class="benefits"><h3>You qualify for ~${total:,.0f}/month in benefits</h3><ul>'
        for p in programs:
            html += f'<li>{p}</li>'
        html += '</ul></div>'

    for i, step in enumerate(steps, 1):
        r = step.get("top_resource") or {}
        rname = r.get("name", "No resource found")
        raddr = r.get("address", "")
        dir_info = step.get("directions") or {}
        rights = step.get("rights", [])

        html += f'<div class="step"><span class="timeframe">{step["timeframe"]}</span>'
        html += f'<h2 style="margin-top:8px;">Step {i}: {step["category"].replace("_"," ").title()}</h2>'
        html += f'<div class="resource-name">→ {rname}</div>'
        if raddr:
            html += f'<div class="resource-addr">{raddr}</div>'
        if dir_info.get("walk_min"):
            html += f'<div class="resource-addr">{dir_info.get("distance_miles","?")} miles · {dir_info["walk_min"]} min walk</div>'
        if rights:
            html += '<div class="rights"><div class="rights-title">Your rights:</div>'
            for right in rights:
                html += f'<div>• {right["right"]}: {right["detail"]}</div>'
            html += '</div>'

        alts = step.get("alternatives", [])
        if alts:
            html += '<div style="margin-top:8px;font-size:12px;"><em>Alternatives:</em>'
            for a in alts[:2]:
                html += f'<div class="alt">• {a.get("name","?")} at {a.get("address","")}</div>'
            html += '</div>'
        html += '</div>'

    html += f'<div class="created">Generated {agent_result.get("created_at","")} by NYC Social Services Engine</div>'
    html += '</body></html>'

    if output_path:
        with open(output_path, "w") as f:
            f.write(html)
        return output_path
    return html


if __name__ == "__main__":
    # Quick test
    print("Testing autonomous agent...\n")
    result = run_autonomous_agent(
        query="I'm Tina, I have 4 kids ages 12-16, my income is $28K, my sister is kicking us out next week",
        location={"lat": 40.65, "lon": -73.95},
    )
    print(f"Summary: {result['summary']}\n")
    print(f"Steps: {len(result['steps'])}")
    for i, s in enumerate(result["steps"], 1):
        r = s.get("top_resource") or {}
        print(f"  {i}. [{s['timeframe']}] {s['category']} → {r.get('name', 'none')}")
    print(f"\nMonthly benefits: ${result['total_monthly_benefits']}")
    print(f"\nTrace:")
    for step in result["trace"]:
        print(f"  {step['step']}. {step['action']} ({step['duration_s']}s)")
    print(f"\nTotal: {result['total_time_s']}s")
