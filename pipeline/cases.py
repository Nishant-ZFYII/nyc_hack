"""
pipeline/cases.py — Case management: track users across visits.

Each case has:
  - case_id (from name or ID number)
  - visits: list of {timestamp, location, query, resources_shown, feedback}
  - needs: accumulated identified needs + status (open/resolved)
  - current_location: last known location

Storage: JSON file in data/cases/ directory.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from datetime import datetime

CASES_DIR = Path(__file__).resolve().parent.parent / "data" / "cases"
CASES_DIR.mkdir(parents=True, exist_ok=True)


def _case_path(case_id: str) -> Path:
    """Get file path for a case. Sanitize the ID."""
    safe_id = "".join(c for c in case_id.lower().strip() if c.isalnum() or c in "-_").strip("_")
    if not safe_id:
        safe_id = "anonymous"
    return CASES_DIR / f"{safe_id}.json"


def load_case(case_id: str) -> dict | None:
    """Load an existing case by ID or name."""
    path = _case_path(case_id)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def create_case(case_id: str, name: str = "", location: dict = None) -> dict:
    """Create a new case or return existing."""
    existing = load_case(case_id)
    if existing:
        return existing

    case = {
        "case_id": case_id,
        "name": name or case_id,
        "created_at": datetime.now().isoformat(),
        "last_visit": datetime.now().isoformat(),
        "current_location": location,
        "visits": [],
        "needs": [],
        "resources_visited": [],
        "feedback": [],
        "notes": "",
    }
    _save_case(case)
    return case


def add_visit(case_id: str, query: str, answer: str, resources: list,
              location: dict = None, plan: dict = None) -> dict:
    """Record a visit (query + response) to an existing case."""
    case = load_case(case_id)
    if not case:
        case = create_case(case_id)

    visit = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer_summary": answer[:300] if answer else "",
        "resources_shown": [
            {"name": r.get("name", ""), "type": r.get("type", ""),
             "address": r.get("address", ""), "borough": r.get("borough", "")}
            for r in (resources or [])[:10]
        ],
        "location": location,
    }
    case["visits"].append(visit)
    case["last_visit"] = datetime.now().isoformat()

    if location:
        case["current_location"] = location

    # Extract needs from plan if available
    if plan and plan.get("intent") == "needs_assessment":
        for need in plan.get("identified_needs", []):
            cat = need.get("category", "")
            # Don't duplicate
            existing_cats = [n["category"] for n in case["needs"]]
            if cat and cat not in existing_cats:
                case["needs"].append({
                    "category": cat,
                    "priority": need.get("priority", 99),
                    "status": "open",
                    "identified_at": datetime.now().isoformat(),
                })

    _save_case(case)
    return case


def mark_resource_visited(case_id: str, resource_name: str, feedback: str = None) -> dict:
    """Mark that the user actually visited a resource."""
    case = load_case(case_id)
    if not case:
        return {"error": "Case not found"}

    case["resources_visited"].append({
        "name": resource_name,
        "visited_at": datetime.now().isoformat(),
        "feedback": feedback,
    })

    if feedback:
        case["feedback"].append({
            "resource": resource_name,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
        })

    _save_case(case)
    return case


def resolve_need(case_id: str, category: str) -> dict:
    """Mark a need as resolved."""
    return update_need_status(case_id, category, "resolved")


def update_need_status(case_id: str, category: str, status: str) -> dict:
    """Update a need's status. Accepts 'open', 'in_progress', or 'resolved'."""
    case = load_case(case_id)
    if not case:
        return {"error": "Case not found"}

    for need in case["needs"]:
        if need["category"] == category:
            need["status"] = status
            if status == "resolved":
                need["resolved_at"] = datetime.now().isoformat()

    _save_case(case)
    return case


def sync_needs_from_plan(case: dict, plan: dict) -> dict:
    """Merge identified_needs from plan into case, deduplicating by category."""
    existing_cats = {n["category"] for n in case.get("needs", [])}
    for need in plan.get("identified_needs", []):
        cat = need.get("category", "")
        if cat and cat not in existing_cats:
            case.setdefault("needs", []).append({
                "category": cat,
                "priority": need.get("priority", 99),
                "status": "open",
                "identified_at": datetime.now().isoformat(),
            })
            existing_cats.add(cat)
    _save_case(case)
    return case


def get_case_summary(case_id: str) -> str:
    """Generate a human-readable summary of a case for the next visit."""
    case = load_case(case_id)
    if not case:
        return ""

    name = case.get("name", case_id)
    visits = case.get("visits", [])
    needs = case.get("needs", [])
    visited = case.get("resources_visited", [])
    feedback = case.get("feedback", [])

    lines = [f"Welcome back, {name}."]

    # Visit history
    if visits:
        last = visits[-1]
        last_time = last.get("timestamp", "")[:10]
        lines.append(f"Your last visit was on {last_time}.")
        if last.get("query"):
            lines.append(f"You asked about: \"{last['query'][:80]}\"")

    # Open needs
    open_needs = [n for n in needs if n.get("status") == "open"]
    resolved = [n for n in needs if n.get("status") == "resolved"]
    if open_needs:
        cats = ", ".join(n["category"] for n in sorted(open_needs, key=lambda x: x.get("priority", 99)))
        lines.append(f"Open needs: {cats}.")
    if resolved:
        cats = ", ".join(n["category"] for n in resolved)
        lines.append(f"Resolved: {cats}.")

    # Resources visited
    if visited:
        names = ", ".join(v["name"] for v in visited[-3:])
        lines.append(f"You've visited: {names}.")
        # Check for negative feedback
        neg = [f for f in feedback if any(w in (f.get("feedback", "")).lower()
               for w in ["full", "closed", "wrong", "unsafe"])]
        if neg:
            lines.append(f"You reported issues with {len(neg)} resource(s).")

    # Location
    loc = case.get("current_location")
    if loc and loc.get("display_name"):
        lines.append(f"Last location: {loc['display_name']}.")

    return " ".join(lines)


def list_cases() -> list:
    """List all cases (for admin view)."""
    cases = []
    for path in CASES_DIR.glob("*.json"):
        try:
            with open(path) as f:
                c = json.load(f)
                cases.append({
                    "case_id": c.get("case_id", ""),
                    "name": c.get("name", ""),
                    "last_visit": c.get("last_visit", ""),
                    "open_needs": len([n for n in c.get("needs", []) if n.get("status") == "open"]),
                    "total_visits": len(c.get("visits", [])),
                })
        except Exception:
            pass
    return sorted(cases, key=lambda x: x.get("last_visit", ""), reverse=True)


def add_destination_intent(case_id: str, resource: dict,
                           state: str = "intent_confirmed") -> dict:
    """Record that the user confirmed intent to visit a resource."""
    case = load_case(case_id)
    if not case:
        case = create_case(case_id)

    intent = {
        "resource_name": resource.get("name", ""),
        "resource_type": resource.get("resource_type", resource.get("type", "")),
        "address": resource.get("address", ""),
        "borough": resource.get("borough", ""),
        "category": resource.get("category", ""),
        "state": state,
        "intent_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "thread_id": None,
        "thread_url": None,
        "acknowledged": False,
    }

    intents = case.setdefault("destination_intents", [])
    today = datetime.now().date().isoformat()
    # Same resource + same day → update in place instead of appending
    for ex in intents:
        if (ex.get("resource_name") == intent["resource_name"] and
                ex.get("intent_at", "")[:10] == today):
            ex["state"] = state
            ex["updated_at"] = intent["updated_at"]
            _save_case(case)
            return case
    intents.append(intent)
    _save_case(case)
    return case


def update_destination_state(case_id: str, resource_name: str,
                              new_state: str) -> dict:
    """Advance lifecycle state for a destination intent."""
    case = load_case(case_id)
    if not case:
        return {"error": "Case not found"}
    for intent in case.get("destination_intents", []):
        if intent.get("resource_name") == resource_name:
            intent["state"] = new_state
            intent["updated_at"] = datetime.now().isoformat()
            if new_state in ("arrived", "resolved"):
                intent[f"{new_state}_at"] = datetime.now().isoformat()
    _save_case(case)
    return case


def get_active_destinations(case_id: str) -> list:
    """Return destination intents not yet resolved or cancelled."""
    case = load_case(case_id)
    if not case:
        return []
    terminal = {"resolved", "cancelled"}
    return [i for i in case.get("destination_intents", [])
            if i.get("state") not in terminal]


def _save_case(case: dict):
    """Save case to disk."""
    path = _case_path(case["case_id"])
    with open(path, "w") as f:
        json.dump(case, f, indent=2)
