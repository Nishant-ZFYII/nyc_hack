"""
pipeline/cases.py — Case management: track users across visits.

Each case has:
  - case_id (from name or ID number)
  - visits: list of {timestamp, location, query, resources_shown, feedback}
  - needs: accumulated identified needs + status (open/resolved)
  - current_location: last known location

Storage: JSON file in data/cases/ directory.
"""
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
    case = load_case(case_id)
    if not case:
        return {"error": "Case not found"}

    for need in case["needs"]:
        if need["category"] == category:
            need["status"] = "resolved"
            need["resolved_at"] = datetime.now().isoformat()

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


# Map resource types back to need categories
_TYPE_TO_NEED = {
    "hospital": "medical", "clinic": "medical", "mental_health": "medical",
    "shelter": "housing", "nycha": "housing", "dropin_center": "housing",
    "food_bank": "food",
    "school": "school", "childcare": "school", "education": "school",
    "benefits_center": "benefits",
    "domestic_violence": "safety",
    "legal_aid": "legal",
    "senior_services": "senior",
    "community_center": "employment",
    "emergency_services": "safety",
}


def choose_resource(case_id: str, need_category: str, resource_name: str,
                    resource_address: str = "", resource_type: str = "") -> dict:
    """User selects a specific resource for a need. Saves the choice and marks need as in_progress."""
    case = load_case(case_id)
    if not case:
        return {"error": "Case not found"}

    # Map resource type to need category if the exact category doesn't match
    matched = False
    for need in case["needs"]:
        if need["category"] == need_category and need["status"] == "open":
            need["status"] = "in_progress"
            need["chosen_resource"] = resource_name
            need["chosen_address"] = resource_address
            need["chosen_type"] = resource_type
            need["chosen_at"] = datetime.now().isoformat()
            matched = True
            break

    # If no exact match, try mapping resource type → need category
    if not matched:
        mapped_cat = _TYPE_TO_NEED.get(need_category, need_category)
        for need in case["needs"]:
            if need["category"] == mapped_cat and need["status"] == "open":
                need["status"] = "in_progress"
                need["chosen_resource"] = resource_name
                need["chosen_address"] = resource_address
                need["chosen_type"] = resource_type
                need["chosen_at"] = datetime.now().isoformat()
                matched = True
                break

    # If still no match, try matching by resource_type parameter
    if not matched and resource_type:
        mapped_cat = _TYPE_TO_NEED.get(resource_type, resource_type)
        for need in case["needs"]:
            if need["category"] == mapped_cat and need["status"] == "open":
                need["status"] = "in_progress"
                need["chosen_resource"] = resource_name
                need["chosen_address"] = resource_address
                need["chosen_type"] = resource_type
                need["chosen_at"] = datetime.now().isoformat()
                break

    _save_case(case)
    return case


def checkin(case_id: str, arrived: bool, resource_name: str = "",
            feedback: str = "", location: dict = None) -> dict:
    """User confirms arrival at a resource. Updates need status and location."""
    case = load_case(case_id)
    if not case:
        return {"error": "Case not found"}

    if location:
        case["current_location"] = location
    case["last_visit"] = datetime.now().isoformat()

    if arrived:
        # Mark the resource as visited
        case["resources_visited"].append({
            "name": resource_name,
            "visited_at": datetime.now().isoformat(),
            "feedback": feedback or "arrived",
        })
        # Find and resolve the matching need
        for need in case["needs"]:
            if need.get("chosen_resource") == resource_name and need["status"] == "in_progress":
                need["status"] = "resolved"
                need["resolved_at"] = datetime.now().isoformat()
    else:
        # User did NOT arrive — resource might be full/closed
        if feedback:
            case["feedback"].append({
                "resource": resource_name,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat(),
            })
        # Reset the need to open so they can pick another
        for need in case["needs"]:
            if need.get("chosen_resource") == resource_name and need["status"] == "in_progress":
                need["status"] = "open"
                need["failed_resources"] = need.get("failed_resources", [])
                need["failed_resources"].append({
                    "name": resource_name,
                    "reason": feedback or "did not arrive",
                    "timestamp": datetime.now().isoformat(),
                })
                need.pop("chosen_resource", None)
                need.pop("chosen_address", None)

    _save_case(case)
    return case


def get_failed_resources(case_id: str) -> list:
    """Get all resources that were reported as full/closed/failed for this case."""
    case = load_case(case_id)
    if not case:
        return []
    failed = []
    for need in case.get("needs", []):
        for fr in need.get("failed_resources", []):
            failed.append(fr["name"])
    # Also include feedback-reported resources
    for fb in case.get("feedback", []):
        if fb.get("resource") and fb["resource"] not in failed:
            failed.append(fb["resource"])
    return failed


def get_progress(case_id: str) -> dict:
    """Get a structured progress report for the case."""
    case = load_case(case_id)
    if not case:
        return {"error": "Case not found"}

    needs = case.get("needs", [])
    open_needs = [n for n in needs if n["status"] == "open"]
    in_progress = [n for n in needs if n["status"] == "in_progress"]
    resolved = [n for n in needs if n["status"] == "resolved"]

    progress = {
        "case_id": case["case_id"],
        "name": case.get("name", ""),
        "total_needs": len(needs),
        "resolved_count": len(resolved),
        "in_progress_count": len(in_progress),
        "open_count": len(open_needs),
        "needs": [],
    }

    for n in sorted(needs, key=lambda x: x.get("priority", 99)):
        item = {
            "category": n["category"],
            "status": n["status"],
            "priority": n.get("priority", 99),
        }
        if n.get("chosen_resource"):
            item["chosen_resource"] = n["chosen_resource"]
            item["chosen_address"] = n.get("chosen_address", "")
        if n.get("failed_resources"):
            item["failed_resources"] = [f["name"] for f in n["failed_resources"]]
        if n.get("resolved_at"):
            item["resolved_at"] = n["resolved_at"]
        progress["needs"].append(item)

    return progress


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


# ── Functions used by admin_server.py (teammate's admin portal) ──────────────

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
    """Advance lifecycle state: intent_confirmed → acknowledged → en_route → arrived → resolved."""
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


def save_admin_notes(case_id: str, notes: str) -> dict:
    """Persist admin notes to a case (called from admin portal)."""
    case = load_case(case_id)
    if not case:
        return {"error": "Case not found"}
    case["admin_notes"] = notes
    _save_case(case)
    return case


def raise_ticket(case_id: str, ticket_type: str = "sponsored_ride",
                 reason: str = "") -> dict:
    """Raise a support ticket on a case (e.g. to unlock sponsored ride).

    Idempotent — if a ticket of the same type already exists and is open,
    it's returned rather than duplicated. Tickets are stored on the case
    under `tickets: [{type, raised_at, status, reason}]`.
    """
    from datetime import datetime
    case = load_case(case_id)
    if not case:
        return {"error": f"Case '{case_id}' not found"}
    tickets = case.setdefault("tickets", [])
    # Don't duplicate an open ticket of the same type
    existing = next((t for t in tickets
                     if t.get("type") == ticket_type
                     and t.get("status") == "open"), None)
    if existing:
        return {"case_id": case_id, "ticket": existing, "already_raised": True}
    ticket = {
        "type": ticket_type,
        "raised_at": datetime.utcnow().isoformat() + "Z",
        "status": "open",
        "reason": reason,
    }
    tickets.append(ticket)
    _save_case(case)
    return {"case_id": case_id, "ticket": ticket, "already_raised": False}


def get_tickets(case_id: str) -> list:
    """Return all tickets on a case."""
    case = load_case(case_id)
    if not case:
        return []
    return case.get("tickets", [])


def has_open_ticket(case_id: str, ticket_type: str = "sponsored_ride") -> bool:
    """Check if a case has an open ticket of the given type."""
    for t in get_tickets(case_id):
        if t.get("type") == ticket_type and t.get("status") == "open":
            return True
    return False


def _save_case(case: dict):
    """Save case to disk."""
    path = _case_path(case["case_id"])
    with open(path, "w") as f:
        json.dump(case, f, indent=2)
