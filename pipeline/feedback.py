"""
pipeline/feedback.py — User-in-the-loop ground truth correction.

When a user reports "this shelter is full" or "wrong address", the system:
1. Extracts which resource is being flagged
2. Classifies the issue (full, closed, wrong_address, unsafe, other)
3. Adds to session exclusion list
4. Re-runs pipeline excluding that resource
5. Shows alternatives

This closes the loop: AI recommends → user verifies → AI adapts.
"""
import sys
import re
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.client import chat


FEEDBACK_PROMPT = """You are parsing user feedback about an NYC resource recommendation.
The user is reporting a real-world issue with a recommended resource.

Given the user's feedback and the list of resources that were recommended,
identify:
1. Which resource they're reporting about (match by name or address)
2. What the issue is

Output ONLY JSON:
{"resource_name":"exact name from the list","issue":"full|closed|wrong_address|unsafe|not_helpful|other","detail":"brief description"}

If you can't identify which resource, use: {"resource_name":"unknown","issue":"other","detail":"..."}"""


def parse_feedback(feedback_text: str, recommended_resources: list[dict]) -> dict:
    """
    Parse user feedback to identify which resource has an issue.

    Parameters
    ----------
    feedback_text : str — e.g. "The shelter at 66 Boerum Place is full"
    recommended_resources : list of dicts with 'name' and 'address' keys

    Returns
    -------
    {"resource_name": str, "issue": str, "detail": str}
    """
    # Build resource list for context
    resource_list = "\n".join(
        f"- {r.get('name', '?')} at {r.get('address', '?')}"
        for r in recommended_resources
    )

    messages = [
        {"role": "system", "content": FEEDBACK_PROMPT},
        {"role": "user", "content": (
            f"Recommended resources:\n{resource_list}\n\n"
            f"User feedback: {feedback_text}"
        )},
    ]

    raw = chat(messages, temperature=0.0, max_tokens=200)

    # Parse JSON
    import json
    try:
        raw = re.sub(r'```(?:json)?\s*', '', raw or "").strip()
        raw = re.sub(r'```\s*$', '', raw).strip()
        start = raw.find('{')
        if start >= 0:
            end = raw.find('}', start)
            if end >= 0:
                return json.loads(raw[start:end+1])
    except Exception:
        pass

    # Fallback: try to match resource name directly
    feedback_lower = feedback_text.lower()
    for r in recommended_resources:
        name = r.get("name", "").lower()
        addr = r.get("address", "").lower()
        if (name and name in feedback_lower) or (addr and addr in feedback_lower):
            issue = "full" if any(w in feedback_lower for w in ["full", "no beds", "no room", "capacity"]) else \
                    "closed" if any(w in feedback_lower for w in ["closed", "shut", "not open"]) else \
                    "wrong_address" if any(w in feedback_lower for w in ["wrong", "not there", "doesn't exist", "moved"]) else \
                    "unsafe" if any(w in feedback_lower for w in ["unsafe", "dangerous", "scary"]) else "other"
            return {"resource_name": r.get("name", "unknown"), "issue": issue, "detail": feedback_text}

    return {"resource_name": "unknown", "issue": "other", "detail": feedback_text}


def get_excluded_resources(session_state) -> list[str]:
    """Get list of excluded resource names from session state."""
    return session_state.get("excluded_resources", [])


def add_exclusion(session_state, resource_name: str, issue: str, detail: str):
    """Add a resource to the exclusion list."""
    if "excluded_resources" not in session_state:
        session_state["excluded_resources"] = []
    if "feedback_log" not in session_state:
        session_state["feedback_log"] = []

    if resource_name not in session_state["excluded_resources"]:
        session_state["excluded_resources"].append(resource_name)
        session_state["feedback_log"].append({
            "resource": resource_name,
            "issue": issue,
            "detail": detail,
        })


def filter_excluded(df: pd.DataFrame, excluded: list[str]) -> pd.DataFrame:
    """Remove excluded resources from a DataFrame."""
    if not excluded or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "name" in df.columns:
        return df[~df["name"].isin(excluded)].reset_index(drop=True)
    return df


def generate_alternative_response(original_query: str, feedback: dict,
                                   excluded: list[str]) -> str:
    """
    Generate a response that acknowledges the feedback and explains
    what we're doing about it.
    """
    issue_messages = {
        "full": f"I understand that **{feedback['resource_name']}** is at capacity.",
        "closed": f"Thank you for letting me know that **{feedback['resource_name']}** is currently closed.",
        "wrong_address": f"I'm sorry — the address for **{feedback['resource_name']}** appears to be incorrect in our records.",
        "unsafe": f"I've noted your safety concern about **{feedback['resource_name']}**. Your safety is the top priority.",
        "other": f"I've noted the issue with **{feedback['resource_name']}**.",
    }

    msg = issue_messages.get(feedback.get("issue", "other"),
                             f"I've noted the issue with **{feedback.get('resource_name', 'that resource')}**.")

    msg += (f" I'm excluding it from recommendations and searching for alternatives. "
            f"({len(excluded)} resource(s) excluded so far.)")

    return msg
