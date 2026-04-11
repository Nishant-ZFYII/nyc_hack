"""
pipeline/case_notify.py — Discord webhook follow-up notifications.

Sends an embed to a Discord channel after a configurable delay.
No bot or OAuth required — just a webhook URL.
"""
import os
import threading
import time
from datetime import datetime

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# Guards against duplicate scheduling within the same process
_scheduled_cases: set = set()
_lock = threading.Lock()


def schedule_followup(case: dict, delay_minutes: int = 30,
                      webhook_url: str = "") -> bool:
    """
    Schedule a Discord follow-up embed after delay_minutes.
    Returns True if scheduled, False if skipped (already scheduled or no webhook).
    """
    url = webhook_url or DISCORD_WEBHOOK_URL
    if not url:
        return False

    case_id = case.get("case_id", "")
    with _lock:
        if case_id in _scheduled_cases:
            return False
        _scheduled_cases.add(case_id)

    def _run():
        time.sleep(delay_minutes * 60)
        payload = {"embeds": [_build_embed(case)]}
        _post_webhook(payload, url)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return True


def _build_embed(case: dict) -> dict:
    """Build a Discord embed dict from a case."""
    name = case.get("name", "User")
    case_id = case.get("case_id", "")
    needs = case.get("needs", [])
    open_needs = [n for n in needs if n.get("status") == "open"]
    all_resolved = len(needs) > 0 and len(open_needs) == 0

    color = 0x76B900 if all_resolved else 0xFF6347  # NVIDIA green or tomato red

    if all_resolved:
        description = "All your needs have been addressed. Checking in to make sure everything is still okay."
    elif open_needs:
        description = f"You have **{len(open_needs)}** open need(s) that still need attention."
    else:
        description = "Following up on your NYC services session."

    fields = []
    for n in sorted(open_needs, key=lambda x: x.get("priority", 99)):
        fields.append({
            "name": f"{'🔴' if n.get('status') == 'open' else '🟡'} {n['category'].replace('_', ' ').title()}",
            "value": f"Priority: {n.get('priority', '?')} · Status: {n.get('status', 'open')}",
            "inline": True,
        })

    last_visit = case.get("last_visit", "")
    if last_visit:
        try:
            dt = datetime.fromisoformat(last_visit)
            last_visit_str = dt.strftime("%b %d, %Y at %I:%M %p")
        except Exception:
            last_visit_str = last_visit[:16]
    else:
        last_visit_str = "Unknown"

    # Surface any in-progress destination intents
    terminal = {"resolved", "cancelled"}
    active_dests = [
        i for i in case.get("destination_intents", [])
        if i.get("state") not in terminal
    ]
    for dest in active_dests[:2]:
        fields.append({
            "name": f"🚶 You were heading to: {dest.get('resource_name','')}",
            "value": f"Status: {dest.get('state','').replace('_',' ').title()} · Did you receive help there?",
            "inline": False,
        })

    return {
        "title": f"NYC Services Follow-Up: {name}",
        "description": description,
        "color": color,
        "fields": fields,
        "footer": {
            "text": f"Re-enter your case ID to continue: {case_id} · Last session: {last_visit_str}"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _post_webhook(payload: dict, url: str) -> bool:
    """POST to Discord webhook. Returns True on success."""
    if not _HAS_REQUESTS or not url:
        return False
    try:
        resp = _requests.post(url, json=payload, timeout=10)
        return resp.status_code in (200, 204)
    except Exception:
        return False
