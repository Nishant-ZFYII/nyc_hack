"""
pipeline/case_notify.py — Telegram follow-up notifications.

Sends a case status message to the coordination group after a configurable delay.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

TELEGRAM_BOT_TOKEN     = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_COORD_CHAT_ID = os.environ.get("TELEGRAM_COORD_CHAT_ID", "")

_TG_BASE = "https://api.telegram.org/bot{token}"

_scheduled_cases: set = set()
_lock = threading.Lock()


def schedule_followup(case: dict, delay_minutes: int = 30,
                      webhook_url: str = "", coord_chat_id: str = "") -> bool:
    """
    Schedule a Telegram follow-up message to the coord group after delay_minutes.
    webhook_url is ignored (kept for API compatibility).
    Returns True if scheduled, False if skipped.
    """
    token = TELEGRAM_BOT_TOKEN
    chat_id = str(coord_chat_id or TELEGRAM_COORD_CHAT_ID)
    if not token or not chat_id:
        return False

    case_id = case.get("case_id", "")
    with _lock:
        if case_id in _scheduled_cases:
            return False
        _scheduled_cases.add(case_id)

    def _run():
        time.sleep(delay_minutes * 60)
        text = _build_followup_text(case)
        _tg_send(token, chat_id, text)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return True


def _build_followup_text(case: dict) -> str:
    name = case.get("name", "User")
    case_id = case.get("case_id", "")
    needs = case.get("needs", [])
    open_needs = [n for n in needs if n.get("status") == "open"]
    all_resolved = len(needs) > 0 and len(open_needs) == 0

    last_visit = case.get("last_visit", "")
    if last_visit:
        try:
            dt = datetime.fromisoformat(last_visit)
            last_visit_str = dt.strftime("%b %d, %Y at %I:%M %p")
        except Exception:
            last_visit_str = last_visit[:16]
    else:
        last_visit_str = "Unknown"

    if all_resolved:
        status_line = "All needs have been addressed."
    elif open_needs:
        cats = ", ".join(
            n["category"].replace("_", " ").title()
            for n in sorted(open_needs, key=lambda x: x.get("priority", 99))
        )
        status_line = f"{len(open_needs)} open need(s): {cats}"
    else:
        status_line = "Following up on NYC services session."

    # Active destinations
    terminal = {"resolved", "cancelled"}
    active_dests = [
        i for i in case.get("destination_intents", [])
        if i.get("state") not in terminal
    ]
    dest_lines = ""
    for dest in active_dests[:2]:
        dest_lines += (
            f"\n• Heading to: <b>{dest.get('resource_name','')}</b> "
            f"— {dest.get('state','').replace('_',' ').title()}"
        )

    return (
        f"<b>Follow-Up: {name}</b>\n\n"
        f"{status_line}"
        f"{dest_lines}\n\n"
        f"<b>Last session:</b> {last_visit_str}\n"
        f"<b>Case ID:</b> <code>{case_id}</code>"
    )


def _tg_send(token: str, chat_id: str, text: str) -> bool:
    if not _HAS_REQUESTS or not token or not chat_id:
        return False
    try:
        r = _req.post(
            f"{_TG_BASE.format(token=token)}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200 and r.json().get("ok", False)
    except Exception:
        return False
