"""
pipeline/destination_notify.py — Telegram-based emergency contact + destination coordination.

When a user confirms "I'm going here", this module:
1. Posts a message to the Telegram coordination group
2. DMs the emergency contact if they have registered their chat_id
3. Schedules an SLA check — if no state advance within sla_minutes, escalates

All steps degrade gracefully — if no config is provided the intent is still
recorded to the case JSON; only the notifications are skipped.
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
TELEGRAM_BOT_NAME      = os.environ.get("TELEGRAM_BOT_NAME", "")

_TG_BASE = "https://api.telegram.org/bot{token}"

_sla_scheduled: set = set()
_sla_lock = threading.Lock()

LIFECYCLE = ["intent_confirmed", "notified", "acknowledged",
             "en_route", "arrived", "resolved"]


# ── Public API ────────────────────────────────────────────────────────────────

def notify_ec_added(case: dict, ec_username: str,
                    bot_token: str, coord_chat_id: str) -> dict:
    """
    Called immediately when an EC username is saved during onboarding.

    - If EC already has telegram_chat_id: DMs them directly.
    - If not: generates a deep link and posts it to the coord group.

    Returns: {"dm_sent": bool, "deep_link": str | None}
    """
    result: dict = {"dm_sent": False, "deep_link": None}
    token = bot_token or TELEGRAM_BOT_TOKEN
    chat_id = str(coord_chat_id or TELEGRAM_COORD_CHAT_ID)
    if not token:
        return result

    name = case.get("name", "Someone")
    case_id = case.get("case_id", "")
    ec = case.get("emergency_contact", {})
    if isinstance(ec, str):
        ec = {}

    existing_chat_id = ec.get("telegram_chat_id")

    # EC already registered — DM directly
    if existing_chat_id:
        text = (
            f"Hi! You are the emergency contact for <b>{name}</b> on NYC Help Finder.\n\n"
            f"You will receive a message here if they confirm they are heading to a service location."
        )
        result["dm_sent"] = _tg_send(token, existing_chat_id, text)
        return result

    # EC not yet registered — generate deep link and post to coord group
    bot_name = TELEGRAM_BOT_NAME
    if not bot_name:
        # Try to fetch it from the API
        try:
            r = _req.get(f"{_TG_BASE.format(token=token)}/getMe", timeout=10)
            if r.status_code == 200:
                bot_name = r.json().get("result", {}).get("username", "")
        except Exception:
            pass

    if bot_name and case_id:
        deep_link = _deep_link_url(bot_name, f"ec_{case_id}")
        result["deep_link"] = deep_link
        if chat_id:
            ec_display = ec_username.lstrip("@") or "the emergency contact"
            text = (
                f"<b>Emergency Contact Registration</b>\n\n"
                f"<b>{name}</b> has added <b>@{ec_display}</b> as their emergency contact.\n\n"
                f"Please forward this link to them so they can register:\n"
                f"{deep_link}\n\n"
                f"Once they click Start, they will automatically receive notifications "
                f"whenever {name} confirms a destination."
            )
            _tg_send(token, chat_id, text)

    return result


def confirm_destination_intent(case: dict, resource: dict,
                                config: dict | None = None) -> dict:
    """
    Orchestrator called when the user clicks "Yes, I'm going there".

    config keys (all optional):
        bot_token      — Telegram bot token
        coord_chat_id  — Telegram group chat ID (negative int as string)
        sla_minutes    — How long before escalating (default 15)

    Returns: {"notifications_sent": list}
    """
    cfg = config or {}
    result: dict = {"notifications_sent": []}

    token = cfg.get("bot_token", TELEGRAM_BOT_TOKEN)
    coord_chat_id = str(cfg.get("coord_chat_id", TELEGRAM_COORD_CHAT_ID))
    sla_minutes = int(cfg.get("sla_minutes", 15))
    case_id = case.get("case_id", "")
    resource_name = resource.get("name", "Unknown")

    # ── 1. Post to coordination group ─────────────────────────────────────────
    if token and coord_chat_id:
        text = _build_coord_group_text(case, resource)
        if _tg_send(token, coord_chat_id, text):
            result["notifications_sent"].append("coord_group")

    # ── 2. DM emergency contact if chat_id registered ─────────────────────────
    ec = case.get("emergency_contact", {})
    if isinstance(ec, str):
        ec = {}
    ec_chat_id = ec.get("telegram_chat_id")

    if token and ec_chat_id:
        text = _build_ec_dm_text(case, resource)
        if _tg_send(token, ec_chat_id, text):
            result["notifications_sent"].append("emergency_contact")
    elif token and coord_chat_id:
        # EC hasn't registered yet — post reminder to group
        ec_user = ec.get("telegram_username", "")
        if ec_user:
            reminder = (
                f"<b>Note:</b> Emergency contact <b>@{ec_user}</b> has not yet registered "
                f"with the bot and cannot be reached directly. "
                f"Please ensure they click the registration link sent earlier."
            )
            _tg_send(token, coord_chat_id, reminder)

    # ── 3. Advance case state to "notified" ───────────────────────────────────
    if result["notifications_sent"]:
        try:
            from pipeline.cases import update_destination_state
            update_destination_state(case_id, resource_name, "notified")
        except Exception:
            pass

    # ── 4. Schedule SLA check ─────────────────────────────────────────────────
    if sla_minutes > 0 and result["notifications_sent"]:
        _schedule_sla_check(case_id, resource_name, sla_minutes, token, coord_chat_id)

    return result


# ── Message builders ──────────────────────────────────────────────────────────

def _build_coord_group_text(case: dict, resource: dict) -> str:
    name = case.get("name", "User")
    case_id = case.get("case_id", "")
    needs = case.get("needs", [])
    open_cats = [n["category"].replace("_", " ").title()
                 for n in needs if n.get("status") != "resolved"]
    resource_name = resource.get("name", "")
    address = resource.get("address", "NYC")
    rtype = resource.get("resource_type", resource.get("type", "")).replace("_", " ").title()
    now = datetime.now().strftime("%b %d at %I:%M %p")

    needs_line = ", ".join(open_cats) if open_cats else "General assistance"

    return (
        f"<b>Incoming: {name} is heading to {resource_name}</b>\n\n"
        f"<b>Needs:</b> {needs_line}\n"
        f"<b>Destination:</b> {resource_name} ({rtype})\n"
        f"<b>Address:</b> {address}\n"
        f"<b>Time:</b> {now}\n"
        f"<b>Case ID:</b> <code>{case_id}</code>"
    )


def _build_ec_dm_text(case: dict, resource: dict) -> str:
    name = case.get("name", "User")
    case_id = case.get("case_id", "")
    ec = case.get("emergency_contact", {})
    ec_name = ec.get("name", "") if isinstance(ec, dict) else ""
    greeting = f"Hi {ec_name}, " if ec_name else ""
    resource_name = resource.get("name", "")
    address = resource.get("address", "NYC")
    now = datetime.now().strftime("%b %d at %I:%M %p")

    return (
        f"{greeting}<b>{name}</b> has confirmed they are heading to "
        f"<b>{resource_name}</b> ({address}) to get help.\n\n"
        f"<b>Time:</b> {now}\n"
        f"Please check in with them to confirm they arrived safely.\n"
        f"<b>Case ID:</b> <code>{case_id}</code>"
    )


def _build_sla_text(case_id: str, resource_name: str,
                    name: str, sla_minutes: int) -> str:
    return (
        f"<b>SLA Alert</b>\n\n"
        f"<b>{resource_name}</b> has not acknowledged <b>{name}</b> "
        f"after {sla_minutes} minutes.\n\n"
        f"Consider suggesting alternative resources.\n"
        f"<b>Case:</b> <code>{case_id}</code>"
    )


# ── SLA escalation ────────────────────────────────────────────────────────────

def _schedule_sla_check(case_id: str, resource_name: str,
                         sla_minutes: int, token: str, coord_chat_id: str):
    key = f"{case_id}:{resource_name}"
    with _sla_lock:
        if key in _sla_scheduled:
            return
        _sla_scheduled.add(key)

    def _check():
        time.sleep(sla_minutes * 60)
        try:
            from pipeline.cases import load_case
            case = load_case(case_id)
            if not case:
                return
            for intent in case.get("destination_intents", []):
                if intent.get("resource_name") != resource_name:
                    continue
                if intent.get("state") not in ("intent_confirmed", "notified"):
                    return
                name = case.get("name", "User")
                sla_text = _build_sla_text(case_id, resource_name, name, sla_minutes)
                if coord_chat_id:
                    _tg_send(token, coord_chat_id, sla_text)
                ec = case.get("emergency_contact", {})
                ec_chat_id = ec.get("telegram_chat_id") if isinstance(ec, dict) else None
                if ec_chat_id:
                    _tg_send(token, ec_chat_id, sla_text)
        except Exception:
            pass

    t = threading.Thread(target=_check, daemon=True)
    t.start()


# ── Telegram utilities ────────────────────────────────────────────────────────

def _tg_send(token: str, chat_id: int | str, text: str,
              parse_mode: str = "HTML") -> bool:
    """POST a message to a Telegram chat. Returns True on success."""
    if not _HAS_REQUESTS or not token or not chat_id:
        return False
    try:
        r = _req.post(
            f"{_TG_BASE.format(token=token)}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode},
            timeout=10,
        )
        return r.status_code == 200 and r.json().get("ok", False)
    except Exception:
        return False


def _deep_link_url(bot_name: str, payload: str) -> str:
    return f"https://t.me/{bot_name}?start={payload}"
