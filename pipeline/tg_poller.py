"""
pipeline/tg_poller.py — Telegram long-poll listener for EC self-registration.

When an emergency contact clicks the deep link and presses Start, this
poller captures their chat_id and saves it to the case JSON so the bot
can DM them directly in future notifications.
"""
from __future__ import annotations

import threading
import time

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

_TG_BASE = "https://api.telegram.org/bot{token}"

_poll_started = False
_poll_lock = threading.Lock()


def start_polling(bot_token: str):
    """Start the background long-poll daemon (no-op if already running)."""
    global _poll_started
    if not _HAS_REQUESTS or not bot_token:
        return
    with _poll_lock:
        if _poll_started:
            return
        _poll_started = True
    t = threading.Thread(target=_poll_loop, args=(bot_token,), daemon=True)
    t.start()


def _poll_loop(bot_token: str):
    base = _TG_BASE.format(token=bot_token)
    offset = 0
    while True:
        try:
            r = _req.get(
                f"{base}/getUpdates",
                params={"offset": offset, "timeout": 30,
                        "allowed_updates": ["message"]},
                timeout=35,
            )
            if r.status_code != 200:
                time.sleep(5)
                continue
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                _handle_update(upd, bot_token)
        except Exception:
            time.sleep(5)


def _handle_update(upd: dict, bot_token: str):
    msg = upd.get("message", {})
    text = (msg.get("text") or "").strip()
    chat_id = msg.get("chat", {}).get("id")
    if not chat_id:
        return
    # Deep link payload: /start ec_<case_id>
    if text.startswith("/start ec_"):
        case_id = text[len("/start ec_"):]
        if case_id:
            _register_ec(case_id, chat_id, bot_token)


def _register_ec(case_id: str, chat_id: int, bot_token: str):
    """Save EC's chat_id to case JSON and confirm via DM."""
    try:
        from pipeline.cases import load_case, _save_case
        case = load_case(case_id)
        if not case:
            return
        ec = case.get("emergency_contact", {})
        if isinstance(ec, dict):
            ec["telegram_chat_id"] = chat_id
            case["emergency_contact"] = ec
            _save_case(case)
        name = case.get("name", "someone")
        _req.post(
            f"{_TG_BASE.format(token=bot_token)}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": (
                    f"You are now registered as the emergency contact for "
                    f"<b>{name}</b> on NYC Help Finder.\n\n"
                    f"You will receive a message here if they confirm they are "
                    f"heading to a service location."
                ),
                "parse_mode": "HTML",
            },
            timeout=10,
        )
    except Exception:
        pass
