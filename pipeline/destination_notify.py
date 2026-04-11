"""
pipeline/destination_notify.py — Emergency contact + destination coordination.

When a user confirms "I'm going here", this module:
1. POSTs a notification embed to the destination's Discord webhook
2. POSTs a notification embed to the emergency contact's Discord webhook
3. Optionally creates a coordination thread via the Discord bot API
4. Schedules an SLA check — if the destination doesn't advance state
   within sla_minutes, fires an orange escalation embed.

All Discord steps degrade gracefully — if no config is provided the intent
is still recorded to the case JSON; only the notifications are skipped.
"""
import os
import threading
import time
from datetime import datetime

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_COORD_CHANNEL_ID = os.environ.get("DISCORD_COORD_CHANNEL_ID", "")

_sla_scheduled: set = set()
_sla_lock = threading.Lock()

# Lifecycle order — used for display; terminal states filtered in cases.py
LIFECYCLE = ["intent_confirmed", "notified", "acknowledged",
             "en_route", "arrived", "resolved"]


def confirm_destination_intent(case: dict, resource: dict,
                                config: dict = None) -> dict:
    """
    Orchestrator called when the user clicks "Yes, I'm going there".

    config keys (all optional):
        dest_webhook      — Discord webhook URL for the service provider
        ec_webhook        — Discord webhook URL override for emergency contact
        bot_token         — Discord bot token for thread creation
        coord_channel_id  — Channel ID to create the coordination thread in
        sla_minutes       — How long to wait before escalating (default 15)

    Returns:
        {notifications_sent: list, thread_id: str|None, thread_url: str|None}
    """
    cfg = config or {}
    result: dict = {"notifications_sent": [], "thread_id": None, "thread_url": None}

    case_id = case.get("case_id", "")
    resource_name = resource.get("name", "Unknown")

    # ── 1. Notify destination ──────────────────────────────────────────────────
    dest_webhook = cfg.get("dest_webhook", "")
    if dest_webhook:
        payload = {"embeds": [_build_destination_embed(case, resource)]}
        if _post_webhook(payload, dest_webhook):
            result["notifications_sent"].append("destination")

    # ── 2. Notify emergency contact via DM (only when they're actually involved) ─
    ec = case.get("emergency_contact", {})
    if isinstance(ec, str):
        ec = {"name": ec}
    ec_username = cfg.get("ec_discord_username", "") or ec.get("discord_username", "")
    bot_token_for_dm = cfg.get("bot_token", DISCORD_BOT_TOKEN)
    guild_id_for_dm = cfg.get("coord_channel_id", DISCORD_COORD_CHANNEL_ID)
    if ec_username and bot_token_for_dm:
        dm_sent = _dm_user_by_username(
            ec_username, _build_ec_embed(case, resource),
            bot_token_for_dm, guild_id_for_dm,
        )
        if dm_sent:
            result["notifications_sent"].append("emergency_contact")

    # ── 3. Create coordination thread (requires bot token + channel ID) ────────
    bot_token = cfg.get("bot_token", DISCORD_BOT_TOKEN)
    channel_id = cfg.get("coord_channel_id", DISCORD_COORD_CHANNEL_ID)
    if bot_token and channel_id:
        thread_data = _create_thread(case, resource, bot_token, channel_id)
        result["thread_id"] = thread_data.get("id")
        result["thread_url"] = thread_data.get("url")
        if result["thread_id"]:
            result["notifications_sent"].append("thread_created")
            # Write thread URL back to the case (best-effort)
            try:
                from pipeline.cases import load_case, _save_case
                c = load_case(case_id)
                if c:
                    for intent in c.get("destination_intents", []):
                        if intent.get("resource_name") == resource_name:
                            intent["thread_id"] = result["thread_id"]
                            intent["thread_url"] = result["thread_url"]
                    _save_case(c)
            except Exception:
                pass

    # ── 4. Schedule SLA check ──────────────────────────────────────────────────
    sla_minutes = int(cfg.get("sla_minutes", 15))
    if sla_minutes > 0 and (dest_webhook or ec_username):
        _schedule_sla_check(case_id, resource_name, sla_minutes, cfg)

    # Advance state to "notified" if at least one notification was sent
    if result["notifications_sent"]:
        try:
            from pipeline.cases import update_destination_state
            update_destination_state(case_id, resource_name, "notified")
        except Exception:
            pass

    return result


# ── Embed builders ────────────────────────────────────────────────────────────

def _build_destination_embed(case: dict, resource: dict) -> dict:
    """Red embed for the service provider — user is incoming."""
    name = case.get("name", "User")
    case_id = case.get("case_id", "")
    needs = case.get("needs", [])
    open_cats = [n["category"].replace("_", " ").title()
                 for n in needs if n.get("status") != "resolved"]

    return {
        "title": f"Incoming: {name} is on their way",
        "description": (
            f"A user has confirmed they are heading to **{resource.get('name', '')}**.\n\n"
            f"**Situation:** {', '.join(open_cats) if open_cats else 'General assistance needed'}"
        ),
        "color": 0xFF6347,
        "fields": [
            {"name": "Resource", "value": resource.get("name", ""), "inline": True},
            {"name": "Type",
             "value": resource.get("resource_type", resource.get("type", "")).replace("_", " ").title(),
             "inline": True},
            {"name": "Address", "value": resource.get("address", "NYC"), "inline": False},
            {"name": "Intent Time",
             "value": datetime.now().strftime("%b %d at %I:%M %p"), "inline": True},
        ],
        "footer": {"text": f"Case ID: {case_id} · NYC Help Finder"},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _build_ec_embed(case: dict, resource: dict) -> dict:
    """Blue informational embed for the emergency contact."""
    name = case.get("name", "User")
    ec = case.get("emergency_contact", {})
    if isinstance(ec, str):
        ec_name = ec
    else:
        ec_name = ec.get("name", "")

    case_id = case.get("case_id", "")
    greeting = f"Hi {ec_name}, " if ec_name else ""

    return {
        "title": f"Update: {name} is heading to get help",
        "description": (
            f"{greeting}**{name}** has confirmed they are heading to a service location "
            f"through NYC Help Finder."
        ),
        "color": 0x2196F3,
        "fields": [
            {"name": "Destination", "value": resource.get("name", ""), "inline": True},
            {"name": "Type",
             "value": resource.get("resource_type", resource.get("type", "")).replace("_", " ").title(),
             "inline": True},
            {"name": "Address", "value": resource.get("address", "NYC"), "inline": False},
            {"name": "Time",
             "value": datetime.now().strftime("%b %d at %I:%M %p"), "inline": True},
        ],
        "footer": {
            "text": (
                f"Case ID: {case_id} · "
                f"Please check in with {name} to confirm they arrived safely."
            )
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _build_thread_starter(case: dict, resource: dict) -> dict:
    """Coordination thread starter embed — three parties visible."""
    name = case.get("name", "User")
    case_id = case.get("case_id", "")
    needs = case.get("needs", [])
    open_cats = [n["category"].replace("_", " ").title()
                 for n in needs if n.get("status") != "resolved"]
    ec = case.get("emergency_contact", {})
    ec_name = ec.get("name", "") if isinstance(ec, dict) else str(ec)

    desc = "\n".join([
        f"**{name}** is heading to **{resource.get('name', '')}**.",
        "",
        f"**Open needs:** {', '.join(open_cats) if open_cats else 'General assistance'}",
        f"**Address:** {resource.get('address', 'NYC')}",
        "",
        "This thread is the single coordination layer for this interaction.",
        f"Emergency contact on file: **{ec_name or 'Not provided'}**",
    ])

    return {
        "title": f"Coordination: {name} → {resource.get('name', '')}",
        "description": desc,
        "color": 0xFF6347,
        "fields": [
            {"name": "Status", "value": "En Route", "inline": True},
            {"name": "Case ID", "value": case_id, "inline": True},
        ],
        "footer": {
            "text": "Update this thread with arrivals, acknowledgments, and outcomes."
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ── Discord bot API ───────────────────────────────────────────────────────────

def _create_thread(case: dict, resource: dict,
                   bot_token: str, channel_id: str) -> dict:
    """
    Create a Discord public thread via bot API.
    Returns {id, url} on success, {} on failure.
    """
    if not _HAS_REQUESTS or not bot_token or not channel_id:
        return {}

    name = case.get("name", "User")
    resource_name = resource.get("name", "Service")
    thread_name = f"{name} → {resource_name}"[:100]

    headers = {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }
    url = f"https://discord.com/api/v10/channels/{channel_id}/threads"
    payload = {
        "name": thread_name,
        "type": 11,               # PUBLIC_THREAD
        "auto_archive_duration": 1440,
        "message": {"embeds": [_build_thread_starter(case, resource)]},
    }

    try:
        resp = _req.post(url, json=payload, headers=headers, timeout=10)
        if resp.status_code in (200, 201):
            data = resp.json()
            thread_id = data.get("id", "")
            guild_id = data.get("guild_id", "")
            return {
                "id": thread_id,
                "url": f"https://discord.com/channels/{guild_id}/{thread_id}",
            }
    except Exception:
        pass
    return {}


# ── SLA escalation ────────────────────────────────────────────────────────────

def _schedule_sla_check(case_id: str, resource_name: str,
                         sla_minutes: int, cfg: dict):
    """
    After sla_minutes, re-reads the case. If the destination intent is still
    in intent_confirmed / notified, fires an orange escalation embed.
    """
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
                    return  # already advanced — no escalation needed
                name = case.get("name", "User")
                sla_embed = {
                    "title": f"SLA Alert — No acknowledgment from {resource_name}",
                    "description": (
                        f"**{resource_name}** has not acknowledged the incoming user "
                        f"**{name}** after {sla_minutes} minutes.\n\n"
                        "Consider suggesting alternative resources."
                    ),
                    "color": 0xFF9800,
                    "fields": [
                        {"name": "Case ID", "value": case_id, "inline": True},
                        {"name": "SLA", "value": f"{sla_minutes} min", "inline": True},
                    ],
                    "footer": {"text": "NYC Help Finder · Automated SLA check"},
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                dest_wh = cfg.get("dest_webhook", "")
                if dest_wh:
                    _post_webhook({"embeds": [sla_embed]}, dest_wh)
                # Also DM the EC if they're on file
                ec_username = cfg.get("ec_discord_username", "")
                bot_token = cfg.get("bot_token", DISCORD_BOT_TOKEN)
                guild_id = cfg.get("coord_channel_id", DISCORD_COORD_CHANNEL_ID)
                if ec_username and bot_token:
                    _dm_user_by_username(ec_username, sla_embed, bot_token, guild_id)
        except Exception:
            pass

    t = threading.Thread(target=_check, daemon=True)
    t.start()


# ── Discord DM utility ────────────────────────────────────────────────────────

def _dm_user_by_username(username: str, embed: dict,
                          bot_token: str, guild_id: str = "") -> bool:
    """
    Send a DM embed to a Discord user identified by username.

    Flow:
      1. If guild_id is set: search guild members for the username to get user ID.
      2. Open a DM channel via POST /users/@me/channels.
      3. POST the embed to that channel.

    Returns True if the DM was sent successfully.
    """
    if not _HAS_REQUESTS or not bot_token or not username:
        return False

    headers = {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }
    username = username.lstrip("@").strip()
    user_id = None

    # Step 1 — resolve username → user ID via guild member search
    if guild_id:
        try:
            r = _req.get(
                f"https://discord.com/api/v10/guilds/{guild_id}/members/search",
                params={"query": username, "limit": 5},
                headers=headers,
                timeout=10,
            )
            if r.status_code == 200:
                for member in r.json():
                    u = member.get("user", {})
                    # Match on username or global_name (display name)
                    if (u.get("username", "").lower() == username.lower() or
                            u.get("global_name", "").lower() == username.lower()):
                        user_id = u.get("id")
                        break
        except Exception:
            pass

    if not user_id:
        return False

    # Step 2 — open DM channel
    try:
        r = _req.post(
            "https://discord.com/api/v10/users/@me/channels",
            json={"recipient_id": user_id},
            headers=headers,
            timeout=10,
        )
        if r.status_code not in (200, 201):
            return False
        dm_channel_id = r.json().get("id")
    except Exception:
        return False

    # Step 3 — send the message
    try:
        r = _req.post(
            f"https://discord.com/api/v10/channels/{dm_channel_id}/messages",
            json={"embeds": [embed]},
            headers=headers,
            timeout=10,
        )
        return r.status_code in (200, 201)
    except Exception:
        return False


# ── Webhook utility ───────────────────────────────────────────────────────────

def _post_webhook(payload: dict, url: str) -> bool:
    """POST JSON payload to a Discord webhook. Returns True on success."""
    if not _HAS_REQUESTS or not url:
        return False
    try:
        resp = _req.post(url, json=payload, timeout=10)
        return resp.status_code in (200, 204)
    except Exception:
        return False
