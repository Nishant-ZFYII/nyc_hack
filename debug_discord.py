"""
Run this on the DGX to diagnose Discord DM issues:
  python debug_discord.py <ec_username>
"""
import sys
import os
import requests

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pip install tomli on Python 3.9

_secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
try:
    with open(_secrets_path, "rb") as _f:
        _secrets = tomllib.load(_f)
except Exception:
    _secrets = {}

BOT_TOKEN        = _secrets.get("DISCORD_BOT_TOKEN", os.environ.get("DISCORD_BOT_TOKEN", ""))
COORD_CHANNEL_ID = str(_secrets.get("DISCORD_COORD_CHANNEL_ID", os.environ.get("DISCORD_COORD_CHANNEL_ID", "")))
EC_USERNAME      = sys.argv[1] if len(sys.argv) > 1 else "testuser"

if not BOT_TOKEN:
    print("ERROR: DISCORD_BOT_TOKEN not found in .streamlit/secrets.toml or env")
    sys.exit(1)
if not COORD_CHANNEL_ID:
    print("ERROR: DISCORD_COORD_CHANNEL_ID not found in .streamlit/secrets.toml or env")
    sys.exit(1)

print(f"Using channel ID: {COORD_CHANNEL_ID}")
print(f"Looking up EC username: {EC_USERNAME}")

HEADERS = {"Authorization": f"Bot {BOT_TOKEN}", "Content-Type": "application/json"}

def check(label, r):
    print(f"\n{'='*50}")
    print(f"Step: {label}")
    print(f"Status: {r.status_code}")
    try:
        print(f"Body:   {r.json()}")
    except Exception:
        print(f"Body:   {r.text}")
    return r.status_code in (200, 201)

# Step 1 — verify bot identity
r = requests.get("https://discord.com/api/v10/users/@me", headers=HEADERS, timeout=10)
ok = check("Bot identity", r)
if not ok:
    print("\nFAIL: Bot token is invalid.")
    sys.exit(1)

# Step 2 — fetch channel to get guild_id
r = requests.get(f"https://discord.com/api/v10/channels/{COORD_CHANNEL_ID}", headers=HEADERS, timeout=10)
ok = check("Fetch channel → guild_id", r)
if not ok:
    print("\nFAIL: Bot cannot see that channel. Is it in the server?")
    sys.exit(1)
guild_id = r.json().get("guild_id", "")
print(f"\nGuild ID resolved: {guild_id}")

# Step 3 — search guild members for EC username
r = requests.get(
    f"https://discord.com/api/v10/guilds/{guild_id}/members/search",
    params={"query": EC_USERNAME, "limit": 5},
    headers=HEADERS, timeout=10,
)
ok = check(f"Search guild members for '{EC_USERNAME}'", r)
if not ok:
    print("\nFAIL: Member search failed.")
    print("Check: Server Members Intent enabled at discord.com/developers/applications ?")
    sys.exit(1)

members = r.json()
user_id = None
for m in members:
    u = m.get("user", {})
    if (u.get("username", "").lower() == EC_USERNAME.lower() or
            u.get("global_name", "").lower() == EC_USERNAME.lower()):
        user_id = u["id"]
        print(f"\nFound user: {u.get('username')} (ID: {user_id})")
        break

if not user_id:
    print(f"\nFAIL: '{EC_USERNAME}' not found in guild.")
    print(f"Members returned: {[m.get('user', {}).get('username') for m in members]}")
    sys.exit(1)

# Step 4 — open DM channel
r = requests.post(
    "https://discord.com/api/v10/users/@me/channels",
    json={"recipient_id": user_id},
    headers=HEADERS, timeout=10,
)
ok = check("Open DM channel", r)
if not ok:
    print("\nFAIL: Could not open DM. User may have DMs disabled.")
    sys.exit(1)
dm_channel_id = r.json().get("id")

# Step 5 — send test message
r = requests.post(
    f"https://discord.com/api/v10/channels/{dm_channel_id}/messages",
    json={"content": "Test from NYC Help Finder — Discord DM flow is working!"},
    headers=HEADERS, timeout=10,
)
ok = check("Send DM", r)
if ok:
    print(f"\nSUCCESS: DM sent to {EC_USERNAME}!")
else:
    print("\nFAIL: Could not send DM.")
