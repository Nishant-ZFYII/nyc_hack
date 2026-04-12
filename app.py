"""
app.py — NYC Social Services Intelligence Engine (Streamlit demo)

Run: streamlit run app.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))

from pipeline.planner  import generate_plan
from pipeline.executor import execute, load_state
from pipeline.synth    import answer
from pipeline.clarify  import get_clarifying_question, merge_query
from pipeline.verify   import verify_answer, build_reasoning_path, summarize_reasoning
from pipeline.feedback import parse_feedback, add_exclusion, get_excluded_resources, generate_alternative_response
from pipeline.cases    import (create_case, load_case, add_visit, mark_resource_visited,
                                update_need_status, sync_needs_from_plan,
                                add_destination_intent, update_destination_state,
                                get_active_destinations)
from pipeline.case_notify      import schedule_followup
from pipeline.destination_notify import (confirm_destination_intent,
                                          notify_ec_added,
                                          DISCORD_BOT_TOKEN, DISCORD_COORD_CHANNEL_ID)
from llm.client        import get_active_provider
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "kg_embeddings",
    str(ROOT / "engine" / "embeddings.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
find_similar = _mod.find_similar
find_similar_to_query = _mod.find_similar_to_query
build_embeddings = _mod.build_embeddings

# ── Map helpers ───────────────────────────────────────────────────────────────

# Color per resource type (RGBA)
RESOURCE_COLORS = {
    "shelter":          [255, 99,  71,  220],  # tomato red
    "food_bank":        [255, 165,  0,  220],  # orange
    "hospital":         [0,   191, 255, 220],  # sky blue
    "benefits_center":  [0,   255, 127, 220],  # spring green
    "domestic_violence":[255, 105, 180, 220],  # hot pink
    "school":           [255, 215,   0, 220],  # gold
    "senior_services":  [148,   0, 211, 220],  # purple
    "childcare":        [0,   255, 255, 220],  # cyan
    "community_center": [50,  205,  50, 220],  # lime
    "transit_station":  [180, 180, 180, 200],  # grey
}
DEFAULT_COLOR = [200, 200, 200, 200]

NYC_CENTER = {"lat": 40.7128, "lon": -74.0060, "zoom": 11}


def make_resource_map(frames: list[pd.DataFrame]) -> pdk.Deck:
    """Build a PyDeck scatter map from one or more result DataFrames."""
    rows = []
    for df in frames:
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for _, r in df.iterrows():
            lat = r.get("latitude")
            lon = r.get("longitude")
            if lat is None or lon is None:
                continue
            try:
                lat, lon = float(lat), float(lon)
            except (ValueError, TypeError):
                continue
            rtype = r.get("resource_type", "")
            rows.append({
                "lat": lat, "lon": lon,
                "name": r.get("name", ""),
                "type": rtype,
                "address": r.get("address", ""),
                "borough": r.get("borough", ""),
                "color": RESOURCE_COLORS.get(rtype, DEFAULT_COLOR),
            })

    if not rows:
        return None

    df_map = pd.DataFrame(rows)
    # Center on mean of results
    center_lat = df_map["lat"].mean()
    center_lon = df_map["lon"].mean()

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_line_color=[0, 0, 0, 180],
        get_radius=120,
        radius_min_pixels=6,
        radius_max_pixels=18,
        pickable=True,
        stroked=True,
        line_width_min_pixels=1,
    )

    view = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=12,
        pitch=0,
    )

    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{name}\n{type}\n{address}\n{borough}"},
    )


def render_map_legend(resource_types: list[str]):
    """Inline color dot legend."""
    items = []
    for rtype in resource_types:
        color = RESOURCE_COLORS.get(rtype, DEFAULT_COLOR)
        items.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;">'
            f'<span style="color:rgb({color[0]},{color[1]},{color[2]});font-size:18px;">●</span>'
            f'<span style="color:#ccc;font-size:12px;">{rtype.replace("_"," ")}</span></span>'
        )
    st.markdown('<div style="padding:4px 0">' + "".join(items) + "</div>", unsafe_allow_html=True)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Social Services Intelligence Engine",
    page_icon="🗽",
    layout="wide",
)

# ── Dark NVIDIA Theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* NVIDIA Dark Theme */
[data-testid="stAppViewContainer"] {
    background-color: #0a0a0f;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #111118;
    border-right: 1px solid #1e1e2a;
}
[data-testid="stHeader"] {
    background-color: #0a0a0f;
}

/* NVIDIA Green accent */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #76b900, #5a8f00);
    color: white;
    border: none;
    font-weight: bold;
    box-shadow: 0 2px 8px rgba(118, 185, 0, 0.3);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8cd600, #76b900);
    box-shadow: 0 4px 12px rgba(118, 185, 0, 0.5);
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background-color: #1a1a24;
    color: #e0e0e0;
    border: 1px solid #2a2a3a;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #76b900;
    color: #76b900;
    transform: translateX(2px);
}

/* Title styling */
h1 {
    color: #76b900 !important;
    text-shadow: 0 0 20px rgba(118, 185, 0, 0.3);
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #12121a;
    border: 1px solid #1e1e2a;
    border-radius: 12px;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #12121a;
    border: 1px solid #1e1e2a;
    border-radius: 8px;
}

/* Text inputs */
textarea, input[type="text"] {
    background-color: #12121a !important;
    color: #e0e0e0 !important;
    border: 1px solid #2a2a3a !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #76b900 !important;
    box-shadow: 0 0 8px rgba(118, 185, 0, 0.2) !important;
}

/* Success/info/warning boxes */
[data-testid="stAlert"] {
    background-color: #12121a;
    border: 1px solid #2a2a3a;
}

/* Metrics / stats */
[data-testid="stMetric"] {
    background-color: #12121a;
    border: 1px solid #1e1e2a;
    border-radius: 8px;
    padding: 12px;
}
[data-testid="stMetricValue"] {
    color: #76b900 !important;
}

/* Dividers */
hr {
    border-color: #1e1e2a !important;
}

/* Caption text */
.stCaption, small {
    color: #888 !important;
}

/* Toggle */
[data-testid="stToggle"] label span {
    color: #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:20px 0 10px 0;">
    <div style="font-size:42px;font-weight:bold;color:#76b900;">Need Help?</div>
    <div style="font-size:18px;color:#aaa;margin-top:4px;">Tell us what's going on. We'll find the right resources for you.</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Preload data silently
    try:
        mart, payload = load_state()
    except Exception:
        mart, payload = None, {}

    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px 0;">
        <div style="font-size:24px;font-weight:bold;color:#76b900;">NYC Help Finder</div>
        <div style="font-size:12px;color:#888;margin-top:2px;">Shelters · Food · Healthcare · Benefits</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1a1a24;border:1px solid #2a2a3a;border-radius:10px;padding:14px;margin:8px 0;">
        <div style="color:#ccc;font-size:14px;line-height:1.7;">
            Just describe what you need — in your own words. We'll find shelters, food, hospitals, benefits, schools, and more near you.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="color:#76b900;font-size:12px;text-transform:uppercase;letter-spacing:1px;">I need help with...</div>', unsafe_allow_html=True)

    EXAMPLES = {
        "I'm about to lose my housing": "I'm Tina, I have 4 kids ages 12-16, my income is $28K, and my sister is kicking us out of her Flatbush apartment next week. What do we do?",
        "I don't feel safe at home": "Someone broke into my apartment last night. I don't feel safe going back. I have a 6 year old. What should I do?",
        "I just arrived in NYC": "I just arrived from Haiti with my two children. We speak Haitian Creole and need shelter tonight near Flatbush.",
        "I need a shelter tonight": "What shelters in Brooklyn have available beds right now?",
        "I need to see a doctor": "Find wheelchair accessible hospitals near Jamaica Queens that accept Medicaid",
        "I need food for my family": "How many food banks are open in the Bronx?",
    }

    for label, eq in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["query_input"] = eq

    st.divider()
    st.markdown('<div style="color:#76b900;font-size:12px;text-transform:uppercase;letter-spacing:1px;">For coordinators</div>', unsafe_allow_html=True)

    OPS_EXAMPLES = {
        "Cold emergency — people outside": "A cold emergency is declared. 3 Brooklyn shelters just hit capacity. 200 people are still outside. It's 15°F. What do we do?",
        "Which areas need more resources?": "Which NYC boroughs are most underserved by social services?",
        "What if we add shelter beds?": "What happens if we add 500 shelter beds in the Bronx?",
        "Migrant bus just arrived": "A migrant bus just arrived with 80 people who speak Spanish and Mandarin. They need shelter, food, and schools for children.",
    }

    for label, eq in OPS_EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["query_input"] = eq

    st.divider()
    demo_mode = st.toggle("Fast Mode", value=False,
                          help="Skip claim verification for faster responses.")
    st.session_state["demo_mode"] = demo_mode

    if st.button("Start Over", use_container_width=True):
        # Clears query state but NOT case state
        for k in ["conv_history", "active_query", "_pending_clarify_q",
                  "_pending_clarify_turn", "_clarify_rerun", "query_input",
                  "excluded_resources", "feedback_log", "_followup_scheduled"]:
            st.session_state.pop(k, None)
        st.rerun()

    # ── Case info ────────────────────────────────────────────────────────────
    if st.session_state.get("case_id"):
        _sc = st.session_state.get("case", {})
        _sc_name = _sc.get("name", st.session_state["case_id"])
        _sc_open = len([n for n in _sc.get("needs", []) if n.get("status") == "open"])
        st.divider()
        st.markdown(f"""
        <div style="background:#1a1a24;border:1px solid #2a2a3a;border-radius:8px;padding:12px;margin:4px 0;">
            <div style="color:#76b900;font-size:12px;text-transform:uppercase;letter-spacing:1px;">Active Case</div>
            <div style="color:#e0e0e0;font-weight:bold;margin:4px 0;">{_sc_name}</div>
            <div style="color:#888;font-size:11px;font-family:monospace;">{st.session_state["case_id"]}</div>
            <div style="color:#aaa;font-size:12px;margin-top:4px;">{_sc_open} open need(s)</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Log Out", use_container_width=True, key="_sidebar_logout"):
            for k in ["case_id", "case", "onboarding_done", "show_welcome_back",
                      "_followup_scheduled", "conv_history", "active_query",
                      "_pending_clarify_q", "_pending_clarify_turn", "_clarify_rerun",
                      "query_input", "excluded_resources", "feedback_log"]:
                st.session_state.pop(k, None)
            st.rerun()
        with st.expander("Discord Follow-Up", expanded=False):
            dw = st.text_input("Webhook URL", type="password", key="_discord_webhook",
                               placeholder="https://discord.com/api/webhooks/...")
            if dw:
                st.session_state["_discord_webhook_url"] = dw

        with st.expander("Discord Coordination", expanded=False):
            st.caption("For 'I'm going here' notifications and thread creation.")
            _dest_wh = st.text_input("Destination webhook URL", type="password",
                                     key="_dest_webhook_input",
                                     placeholder="Webhook for the service provider")
            _bot_tok = st.text_input("Bot token", type="password",
                                     key="_bot_token_input",
                                     placeholder="Discord bot token")
            _coord_ch = st.text_input("Coordination channel ID", key="_coord_channel_input",
                                      placeholder="Channel ID for coordination threads")
            _sla = st.number_input("SLA (minutes before escalation)", min_value=1,
                                   max_value=120, value=15, key="_sla_min_input")
            if _dest_wh:
                st.session_state["_dest_webhook"] = _dest_wh
            if _bot_tok:
                st.session_state["_bot_token"] = _bot_tok
            if _coord_ch:
                st.session_state["_coord_channel"] = _coord_ch
            st.session_state["_sla_min"] = int(_sla)

    # Tech details tucked away
    with st.expander("System Info", expanded=False):
        try:
            st.caption(f"LLM: {get_active_provider()}")
        except Exception:
            pass
        try:
            _mc = len(mart) if hasattr(mart, '__len__') else 0
            _ec = payload.get("edges")
            _ec = len(_ec) if _ec is not None else 0
            st.caption(f"Resources: {_mc:,} | Graph: {_ec:,} edges")
            st.caption(f"Backend: {payload.get('backend', 'networkx')}")
        except Exception:
            pass


# ── Pre-populate Discord config from Streamlit secrets (once per session) ────
if "_discord_secrets_loaded" not in st.session_state:
    try:
        if "DISCORD_DEST_WEBHOOK" in st.secrets and not st.session_state.get("_dest_webhook"):
            st.session_state["_dest_webhook"] = st.secrets["DISCORD_DEST_WEBHOOK"]
        if "DISCORD_BOT_TOKEN" in st.secrets and not st.session_state.get("_bot_token"):
            st.session_state["_bot_token"] = st.secrets["DISCORD_BOT_TOKEN"]
        if "DISCORD_COORD_CHANNEL_ID" in st.secrets and not st.session_state.get("_coord_channel"):
            st.session_state["_coord_channel"] = st.secrets["DISCORD_COORD_CHANNEL_ID"]
        if "DISCORD_FOLLOWUP_WEBHOOK" in st.secrets and not st.session_state.get("_discord_webhook_url"):
            st.session_state["_discord_webhook_url"] = st.secrets["DISCORD_FOLLOWUP_WEBHOOK"]
    except Exception:
        pass
    st.session_state["_discord_secrets_loaded"] = True

# ── Block A: Onboarding Phase Gate ───────────────────────────────────────────
if not st.session_state.get("onboarding_done"):
    st.markdown("""
    <div style="max-width:640px;margin:0 auto;padding:20px 0;">
        <div style="font-size:22px;font-weight:bold;color:#76b900;margin-bottom:8px;">
            Welcome to NYC Help Finder
        </div>
        <div style="color:#aaa;margin-bottom:24px;">
            Sign in to track your case across visits, or continue as a guest.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_new, col_ret = st.columns(2)

    with col_new:
        st.markdown("**New? Create a case**")
        new_name = st.text_input("Your first name (required)", key="_ob_name",
                                 placeholder="e.g. Maria")
        new_ec_name = st.text_input("Emergency contact name (optional)", key="_ob_ec_name",
                                    placeholder="e.g. Ana (sister)")
        new_ec_discord = st.text_input("Their Discord username (optional)",
                                       key="_ob_ec_discord",
                                       placeholder="e.g. ana_nyc  (no @ needed)")
        if st.button("Get Started", type="primary", use_container_width=True, key="_ob_create"):
            if not new_name.strip():
                st.error("Please enter your first name.")
            else:
                import uuid
                cid = new_name.strip().lower().replace(" ", "_") + "_" + uuid.uuid4().hex[:6]
                case = create_case(cid, name=new_name.strip())
                if new_ec_name.strip() or new_ec_discord.strip():
                    _ec_user = new_ec_discord.strip().lstrip("@")
                    case["emergency_contact"] = {
                        "name": new_ec_name.strip(),
                        "discord_username": _ec_user,
                    }
                    from pipeline.cases import _save_case
                    _save_case(case)
                    # Notify EC immediately and generate invite
                    if _ec_user:
                        _bot = st.session_state.get("_bot_token", DISCORD_BOT_TOKEN)
                        _ch  = st.session_state.get("_coord_channel", DISCORD_COORD_CHANNEL_ID)
                        _ec_result = notify_ec_added(case, _ec_user, _bot, str(_ch))
                        if _ec_result.get("invite_url"):
                            st.session_state["_ec_invite_url"] = _ec_result["invite_url"]
                        if _ec_result.get("dm_sent"):
                            st.session_state["_ec_dm_sent"] = True
                st.session_state["case_id"] = cid
                st.session_state["case"] = case
                st.session_state["onboarding_done"] = True
                st.rerun()

    with col_ret:
        st.markdown("**Returning? Resume your case**")
        ret_id = st.text_input("Enter your case ID", key="_ob_resume_id",
                               placeholder="e.g. maria_a1b2c3")
        if st.button("Resume", use_container_width=True, key="_ob_resume"):
            if not ret_id.strip():
                st.error("Please enter your case ID.")
            else:
                existing = load_case(ret_id.strip())
                if existing:
                    st.session_state["case_id"] = existing["case_id"]
                    st.session_state["case"] = existing
                    st.session_state["onboarding_done"] = True
                    st.session_state["show_welcome_back"] = True
                    st.rerun()
                else:
                    st.error(f"No case found with ID '{ret_id.strip()}'. Check your ID and try again.")

    st.markdown("---")
    if st.button("Continue without signing in (guest mode)", use_container_width=True, key="_ob_skip"):
        st.session_state["onboarding_done"] = True
        st.rerun()

    st.stop()

# ── Block B: Sidebar Case Info ────────────────────────────────────────────────
# (added inside the sidebar block below)

# ── Block E: Welcome-Back Summary ────────────────────────────────────────────
if st.session_state.get("show_welcome_back"):
    _case = st.session_state.get("case", {})
    _name = _case.get("name", "")
    _needs = _case.get("needs", [])
    _open = [n for n in _needs if n.get("status") == "open"]
    _resolved = [n for n in _needs if n.get("status") == "resolved"]
    _visits = _case.get("visits", [])

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a2a0a,#0a1a0a);border:1px solid #76b900;
        border-radius:12px;padding:20px;margin:8px 0;">
        <div style="font-size:20px;font-weight:bold;color:#76b900;">
            Welcome back, {_name}!
        </div>
    """, unsafe_allow_html=True)

    if _open:
        cats = ", ".join(
            f"**{n['category'].replace('_',' ').title()}** (P{n.get('priority','?')})"
            for n in sorted(_open, key=lambda x: x.get("priority", 99))
        )
        st.markdown(f"**{len(_open)} open need(s):** {cats}", unsafe_allow_html=True)
    if _resolved:
        st.markdown(f"**{len(_resolved)} resolved:** {', '.join(n['category'] for n in _resolved)}")
    if _visits:
        last = _visits[-1]
        ts = last.get("timestamp", "")[:10]
        q = last.get("query", "")[:80]
        st.markdown(f"**Last visit:** {ts} — *\"{q}\"*")

    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.pop("show_welcome_back", None)

# ── EC notification status (shown once after onboarding) ─────────────────────
st.session_state.pop("_ec_invite_url", None)
_ec_dm_sent = st.session_state.pop("_ec_dm_sent", None)
if _ec_dm_sent:
    _ec_name = st.session_state.get("case", {}).get(
        "emergency_contact", {}).get("name", "your emergency contact")
    st.success(f"Discord DM sent to {_ec_name} — they've been notified.")

# ── Main query interface ──────────────────────────────────────────────────────
query = st.text_area(
    "What's going on?",
    value=st.session_state.get("query_input", ""),
    height=120,
    placeholder="Tell us in your own words... e.g. 'I have 4 kids and we're about to lose our apartment. I make $28K a year. What help is there?'",
    key="query_input",
)

run = st.button("Find Help For Me", type="primary", use_container_width=True)

# ── Detect clarification answer submitted via Enter key ──────────────────────
# Must happen BEFORE the pipeline block so the rerun flag is set first.
_pending_q    = st.session_state.get("_pending_clarify_q", "")
_pending_turn = st.session_state.get("_pending_clarify_turn", -1)
if _pending_q and _pending_turn >= 0:
    _cval = st.session_state.get(f"clarify_{_pending_turn}", "").strip()
    _history = st.session_state.setdefault("conv_history", [])
    if _cval and len(_history) == _pending_turn:
        _history.append((_pending_q, _cval))
        st.session_state.active_query = merge_query(
            st.session_state.get("active_query", query), _pending_q, _cval
        )
        st.session_state["_clarify_rerun"] = True
        st.session_state.pop("_pending_clarify_q", None)
        st.session_state.pop("_pending_clarify_turn", None)
        st.rerun()

_clarify_rerun = st.session_state.pop("_clarify_rerun", False)

if (run or _clarify_rerun) and query.strip():
    t0 = time.time()

    # Use enriched query for clarification reruns, original for fresh runs
    effective_query = st.session_state.get("active_query", query) if _clarify_rerun else query

    # ── Session state: reset on manual Run, keep on clarify rerun ────────────
    if "conv_history" not in st.session_state:
        st.session_state.conv_history = []
    if "active_query" not in st.session_state:
        st.session_state.active_query = ""
    if run:
        st.session_state.conv_history = []
        st.session_state.active_query = query

    # ── Apply any user feedback exclusions ──────────────────────────────────
    from pipeline.executor import set_excluded_resources
    set_excluded_resources(get_excluded_resources(st.session_state))

    # ── Run pipeline (plan → execute → synthesize) ────────────────────────────
    with st.spinner("Analyzing situation…"):
        t1 = time.time()
        try:
            plan = generate_plan(effective_query)
            plan_time = time.time() - t1
        except Exception as e:
            st.error(f"Planner error: {e}")
            st.stop()

        t2 = time.time()
        try:
            result = execute(plan)
            exec_time = time.time() - t2
        except Exception as e:
            st.error(f"Executor error: {e}")
            st.stop()

        t3 = time.time()
        try:
            response = answer(effective_query, plan, result)
            synth_time = time.time() - t3
        except Exception as e:
            response = f"Synthesis error: {e}"
            synth_time = 0

    total_time = time.time() - t0

    # ── Block C: Wire Case Persistence ──────────────────────────────────────
    if st.session_state.get("case_id"):
        _cid = st.session_state["case_id"]
        # Collect resource names from results
        _res_names = []
        if result.get("intent") == "lookup":
            _df = result.get("results")
            if isinstance(_df, pd.DataFrame) and len(_df):
                _res_names = _df["name"].tolist()[:10]
        elif result.get("intent") == "needs_assessment":
            for _df in result.get("results_by_need", {}).values():
                if isinstance(_df, pd.DataFrame) and len(_df):
                    _res_names += _df["name"].tolist()[:5]
        _resource_dicts = [{"name": n} for n in _res_names]
        _updated_case = add_visit(_cid, effective_query, response[:300], _resource_dicts, plan=plan)
        if plan.get("intent") == "needs_assessment":
            _updated_case = sync_needs_from_plan(_updated_case, plan)
        st.session_state["case"] = _updated_case

    # ── PRIMARY VIEW: Conversation thread + Answer ────────────────────────────
    # Show prior Q&A turns
    if st.session_state.conv_history:
        for cq, ca in st.session_state.conv_history:
            with st.chat_message("assistant"):
                st.write(cq)
            with st.chat_message("user"):
                st.write(ca)

    # Show current answer as chat message
    with st.chat_message("assistant", avatar="🗽"):
        st.write(response)

    # ── Verification + Reasoning (for ALL intents) ─────────────────────────────
    _demo_mode = st.session_state.get("demo_mode", False)

    if not _demo_mode:
        with st.spinner("Phase 2 — Verifying claims against data…"):
            t4 = time.time()
            verification = verify_answer(response, result)
            verify_time = time.time() - t4
    else:
        verification = {"verified": True, "confidence": "SKIPPED", "claims": []}
        verify_time = 0.0

    total_time = time.time() - t0

    # Verification banner
    v_conf = verification.get("confidence", "LOW")
    if _demo_mode:
        st.markdown(
            '<div style="background:linear-gradient(90deg,#9c27b0,#ba68c8);color:white;'
            'padding:10px 16px;border-radius:8px;font-weight:bold;text-align:center;'
            'margin:8px 0;box-shadow:0 2px 4px rgba(156,39,176,0.3);">'
            '⚡ DEMO MODE — Verification skipped for speed</div>',
            unsafe_allow_html=True)
    elif verification.get("verified"):
        if v_conf == "HIGH":
            st.markdown(
                '<div style="background:linear-gradient(90deg,#00c853,#00e676);color:white;'
                'padding:10px 16px;border-radius:8px;font-weight:bold;text-align:center;'
                'margin:8px 0;box-shadow:0 2px 4px rgba(0,200,83,0.3);">'
                '✅ VERIFIED — High confidence, claims backed by evidence</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="background:linear-gradient(90deg,#2196f3,#42a5f5);color:white;'
                'padding:10px 16px;border-radius:8px;font-weight:bold;text-align:center;'
                'margin:8px 0;box-shadow:0 2px 4px rgba(33,150,243,0.3);">'
                '✅ VERIFIED — Medium confidence, most claims supported</div>',
                unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="background:linear-gradient(90deg,#ff9800,#ffc107);color:white;'
            'padding:10px 16px;border-radius:8px;font-weight:bold;text-align:center;'
            'margin:8px 0;box-shadow:0 2px 4px rgba(255,152,0,0.3);">'
            '⚠️ NEEDS REVIEW — Some claims may exceed available evidence</div>',
            unsafe_allow_html=True)

    st.caption(
        f"⏱ {total_time:.1f}s · Plan {plan_time:.1f}s · Execute {exec_time:.2f}s · "
        f"Synth {synth_time:.1f}s · Verify {verify_time:.1f}s · {get_active_provider()}"
    )

    # Reasoning path (for ALL intents)
    reasoning_path = build_reasoning_path(plan, result)
    if result.get("intent") == "explain" and result.get("reasoning_path"):
        reasoning_path = result["reasoning_path"]  # use the explain engine's path

    with st.expander("🧠 Reasoning Path — How we got this answer", expanded=False):
        # Plain-English summary first
        reasoning_summary = summarize_reasoning(reasoning_path, plan, result)
        if reasoning_summary:
            st.markdown(reasoning_summary)
            st.divider()
            st.caption("Technical details (hop-by-hop data provenance):")

        for step in reasoning_path:
            conf = step.get("confidence", 0)
            cum = step.get("cumulative", 0)
            color = "🟢" if cum >= 0.7 else ("🟡" if cum >= 0.5 else "🔴")
            fact = step.get("fact", "")
            source = step.get("source", "")
            st.markdown(
                f"**Hop {step.get('hop', '?')}** {color} {fact}  \n"
                f"*Conf: {conf} · Cumulative: {cum} · Source: {source}*"
            )
        overall_cum = reasoning_path[-1]["cumulative"] if reasoning_path else 0
        if overall_cum >= 0.7:
            st.success(f"Path confidence: **{overall_cum}**")
        elif overall_cum >= 0.4:
            st.warning(f"Path confidence: **{overall_cum}**")
        else:
            st.error(f"Path confidence: **{overall_cum}**")

    # Claim-by-claim verification
    claims = verification.get("claims", [])
    if claims:
        with st.expander(
            f"🔍 Claim Verification — {verification.get('verified_count',0)}/{verification.get('total_count',0)} verified",
            expanded=False
        ):
            for c in claims:
                is_v = c["verdict"] == "VERIFIED"
                if is_v:
                    st.markdown(
                        f'<div style="background:#1b5e20;color:white;padding:8px 12px;'
                        f'border-radius:6px;margin:6px 0;">✅ <b>VERIFIED</b>: {c["claim"]}'
                        f'<br><small>Evidence: {c.get("evidence","")}</small></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div style="background:#e65100;color:white;padding:8px 12px;'
                        f'border-radius:6px;margin:6px 0;">⚠️ <b>UNVERIFIED</b>: {c["claim"]}'
                        f'<br><small>Evidence: {c.get("evidence","")}</small></div>',
                        unsafe_allow_html=True)

            st.markdown(f"**Summary:** {verification.get('summary','')}")

    # ── User feedback / ground truth correction ─────────────────────────────
    # Show previously excluded resources
    if st.session_state.get("feedback_log"):
        with st.expander(f"📝 Your feedback ({len(st.session_state['feedback_log'])} issues reported)", expanded=False):
            for fb in st.session_state["feedback_log"]:
                st.markdown(f"~~{fb['resource']}~~ — *{fb['issue']}*: {fb['detail']}")

    st.markdown("---")
    feedback_text = st.text_input(
        "⚠️ Report an issue with a recommended resource:",
        key="feedback_input",
        placeholder="e.g. 'The shelter at 66 Boerum Place is full' or 'This address is wrong'",
    )
    if feedback_text and feedback_text.strip():
        # Collect recommended resources from result
        rec_resources = []
        if result.get("intent") == "lookup":
            df = result.get("results")
            if isinstance(df, pd.DataFrame) and len(df):
                rec_resources = df[["name", "address"]].to_dict("records")
        elif result.get("intent") == "needs_assessment":
            for df in result.get("results_by_need", {}).values():
                if isinstance(df, pd.DataFrame) and len(df):
                    rec_resources += df[["name", "address"]].to_dict("records")

        with st.spinner("Processing your feedback…"):
            fb = parse_feedback(feedback_text, rec_resources)

        if fb.get("resource_name") and fb["resource_name"] != "unknown":
            add_exclusion(st.session_state, fb["resource_name"], fb["issue"], fb["detail"])
            # Block G: persist feedback to case
            if st.session_state.get("case_id"):
                mark_resource_visited(st.session_state["case_id"], fb["resource_name"], fb["issue"])
            alt_msg = generate_alternative_response(
                effective_query, fb, get_excluded_resources(st.session_state)
            )
            with st.chat_message("assistant", avatar="🗽"):
                st.write(alt_msg)
                st.write("Searching for alternatives…")

            # Set exclusions in executor and trigger re-run
            from pipeline.executor import set_excluded_resources
            set_excluded_resources(get_excluded_resources(st.session_state))
            st.session_state["_clarify_rerun"] = True
            st.rerun()
        else:
            st.warning("I couldn't identify which resource you're reporting about. "
                       "Try mentioning the resource name or address specifically.")

    # ── Block D: Multi-Issue Tracking Cards ──────────────────────────────────
    _active_case = st.session_state.get("case")
    if _active_case and _active_case.get("needs"):
        _cid = st.session_state["case_id"]
        _case_needs = _active_case["needs"]
        # Sort: open first (by priority), then in_progress, then resolved
        _status_order = {"open": 0, "in_progress": 1, "resolved": 2}
        _sorted_needs = sorted(
            _case_needs,
            key=lambda x: (_status_order.get(x.get("status", "open"), 0), x.get("priority", 99))
        )
        st.markdown("### Your Issues")
        for _i, _need in enumerate(_sorted_needs):
            _cat = _need["category"]
            _status = _need.get("status", "open")
            _pri = _need.get("priority", "?")
            _icon = {"open": "🔴", "in_progress": "🟡", "resolved": "🟢"}.get(_status, "🔴")
            _label = f"{_icon} {_cat.replace('_', ' ').title()} [P{_pri}] — {_status.replace('_', ' ')}"
            _expanded = (_status == "open" and _i == 0)
            with st.expander(_label, expanded=_expanded):
                # Resources recommended for this need — show rows with "I'm going here"
                _need_key = f"{_cat}_1"
                _need_df = result.get("results_by_need", {}).get(_need_key)
                if isinstance(_need_df, pd.DataFrame) and len(_need_df):
                    for _ri, _rrow in _need_df.head(3).iterrows():
                        _rname = str(_rrow.get("name", ""))
                        _raddr = str(_rrow.get("address", ""))
                        _rtype = str(_rrow.get("resource_type", ""))
                        _rboro = str(_rrow.get("borough", ""))
                        _rc1, _rc2 = st.columns([3, 1])
                        with _rc1:
                            st.markdown(
                                f"**{_rname}** · "
                                f"<span style='color:#888;font-size:12px;'>"
                                f"{_rtype.replace('_',' ')}</span>",
                                unsafe_allow_html=True)
                            if _raddr:
                                st.caption(_raddr)
                        with _rc2:
                            if st.session_state.get("case_id") and _status != "resolved":
                                if st.button("I'm going here",
                                             key=f"_going_{_cat}_{_i}_{_ri}",
                                             use_container_width=True):
                                    st.session_state["_confirm_dest_pending"] = {
                                        "name": _rname,
                                        "resource_type": _rtype,
                                        "address": _raddr,
                                        "borough": _rboro,
                                        "category": _cat,
                                    }
                                    st.rerun()
                        st.markdown(
                            '<hr style="border:none;border-top:1px solid #1e1e2a;margin:4px 0;">',
                            unsafe_allow_html=True)

                _btn_cols = st.columns(3)
                if _status != "in_progress" and _status != "resolved":
                    if _btn_cols[0].button("Mark In Progress", key=f"_ip_{_cat}_{_i}"):
                        _updated = update_need_status(_cid, _cat, "in_progress")
                        st.session_state["case"] = _updated
                        st.rerun()
                if _status != "resolved":
                    if _btn_cols[1].button("Mark Resolved", key=f"_res_{_cat}_{_i}"):
                        _updated = update_need_status(_cid, _cat, "resolved")
                        st.session_state["case"] = _updated
                        st.rerun()

        # ── Destination intent confirmation ───────────────────────────────────
        if st.session_state.get("_confirm_dest_pending") and st.session_state.get("case_id"):
            _pdest = st.session_state["_confirm_dest_pending"]
            st.markdown(f"""
            <div style="background:#1a2030;border:2px solid #76b900;border-radius:12px;
                padding:16px;margin:12px 0;">
                <div style="color:#76b900;font-weight:bold;font-size:16px;">
                    Confirm: heading to {_pdest['name']}?
                </div>
                <div style="color:#ccc;font-size:13px;margin-top:6px;">
                    {_pdest.get('address','')}
                </div>
                <div style="color:#aaa;font-size:12px;margin-top:4px;">
                    We'll notify your emergency contact and the destination.
                </div>
            </div>
            """, unsafe_allow_html=True)
            _cf1, _cf2 = st.columns(2)
            if _cf1.button("Yes, I'm going there", type="primary",
                           key="_dest_confirm_yes", use_container_width=True):
                _cid2 = st.session_state["case_id"]
                _case2 = st.session_state["case"]
                # Record intent to case
                _case2 = add_destination_intent(_cid2, _pdest)
                st.session_state["case"] = _case2
                # Build Discord config
                _ec2 = _case2.get("emergency_contact", {})
                _ec_username = (_ec2.get("discord_username", "")
                                if isinstance(_ec2, dict) else "")
                _discord_cfg = {
                    "dest_webhook": st.session_state.get("_dest_webhook", ""),
                    "ec_discord_username": _ec_username,
                    "bot_token": st.session_state.get("_bot_token", DISCORD_BOT_TOKEN),
                    "coord_channel_id": st.session_state.get("_coord_channel",
                                                               DISCORD_COORD_CHANNEL_ID),
                    "sla_minutes": st.session_state.get("_sla_min", 15),
                }
                _notif = confirm_destination_intent(_case2, _pdest, _discord_cfg)
                st.session_state["_last_dest_notif"] = _notif
                st.session_state.pop("_confirm_dest_pending", None)
                # Refresh case from disk (state may have advanced to "notified")
                from pipeline.cases import load_case as _lc
                _refreshed = _lc(_cid2)
                if _refreshed:
                    st.session_state["case"] = _refreshed
                st.rerun()
            if _cf2.button("Cancel", key="_dest_confirm_no", use_container_width=True):
                st.session_state.pop("_confirm_dest_pending", None)
                st.rerun()

        # Show last notification result (transient, clears on next rerun cycle)
        if st.session_state.get("_last_dest_notif"):
            _ln = st.session_state.pop("_last_dest_notif")
            _sent = _ln.get("notifications_sent", [])
            _turl = _ln.get("thread_url")
            if _sent:
                _labels = {"destination": "Destination", "emergency_contact": "Emergency Contact",
                           "thread_created": "Coordination Thread"}
                st.success("Confirmed! Notified: " +
                           ", ".join(_labels.get(s, s) for s in _sent))
            if _turl:
                st.markdown(f"[Open coordination thread]({_turl})")

        # ── Block D2: Active Destinations tracker ─────────────────────────────
        if st.session_state.get("case_id"):
            _active_dests = get_active_destinations(st.session_state["case_id"])
            if _active_dests:
                st.markdown("### Active Destinations")
                _state_icons = {
                    "intent_confirmed": "📍",
                    "notified": "📨",
                    "acknowledged": "✅",
                    "en_route": "🚶",
                    "arrived": "🏠",
                    "resolved": "🟢",
                }
                _state_next = {
                    "intent_confirmed": "en_route",
                    "notified": "en_route",
                    "acknowledged": "en_route",
                    "en_route": "arrived",
                    "arrived": "resolved",
                }
                for _di, _dest in enumerate(_active_dests):
                    _dstate = _dest.get("state", "intent_confirmed")
                    _dname = _dest.get("resource_name", "")
                    _dicon = _state_icons.get(_dstate, "📍")
                    _dlabel = (f"{_dicon} {_dname} — "
                               f"{_dstate.replace('_', ' ').title()}")
                    with st.expander(_dlabel, expanded=True):
                        _da1, _da2 = st.columns(2)
                        _da1.caption(f"Type: {_dest.get('resource_type','').replace('_',' ').title()}")
                        _da1.caption(f"Address: {_dest.get('address','')}")
                        _da2.caption(f"Category: {_dest.get('category','').replace('_',' ').title()}")
                        _da2.caption(f"Since: {_dest.get('intent_at','')[:16]}")
                        _thread_url = _dest.get("thread_url")
                        if _thread_url:
                            st.markdown(f"[Open coordination thread]({_thread_url})")
                        _dc1, _dc2 = st.columns(2)
                        _next = _state_next.get(_dstate)
                        if _next:
                            _btn_lbl = f"Mark {_next.replace('_', ' ').title()}"
                            if _dc1.button(_btn_lbl, key=f"_dnext_{_di}",
                                           use_container_width=True):
                                _updated_c = update_destination_state(
                                    st.session_state["case_id"], _dname, _next)
                                st.session_state["case"] = _updated_c
                                st.rerun()
                        if _dstate != "resolved":
                            if _dc2.button("Mark Resolved", key=f"_dresolve_{_di}",
                                           use_container_width=True):
                                _updated_c = update_destination_state(
                                    st.session_state["case_id"], _dname, "resolved")
                                st.session_state["case"] = _updated_c
                                st.rerun()

        # ── Block F: Adaptive Continuation ───────────────────────────────────
        _refreshed_needs = st.session_state.get("case", {}).get("needs", [])
        _open_count = len([n for n in _refreshed_needs if n.get("status") == "open"])
        _total_needs = len(_refreshed_needs)
        if not st.session_state.get("_followup_scheduled"):
            _webhook = st.session_state.get("_discord_webhook_url", "")
            if _total_needs > 0 and _open_count == 0:
                st.balloons()
                st.success("You're all set! All your needs have been addressed.")
                schedule_followup(st.session_state["case"], delay_minutes=60, webhook_url=_webhook)
            elif _open_count > 0:
                _highest = sorted(
                    [n for n in _refreshed_needs if n.get("status") == "open"],
                    key=lambda x: x.get("priority", 99)
                )[0]
                st.info(
                    f"Next priority: **{_highest['category'].replace('_', ' ').title()}** — "
                    "use the tracking cards above to update your status."
                )
                schedule_followup(st.session_state["case"], delay_minutes=30, webhook_url=_webhook)
            st.session_state["_followup_scheduled"] = True

    # ── Multi-turn clarification ──────────────────────────────────────────────
    turn = len(st.session_state.conv_history)
    if turn < 2:
        with st.spinner("Thinking of a follow-up…"):
            clarify_q = get_clarifying_question(
                st.session_state.active_query, response, turn
            )
        if clarify_q:
            st.session_state["_pending_clarify_q"]    = clarify_q
            st.session_state["_pending_clarify_turn"] = turn
            with st.chat_message("assistant", avatar="🗽"):
                st.write(clarify_q)
            st.text_input(
                "Your answer:",
                key=f"clarify_{turn}",
                placeholder="Type your answer and press Enter…",
            )

    # ── SECONDARY VIEW: Technical details in expanders ────────────────────────
    st.divider()
    intent = plan.get("intent", "unknown")
    intent_icon = {"lookup": "🔍", "needs_assessment": "🏥", "simulate": "⚡", "explain": "🧠"}.get(intent, "❓")

    with st.expander(f"{intent_icon} Query Plan — `{intent}`", expanded=False):
        if intent == "needs_assessment":
            profile = plan.get("client_profile", {})
            needs   = plan.get("identified_needs", [])
            st.write(f"**Situation:** {profile.get('situation', '—')}  |  "
                     f"**Borough:** {profile.get('borough', '?')}  |  "
                     f"**Household:** {profile.get('household_size', '?')}")
            for n in sorted(needs, key=lambda x: x.get("priority", 99)):
                st.write(f"  {n['priority']}. **{n['category']}** — {n.get('reasoning', '')}")
        elif intent == "lookup":
            st.write(f"**Types:** {', '.join(plan.get('resource_types', []))}  |  "
                     f"**Filters:** {plan.get('filters', {})}")
        elif intent == "simulate":
            st.write(f"**Scenario:** {plan.get('scenario')}  |  **Params:** {plan.get('params', {})}")
        if "_parse_error" in plan:
            st.warning(f"Fallback plan. Raw: {plan['_parse_error']}")
        st.json(plan, expanded=False)

    result_intent = result.get("intent", "")
    with st.expander("📋 Resource Results", expanded=False):
        if result_intent == "lookup":
            df = result.get("results", pd.DataFrame())
            if isinstance(df, pd.DataFrame) and len(df):
                display_cols = [c for c in ["name", "resource_type", "borough", "address",
                                            "safety_score", "quality_score",
                                            "nearest_transit_name", "nearest_transit_walk_min"]
                                if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True)
            else:
                st.warning("No matching resources found.")
        elif result_intent == "needs_assessment":
            for need_key, df in result.get("results_by_need", {}).items():
                st.markdown(f"**{need_key}**")
                if isinstance(df, pd.DataFrame) and len(df):
                    display_cols = [c for c in ["name", "resource_type", "borough",
                                                 "address", "nearest_transit_walk_min"]
                                    if c in df.columns]
                    st.dataframe(df[display_cols], use_container_width=True)
                else:
                    st.write("No resources found.")
        elif result_intent == "simulate":
            scenario = result.get("scenario", "")
            if scenario == "cold_emergency":
                c1, c2, c3 = st.columns(3)
                c1.metric("People displaced", result.get("people_displaced"))
                c2.metric("Temperature", f"{result.get('temperature_f')}°F")
                c3.metric("Overflow sites", len(result.get("overflow_sites", [])))
                st.write(f"**Recommendation:** {result.get('recommendation', '')}")

                shelters_df = pd.DataFrame(result.get("available_shelters", []))
                if not shelters_df.empty:
                    st.markdown("**🏠 Available Shelters**")
                    show_cols = [c for c in ["name", "address", "borough", "capacity"] if c in shelters_df.columns]
                    display_s = shelters_df[show_cols].copy()
                    if "capacity" in display_s.columns:
                        display_s["capacity"] = display_s["capacity"].replace(0, None).fillna("—")
                    st.dataframe(display_s, use_container_width=True)

                overflow_df = pd.DataFrame(result.get("overflow_sites", []))
                if not overflow_df.empty:
                    st.markdown("**🏛 PLUTO Overflow Sites** *(assembly-zoned buildings)*")
                    overflow_df = overflow_df.rename(columns={
                        "ownername": "Owner / Operator",
                        "address": "Address",
                        "landuse": "Zoning",
                        "borough": "Borough",
                    })
                    overflow_df["Zoning"] = overflow_df.get("Zoning", "08").apply(
                        lambda x: "Assembly / Community Facility" if str(x) == "08" else x
                    )
                    show_cols = [c for c in ["Address", "Owner / Operator", "Zoning", "Borough"]
                                 if c in overflow_df.columns]
                    st.dataframe(overflow_df[show_cols], use_container_width=True)
            elif scenario == "resource_gap":
                st.write(f"**Most underserved:** {result.get('most_underserved', '?')}")
                st.dataframe(pd.DataFrame(result.get("gaps", [])))
            elif scenario == "capacity_change":
                st.write(f"**Recommendation:** {result.get('recommendation','')}")
                st.dataframe(pd.DataFrame(result.get("summary", [])), use_container_width=True)
            elif scenario == "migrant_allocation":
                st.write(f"**Recommendation:** {result.get('recommendation','')}")
                st.markdown("**Shelter allocation:**")
                st.dataframe(pd.DataFrame(result.get("allocation", [])), use_container_width=True)
                for need, rows in result.get("resources_by_need", {}).items():
                    st.markdown(f"**{need.replace('_',' ').title()}:**")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Explain intent display
    if result_intent == "explain":
        with st.expander("🧠 Reasoning Path — Explainable AI", expanded=True):
            reasoning = result.get("reasoning_path", [])
            if reasoning:
                for step in reasoning:
                    conf = step.get("confidence", 0)
                    cum = step.get("cumulative", 0)
                    # Color: green >0.8, yellow 0.5-0.8, red <0.5
                    if cum >= 0.5:
                        color = "🟢" if cum >= 0.7 else "🟡"
                    else:
                        color = "🔴"
                    fact = step.get("fact", step.get("predicate", ""))
                    source = step.get("source", "")
                    st.markdown(
                        f"**Hop {step.get('hop', '?')}** {color} "
                        f"{fact}  \n"
                        f"*Confidence: {conf} · Cumulative: {cum} · Source: {source}*"
                    )

            overall = result.get("overall_confidence", 0)
            if overall >= 0.7:
                st.success(f"Overall confidence: **{overall}** — High confidence answer")
            elif overall >= 0.4:
                st.warning(f"Overall confidence: **{overall}** — Moderate confidence, some derived data")
            else:
                st.error(f"Overall confidence: **{overall}** — Low confidence, verify independently")

            # Show additional explain-specific details
            if result.get("question") == "why_underserved":
                ranking = result.get("ranking", [])
                if ranking:
                    st.markdown("**Borough ranking (resources per 100K):**")
                    rank_df = pd.DataFrame(ranking, columns=["Borough", "Per 100K"])
                    st.dataframe(rank_df, use_container_width=True)
                weakest = result.get("weakest_types", [])
                if weakest:
                    st.markdown("**Most lacking resource types:**")
                    for rtype, val in weakest:
                        st.write(f"  - **{rtype}**: {val} per 100K")

    with st.expander("🗺 Map", expanded=False):
        map_frames = []
        map_rtypes = []
        if result_intent == "lookup":
            df = result.get("results")
            if isinstance(df, pd.DataFrame) and len(df):
                map_frames.append(df)
                map_rtypes = df["resource_type"].unique().tolist()
        elif result_intent == "needs_assessment":
            for df in result.get("results_by_need", {}).values():
                if isinstance(df, pd.DataFrame) and len(df):
                    map_frames.append(df)
                    map_rtypes += df["resource_type"].unique().tolist()
            map_rtypes = list(dict.fromkeys(map_rtypes))
        elif result_intent == "simulate":
            shelters = result.get("available_shelters", [])
            if shelters:
                map_frames.append(pd.DataFrame(shelters))

        if map_frames:
            render_map_legend(map_rtypes)
            deck = make_resource_map(map_frames)
            if deck:
                st.pydeck_chart(deck, use_container_width=True)
            else:
                st.info("No coordinates available for map.")
        else:
            st.info("No resources to map.")

    # ── KGE Similar Resources ────────────────────────────────────────────────
    with st.expander("🔗 KGE — Similar Resources", expanded=False):
        try:
            if result_intent in ("lookup", "needs_assessment"):
                # Get first resource_id from results
                first_df = None
                if result_intent == "lookup":
                    first_df = result.get("results")
                elif result_intent == "needs_assessment":
                    for df in result.get("results_by_need", {}).values():
                        if isinstance(df, pd.DataFrame) and len(df):
                            first_df = df
                            break

                if first_df is not None and isinstance(first_df, pd.DataFrame) and len(first_df):
                    if "resource_id" in first_df.columns:
                        rid = first_df.iloc[0]["resource_id"]
                        rname = first_df.iloc[0].get("name", rid)
                        st.markdown(f"**Resources similar to:** {rname}")
                        similar = find_similar(rid, k=5)
                        if not similar.empty:
                            show_cols = [c for c in ["name", "resource_type", "borough",
                                                      "address", "similarity"]
                                         if c in similar.columns]
                            st.dataframe(similar[show_cols], use_container_width=True)
                        else:
                            st.info("No similar resources found.")
                    else:
                        # Fallback: query-based similarity
                        rtypes = plan.get("resource_types", [])
                        boro = plan.get("filters", {}).get("borough")
                        if rtypes:
                            st.markdown(f"**Resources matching profile:** {', '.join(rtypes)} in {boro or 'NYC'}")
                            similar = find_similar_to_query(rtypes, boro, k=5)
                            if not similar.empty:
                                show_cols = [c for c in ["name", "resource_type", "borough",
                                                          "address", "similarity"]
                                             if c in similar.columns]
                                st.dataframe(similar[show_cols], use_container_width=True)
                else:
                    st.info("No results to compute similarity from.")
            else:
                st.info("Similarity search available for lookup and needs assessment queries.")
        except Exception as e:
            st.warning(f"KGE embeddings not available: {e}")
