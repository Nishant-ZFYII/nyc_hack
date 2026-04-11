"""
UI_testing.py — NYC Care Network
Caseworker · Agent · Resident Decision-Support System

Run: streamlit run UI_testing.py
"""
import sys
import time
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent  # repo root (one level up from tests/)
sys.path.insert(0, str(ROOT))

from pipeline.planner  import generate_plan
from pipeline.executor import execute, load_state
from pipeline.synth    import answer
from pipeline.clarify  import get_clarifying_question, merge_query
from pipeline.verify   import verify_answer, build_reasoning_path, summarize_reasoning
from pipeline.feedback import parse_feedback, add_exclusion, get_excluded_resources, generate_alternative_response
from llm.client        import get_active_provider
import importlib.util

_spec = importlib.util.spec_from_file_location("kg_embeddings", str(ROOT / "engine" / "embeddings.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
find_similar = _mod.find_similar
find_similar_to_query = _mod.find_similar_to_query
build_embeddings = _mod.build_embeddings

# ── Map helpers ────────────────────────────────────────────────────────────────
RESOURCE_COLORS = {
    "shelter":           [255, 99,  71,  220],
    "food_bank":         [255, 165,  0,  220],
    "hospital":          [0,   191, 255, 220],
    "benefits_center":   [0,   255, 127, 220],
    "domestic_violence": [255, 105, 180, 220],
    "school":            [255, 215,   0, 220],
    "senior_services":   [148,   0, 211, 220],
    "childcare":         [0,   255, 255, 220],
    "community_center":  [50,  205,  50, 220],
    "transit_station":   [180, 180, 180, 200],
}
DEFAULT_COLOR = [200, 200, 200, 200]
NYC_CENTER = {"lat": 40.7128, "lon": -74.0060, "zoom": 11}


def make_resource_map(frames: list) -> pdk.Deck:
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
        latitude=df_map["lat"].mean(),
        longitude=df_map["lon"].mean(),
        zoom=12,
        pitch=0,
    )
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{name}\n{type}\n{address}\n{borough}"},
    )


def render_map_legend(resource_types: list):
    items = []
    for rtype in resource_types:
        color = RESOURCE_COLORS.get(rtype, DEFAULT_COLOR)
        items.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;">'
            f'<span style="color:rgb({color[0]},{color[1]},{color[2]});font-size:16px;">●</span>'
            f'<span style="color:#475569;font-size:12px;">{rtype.replace("_"," ")}</span></span>'
        )
    st.markdown('<div style="padding:4px 0">' + "".join(items) + "</div>", unsafe_allow_html=True)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Care Network",
    page_icon="🗽",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* App background */
    .stApp { background-color: #F0F4F8; }
    .main .block-container { padding-top: 20px; padding-bottom: 40px; max-width: 960px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #E2E8F0; }
    [data-testid="stSidebar"] .block-container { padding-top: 20px; }

    /* Primary buttons */
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 15px !important;
        padding: 10px 24px !important;
        cursor: pointer !important;
        transition: background-color 0.2s ease !important;
        min-height: 44px !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover,
    div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
        background-color: #1E40AF !important;
    }

    /* Secondary buttons */
    div[data-testid="stButton"] > button[kind="secondary"] {
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        background: #FFFFFF !important;
        color: #374151 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        min-height: 40px !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        border-color: #1D4ED8 !important;
        color: #1D4ED8 !important;
        background: #EFF6FF !important;
    }

    /* Link buttons */
    div[data-testid="stLinkButton"] > a {
        background: #FFFFFF !important;
        color: #374151 !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        text-decoration: none !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        min-height: 40px !important;
        display: inline-flex !important;
        align-items: center !important;
    }

    /* Download buttons */
    div[data-testid="stDownloadButton"] > button {
        background: #FFFFFF !important;
        color: #374151 !important;
        border: 1.5px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        min-height: 40px !important;
        cursor: pointer !important;
    }

    /* Text inputs */
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input {
        border-radius: 8px !important;
        border: 1.5px solid #CBD5E1 !important;
        font-size: 14px !important;
        background: #FFFFFF !important;
        min-height: 44px !important;
    }
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #1D4ED8 !important;
        box-shadow: 0 0 0 3px rgba(29,78,216,0.1) !important;
    }

    /* Selectbox */
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 8px !important;
        border: 1.5px solid #CBD5E1 !important;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
    }

    /* Tabs */
    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    /* Status indicators */
    .status-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: #16A34A;
        font-weight: 600;
        font-size: 13px;
        padding: 4px 10px;
        background: #F0FDF4;
        border-radius: 20px;
        border: 1px solid #BBF7D0;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background: #22C55E;
        border-radius: 50%;
        display: inline-block;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.35; }
    }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }

    /* Toast */
    div[data-testid="stToast"] { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
_DEFAULTS = {
    "user_role": "Caseworker",
    "selected_category": None,
    "saved_resources": [],
    "case_notes": "",
    "conv_history": [],
    "active_query": "",
    "excluded_resources": [],
    "feedback_log": [],
    "query_input": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Helpers ────────────────────────────────────────────────────────────────────
BOROUGH_MAP = {
    "Any Borough": None,
    "Manhattan": "MN",
    "Brooklyn": "BK",
    "Queens": "QN",
    "The Bronx": "BX",
    "Staten Island": "SI",
}
BOROUGH_FULL = {v: k for k, v in BOROUGH_MAP.items() if v}

URGENCY_OPTIONS = [
    "Planning ahead — not urgent",
    "Need help soon — within a week",
    "Urgent — within 24-48 hours",
    "Emergency — right now",
]

CATEGORIES = [
    ("Housing",     "Shelter, eviction help, emergency housing"),
    ("Food",        "Food banks, meal programs, SNAP"),
    ("Healthcare",  "Clinics, hospitals, mental health"),
    ("Immigration", "Legal aid, asylum, documentation"),
    ("Benefits",    "Medicaid, cash assistance, SNAP"),
    ("Safety",      "Domestic violence, crime victims"),
    ("Children",    "Childcare, schools, family services"),
    ("Seniors",     "Senior centers, elder care, home support"),
    ("Jobs",        "Workforce training, job placement"),
    ("Emergency",   "Crisis, cold emergency, acute need"),
]

QUICK_SCENARIOS = [
    ("Family facing eviction",       "I have 4 kids ages 12-16, my income is $28K, and I'm losing my Flatbush apartment next week. What do we do?"),
    ("Crime victim needs safety",    "Someone broke into my apartment last night. I don't feel safe going back. I have a 6-year-old. What help is available?"),
    ("Migrant family just arrived",  "I just arrived from Haiti with my two children. We speak Haitian Creole and need shelter tonight near Flatbush."),
    ("Cold emergency response",      "A cold emergency is declared. 3 Brooklyn shelters just hit capacity. 200 people are still outside. It's 15°F. What do we do?"),
    ("Why is the Bronx underserved?", "Why is the Bronx the most underserved borough for social services?"),
    ("Bronx food banks",             "How many food banks are open in the Bronx right now?"),
    ("Wheelchair-accessible hospitals", "Find wheelchair accessible hospitals near Jamaica Queens that accept Medicaid"),
    ("Migrant allocation",           "A migrant bus arrived with 80 people speaking Spanish and Mandarin. They need shelter, food, and schools for children."),
]

EMERGENCY_RTYPES = {
    "shelter", "hospital", "domestic violence", "domestic_violence",
    "domestic_violence_shelter", "drop_in_center", "cooling_center",
}


def _score_color(val: float) -> str:
    if val >= 0.70:
        return "#16A34A"
    if val >= 0.40:
        return "#CA8A04"
    return "#DC2626"


def render_resource_cards(df: pd.DataFrame, section_label: str = ""):
    """Render resource results as structured, scannable cards."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No services found for this need.")
        return

    if section_label:
        st.markdown(
            f'<h3 style="font-size:16px;font-weight:700;color:#0F172A;'
            f'margin:24px 0 12px;padding-bottom:8px;border-bottom:2px solid #E2E8F0;">'
            f'{section_label}</h3>',
            unsafe_allow_html=True,
        )

    is_emergency_query = section_label.lower() in ("shelter", "emergency", "safety", "domestic violence")

    for idx, (_, row) in enumerate(df.iterrows()):
        name      = str(row.get("name", "Unknown Service"))
        rtype     = str(row.get("resource_type", "")).replace("_", " ").title()
        address   = str(row.get("address", ""))
        borough   = str(row.get("borough", ""))
        safety    = float(row.get("safety_score", 0) or 0)
        quality   = float(row.get("quality_score", 0) or 0)
        transit_m = row.get("nearest_transit_walk_min")
        phone     = str(row.get("phone", "") or "")

        rtype_lower = str(row.get("resource_type", "")).lower()
        is_urgent   = rtype_lower in EMERGENCY_RTYPES or is_emergency_query
        is_24_7     = rtype_lower in {"shelter", "hospital", "domestic_violence_shelter",
                                       "drop_in_center", "cooling_center"}

        border_color = "#DC2626" if is_urgent else "#1D4ED8"
        card_bg      = "#FFF5F5" if is_urgent else "#FFFFFF"
        avail_color  = "#16A34A" if is_24_7 else "#92400E"
        avail_text   = "Open 24/7" if is_24_7 else "Call ahead for hours"

        urgent_badge = (
            '<span style="background:#FEE2E2;color:#DC2626;padding:2px 8px;border-radius:4px;'
            'font-size:11px;font-weight:700;margin-right:6px;letter-spacing:0.5px;">URGENT</span>'
            if is_urgent else ""
        )
        transit_str = (
            f' &nbsp;·&nbsp; {int(transit_m)} min walk to transit'
            if isinstance(transit_m, (int, float)) and transit_m > 0 else ""
        )
        phone_str = (
            f'<div style="margin-top:4px;font-size:13px;color:#475569;">Call: {phone}</div>'
            if phone and phone != "nan" else ""
        )

        st.markdown(f"""
        <div style="background:{card_bg};border:1px solid #E2E8F0;border-left:4px solid {border_color};
             border-radius:12px;padding:18px 20px;margin-bottom:10px;
             box-shadow:0 1px 6px rgba(0,0,0,0.06);">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;
                        flex-wrap:wrap;gap:6px;margin-bottom:8px;">
                <div>
                    {urgent_badge}
                    <strong style="font-size:16px;color:#0F172A;line-height:1.3;">{name}</strong>
                </div>
                <span style="background:#DBEAFE;color:#1D4ED8;padding:3px 10px;border-radius:20px;
                             font-size:12px;font-weight:600;white-space:nowrap;">{rtype}</span>
            </div>
            <div style="color:#475569;font-size:13px;line-height:1.7;">
                <div style="margin-bottom:2px;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24"
                         fill="none" stroke="#94A3B8" stroke-width="2" style="vertical-align:middle;margin-right:4px;">
                        <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0118 0z"/>
                        <circle cx="12" cy="10" r="3"/>
                    </svg>
                    {address}{", " + borough if borough and borough not in address else ""}{transit_str}
                </div>
                <div style="margin-top:4px;">
                    <span style="color:{avail_color};font-weight:600;font-size:12px;">
                        &#9679; {avail_text}
                    </span>
                </div>
                {phone_str}
            </div>
            <div style="display:flex;gap:20px;margin-top:10px;padding-top:8px;border-top:1px solid #F1F5F9;">
                <span style="font-size:12px;color:#64748B;">
                    Safety &nbsp;
                    <strong style="color:{_score_color(safety)};">{int(safety * 100)}%</strong>
                </span>
                <span style="font-size:12px;color:#64748B;">
                    Quality &nbsp;
                    <strong style="color:{_score_color(quality)};">{int(quality * 100)}%</strong>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action buttons — must be Streamlit widgets
        uid = f"{section_label}_{idx}_{hash(name) % 9999}"
        c1, c2, c3, _sp = st.columns([2, 2, 2, 4])

        if c1.button("Save to Case", key=f"save_{uid}", use_container_width=True):
            entry = {"name": name, "address": address, "type": rtype, "borough": borough}
            if entry not in st.session_state.saved_resources:
                st.session_state.saved_resources.append(entry)
                st.toast(f"Saved: {name}")
            else:
                st.toast("Already saved.")

        share_txt = (
            f"Service: {name}\n"
            f"Type: {rtype}\n"
            f"Address: {address}, {borough}\n"
            f"Availability: {avail_text}\n"
            f"Safety: {int(safety * 100)}%  Quality: {int(quality * 100)}%"
        )
        c2.download_button(
            "Share Info",
            share_txt,
            file_name=f"{name[:30].replace(' ','_')}.txt",
            key=f"share_{uid}",
            use_container_width=True,
        )

        gmaps = f"https://maps.google.com/?q={address.replace(' ','+')},+{borough.replace(' ','+')},+NYC"
        c3.link_button("Get Directions", gmaps, use_container_width=True)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # User role
    st.session_state.user_role = st.radio(
        "I am a:",
        ["Caseworker", "Agent", "Resident"],
        horizontal=True,
        label_visibility="visible",
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # System status
    try:
        _provider = get_active_provider()
        _mart, _payload = load_state()
        st.markdown(
            f'<div class="status-online">'
            f'<span class="status-dot"></span> System Online</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{_provider} · {len(_mart):,} resources")
    except Exception as _e:
        st.markdown(
            '<div style="color:#DC2626;font-weight:600;font-size:13px;">System Offline</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Current Case panel
    st.markdown(
        '<p style="font-size:14px;font-weight:700;color:#0F172A;margin-bottom:6px;">Current Case</p>',
        unsafe_allow_html=True,
    )
    new_notes = st.text_area(
        "Notes",
        value=st.session_state.case_notes,
        height=80,
        placeholder="Add notes about this case...",
        label_visibility="collapsed",
    )
    st.session_state.case_notes = new_notes

    if st.session_state.saved_resources:
        st.markdown(
            f'<p style="font-size:13px;font-weight:600;color:#374151;margin:10px 0 6px;">'
            f'Saved Resources ({len(st.session_state.saved_resources)})</p>',
            unsafe_allow_html=True,
        )
        for _i, _res in enumerate(st.session_state.saved_resources):
            _c1, _c2 = st.columns([5, 1])
            _c1.markdown(
                f'<p style="font-size:12px;margin:0;color:#0F172A;font-weight:500;">{_res["name"]}</p>'
                f'<p style="font-size:11px;margin:0;color:#64748B;">{_res.get("address","")}</p>',
                unsafe_allow_html=True,
            )
            if _c2.button("✕", key=f"rm_{_i}", help="Remove"):
                st.session_state.saved_resources.pop(_i)
                st.rerun()

    st.divider()

    # Quick scenarios
    st.markdown(
        '<p style="font-size:13px;font-weight:600;color:#374151;margin-bottom:8px;">Quick Scenarios</p>',
        unsafe_allow_html=True,
    )
    for _label, _q in QUICK_SCENARIOS:
        if st.button(_label, use_container_width=True, key=f"qs_{_label}"):
            st.session_state.query_input = _q
            st.session_state.active_query = _q
            for _k in ["_last_plan", "_last_result", "_last_response",
                       "_last_verification", "_last_timing", "conv_history",
                       "excluded_resources", "feedback_log", "_pending_clarify_q",
                       "_pending_clarify_turn", "selected_category"]:
                st.session_state.pop(_k, None)
            st.session_state._run_query = True
            st.rerun()

    st.divider()

    if st.button("Start New Search", use_container_width=True, type="primary"):
        for _k in ["_last_plan", "_last_result", "_last_response", "_last_verification",
                   "_last_timing", "conv_history", "active_query", "excluded_resources",
                   "feedback_log", "_pending_clarify_q", "_pending_clarify_turn",
                   "_clarify_rerun", "query_input", "selected_category", "_run_query"]:
            st.session_state.pop(_k, None)
        st.rerun()


# ── HEADER ─────────────────────────────────────────────────────────────────────
has_results = "_last_result" in st.session_state

st.markdown("""
<div style="background:linear-gradient(135deg,#0D1B2A 0%,#1B3A5C 100%);
     color:white;padding:22px 28px;border-radius:14px;margin-bottom:20px;
     box-shadow:0 4px 20px rgba(13,27,42,0.25);">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
        <div>
            <h1 style="margin:0;font-size:26px;font-weight:800;letter-spacing:-0.5px;">
                NYC Care Network
            </h1>
            <p style="margin:4px 0 0;opacity:0.75;font-size:14px;font-weight:400;">
                We'll help you find the best services available in New York City
            </p>
        </div>
        <div style="opacity:0.55;font-size:11px;text-align:right;line-height:1.5;">
            Powered by NVIDIA Nemotron<br>cuGraph &middot; RAPIDS &middot; DGX Spark
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── CLARIFICATION STATE DETECTION ─────────────────────────────────────────────
_pending_q    = st.session_state.get("_pending_clarify_q", "")
_pending_turn = st.session_state.get("_pending_clarify_turn", -1)
if _pending_q and _pending_turn >= 0:
    _cval    = st.session_state.get(f"clarify_{_pending_turn}", "").strip()
    _history = st.session_state.setdefault("conv_history", [])
    if _cval and len(_history) == _pending_turn:
        _history.append((_pending_q, _cval))
        st.session_state.active_query = merge_query(
            st.session_state.get("active_query", ""), _pending_q, _cval
        )
        st.session_state["_clarify_rerun"] = True
        st.session_state.pop("_pending_clarify_q", None)
        st.session_state.pop("_pending_clarify_turn", None)
        st.rerun()

_clarify_rerun = st.session_state.pop("_clarify_rerun", False)

# ── INPUT SECTION (shown when no results) ──────────────────────────────────────
if not has_results:
    # ── Card: What do you need help with? ──────────────────────────────────────
    st.markdown("""
    <div style="background:#FFFFFF;border-radius:14px;padding:28px;
         box-shadow:0 2px 16px rgba(0,0,0,0.07);margin-bottom:16px;">
        <h2 style="margin:0 0 4px;font-size:20px;font-weight:700;color:#0F172A;">
            What do you need help with?
        </h2>
        <p style="margin:0 0 20px;color:#64748B;font-size:14px;">
            Select a category or describe the situation below.
        </p>
    """, unsafe_allow_html=True)

    # Category grid — 5 per row
    _row1 = st.columns(5)
    _row2 = st.columns(5)
    _all_cols = _row1 + _row2
    for _ci, (_cat, _desc) in enumerate(CATEGORIES):
        _selected = st.session_state.selected_category == _cat
        _btn_type = "primary" if _selected else "secondary"
        if _all_cols[_ci].button(_cat, key=f"cat_{_cat}", use_container_width=True, type=_btn_type):
            st.session_state.selected_category = None if _selected else _cat
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Card: Details form ─────────────────────────────────────────────────────
    _sel_cat = st.session_state.selected_category
    _form_title = (
        f"Tell us more about your {_sel_cat.lower()} need"
        if _sel_cat else "Tell us more so we can find the best match"
    )

    st.markdown(
        f'<div style="background:#FFFFFF;border-radius:14px;padding:28px;'
        f'box-shadow:0 2px 16px rgba(0,0,0,0.07);margin-bottom:16px;">'
        f'<h3 style="margin:0 0 18px;font-size:16px;font-weight:700;color:#0F172A;">'
        f'{_form_title}</h3>',
        unsafe_allow_html=True,
    )

    with st.form("guided_search_form"):
        _fc1, _fc2 = st.columns(2)
        with _fc1:
            _borough_choice = st.selectbox(
                "Location",
                list(BOROUGH_MAP.keys()),
                help="Select the borough where help is needed",
            )
            _household = st.number_input(
                "Household size",
                min_value=1, max_value=20, value=1,
                help="Number of people in the household",
            )
        with _fc2:
            _urgency = st.selectbox("How urgent is this?", URGENCY_OPTIONS)
            _special = st.multiselect(
                "Special circumstances",
                [
                    "Has children under 18",
                    "Senior 65+",
                    "Disability or accessibility needs",
                    "Veteran",
                    "Domestic violence survivor",
                    "Asylum seeker / immigration",
                    "Non-English speaker",
                    "Medicaid / uninsured",
                ],
                help="Select all that apply",
            )

        _extra = st.text_area(
            "Additional context (optional)",
            placeholder="Any details that might help us find the right services — income, specific neighborhood, languages spoken, etc.",
            height=80,
        )

        _form_submitted = st.form_submit_button(
            "Find Help",
            type="primary",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── OR: Free-text input ────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;color:#94A3B8;font-size:13px;margin:4px 0 12px;">— or describe the situation directly —</div>',
        unsafe_allow_html=True,
    )

    _free_query = st.text_area(
        "Describe the situation:",
        value=st.session_state.get("query_input", ""),
        height=90,
        placeholder="e.g. 'Single mother with 3 kids facing eviction in the Bronx next week, income under $30K'",
        key="query_input",
        label_visibility="collapsed",
    )
    _run_free = st.button("Find Help", type="primary", use_container_width=True, key="run_free_btn")

    # ── Build query from guided form ───────────────────────────────────────────
    if _form_submitted:
        _parts = []
        if _sel_cat:
            _parts.append(f"I need help with {_sel_cat.lower()}")
        _bc = BOROUGH_MAP.get(_borough_choice)
        if _bc:
            _parts.append(f"in {BOROUGH_FULL.get(_bc, _borough_choice)}")
        if _household > 1:
            _parts.append(f"for a household of {_household} people")
        if _urgency.startswith("Emergency"):
            _parts.append("This is an emergency — I need immediate help")
        elif _urgency.startswith("Urgent"):
            _parts.append("This is urgent")
        if _special:
            _parts.append("Circumstances include: " + ", ".join(_special))
        if _extra.strip():
            _parts.append(_extra.strip())
        _built_query = ". ".join(_parts) + "."
        st.session_state.query_input = _built_query
        st.session_state.active_query = _built_query
        st.session_state._run_query = True
        st.rerun()

    if _run_free and _free_query.strip():
        st.session_state.active_query = _free_query.strip()
        st.session_state._run_query = True
        st.rerun()


# ── PIPELINE EXECUTION ─────────────────────────────────────────────────────────
_run_query     = st.session_state.pop("_run_query", False)
_effective_q   = (
    st.session_state.get("active_query", "")
    if (_clarify_rerun or _run_query)
    else ""
)

if (_run_query or _clarify_rerun) and _effective_q.strip():
    from pipeline.executor import set_excluded_resources
    set_excluded_resources(get_excluded_resources(st.session_state))

    if not _clarify_rerun:
        st.session_state.conv_history   = []
        st.session_state.active_query   = _effective_q

    _t0 = time.time()
    with st.spinner("Finding the best services for you..."):
        try:
            _plan = generate_plan(_effective_q)
        except Exception as _e:
            st.error(f"Could not process your request: {_e}")
            st.stop()
        try:
            _result = execute(_plan)
        except Exception as _e:
            st.error(f"Could not retrieve services: {_e}")
            st.stop()
        try:
            _response = answer(_effective_q, _plan, _result)
        except Exception as _e:
            _response = f"Unable to generate summary: {_e}"

    with st.spinner("Verifying information accuracy..."):
        _verification = verify_answer(_response, _result)

    _total = time.time() - _t0

    # Store in session state and rerun to render results
    st.session_state._last_plan         = _plan
    st.session_state._last_result       = _result
    st.session_state._last_response     = _response
    st.session_state._last_verification = _verification
    st.session_state._last_timing       = {"total": _total}
    st.rerun()


# ── RESULTS VIEW ───────────────────────────────────────────────────────────────
if has_results:
    plan         = st.session_state._last_plan
    result       = st.session_state._last_result
    response     = st.session_state._last_response
    verification = st.session_state._last_verification
    timing       = st.session_state._last_timing
    active_q     = st.session_state.active_query
    result_intent = result.get("intent", "")

    # ── Verification banner ────────────────────────────────────────────────────
    _v_conf   = verification.get("confidence", "LOW")
    _verified = verification.get("verified", False)
    if _verified and _v_conf == "HIGH":
        _b_bg, _b_icon, _b_txt = (
            "linear-gradient(90deg,#14532D,#16A34A)",
            "&#10003;",
            "High confidence &nbsp;·&nbsp; All claims verified against NYC data",
        )
    elif _verified:
        _b_bg, _b_icon, _b_txt = (
            "linear-gradient(90deg,#1E3A8A,#1D4ED8)",
            "&#10003;",
            "Verified &nbsp;·&nbsp; Most information supported by data",
        )
    else:
        _b_bg, _b_icon, _b_txt = (
            "linear-gradient(90deg,#9A3412,#EA580C)",
            "&#9888;",
            "Needs Review &nbsp;·&nbsp; Some information may require manual confirmation",
        )

    st.markdown(f"""
    <div style="background:{_b_bg};color:#FFFFFF;padding:10px 18px;border-radius:8px;
         margin-bottom:16px;display:flex;align-items:center;gap:10px;
         box-shadow:0 2px 8px rgba(0,0,0,0.15);">
        <span style="font-size:16px;font-weight:700;">{_b_icon}</span>
        <span style="font-size:13px;font-weight:600;">{_b_txt}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Answer summary card ────────────────────────────────────────────────────
    _resp_html = response.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    _resp_html = _resp_html.replace("\n", "<br>")
    st.markdown(f"""
    <div style="background:#FFFFFF;border-radius:12px;padding:22px 24px;
         box-shadow:0 2px 12px rgba(0,0,0,0.07);margin-bottom:20px;
         border-left:4px solid #1D4ED8;">
        <p style="font-size:14px;font-weight:600;color:#64748B;margin:0 0 8px;text-transform:uppercase;
                  letter-spacing:0.5px;">Recommendation</p>
        <div style="font-size:15px;line-height:1.75;color:#1E293B;">{_resp_html}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Resource cards ─────────────────────────────────────────────────────────
    if result_intent == "lookup":
        _df = result.get("results", pd.DataFrame())
        render_resource_cards(_df)

    elif result_intent == "needs_assessment":
        for _need_key, _df in result.get("results_by_need", {}).items():
            render_resource_cards(_df, _need_key.replace("_", " ").title())

    elif result_intent == "simulate":
        _scenario = result.get("scenario", "")
        if _scenario == "cold_emergency":
            _mc1, _mc2, _mc3 = st.columns(3)
            _mc1.metric("People Displaced", result.get("people_displaced", "—"))
            _mc2.metric("Temperature", f"{result.get('temperature_f', '—')}°F")
            _mc3.metric("Overflow Sites", len(result.get("overflow_sites", [])))
            render_resource_cards(pd.DataFrame(result.get("available_shelters", [])), "Available Shelters")
            _of_df = pd.DataFrame(result.get("overflow_sites", []))
            if not _of_df.empty:
                st.markdown(
                    '<h4 style="font-size:14px;font-weight:700;color:#0F172A;margin:16px 0 8px;">'
                    'Assembly / Overflow Sites</h4>',
                    unsafe_allow_html=True,
                )
                st.dataframe(_of_df, use_container_width=True)

        elif _scenario == "resource_gap":
            _gc1, _gc2 = st.columns([1, 2])
            _gc1.metric("Most Underserved", result.get("most_underserved", "?"))
            _gc2.dataframe(pd.DataFrame(result.get("gaps", [])), use_container_width=True)

        elif _scenario == "capacity_change":
            st.dataframe(pd.DataFrame(result.get("summary", [])), use_container_width=True)

        elif _scenario == "migrant_allocation":
            st.dataframe(pd.DataFrame(result.get("allocation", [])), use_container_width=True)
            for _mn, _mrows in result.get("resources_by_need", {}).items():
                render_resource_cards(pd.DataFrame(_mrows), _mn.replace("_", " ").title())

    elif result_intent == "explain":
        _ranking = result.get("ranking", [])
        if _ranking:
            st.markdown(
                '<h4 style="font-size:14px;font-weight:700;color:#0F172A;margin:16px 0 8px;">'
                'Borough Ranking (resources per 100K residents)</h4>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                pd.DataFrame(_ranking, columns=["Borough", "Per 100K"]),
                use_container_width=True,
            )
        _weakest = result.get("weakest_types", [])
        if _weakest:
            st.markdown(
                '<h4 style="font-size:14px;font-weight:700;color:#0F172A;margin:16px 0 8px;">'
                'Most Lacking Resource Types</h4>',
                unsafe_allow_html=True,
            )
            for _wt, _wv in _weakest:
                st.write(f"**{_wt}:** {_wv} per 100K")

    # ── Feedback ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div style="height:12px"></div>'
        '<p style="font-size:13px;color:#64748B;margin-bottom:4px;">'
        'See incorrect or outdated information? Let us know:</p>',
        unsafe_allow_html=True,
    )
    _fb_text = st.text_input(
        "Report an issue:",
        key="feedback_input",
        placeholder="e.g. 'The shelter at 66 Boerum Place is full' or 'This phone number is wrong'",
        label_visibility="collapsed",
    )
    if _fb_text and _fb_text.strip():
        _rec = []
        if result.get("intent") == "lookup":
            _rd = result.get("results")
            if isinstance(_rd, pd.DataFrame) and len(_rd):
                _rec = _rd[["name", "address"]].to_dict("records")
        elif result.get("intent") == "needs_assessment":
            for _rdf in result.get("results_by_need", {}).values():
                if isinstance(_rdf, pd.DataFrame) and len(_rdf):
                    _rec += _rdf[["name", "address"]].to_dict("records")

        with st.spinner("Processing feedback..."):
            _fb = parse_feedback(_fb_text, _rec)

        if _fb.get("resource_name") and _fb["resource_name"] != "unknown":
            add_exclusion(st.session_state, _fb["resource_name"], _fb["issue"], _fb["detail"])
            _alt = generate_alternative_response(active_q, _fb, get_excluded_resources(st.session_state))
            st.info(_alt)
            from pipeline.executor import set_excluded_resources
            set_excluded_resources(get_excluded_resources(st.session_state))
            st.session_state["_clarify_rerun"] = True
            st.rerun()
        else:
            st.warning("Could not identify the resource. Try mentioning its name or address specifically.")

    # ── Multi-turn clarification ───────────────────────────────────────────────
    _turn = len(st.session_state.conv_history)
    if _turn < 2:
        with st.spinner("Checking for follow-up..."):
            _clarify_q = get_clarifying_question(
                st.session_state.active_query, response, _turn
            )
        if _clarify_q:
            st.session_state["_pending_clarify_q"]    = _clarify_q
            st.session_state["_pending_clarify_turn"] = _turn
            with st.chat_message("assistant", avatar="🗽"):
                st.write(_clarify_q)
            st.text_input(
                "Your answer:",
                key=f"clarify_{_turn}",
                placeholder="Type your answer and press Enter...",
            )

    # ── ADVANCED VIEW (collapsed, for admins/auditors) ─────────────────────────
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    with st.expander("Advanced View — Technical Details", expanded=False):
        st.caption(
            "System information for administrators and auditors. "
            f"Response time: {timing.get('total', 0):.1f}s · Provider: {get_active_provider()}"
        )

        _intent       = plan.get("intent", "unknown")
        _intent_label = {
            "lookup": "Resource Lookup",
            "needs_assessment": "Needs Assessment",
            "simulate": "Simulation",
            "explain": "Explanation",
        }.get(_intent, _intent)

        _tab_plan, _tab_verify, _tab_reason, _tab_map, _tab_kge = st.tabs([
            f"Query Plan ({_intent_label})", "Verification", "Reasoning Path", "Map", "Similar Resources",
        ])

        with _tab_plan:
            if _intent == "needs_assessment":
                _profile = plan.get("client_profile", {})
                _needs   = plan.get("identified_needs", [])
                st.write(
                    f"**Situation:** {_profile.get('situation', '—')}  |  "
                    f"**Borough:** {_profile.get('borough', '?')}  |  "
                    f"**Household:** {_profile.get('household_size', '?')}"
                )
                for _n in sorted(_needs, key=lambda x: x.get("priority", 99)):
                    st.write(f"  {_n['priority']}. **{_n['category']}** — {_n.get('reasoning', '')}")
            elif _intent == "lookup":
                st.write(
                    f"**Types:** {', '.join(plan.get('resource_types', []))}  |  "
                    f"**Filters:** {plan.get('filters', {})}"
                )
            elif _intent == "simulate":
                st.write(f"**Scenario:** {plan.get('scenario')}  |  **Params:** {plan.get('params', {})}")
            if "_parse_error" in plan:
                st.warning(f"Fallback plan used. Raw: {plan['_parse_error']}")
            st.json(plan, expanded=False)

        with _tab_verify:
            _claims     = verification.get("claims", [])
            _v_count    = verification.get("verified_count", 0)
            _t_count    = verification.get("total_count", 0)
            st.markdown(
                f"**{_v_count}/{_t_count} claims verified** &nbsp;·&nbsp; "
                f"{verification.get('summary', '')}"
            )
            for _c in _claims:
                _is_v = _c["verdict"] == "VERIFIED"
                if _is_v:
                    st.success(f"VERIFIED: {_c['claim']}  \n*Evidence: {_c.get('evidence', '')}*")
                else:
                    st.warning(f"UNVERIFIED: {_c['claim']}  \n*Evidence: {_c.get('evidence', '')}*")

        with _tab_reason:
            _rpath = build_reasoning_path(plan, result)
            if result.get("intent") == "explain" and result.get("reasoning_path"):
                _rpath = result["reasoning_path"]

            _rsummary = summarize_reasoning(_rpath, plan, result)
            if _rsummary:
                st.markdown(_rsummary)
                st.divider()

            for _step in _rpath:
                _cum = _step.get("cumulative", 0)
                _col = "🟢" if _cum >= 0.7 else ("🟡" if _cum >= 0.5 else "🔴")
                st.markdown(
                    f"**Hop {_step.get('hop','?')}** {_col} {_step.get('fact','')}  \n"
                    f"*Conf: {_step.get('confidence',0)} · Cumulative: {_cum} · {_step.get('source','')}*"
                )
            if _rpath:
                _oc = _rpath[-1]["cumulative"]
                if _oc >= 0.7:
                    st.success(f"Path confidence: **{_oc}**")
                elif _oc >= 0.4:
                    st.warning(f"Path confidence: **{_oc}**")
                else:
                    st.error(f"Path confidence: **{_oc}**")

        with _tab_map:
            _map_frames = []
            _map_rtypes = []
            if result_intent == "lookup":
                _md = result.get("results")
                if isinstance(_md, pd.DataFrame) and len(_md):
                    _map_frames.append(_md)
                    _map_rtypes = _md["resource_type"].unique().tolist()
            elif result_intent == "needs_assessment":
                for _md in result.get("results_by_need", {}).values():
                    if isinstance(_md, pd.DataFrame) and len(_md):
                        _map_frames.append(_md)
                        _map_rtypes += _md["resource_type"].unique().tolist()
                _map_rtypes = list(dict.fromkeys(_map_rtypes))
            elif result_intent == "simulate":
                _sh = result.get("available_shelters", [])
                if _sh:
                    _map_frames.append(pd.DataFrame(_sh))

            if _map_frames:
                render_map_legend(_map_rtypes)
                _deck = make_resource_map(_map_frames)
                if _deck:
                    st.pydeck_chart(_deck, use_container_width=True)
                else:
                    st.info("No coordinates available for mapping.")
            else:
                st.info("No resources to map for this query.")

        with _tab_kge:
            try:
                if result_intent in ("lookup", "needs_assessment"):
                    _first_df = None
                    if result_intent == "lookup":
                        _first_df = result.get("results")
                    else:
                        for _df in result.get("results_by_need", {}).values():
                            if isinstance(_df, pd.DataFrame) and len(_df):
                                _first_df = _df
                                break

                    if _first_df is not None and isinstance(_first_df, pd.DataFrame) and len(_first_df):
                        if "resource_id" in _first_df.columns:
                            _rid   = _first_df.iloc[0]["resource_id"]
                            _rname = _first_df.iloc[0].get("name", _rid)
                            st.markdown(f"**Resources similar to:** {_rname}")
                            _sim = find_similar(_rid, k=5)
                            if not _sim.empty:
                                _sc = [c for c in ["name", "resource_type", "borough", "address", "similarity"] if c in _sim.columns]
                                st.dataframe(_sim[_sc], use_container_width=True)
                            else:
                                st.info("No similar resources found.")
                        else:
                            _rtypes = plan.get("resource_types", [])
                            _boro   = plan.get("filters", {}).get("borough")
                            if _rtypes:
                                st.markdown(f"**Resources matching profile:** {', '.join(_rtypes)} in {_boro or 'NYC'}")
                                _sim = find_similar_to_query(_rtypes, _boro, k=5)
                                if not _sim.empty:
                                    _sc = [c for c in ["name", "resource_type", "borough", "address", "similarity"] if c in _sim.columns]
                                    st.dataframe(_sim[_sc], use_container_width=True)
                    else:
                        st.info("No results to compute similarity from.")
                else:
                    st.info("Similarity search available for lookup and needs assessment queries.")
            except Exception as _ke:
                st.warning(f"Similarity search unavailable: {_ke}")

    # ── Previously reported feedback summary ───────────────────────────────────
    if st.session_state.get("feedback_log"):
        with st.expander(f"Reported Issues ({len(st.session_state['feedback_log'])})", expanded=False):
            for _fb_item in st.session_state["feedback_log"]:
                st.markdown(f"~~{_fb_item['resource']}~~ — *{_fb_item['issue']}*: {_fb_item['detail']}")
