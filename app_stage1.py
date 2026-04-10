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

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.planner  import generate_plan
from pipeline.executor import graph_nearest
from pipeline.executor import execute, load_state
from pipeline.synth    import answer
from llm.client        import get_active_provider

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

st.title("🗽 NYC Social Services Intelligence Engine")
st.caption("Powered by NVIDIA Nemotron · cuGraph · RAPIDS — DGX Spark")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")

    # LLM provider
    try:
        provider = get_active_provider()
        st.success(f"LLM: {provider}")
    except Exception as e:
        st.error(f"LLM: {e}")

    # Data status
    try:
        mart, payload = load_state()
        st.success(f"Mart: {len(mart):,} resources")
        st.success(f"Graph: {payload['graph'].number_of_nodes():,} nodes · "
                   f"{payload['graph'].number_of_edges():,} edges")
        st.info(f"Backend: {payload.get('backend', 'networkx')}")
    except Exception as e:
        st.error(f"Data: {e}")

    st.divider()
    st.subheader("Try these queries")

    EXAMPLES = {
        "🏠 Caseworker — Family at risk": "I'm Tina, I have 4 kids ages 12-16, my income is $28K, and my sister is kicking us out of her Flatbush apartment next week. What do we do?",
        "🚨 Caseworker — Crime victim": "Someone broke into my apartment last night. I don't feel safe going back. I have a 6 year old. What should I do?",
        "🌍 Caseworker — Migrant family": "I just arrived from Haiti with my two children. We speak Haitian Creole and need shelter tonight near Flatbush.",
        "🔍 Lookup — Brooklyn shelters": "What shelters in Brooklyn have available beds right now?",
        "🏥 Lookup — Wheelchair hospitals": "Find wheelchair accessible hospitals near Jamaica Queens that accept Medicaid",
        "🍎 Lookup — Bronx food banks": "How many food banks are open in the Bronx?",
        "❄️  Sim — Cold emergency": "A cold emergency is declared. 3 Brooklyn shelters just hit capacity. 200 people are still outside. It's 15°F. What do we do?",
        "📊 Sim — Resource gap": "Which NYC boroughs are most underserved by social services?",
    }

    for label, query in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["query_input"] = query


# ── Main query interface ──────────────────────────────────────────────────────
query = st.text_area(
    "Describe the situation or ask a question:",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="e.g. 'I'm a family of 4 losing housing next week in Brooklyn. What help is available?'",
    key="query_input",
)

run = st.button("Run Query", type="primary", use_container_width=True)

if run and query.strip():
    t0 = time.time()

    col1, col2 = st.columns([1, 1])

    # ── Step 1: Plan ──────────────────────────────────────────────────────────
    with col1:
        with st.spinner("Step 1/3 — Generating plan…"):
            t1 = time.time()
            try:
                plan = generate_plan(query)
                plan_time = time.time() - t1
            except Exception as e:
                st.error(f"Planner error: {e}")
                st.stop()

        st.subheader("Query Plan")
        intent = plan.get("intent", "unknown")
        intent_color = {"lookup": "🔍", "needs_assessment": "🏥", "simulate": "⚡"}.get(intent, "❓")
        st.write(f"{intent_color} **Intent:** `{intent}`")

        if intent == "needs_assessment":
            profile = plan.get("client_profile", {})
            needs   = plan.get("identified_needs", [])
            st.write(f"**Situation:** {profile.get('situation', '—')}")
            st.write(f"**Borough:** {profile.get('borough', 'Not specified')} · "
                     f"**Household:** {profile.get('household_size', '?')}")
            if needs:
                st.write("**Identified needs:**")
                for n in sorted(needs, key=lambda x: x.get("priority", 99)):
                    st.write(f"  {n['priority']}. **{n['category']}** — {n.get('reasoning', '')}")
        elif intent == "lookup":
            st.write(f"**Resource types:** {', '.join(plan.get('resource_types', []))}")
            filters = plan.get("filters", {})
            if any(filters.values()):
                st.write(f"**Filters:** {filters}")
        elif intent == "simulate":
            st.write(f"**Scenario:** `{plan.get('scenario', '?')}`")
            st.write(f"**Params:** {plan.get('params', {})}")

        if "_parse_error" in plan:
            st.warning(f"Plan parse fallback used. Raw: {plan['_parse_error']}")

        st.caption(f"Plan time: {plan_time:.2f}s")
        st.json(plan, expanded=False)

    # ── Step 2: Execute ───────────────────────────────────────────────────────
    with col2:
        with st.spinner("Step 2/3 — Executing against knowledge graph…"):
            t2 = time.time()
            try:
                result = execute(plan)
                exec_time = time.time() - t2
            except Exception as e:
                st.error(f"Executor error: {e}")
                st.stop()

        st.subheader("Results")
        result_intent = result.get("intent", "")

        if result_intent == "lookup":
            df = result.get("results", pd.DataFrame())
            if isinstance(df, pd.DataFrame) and len(df):
                display_cols = [c for c in ["name", "resource_type", "borough",
                                            "address", "safety_score", "quality_score",
                                            "nearest_transit_name", "nearest_transit_walk_min"]
                                if c in df.columns]
                st.dataframe(df[display_cols], use_container_width=True)
            else:
                st.warning("No matching resources found.")

        elif result_intent == "needs_assessment":
            results_by_need = result.get("results_by_need", {})
            for need_key, df in results_by_need.items():
                with st.expander(f"📌 {need_key}", expanded=True):
                    if isinstance(df, pd.DataFrame) and len(df):
                        display_cols = [c for c in ["name", "resource_type", "borough",
                                                     "address", "nearest_transit_walk_min"]
                                        if c in df.columns]
                        st.dataframe(df[display_cols], use_container_width=True)
                    else:
                        st.write("No resources found for this need.")

        elif result_intent == "simulate":
            scenario = result.get("scenario", "")
            if scenario == "cold_emergency":
                c1, c2, c3 = st.columns(3)
                c1.metric("People displaced", result.get("people_displaced"))
                c2.metric("Temperature", f"{result.get('temperature_f')}°F")
                c3.metric("Overflow sites found", len(result.get("overflow_sites", [])))
                st.write(f"**Recommendation:** {result.get('recommendation', '')}")
                with st.expander("Available shelters"):
                    st.dataframe(pd.DataFrame(result.get("available_shelters", [])))
                with st.expander("Overflow sites (PLUTO assembly buildings)"):
                    st.dataframe(pd.DataFrame(result.get("overflow_sites", [])))
            elif scenario == "resource_gap":
                st.write(f"**Most underserved borough:** {result.get('most_underserved', '?')}")
                st.dataframe(pd.DataFrame(result.get("gaps", [])))

        st.caption(f"Execute time: {exec_time:.2f}s")

    # ── Map ───────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Map")

    map_frames = []
    map_rtypes = []
    if result.get("intent") == "lookup":
        df = result.get("results")
        if isinstance(df, pd.DataFrame) and len(df):
            map_frames.append(df)
            map_rtypes = df["resource_type"].unique().tolist()
    elif result.get("intent") == "needs_assessment":
        for df in result.get("results_by_need", {}).values():
            if isinstance(df, pd.DataFrame) and len(df):
                map_frames.append(df)
                map_rtypes += df["resource_type"].unique().tolist()
        map_rtypes = list(dict.fromkeys(map_rtypes))
    elif result.get("intent") == "simulate":
        shelters = result.get("available_shelters", [])
        overflow = result.get("overflow_sites", [])
        if shelters:
            map_frames.append(pd.DataFrame(shelters))
        if overflow:
            ov_df = pd.DataFrame(overflow)
            if "latitude" not in ov_df.columns and "lat" in ov_df.columns:
                ov_df = ov_df.rename(columns={"lat": "latitude", "lon": "longitude"})
            ov_df["resource_type"] = "community_center"
            map_frames.append(ov_df)

    if map_frames:
        render_map_legend(map_rtypes)
        deck = make_resource_map(map_frames)
        if deck:
            st.pydeck_chart(deck, use_container_width=True)
        else:
            st.info("Resources found but no coordinates available for map.")
    else:
        st.info("No resources to map.")

    # ── Step 3: Synthesize ────────────────────────────────────────────────────
    st.divider()
    with st.spinner("Step 3/3 — Synthesizing answer…"):
        t3 = time.time()
        try:
            response = answer(query, plan, result)
            synth_time = time.time() - t3
        except Exception as e:
            response = f"Synthesis error: {e}"
            synth_time = 0

    total_time = time.time() - t0

    st.subheader("Answer")
    st.write(response)

    st.caption(
        f"⏱ Total: **{total_time:.2f}s** · "
        f"Plan: {plan_time:.2f}s · "
        f"Execute: {exec_time:.2f}s · "
        f"Synth: {synth_time:.2f}s · "
        f"LLM: {get_active_provider()}"
    )
