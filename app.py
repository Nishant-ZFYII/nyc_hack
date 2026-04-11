"""
app.py — NYC Social Services Intelligence Engine (Streamlit demo)

Run: streamlit run app.py
"""
import sys
import time
from pathlib import Path

# GPU-accelerate all pandas operations transparently
try:
    import cudf.pandas
    cudf.pandas.install()
except ImportError:
    pass

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

    # Triples + KGE status
    try:
        triples_df = pd.read_parquet(str(ROOT / "data" / "triples.parquet"))
        st.success(f"Triples: {len(triples_df):,} SPO facts")
        st.info(f"Predicates: {triples_df['predicate'].nunique()} · Sources: {triples_df['source'].nunique()}")
    except Exception:
        st.warning("Triples: not built yet (run build_triples.py)")

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
        "🏗️  Sim — Capacity change": "What happens if we add 500 shelter beds in the Bronx?",
        "🌍 Sim — Migrant allocation": "A migrant bus just arrived with 80 people who speak Spanish and Mandarin. They need shelter, food, and schools for children.",
        "🧠 Explain — Why underserved?": "Why is the Bronx the most underserved borough for social services?",
        "🧠 Explain — Emergency confidence": "How confident are you in the cold emergency plan for Brooklyn?",
    }

    for label, eq in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state["query_input"] = eq

    st.divider()
    if st.button("🔄 Start Over", use_container_width=True):
        for k in ["conv_history", "active_query", "_pending_clarify_q",
                  "_pending_clarify_turn", "_clarify_rerun", "query_input",
                  "excluded_resources", "feedback_log"]:
            st.session_state.pop(k, None)
        st.rerun()


# ── Main query interface ──────────────────────────────────────────────────────
query = st.text_area(
    "Describe the situation or ask a question:",
    value=st.session_state.get("query_input", ""),
    height=100,
    placeholder="e.g. 'I'm a family of 4 losing housing next week in Brooklyn. What help is available?'",
    key="query_input",
)

run = st.button("Run Query", type="primary", use_container_width=True)

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
    with st.spinner("Phase 2 — Verifying claims against data…"):
        t4 = time.time()
        verification = verify_answer(response, result)
        verify_time = time.time() - t4

    total_time = time.time() - t0

    # Verification banner
    v_conf = verification.get("confidence", "LOW")
    if verification.get("verified"):
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
