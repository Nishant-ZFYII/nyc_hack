"""
pipeline/synth.py — Synthesize a human-readable answer from plan + results.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.client import synth_chat

SYNTH_PROMPT = """You are an NYC Department of Social Services AI assistant.
Write a clear, helpful, concise answer.
- Name specific resources from the data with addresses
- Be warm but professional — real people's lives are affected
- Max 150 words. No markdown headers. Just clear prose.

CRITICAL NYC FACTS (always true, never contradict these):
- Emergency shelter is a LEGAL RIGHT in NYC. Nobody can be turned away.
- You can apply for benefits WITHOUT an ID (use Request for Proof form at HRA).
- Children can enroll in NYC public school WITHOUT an address (McKinney-Vento law).
- All NYC public school students get free meals regardless of income.
- Hospitals must treat emergencies regardless of insurance or immigration status (EMTALA).
- Food pantries do NOT check immigration status or require ID.
- NYC agencies cannot share your info with ICE.

If the user mentions NO ID, NO address, undocumented status, or no insurance — REASSURE them that help is still available. DO NOT tell them to get those things first."""


def _format_results(result: dict) -> str:
    """Format executor results into a compact text for the LLM synthesizer."""
    intent = result.get("intent", "")

    if intent == "lookup":
        df = result.get("results", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and len(df):
            lines = []
            for _, row in df.iterrows():
                name = row.get("name", "Unknown")
                addr = row.get("address", "")
                rtype = row.get("resource_type", "")
                boro = row.get("borough", "")
                lines.append(f"- {name} ({rtype}, {boro}): {addr}")
            return "\n".join(lines)
        return "No results found."

    elif intent == "needs_assessment":
        profile = result.get("client_profile", {})
        needs   = result.get("identified_needs", [])
        results_by_need = result.get("results_by_need", {})

        lines = [f"Client: {profile.get('situation', 'unknown situation')}, "
                 f"household {profile.get('household_size', '?')}, borough {profile.get('borough', '?')}"]
        lines.append("\nIdentified needs (prioritized):")
        for n in sorted(needs, key=lambda x: x.get("priority", 99)):
            lines.append(f"  {n['priority']}. {n['category']}: {n.get('reasoning', '')}")
        lines.append("\nResources found:")
        for key, df in results_by_need.items():
            if isinstance(df, pd.DataFrame) and len(df):
                lines.append(f"\n  [{key}]")
                for _, row in df.head(3).iterrows():
                    lines.append(f"    - {row.get('name','?')} ({row.get('borough','?')}): {row.get('address','')}")
        return "\n".join(lines)

    elif intent == "simulate":
        scenario = result.get("scenario", "")
        if scenario == "cold_emergency":
            return (
                f"Cold emergency response for {result.get('people_displaced')} people "
                f"at {result.get('temperature_f')}°F.\n"
                f"Available shelters: {len(result.get('available_shelters', []))}\n"
                f"Overflow sites: {len(result.get('overflow_sites', []))}\n"
                f"Food distribution: {len(result.get('food_distribution', []))}\n"
                f"Recommendation: {result.get('recommendation', '')}"
            )
        elif scenario == "resource_gap":
            gaps = result.get("gaps", [])
            lines = ["Resource gap analysis by borough:"]
            for g in gaps:
                lines.append(f"  {g['borough']}: {g['resources_per_100k']} resources per 100K people")
            lines.append(f"Most underserved: {result.get('most_underserved', '?')}")
            return "\n".join(lines)
        return str(result)

    return str(result)


def _detect_concerns(query: str) -> list:
    """Detect common concerns that need explicit reassurance in the answer."""
    q = query.lower()
    concerns = []
    if any(p in q for p in ["no id", "don't have an id", "without id", "lost my id",
                              "no documents", "no papers"]):
        concerns.append("NO_ID")
    if any(p in q for p in ["undocumented", "no papers", "not legal",
                              "no status", "afraid of ice", "deportation"]):
        concerns.append("UNDOCUMENTED")
    if any(p in q for p in ["no insurance", "don't have insurance", "uninsured"]):
        concerns.append("NO_INSURANCE")
    if any(p in q for p in ["no address", "nowhere to live", "no permanent address"]):
        concerns.append("NO_ADDRESS")
    if any(p in q for p in ["no money", "broke", "no cash", "can't pay"]):
        concerns.append("NO_MONEY")
    return concerns


REASSURANCE = {
    "NO_ID": (
        "You can still get help without an ID. At HRA, ask for a 'Request for Proof' form "
        "— staff will help you apply for benefits and get a new ID."
    ),
    "UNDOCUMENTED": (
        "Immigration status does NOT prevent you from getting emergency shelter, food, "
        "or healthcare in NYC. Your info is protected — NYC agencies cannot share it with ICE."
    ),
    "NO_INSURANCE": (
        "You can still be treated. Hospitals must provide emergency care regardless of "
        "insurance (EMTALA law). NYC Health + Hospitals offers care on a sliding-scale based on income."
    ),
    "NO_ADDRESS": (
        "You can still apply for benefits, enroll children in school (McKinney-Vento law), "
        "and get shelter without a permanent address."
    ),
    "NO_MONEY": (
        "Emergency shelter, food, and emergency care are FREE. For transit, HRA Benefits "
        "Centers provide free MetroCards for people going to benefits appointments."
    ),
}


def answer(nl_query: str, plan: dict, result: dict) -> str:
    """Generate a natural language answer from the query, plan, and results."""
    context = _format_results(result)

    # Detect concerns mentioned in the query → inject reassurance directly
    concerns = _detect_concerns(nl_query)
    reassurance_text = ""
    if concerns:
        lines = ["IMPORTANT — include these facts verbatim in your answer:"]
        for c in concerns:
            lines.append(f"- {REASSURANCE[c]}")
        reassurance_text = "\n".join(lines) + "\n\n"

    messages = [
        {"role": "system", "content": SYNTH_PROMPT},
        {"role": "user",   "content": f"Query: {nl_query}\n\n{reassurance_text}Data:\n{context}"},
    ]
    response = synth_chat(messages)
    # If LLM returns None or empty, fall back to structured summary
    if not response or response.strip().lower() in ("none", ""):
        return _fallback_answer(result, context)

    # Post-hoc: if concerns exist but reassurance is missing, prepend it
    if concerns and response:
        resp_lower = response.lower()
        # If the response doesn't mention the reassurance, prepend it
        for c in concerns:
            key_phrase = REASSURANCE[c][:40].lower()
            if key_phrase not in resp_lower:
                response = f"{REASSURANCE[c]}\n\n{response}"
                break

    return response


def _fallback_answer(result: dict, context: str) -> str:
    """Plain-text fallback when LLM synthesis fails."""
    intent   = result.get("intent", "")
    scenario = result.get("scenario", "")

    if intent == "simulate" and scenario == "cold_emergency":
        shelters  = result.get("available_shelters", [])
        overflow  = result.get("overflow_sites", [])
        food      = result.get("food_distribution", [])
        people    = result.get("people_displaced", "?")
        temp      = result.get("temperature_f", "?")
        lines = [
            f"Cold emergency response for {people} people at {temp}°F:",
            f"• {len(shelters)} available shelters identified in Brooklyn:",
        ]
        for s in shelters:
            lines.append(f"  - {s.get('name','?')} — {s.get('address','')}")
        lines.append(f"• {len(overflow)} PLUTO assembly-zoned overflow sites activated")
        lines.append(f"• {len(food)} food distribution points opened nearby")
        return "\n".join(lines)

    elif intent == "lookup":
        df = result.get("results")
        if df is not None and len(df):
            lines = [f"Found {len(df)} resources:"]
            for _, row in df.iterrows():
                lines.append(f"• {row.get('name','?')} ({row.get('borough','?')}) — {row.get('address','')}")
            return "\n".join(lines)

    return context
