"""
pipeline/synth.py — Synthesize a human-readable answer from plan + results.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm.client import synth_chat

SYNTH_PROMPT = """You are an NYC Department of Social Services AI assistant.
Given a user query and structured data results, write a clear, helpful, concise answer.
- For needs assessments: address each identified need, name specific resources with addresses
- For lookups: list the resources with name, address, and any relevant details
- For simulations: summarize the emergency response plan
- Be warm but professional — real people's lives are affected
- Max 150 words. No markdown headers. Just clear prose."""


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


def answer(nl_query: str, plan: dict, result: dict) -> str:
    """Generate a natural language answer from the query, plan, and results."""
    context = _format_results(result)
    messages = [
        {"role": "system", "content": SYNTH_PROMPT},
        {"role": "user",   "content": f"Query: {nl_query}\n\nData:\n{context}"},
    ]
    response = synth_chat(messages)
    # If LLM returns None or empty, fall back to structured summary
    if not response or response.strip().lower() in ("none", ""):
        return _fallback_answer(result, context)
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
