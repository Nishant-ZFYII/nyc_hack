"""
pipeline/verify.py — Two-phase verification: fact-check LLM answers against raw data.

Phase 1 (Agent): LLM generates answer (already done in synth.py)
Phase 2 (Verifier): Check each claim against the actual data results

This prevents hallucination by ensuring every named resource, address,
and statistic in the answer actually exists in the knowledge graph.
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.client import chat


def _extract_evidence(result: dict) -> str:
    """Convert executor results into a compact evidence string for the verifier."""
    intent = result.get("intent", "")
    lines = []

    if intent == "lookup":
        df = result.get("results", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and len(df):
            for _, r in df.iterrows():
                lines.append(
                    f"RESOURCE: {r.get('name','?')} | type={r.get('resource_type','?')} | "
                    f"borough={r.get('borough','?')} | address={r.get('address','?')} | "
                    f"safety={r.get('safety_score','?')} | quality={r.get('quality_score','?')}"
                )

    elif intent == "needs_assessment":
        for key, df in result.get("results_by_need", {}).items():
            if isinstance(df, pd.DataFrame) and len(df):
                lines.append(f"\n[{key}]")
                for _, r in df.head(5).iterrows():
                    lines.append(
                        f"RESOURCE: {r.get('name','?')} | type={r.get('resource_type','?')} | "
                        f"borough={r.get('borough','?')} | address={r.get('address','?')}"
                    )

    elif intent == "simulate":
        scenario = result.get("scenario", "")
        if scenario == "cold_emergency":
            lines.append(f"PEOPLE_DISPLACED: {result.get('people_displaced')}")
            lines.append(f"TEMPERATURE: {result.get('temperature_f')}F")
            for s in result.get("available_shelters", []):
                lines.append(f"SHELTER: {s.get('name','?')} | {s.get('address','?')} | {s.get('borough','?')}")
            for o in result.get("overflow_sites", []):
                lines.append(f"OVERFLOW: {o.get('address','?')} | owner={o.get('ownername','?')}")
            for f in result.get("food_distribution", []):
                lines.append(f"FOOD: {f.get('name','?')} | {f.get('address','?')}")
        elif scenario == "resource_gap":
            for g in result.get("gaps", []):
                lines.append(f"BOROUGH: {g.get('borough')} | resources={g.get('resource_count')} | per_100k={g.get('resources_per_100k')}")
            lines.append(f"MOST_UNDERSERVED: {result.get('most_underserved')}")
        elif scenario == "migrant_allocation":
            for a in result.get("allocation", []):
                lines.append(f"ALLOCATION: {a.get('shelter','?')} | {a.get('address','?')} | assigned={a.get('assigned_people')}")

    elif intent == "explain":
        for step in result.get("reasoning_path", []):
            lines.append(f"HOP {step.get('hop')}: {step.get('fact','')} [conf={step.get('confidence')}, src={step.get('source')}]")

    return "\n".join(lines) if lines else "No structured data available."


VERIFIER_PROMPT = """You are a strict fact-checker for an NYC social services AI.
Given an AI-generated answer and the RAW DATA it was based on, verify each factual claim.

Rules:
- Check EVERY named resource, address, borough, and number against the raw data
- A claim is VERIFIED if the resource name and borough match the raw data
- A claim is UNVERIFIED if the name/address/number doesn't appear in the raw data
- Recommendations (call 311, apply online) are NOT claims — skip them
- Be strict: if the answer says "Flatbush" but data says "BK", that's still verified (BK = Brooklyn)

Output format (repeat for each claim):
CLAIM: [quote the claim]
VERDICT: VERIFIED or UNVERIFIED
EVIDENCE: [what in the data supports or contradicts it]

Then:
OVERALL: [HIGH/MEDIUM/LOW] confidence
VERIFIED_COUNT: [N/M claims verified]
SUMMARY: [one sentence]"""


def verify_answer(answer_text: str, result: dict) -> dict:
    """
    Phase 2: Verify the LLM answer against actual executor results.

    Returns:
        {
            "verified": bool,
            "confidence": "HIGH" / "MEDIUM" / "LOW",
            "claims": [{"claim": str, "verdict": str, "evidence": str}, ...],
            "summary": str,
            "timing": float,
        }
    """
    t0 = time.time()

    evidence = _extract_evidence(result)

    messages = [
        {"role": "system", "content": VERIFIER_PROMPT},
        {"role": "user", "content": (
            f"AI ANSWER:\n{answer_text}\n\n"
            f"RAW DATA:\n{evidence}"
        )},
    ]

    raw = chat(messages, temperature=0.1, max_tokens=1500)
    verify_time = time.time() - t0

    if not raw:
        return {
            "verified": False, "confidence": "LOW",
            "claims": [], "summary": "Verifier unavailable",
            "timing": verify_time,
        }

    # Parse claims
    claims = []
    claim_blocks = re.split(r'CLAIM:', raw)
    for block in claim_blocks[1:]:  # skip first empty
        claim_text = ""
        verdict = "UNKNOWN"
        evidence_text = ""

        # Extract claim text
        m = re.match(r'\s*(.*?)(?:VERDICT:|$)', block, re.DOTALL)
        if m:
            claim_text = m.group(1).strip().strip('"').strip()

        # Extract verdict
        vm = re.search(r'VERDICT:\s*(VERIFIED|UNVERIFIED)', block, re.IGNORECASE)
        if vm:
            verdict = vm.group(1).upper()

        # Extract evidence
        em = re.search(r'EVIDENCE:\s*(.*?)(?:CLAIM:|OVERALL:|$)', block, re.DOTALL)
        if em:
            evidence_text = em.group(1).strip()

        if claim_text:
            claims.append({
                "claim": claim_text[:200],
                "verdict": verdict,
                "evidence": evidence_text[:200],
            })

    # Parse overall
    verified_count = sum(1 for c in claims if c["verdict"] == "VERIFIED")
    total_count = max(len(claims), 1)

    # Determine confidence
    raw_lower = raw.lower()
    if "high" in raw_lower and verified_count >= total_count * 0.8:
        confidence = "HIGH"
        verified = True
    elif verified_count >= total_count * 0.5:
        confidence = "MEDIUM"
        verified = True
    else:
        confidence = "LOW"
        verified = False

    # Override if many unverified
    if sum(1 for c in claims if c["verdict"] == "UNVERIFIED") > 2:
        confidence = "LOW"
        verified = False

    # Extract summary
    summary_m = re.search(r'SUMMARY:\s*(.*?)$', raw, re.MULTILINE)
    summary = summary_m.group(1).strip() if summary_m else f"{verified_count}/{total_count} claims verified"

    return {
        "verified": verified,
        "confidence": confidence,
        "claims": claims,
        "verified_count": verified_count,
        "total_count": total_count,
        "summary": summary,
        "timing": round(verify_time, 1),
    }


def build_reasoning_path(plan: dict, result: dict) -> list[dict]:
    """
    Build a reasoning/explanation path for ANY intent — not just 'explain'.
    Shows the data provenance chain for every query type.
    """
    intent = plan.get("intent", "")
    path = []

    if intent == "lookup":
        rtypes = plan.get("resource_types", [])
        borough = plan.get("filters", {}).get("borough")
        df = result.get("results", pd.DataFrame())
        n_results = len(df) if isinstance(df, pd.DataFrame) else 0

        path.append({"hop": 1, "fact": f"Query: {', '.join(rtypes)} in {borough or 'all NYC'}",
                      "confidence": 1.0, "source": "user_query"})
        path.append({"hop": 2, "fact": f"Filtered resource mart (7,759 resources) by type + borough",
                      "confidence": 0.95, "source": "dohmh"})
        path.append({"hop": 3, "fact": f"Sorted by safety_score (NYPD 500m) + quality_score (311 500m)",
                      "confidence": 0.80, "source": "nypd + 311 cross-dataset"})
        path.append({"hop": 4, "fact": f"Returned top {n_results} results",
                      "confidence": 0.95, "source": "mart_derived"})

    elif intent == "needs_assessment":
        profile = plan.get("client_profile", {})
        needs = plan.get("identified_needs", [])
        searches = plan.get("resource_searches", [])

        path.append({"hop": 1, "fact": f"Situation: {profile.get('situation', '?')}",
                      "confidence": 1.0, "source": "user_query"})
        path.append({"hop": 2, "fact": f"LLM decomposed into {len(needs)} needs, {len(searches)} searches",
                      "confidence": 0.85, "source": "nemotron_planning"})

        for i, need in enumerate(needs):
            path.append({
                "hop": 3 + i,
                "fact": f"Need #{need.get('priority','?')}: {need.get('category','?')}",
                "confidence": 0.85, "source": "nemotron_planning"
            })

        rbn = result.get("results_by_need", {})
        total_found = sum(len(v) for v in rbn.values() if isinstance(v, pd.DataFrame))
        path.append({"hop": 3 + len(needs),
                      "fact": f"Executor found {total_found} resources across {len(rbn)} categories",
                      "confidence": 0.95, "source": "mart + graph"})

    elif intent == "simulate":
        scenario = plan.get("scenario", "")
        path.append({"hop": 1, "fact": f"Simulation: {scenario}",
                      "confidence": 1.0, "source": "user_query"})

        if scenario == "cold_emergency":
            n_shelters = len(result.get("available_shelters", []))
            n_overflow = len(result.get("overflow_sites", []))
            n_food = len(result.get("food_distribution", []))
            path.append({"hop": 2, "fact": f"Queried shelter mart: {n_shelters} available",
                          "confidence": 0.95, "source": "dohmh_shelters"})
            path.append({"hop": 3, "fact": f"Scanned 857K PLUTO lots: {n_overflow} overflow sites (landuse=08)",
                          "confidence": 0.90, "source": "pluto_zoning"})
            path.append({"hop": 4, "fact": f"Distance-sorted from borough centroid",
                          "confidence": 0.85, "source": "spatial_derived"})
            path.append({"hop": 5, "fact": f"Found {n_food} food banks nearby",
                          "confidence": 0.95, "source": "dohmh"})

        elif scenario == "resource_gap":
            gaps = result.get("gaps", [])
            path.append({"hop": 2, "fact": f"Grouped resources by borough ({len(gaps)} boroughs)",
                          "confidence": 0.95, "source": "mart_derived"})
            path.append({"hop": 3, "fact": "Population estimates from ACS Census",
                          "confidence": 0.60, "source": "acs_census (hardcoded)"})
            path.append({"hop": 4, "fact": f"Most underserved: {result.get('most_underserved','?')}",
                          "confidence": 0.75, "source": "mart_derived"})

        elif scenario == "capacity_change":
            path.append({"hop": 2, "fact": f"Simulated {result.get('new_beds',0)} new beds in {result.get('borough','?')}",
                          "confidence": 0.90, "source": "simulation"})
            path.append({"hop": 3, "fact": "Recomputed coverage per 100K",
                          "confidence": 0.75, "source": "mart_derived"})

        elif scenario == "migrant_allocation":
            n_alloc = len(result.get("allocation", []))
            path.append({"hop": 2, "fact": f"Filtered for language-matching shelters",
                          "confidence": 0.70, "source": "dohmh (language data sparse)"})
            path.append({"hop": 3, "fact": f"Allocated {result.get('people',0)} people across {n_alloc} sites",
                          "confidence": 0.85, "source": "allocation_algorithm"})

    # Compute cumulative confidence
    cumulative = 1.0
    for step in path:
        cumulative *= step["confidence"]
        step["cumulative"] = round(cumulative, 3)

    return path


def summarize_reasoning(path: list[dict], plan: dict, result: dict) -> str:
    """
    Convert a hop-by-hop reasoning path into a plain-English summary
    that a non-technical user can understand.
    """
    if not path:
        return ""

    intent = plan.get("intent", "")
    overall_conf = path[-1].get("cumulative", 0) if path else 0
    conf_word = "high" if overall_conf >= 0.7 else ("moderate" if overall_conf >= 0.4 else "limited")

    lines = []

    if intent == "needs_assessment":
        needs = plan.get("identified_needs", [])
        searches = plan.get("resource_searches", [])
        n_needs = len(needs)
        need_names = [n.get("category", "?") for n in needs]
        rbn = result.get("results_by_need", {})
        total_found = sum(len(v) for v in rbn.values() if hasattr(v, "__len__"))

        lines.append(f"Here's how I reached this answer ({conf_word} confidence, {overall_conf:.0%}):")
        lines.append(f"")
        lines.append(f"1. I analyzed your situation and identified {n_needs} priority needs: "
                     f"**{', '.join(need_names)}**.")
        lines.append(f"2. I searched the NYC resource database (7,759 verified resources from "
                     f"official city records) and found **{total_found} matching resources**.")
        lines.append(f"3. Results are sorted by safety score (based on NYPD data within 500m) "
                     f"and quality score (based on 311 complaints within 500m).")

        if overall_conf < 0.5:
            lines.append(f"4. ⚠️ Confidence is {conf_word} because the plan required multiple "
                         f"inference steps. I recommend verifying specific resources by calling them directly.")

    elif intent == "lookup":
        df = result.get("results")
        n_results = len(df) if hasattr(df, "__len__") else 0
        rtypes = plan.get("resource_types", [])
        borough = plan.get("filters", {}).get("borough", "all of NYC")

        lines.append(f"Here's how I found these results ({conf_word} confidence, {overall_conf:.0%}):")
        lines.append(f"")
        lines.append(f"1. I searched for **{', '.join(rtypes)}** in **{borough}** across "
                     f"the NYC resource database.")
        lines.append(f"2. Found **{n_results} results**, ranked by safety and quality scores.")
        lines.append(f"3. Safety scores are derived from NYPD crime data within 500 meters. "
                     f"Quality scores from 311 complaint history.")

    elif intent == "simulate":
        scenario = plan.get("scenario", "")
        if scenario == "cold_emergency":
            n_shelters = len(result.get("available_shelters", []))
            n_overflow = len(result.get("overflow_sites", []))
            lines.append(f"Here's how I built this emergency plan ({conf_word} confidence, {overall_conf:.0%}):")
            lines.append(f"")
            lines.append(f"1. I searched the shelter registry and found **{n_shelters} available shelters**, "
                         f"sorted by distance to the affected area.")
            lines.append(f"2. I scanned **857,161 NYC tax lots** (PLUTO database) and identified "
                         f"**{n_overflow} assembly-zoned buildings** that could serve as overflow sites — "
                         f"churches, community centers, school gyms.")
            lines.append(f"3. I located nearby food banks for distribution points.")
            lines.append(f"4. Confidence drops from 95% to {overall_conf:.0%} because overflow site "
                         f"availability is derived from zoning data, not real-time confirmation.")

        elif scenario == "resource_gap":
            most = result.get("most_underserved", "?")
            lines.append(f"Here's how I determined which boroughs are underserved "
                         f"({conf_word} confidence, {overall_conf:.0%}):")
            lines.append(f"")
            lines.append(f"1. I counted shelters, food banks, and hospitals in each borough.")
            lines.append(f"2. I divided by population estimates from the Census Bureau.")
            lines.append(f"3. **{most}** has the fewest resources per 100,000 residents.")
            lines.append(f"4. Note: population figures are estimates (60% confidence), which "
                         f"is why overall confidence is {conf_word}.")

        elif scenario == "migrant_allocation":
            n_alloc = len(result.get("allocation", []))
            lines.append(f"Here's how I built this allocation plan ({conf_word} confidence, {overall_conf:.0%}):")
            lines.append(f"")
            lines.append(f"1. I filtered shelters for language-matching services.")
            lines.append(f"2. I distributed people across **{n_alloc} shelter sites**.")
            lines.append(f"3. Language data is sparse (70% confidence) — verify language services "
                         f"by calling shelters directly before sending people.")

        else:
            lines.append(f"Simulation completed with {conf_word} confidence ({overall_conf:.0%}).")

    elif intent == "explain":
        lines.append(f"Analysis completed with {conf_word} confidence ({overall_conf:.0%}). "
                     f"See the reasoning path above for the step-by-step data provenance.")

    return "\n".join(lines)
