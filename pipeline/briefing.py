"""
pipeline/briefing.py — AI-powered admin briefing generator.

Called when a user confirms destination intent. Produces a structured
briefing for frontline staff with:
  - Problem diagnosis + confidence
  - Urgency classification
  - Required documents with gap detection and substitutes
  - Edge case detection with workarounds and fallback paths
  - Service protocol knowledge
  - Pre-arrival instructions for the client
  - Dynamic updates as case state changes
"""
from __future__ import annotations

import json
import re
from datetime import datetime

from llm.client import chat

# ── Optional knowledge augmentation modules ───────────────────────────────────
try:
    from pipeline.eligibility import calculate_eligibility, get_rights, get_stories
    _HAS_ELIGIBILITY = True
except Exception:
    _HAS_ELIGIBILITY = False

try:
    from engine.confidence import explain_resource_recommendation
    _HAS_CONFIDENCE = True
except Exception:
    _HAS_CONFIDENCE = False

try:
    from engine.embeddings import find_similar
    _HAS_EMBEDDINGS = True
except Exception:
    _HAS_EMBEDDINGS = False

# ── Document requirements by resource type ────────────────────────────────────
DOCS_BY_TYPE: dict[str, list[dict]] = {
    "shelter": [
        {"doc": "Photo ID", "required": True,
         "substitutes": ["birth certificate", "passport", "consular ID (Matricula)", "IDNYC card"],
         "note": "DHS accepts a wide range of IDs; no ID should never block emergency shelter"},
        {"doc": "Social Security card or number", "required": False,
         "substitutes": ["ITIN letter", "signed declaration of no SSN", "pending-application notice"],
         "note": "Undocumented individuals may still access shelter — SSN not required for emergency intake"},
    ],
    "food_bank": [
        {"doc": "Proof of address (sometimes)", "required": False,
         "substitutes": ["any mail with name", "caseworker letter", "verbal address", "hotel/motel confirmation"],
         "note": "Most NYC food pantries do not require documents at all"},
    ],
    "hospital": [
        {"doc": "ID (any form)", "required": False,
         "substitutes": ["verbal identification", "emergency waiver", "no ID required for emergency care"],
         "note": "EMTALA mandates emergency treatment regardless of ID or insurance"},
        {"doc": "Insurance card", "required": False,
         "substitutes": ["Medicaid number", "emergency Medicaid application on-site",
                         "sliding-scale self-pay", "NYC Health + Hospitals financial counselor"],
         "note": "NYC H+H hospitals offer financial counseling same-day"},
    ],
    "clinic": [
        {"doc": "ID", "required": False,
         "substitutes": ["any photo ID", "letter from shelter", "utility bill", "verbal ID"],
         "note": "Federally Qualified Health Centers (FQHCs) serve all regardless of documentation"},
        {"doc": "Proof of income or insurance", "required": False,
         "substitutes": ["sliding-scale self-pay", "FHCC eligibility application", "Medicaid application"],
         "note": "Income-based sliding scale available at most city clinics"},
    ],
    "benefits_center": [
        {"doc": "Photo ID", "required": True,
         "substitutes": ["passport", "consular ID", "school ID + utility bill", "IDNYC"],
         "note": "IDNYC is free for NYC residents and accepted for most benefits applications"},
        {"doc": "Proof of NYC address", "required": True,
         "substitutes": ["shelter placement letter", "utility bill", "postmarked mail",
                         "signed statement from landlord or host"],
         "note": "Letter from shelter director is accepted at HRA offices"},
        {"doc": "Proof of income", "required": True,
         "substitutes": ["pay stubs (2 most recent)", "employer letter",
                         "zero-income declaration (signed)", "DHS shelter record as proof of homelessness"],
         "note": "Zero-income declaration is legally accepted; no documents required to prove $0 income"},
        {"doc": "Social Security number", "required": True,
         "substitutes": ["ITIN", "SSN application pending notice",
                         "exempt — undocumented may apply for emergency benefits"],
         "note": "Undocumented immigrants are eligible for emergency Medicaid and limited cash assistance"},
    ],
    "school": [
        {"doc": "Proof of age (birth certificate)", "required": True,
         "substitutes": ["baptismal record", "passport", "hospital birth record",
                         "parent sworn statement (McKinney-Vento)"],
         "note": "Under McKinney-Vento Act, homeless children must be enrolled immediately — documents can follow"},
        {"doc": "Immunization records", "required": True,
         "substitutes": ["parent affidavit pending records", "conditional enrollment up to 30 days",
                         "health center can provide records same-day"],
         "note": "Conditional enrollment allows 30-day grace period under McKinney-Vento"},
        {"doc": "Proof of residency", "required": False,
         "substitutes": ["shelter placement letter", "hotel/motel letter",
                         "McKinney-Vento liaison waiver (immediate enrollment)"],
         "note": "McKinney-Vento liaisons at each school district must facilitate immediate enrollment"},
    ],
    "childcare": [
        {"doc": "Child's birth certificate", "required": True,
         "substitutes": ["passport", "baptismal record", "hospital birth record"],
         "note": ""},
        {"doc": "Parent/guardian photo ID", "required": True,
         "substitutes": ["consular ID", "passport", "IDNYC"],
         "note": ""},
        {"doc": "Income documentation for subsidy", "required": True,
         "substitutes": ["zero-income declaration", "shelter confirmation letter",
                         "TANF/HRA letter as income proxy"],
         "note": "ACS child care subsidies available; processing takes 2–4 weeks"},
    ],
    "legal_aid": [
        {"doc": "Case/court documents (if applicable)", "required": False,
         "substitutes": ["verbal summary accepted", "caseworker notes", "no documents required for intake"],
         "note": "Most legal aid orgs do intake with no documents — they help gather records"},
        {"doc": "ID (preferred but not required)", "required": False,
         "substitutes": ["verbal ID", "any form of ID"],
         "note": ""},
    ],
    "domestic_violence": [
        {"doc": "No documentation required", "required": False, "substitutes": [],
         "note": "Safety planning takes priority — no documents needed to access DV services or shelter"},
    ],
    "mental_health": [
        {"doc": "ID", "required": False,
         "substitutes": ["verbal ID", "most NYC mental health clinics have no-ID policy for crisis"],
         "note": "988 crisis line requires no ID; ER mental health services covered under EMTALA"},
        {"doc": "Insurance", "required": False,
         "substitutes": ["Medicaid", "NYC mental health sliding scale", "uninsured accepted at NYC clinics"],
         "note": ""},
    ],
    "community_center": [
        {"doc": "Proof of borough/neighborhood (sometimes)", "required": False,
         "substitutes": ["verbal address", "any mail piece"],
         "note": "Most community centers are open to all NYC residents"},
    ],
    "senior_services": [
        {"doc": "Proof of age (60+)", "required": True,
         "substitutes": ["any ID showing birthdate", "Medicare card", "Social Security statement"],
         "note": ""},
        {"doc": "Proof of income (for some programs)", "required": False,
         "substitutes": ["zero-income declaration", "SSI/SSDI letter"],
         "note": ""},
    ],
    "default": [
        {"doc": "ID (preferred)", "required": False,
         "substitutes": ["any government-issued ID", "verbal identification"],
         "note": "Contact intake staff if no ID available"},
    ],
}

# ── Service protocols by need category ────────────────────────────────────────
SERVICE_PROTOCOLS: dict[str, str] = {
    "housing": (
        "Intake: assess shelter type needed (single adult vs. family with children). "
        "Families: HRA PATH intake center (151 E. 151st St, Bronx). "
        "Single adults: DHS drop-in centers or call 311. "
        "McKinney-Vento protections for families with school-age children. "
        "If fleeing DV, specialized shelter available — do not list previous address."
    ),
    "shelter": (
        "Emergency shelter is a legal right in NYC under the Callahan Consent Decree. "
        "No one can be turned away in cold weather (below 32°F). "
        "DHS hotline: 311 or 1-800-994-6494. "
        "Safe Haven and drop-in centers for those not ready for shelter system."
    ),
    "food": (
        "No eligibility check required at most NYC food pantries and soup kitchens. "
        "SNAP application assistance often available on-site. "
        "Food Bank for NYC: 1-866-888-8777 for nearest pantry. "
        "WIC available for pregnant women, new mothers, and children under 5."
    ),
    "medical": (
        "EMTALA: hospitals must provide emergency treatment regardless of ability to pay or documentation. "
        "NYC Health + Hospitals financial counselors available same-day. "
        "Ask about NYC Care program (low-cost care for uninsured NYC residents). "
        "Insurance application can be submitted while receiving care."
    ),
    "mental_health": (
        "Mental health crisis: call 988 (Suicide & Crisis Lifeline) or go to any ER. "
        "Mobile Crisis Teams: 1-800-LIFENET for same-day outreach. "
        "Outpatient: most clinics accept Medicaid or offer sliding scale. "
        "No ID required for crisis services. Language access required by law."
    ),
    "benefits": (
        "SNAP, Medicaid, Cash Assistance applications take 30–45 min at HRA. "
        "Apply online at ACCESS HRA app or nyc.gov/hra. "
        "Emergency SNAP available same-day for eligible households. "
        "Fair Fares (50% MetroCard) available for income-eligible New Yorkers."
    ),
    "legal": (
        "Most legal aid is first-come first-served or by appointment. "
        "Emergency legal issues (eviction, custody): call 311 for referral. "
        "Immigration: note DACA/asylum/TPS timelines if relevant. "
        "Language access required — request interpreter if needed."
    ),
    "safety": (
        "Domestic violence: NYC DV Hotline 1-800-621-HOPE (4673), 24/7. "
        "If in immediate danger: call 911. "
        "Safety planning is the first priority before any intake paperwork. "
        "Mandatory reporter obligations may apply for staff members. "
        "Address confidentiality program available for DV survivors."
    ),
    "school": (
        "McKinney-Vento Act: homeless children must be enrolled immediately, even without documents. "
        "Each school district has a McKinney-Vento liaison — contact them directly. "
        "Children in shelter have right to attend school of origin (previous school) if feasible. "
        "Bus transportation available for children in temporary housing."
    ),
    "childcare": (
        "ACS child care subsidies available for income-eligible families. "
        "Processing takes 2–4 weeks; emergency placements sometimes available. "
        "Head Start: free early education for income-eligible children 0–5. "
        "Apply at childcareny.org or through ACS."
    ),
    "employment": (
        "NYC Workforce1 Career Centers: free job placement, resume help, training. "
        "HireNYC: prioritizes NYC residents for city-contracted jobs. "
        "SNAP recipients may qualify for employment and training programs."
    ),
    "senior": (
        "NYC Department for the Aging: 311 or nyc.gov/aging. "
        "SNAP eligibility for seniors often higher than general public. "
        "NORC programs provide services in buildings with many seniors."
    ),
}

# ── Urgency heuristics ────────────────────────────────────────────────────────
_CRITICAL_CATS = {"safety", "domestic_violence", "emergency_services"}
_HIGH_CATS = {"housing", "shelter", "medical", "mental_health"}


def _estimate_urgency(needs: list, failed_resources: list) -> str:
    """Heuristic urgency classification without LLM."""
    open_needs = [n for n in needs if n.get("status") != "resolved"]
    cats = {n.get("category", "") for n in open_needs}

    if cats & _CRITICAL_CATS:
        return "critical"
    if cats & _HIGH_CATS:
        return "high" if len(failed_resources) < 2 else "critical"
    if open_needs:
        return "medium"
    return "low"


# ── Main briefing generator ───────────────────────────────────────────────────

def generate_briefing(case: dict, resource: dict) -> dict:
    """
    Generate an AI-powered admin briefing for an incoming client.

    Parameters
    ----------
    case     : loaded case dict from cases.py
    resource : dict with name, resource_type, address (from mart or user intent)

    Returns
    -------
    Structured briefing dict with all sections populated.
    """
    case_id = case.get("case_id", "")
    name = case.get("name", "Unknown")
    needs = case.get("needs", [])
    visits = case.get("visits", [])
    feedback = case.get("feedback", [])
    resources_visited = case.get("resources_visited", [])
    destination_intents = case.get("destination_intents", [])

    # Collect failed resources across all needs
    failed_resources: list[dict] = []
    for n in needs:
        for fr in n.get("failed_resources", []):
            failed_resources.append(fr)

    resource_name = resource.get("name", "Unknown")
    resource_type = resource.get("resource_type", resource.get("type", "default"))
    resource_address = resource.get("address", "")

    # Document requirements for this resource type
    docs_required = DOCS_BY_TYPE.get(resource_type, DOCS_BY_TYPE["default"])

    # Service protocol — prefer open need category match over resource type
    open_need_cats = [n["category"] for n in needs if n.get("status") != "resolved"]
    protocol = ""
    for cat in open_need_cats:
        if cat in SERVICE_PROTOCOLS:
            protocol = SERVICE_PROTOCOLS[cat]
            break
    if not protocol:
        protocol = SERVICE_PROTOCOLS.get(resource_type, "Standard intake process applies.")

    # Build history timeline (most recent first in context, oldest first in display)
    history_lines: list[str] = []
    for v in visits[-8:]:
        ts = v.get("timestamp", "")[:16].replace("T", " ")
        q = (v.get("query") or "")[:100]
        history_lines.append(f"[{ts}] Query: {q}")
    for fb in feedback[-5:]:
        ts = fb.get("timestamp", "")[:16].replace("T", " ")
        history_lines.append(
            f"[{ts}] Feedback — {fb.get('resource', '?')}: {fb.get('feedback', '')}"
        )
    for fr in failed_resources[-5:]:
        ts = (fr.get("timestamp") or "")[:16].replace("T", " ")
        history_lines.append(
            f"[{ts}] Failed resource: {fr.get('name', '?')} — {fr.get('reason', '')}"
        )
    for rv in resources_visited[-5:]:
        ts = (rv.get("visited_at") or "")[:16].replace("T", " ")
        history_lines.append(
            f"[{ts}] Visited: {rv.get('name', '?')} — {rv.get('feedback', 'arrived')}"
        )
    history_text = "\n".join(sorted(history_lines)) if history_lines else "No prior history on record."

    # Build formatted need summaries
    open_needs_text = ", ".join(
        f"{n['category']} (P{n.get('priority', '?')})"
        for n in sorted(needs, key=lambda x: x.get("priority", 99))
        if n.get("status") != "resolved"
    ) or "None identified yet"
    resolved_needs_text = ", ".join(
        n["category"] for n in needs if n.get("status") == "resolved"
    ) or "None"

    # EC info
    ec = case.get("emergency_contact", {})
    ec_line = ""
    if isinstance(ec, dict) and ec.get("name"):
        ec_line = (
            f"Emergency contact: {ec.get('name', '')} "
            f"(@{ec.get('telegram_username', '')}), "
            f"{'DM registered' if ec.get('telegram_chat_id') else 'not yet registered with bot'}"
        )

    # Docs for prompt
    docs_text = "\n".join(
        f"- {d['doc']} ({'REQUIRED' if d['required'] else 'preferred'}): "
        f"substitutes: {', '.join(d['substitutes']) if d['substitutes'] else 'none'}. "
        f"{d.get('note','')}"
        for d in docs_required
    )

    # Previous destinations
    prev_dests = [
        f"{i.get('resource_name', '?')} ({i.get('state', '?')})"
        for i in destination_intents[-3:]
    ]
    prev_dests_text = ", ".join(prev_dests) if prev_dests else "none"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    system_prompt = (
        "You are a senior NYC social services coordinator generating a structured intake briefing. "
        "Be concise, specific, and actionable. Prioritize ENABLING assistance over gatekeeping. "
        "When documents are missing, always identify a workaround. "
        "Output valid JSON only — no markdown, no commentary."
    )

    user_prompt = f"""Generate an admin intake briefing for this incoming client.

CLIENT: {name} (case_id: {case_id})
ARRIVING AT: {resource_name} ({resource_type}) — {resource_address}
TIME: {now}
{ec_line}

OPEN NEEDS: {open_needs_text}
RESOLVED NEEDS: {resolved_needs_text}
PREVIOUS DESTINATIONS: {prev_dests_text}
FAILED RESOURCES ({len(failed_resources)}): {', '.join(fr.get('name','') for fr in failed_resources) or 'none'}

CASE HISTORY:
{history_text}

STANDARD DOCUMENTS FOR {resource_type.upper()}:
{docs_text}

SERVICE PROTOCOL:
{protocol}

Output this exact JSON structure:
{{
  "diagnosis": "2–3 sentences analyzing the client's situation, likely root causes, and confidence level (high/medium/low)",
  "urgency": "critical|high|medium|low",
  "urgency_reason": "one sentence explaining urgency classification",
  "likely_needs": ["need description 1", "need description 2"],
  "recommended_approach": "2–3 sentences on how staff should approach this intake — tone, priorities, specific actions",
  "pre_arrival_instructions": "1–2 sentences to tell the client before they walk in",
  "missing_info": ["unknown 1", "unknown 2"],
  "required_docs": [
    {{
      "doc": "document name",
      "required": true,
      "gap_risk": "high|medium|low",
      "gap_reason": "why this might be missing based on client context",
      "substitute": "best acceptable alternative",
      "fallback_process": "what intake staff should do if no substitute available"
    }}
  ],
  "edge_cases": [
    {{
      "issue": "specific edge case that may arise",
      "workaround": "concrete workaround or alternative pathway",
      "risk": "consequence if unaddressed",
      "next_action": "immediate step for staff"
    }}
  ],
  "knowledge_notes": "key legal rights, service protocols, or best practices specific to this case"
}}"""

    briefing_data: dict = {}
    llm_succeeded = False

    try:
        raw = chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1400,
        )
        # Extract JSON (handle any wrapping text)
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            briefing_data = json.loads(json_match.group())
            llm_succeeded = True
    except Exception:
        pass

    # Fallback: rule-based briefing if LLM unavailable
    if not llm_succeeded or not briefing_data.get("diagnosis"):
        urgency = _estimate_urgency(needs, failed_resources)
        briefing_data = {
            "diagnosis": (
                f"Client has {len(open_need_cats)} open need(s): {open_needs_text}. "
                f"Arriving at {resource_name} after "
                f"{'failing at ' + str(len(failed_resources)) + ' prior resource(s)' if failed_resources else 'first attempt'}. "
                f"Confidence: medium (based on case data, no LLM analysis available)."
            ),
            "urgency": urgency,
            "urgency_reason": (
                "Safety/housing/medical need present with multiple failed resources."
                if failed_resources else
                "Based on open need categories."
            ),
            "likely_needs": open_need_cats[:5] or ["general assistance"],
            "recommended_approach": (
                f"Follow standard {resource_type} intake. "
                f"Review {len(open_need_cats)} open need(s) before starting. "
                "Be patient — client may have experienced multiple service failures."
            ),
            "pre_arrival_instructions": (
                f"Please bring any ID you have and go to {resource_address or resource_name}. "
                "Ask for the intake worker when you arrive."
            ),
            "missing_info": [
                "Household composition unknown",
                "Immigration status unknown",
                "Income level unknown",
            ],
            "required_docs": [],
            "edge_cases": [],
            "knowledge_notes": protocol,
        }

    # Ensure required_docs is populated (fill from static data if LLM skipped it)
    if not briefing_data.get("required_docs"):
        briefing_data["required_docs"] = [
            {
                "doc": d["doc"],
                "required": d["required"],
                "gap_risk": "high" if d["required"] else "low",
                "gap_reason": "Unknown — insufficient household profile data",
                "substitute": d["substitutes"][0] if d["substitutes"] else "Contact intake supervisor",
                "fallback_process": (
                    "Request supervisor override or document waiver. "
                    + (d.get("note") or "")
                ),
            }
            for d in docs_required
        ]

    # ── Knowledge augmentation ───────────────────────────────────────────────

    # 1. Eligibility & rights
    rights_info: list = []
    success_stories: list = []
    eligibility_data: dict = {}
    if _HAS_ELIGIBILITY:
        try:
            rights_info = get_rights(resource_type)
        except Exception:
            pass
        try:
            primary_cat = open_need_cats[0] if open_need_cats else None
            success_stories = get_stories(primary_cat, k=2)
        except Exception:
            pass
        # If case has household profile, calculate eligibility
        household = case.get("household_profile", {})
        if household:
            try:
                eligibility_data = calculate_eligibility(**household)
            except Exception:
                pass

    # 2. Resource recommendation reasoning (confidence engine)
    reasoning_path: list = []
    resource_id = resource.get("resource_id", "")
    if _HAS_CONFIDENCE and resource_id:
        try:
            explanation = explain_resource_recommendation(resource_id)
            reasoning_path = explanation.get("reasoning_path", [])
        except Exception:
            pass

    # 3. Alternative resources (KGE embeddings)
    alternative_resources: list = []
    if _HAS_EMBEDDINGS and resource_id:
        try:
            import pandas as _pd
            similar_df = find_similar(resource_id, k=3, same_borough=True)
            if isinstance(similar_df, _pd.DataFrame) and len(similar_df):
                cols = [c for c in ["name", "resource_type", "borough", "address", "similarity"]
                        if c in similar_df.columns]
                alternative_resources = similar_df[cols].to_dict("records")
        except Exception:
            pass

    # ── Enrich with static metadata + knowledge augmentation ─────────────
    briefing_data.update({
        "case_id": case_id,
        "client_name": name,
        "resource_name": resource_name,
        "resource_type": resource_type,
        "resource_address": resource_address,
        "generated_at": now,
        "llm_powered": llm_succeeded,
        "service_protocol": protocol,
        "history_timeline": sorted(history_lines),
        "open_needs_count": len(open_need_cats),
        "resolved_needs_count": len([n for n in needs if n.get("status") == "resolved"]),
        "total_visits": len(visits),
        "failed_resource_count": len(failed_resources),
        "emergency_contact": ec if isinstance(ec, dict) else {},
        "current_location": case.get("current_location"),
        "all_needs": [
            {
                "category": n.get("category", ""),
                "status": n.get("status", "open"),
                "priority": n.get("priority", 99),
                "chosen_resource": n.get("chosen_resource"),
                "failed_resources": [fr.get("name") for fr in n.get("failed_resources", [])],
            }
            for n in sorted(needs, key=lambda x: x.get("priority", 99))
        ],
        # Knowledge augmentation
        "rights_info": rights_info,
        "success_stories": success_stories,
        "eligibility": eligibility_data,
        "reasoning_path": reasoning_path,
        "alternative_resources": alternative_resources,
    })

    return briefing_data
