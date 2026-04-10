"""
engine/txt2kg.py — Extract structured triples from 311 complaint text.

Two modes:
  1. Rule-based bulk: complaint_type + descriptor → predefined triples (300K records, <5s)
  2. LLM-powered sample: send a batch of complaints to Nemotron/Claude → rich SPO triples

On DGX Spark: replace rule-based with NVIDIA txt2kg for full NLP extraction.

Usage:
    from engine.txt2kg import extract_311_triples, llm_extract_sample
"""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

STAGE = Path("/media/nishant/SeeGayt2/nyc_hack_data/stage")

# ── Mapping: complaint_type + descriptor → structured issue triples ───────────

# Impact categories: how does this complaint affect a vulnerable person?
IMPACT_MAP = {
    # Safety-critical (affects DV victims, elderly, children)
    ("SAFETY", None):                 {"issue": "safety_hazard",       "severity": "high",   "affects": "all"},
    ("ELEVATOR", None):               {"issue": "elevator_broken",     "severity": "high",   "affects": "elderly,disabled"},
    ("ELECTRIC", "NO LIGHTING"):      {"issue": "no_lighting",         "severity": "high",   "affects": "all"},
    ("ELECTRIC", "POWER OUTAGE"):     {"issue": "power_outage",        "severity": "critical","affects": "all"},

    # Health hazards (affects children, immunocompromised)
    ("UNSANITARY CONDITION", "PESTS"):         {"issue": "pest_infestation",  "severity": "high",   "affects": "children,health"},
    ("UNSANITARY CONDITION", "MOLD"):          {"issue": "mold",             "severity": "high",   "affects": "children,health"},
    ("UNSANITARY CONDITION", "GARBAGE/RECYCLING STORAGE"): {"issue": "garbage_hazard", "severity": "medium", "affects": "all"},

    # Habitability (affects families)
    ("HEAT/HOT WATER", "ENTIRE BUILDING"):     {"issue": "no_heat_building", "severity": "critical","affects": "all"},
    ("HEAT/HOT WATER", "APARTMENT ONLY"):      {"issue": "no_heat_unit",     "severity": "high",   "affects": "all"},
    ("WATER LEAK", "HEAVY FLOW"):              {"issue": "major_water_leak", "severity": "high",   "affects": "all"},
    ("WATER LEAK", "SLOW LEAK"):               {"issue": "minor_water_leak", "severity": "low",    "affects": "property"},
    ("PLUMBING", "WATER SUPPLY"):              {"issue": "no_water",         "severity": "critical","affects": "all"},
    ("PLUMBING", "TOILET"):                    {"issue": "toilet_broken",    "severity": "high",   "affects": "all"},
    ("PLUMBING", "BATHTUB/SHOWER"):            {"issue": "plumbing_broken",  "severity": "medium", "affects": "all"},

    # Structural (affects safety)
    ("FLOORING/STAIRS", "FLOOR"):              {"issue": "floor_damage",     "severity": "medium", "affects": "elderly,children"},
    ("DOOR/WINDOW", "DOOR"):                   {"issue": "broken_door",      "severity": "medium", "affects": "security"},
    ("DOOR/WINDOW", "WINDOW FRAME"):           {"issue": "broken_window",    "severity": "medium", "affects": "all"},
    ("PAINT/PLASTER", "CEILING"):              {"issue": "ceiling_damage",   "severity": "medium", "affects": "all"},
    ("PAINT/PLASTER", "WALL"):                 {"issue": "wall_damage",      "severity": "low",    "affects": "all"},
}

# Fallback: complaint_type only
COMPLAINT_TYPE_MAP = {
    "HEAT/HOT WATER":       {"issue": "heating_problem",    "severity": "high",   "affects": "all"},
    "UNSANITARY CONDITION":  {"issue": "unsanitary",         "severity": "high",   "affects": "children,health"},
    "PLUMBING":              {"issue": "plumbing_issue",     "severity": "medium", "affects": "all"},
    "PAINT/PLASTER":         {"issue": "structural_wear",    "severity": "low",    "affects": "all"},
    "DOOR/WINDOW":           {"issue": "access_issue",       "severity": "medium", "affects": "security"},
    "WATER LEAK":            {"issue": "water_leak",         "severity": "medium", "affects": "all"},
    "GENERAL":               {"issue": "general_complaint",  "severity": "low",    "affects": "all"},
    "ELECTRIC":              {"issue": "electrical_issue",    "severity": "medium", "affects": "all"},
    "FLOORING/STAIRS":       {"issue": "structural_hazard",  "severity": "medium", "affects": "elderly,children"},
    "APPLIANCE":             {"issue": "appliance_broken",   "severity": "low",    "affects": "all"},
    "SAFETY":                {"issue": "safety_hazard",      "severity": "high",   "affects": "all"},
    "ELEVATOR":              {"issue": "elevator_broken",    "severity": "high",   "affects": "elderly,disabled"},
    "OUTSIDE BUILDING":      {"issue": "exterior_damage",    "severity": "low",    "affects": "all"},
}

SEVERITY_CONFIDENCE = {"critical": 0.95, "high": 0.85, "medium": 0.75, "low": 0.65}


def extract_311_triples(limit: Optional[int] = None) -> list[dict]:
    """
    Rule-based extraction: 311 complaint_type + descriptor → structured triples.
    Returns list of triple dicts.
    """
    df = pd.read_parquet(STAGE / "311_complaints.parquet")
    df = df.dropna(subset=["latitude", "longitude", "complaint_type"])
    if limit:
        df = df.head(limit)

    triples = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="txt2kg (311)"):
        ctype = row["complaint_type"]
        desc = row.get("descriptor", "")
        addr = row.get("incident_address", "unknown")
        boro = row.get("borough", "")
        date = str(row.get("created_date", ""))[:10]

        # Look up impact
        impact = IMPACT_MAP.get((ctype, desc))
        if not impact:
            impact = IMPACT_MAP.get((ctype, None))
        if not impact:
            impact = COMPLAINT_TYPE_MAP.get(ctype, {"issue": ctype.lower().replace("/","_"),
                                                      "severity": "low", "affects": "all"})

        subject = f"addr_{addr.replace(' ','_')[:50]}_{boro}"
        conf = SEVERITY_CONFIDENCE.get(impact["severity"], 0.7)

        # Triple 1: location has issue
        triples.append({
            "subject": subject,
            "predicate": "HAS_ISSUE",
            "object_val": impact["issue"],
            "confidence": round(conf * 0.85, 3),  # cross-dataset derivation
            "source": "311_txt2kg",
        })

        # Triple 2: issue severity
        triples.append({
            "subject": subject,
            "predicate": "ISSUE_SEVERITY",
            "object_val": impact["severity"],
            "confidence": round(conf * 0.90, 3),
            "source": "311_txt2kg",
        })

        # Triple 3: who it affects
        triples.append({
            "subject": subject,
            "predicate": "AFFECTS_POPULATION",
            "object_val": impact["affects"],
            "confidence": round(conf * 0.80, 3),
            "source": "311_txt2kg",
        })

        # Triple 4: temporal
        if date and date != "nan":
            triples.append({
                "subject": subject,
                "predicate": "ISSUE_DATE",
                "object_val": date,
                "confidence": 0.95,
                "source": "311_txt2kg",
            })

    return triples


def llm_extract_sample(n: int = 10) -> list[dict]:
    """
    LLM-powered extraction: send a sample of 311 complaints to Nemotron/Claude
    and extract rich triples. For demo purposes.

    On DGX: replace with NVIDIA txt2kg pipeline.
    """
    sys.path.insert(0, "/home/nishant/MS_Project/temp_proj/Spark")
    from llm.client import chat

    df = pd.read_parquet(STAGE / "311_complaints.parquet")
    df = df.dropna(subset=["complaint_type", "incident_address"]).head(n)

    complaints_text = ""
    for i, (_, row) in enumerate(df.iterrows()):
        complaints_text += (
            f"{i+1}. Address: {row['incident_address']}, Borough: {row.get('borough','?')}, "
            f"Type: {row['complaint_type']}, Detail: {row.get('descriptor','?')}, "
            f"Date: {str(row.get('created_date','?'))[:10]}\n"
        )

    system_prompt = """You are a knowledge graph extractor. Given 311 complaints, extract SPO triples.
Output ONLY a JSON array of triples. Each triple: {"s":"subject","p":"predicate","o":"object"}

Predicates to use:
- HAS_ISSUE (specific issue: pest_infestation, no_heat, mold, broken_door, etc.)
- SEVERITY (critical/high/medium/low)
- AFFECTS (elderly, children, disabled, all)
- RISK_FACTOR (what could go wrong: fire_risk, health_hazard, fall_risk, security_risk)
- RELATED_SERVICE (what service is needed: exterminator, plumber, electrician, HPD_inspection)

Example output:
[{"s":"123_Main_St_BK","p":"HAS_ISSUE","o":"pest_infestation"},{"s":"123_Main_St_BK","p":"SEVERITY","o":"high"},{"s":"123_Main_St_BK","p":"AFFECTS","o":"children"},{"s":"123_Main_St_BK","p":"RISK_FACTOR","o":"health_hazard"},{"s":"123_Main_St_BK","p":"RELATED_SERVICE","o":"exterminator"}]"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract triples from these 311 complaints:\n{complaints_text}"},
    ]

    import json
    raw = chat(messages, temperature=0.0, max_tokens=2048)

    # Parse JSON array from response
    try:
        # Strip markdown fences if present
        import re
        raw = re.sub(r'```(?:json)?\s*', '', raw).strip()
        raw = re.sub(r'```\s*$', '', raw).strip()
        if not raw.startswith('['):
            start = raw.find('[')
            if start >= 0:
                raw = raw[start:]
        parsed = json.loads(raw)

        triples = []
        for t in parsed:
            triples.append({
                "subject": str(t.get("s", "")),
                "predicate": str(t.get("p", "")),
                "object_val": str(t.get("o", "")),
                "confidence": 0.70,  # LLM-extracted
                "source": "311_llm_txt2kg",
            })
        return triples
    except Exception as e:
        print(f"LLM txt2kg parse error: {e}")
        return []


def aggregate_address_issues(triples: list[dict]) -> pd.DataFrame:
    """
    Aggregate txt2kg triples into per-address issue summaries.
    Used by the executor to enrich resource recommendations.
    """
    df = pd.DataFrame(triples)
    if df.empty:
        return pd.DataFrame()

    issues = df[df["predicate"] == "HAS_ISSUE"]
    severity = df[df["predicate"] == "ISSUE_SEVERITY"]

    # Count issues per address
    addr_issues = issues.groupby("subject").agg(
        issue_count=("object_val", "count"),
        issues=("object_val", lambda x: ", ".join(x.unique())),
    ).reset_index()

    # Get worst severity per address
    sev_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    if not severity.empty:
        severity = severity.copy()
        severity["sev_rank"] = severity["object_val"].map(sev_order).fillna(0)
        worst = severity.groupby("subject")["sev_rank"].max().reset_index()
        worst["worst_severity"] = worst["sev_rank"].map({v:k for k,v in sev_order.items()})
        addr_issues = addr_issues.merge(worst[["subject","worst_severity"]], on="subject", how="left")

    return addr_issues


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("=" * 60)
    print("txt2kg — 311 Complaint Triple Extraction")
    print("=" * 60)

    # Rule-based bulk extraction
    triples = extract_311_triples()
    print(f"\nRule-based triples: {len(triples):,}")

    # Save
    df = pd.DataFrame(triples)
    outpath = Path("/media/nishant/SeeGayt2/nyc_hack_data/data/triples_311.parquet")
    df.to_parquet(outpath, index=False)
    print(f"Saved: {outpath} ({outpath.stat().st_size / 1e6:.1f} MB)")

    # Aggregate
    agg = aggregate_address_issues(triples)
    print(f"Unique addresses with issues: {len(agg):,}")
    print(f"Top addresses by issue count:")
    print(agg.nlargest(10, "issue_count")[["subject", "issue_count", "issues"]].to_string(index=False))

    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # LLM sample (optional — needs API key)
    try:
        print("\n--- LLM txt2kg sample (10 complaints) ---")
        llm_triples = llm_extract_sample(10)
        print(f"LLM-extracted triples: {len(llm_triples)}")
        for t in llm_triples[:10]:
            print(f"  ({t['subject']}, {t['predicate']}, {t['object_val']})")
    except Exception as e:
        print(f"LLM sample skipped: {e}")
