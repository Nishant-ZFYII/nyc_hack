"""
pipeline/planner.py — NL query → JSON plan via Nemotron.

Two output modes:
  1. needs_assessment — full caseworker decomposition for life-situation queries
  2. lookup           — simple filter for direct resource queries

Handles Nemotron's thinking-model quirk (reasoning text before JSON).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.client import plan_chat

SYSTEM_PROMPT = """You are a JSON-only API. You receive a user query about NYC social services and respond with EXACTLY one JSON object. Do not write any text before or after the JSON. Do not explain. Do not use markdown. Do not think out loud. Just output the JSON.

Pick one of these 4 JSON formats based on the query:

FORMAT 1 - Simple resource search:
{"intent":"lookup","resource_types":["shelter"],"filters":{"borough":"BK"},"limit":5}

FORMAT 2 - Person describing a life situation (USE THIS for stories about people needing help):
{"intent":"needs_assessment","client_profile":{"borough":"BK","situation":"brief summary"},"identified_needs":[{"category":"housing","priority":1}],"resource_searches":[{"resource_types":["shelter"],"filters":{"borough":"BK"},"limit":5}]}

FORMAT 3 - Emergency simulation:
{"intent":"simulate","scenario":"cold_emergency","params":{"borough":"BK","people_displaced":200,"temperature_f":15}}

FORMAT 4 - Explanation question:
{"intent":"explain","question":"why_underserved","target":"BX"}

RULES:
- Max 3 identified_needs, max 3 resource_searches
- Boroughs: MN BK QN BX SI
- Types: shelter food_bank hospital school domestic_violence benefits_center senior_services childcare community_center
- Scenarios: cold_emergency resource_gap capacity_change migrant_allocation
- If someone describes their personal situation, ALWAYS use FORMAT 2 (needs_assessment)
- Include benefits_center if income is mentioned
- Include school if children are mentioned
- Housing/shelter is always priority 1 if someone is losing their home
Explain questions: why_underserved (target=borough), why_recommend (target=resource_id), confidence_emergency (target=borough)

Priority rules:
- Safety first (domestic_violence, crime victim → shelter + DV services)
- Housing before food before medical before benefits
- Always include benefits_center if income mentioned
- Add hospital if disability/medical/pregnancy mentioned
- Add school/childcare if children mentioned

Examples:

Q: I'm Tina, 4 kids ages 12-16, income $28K, sister kicking us out of Flatbush next week
A: {"intent":"needs_assessment","client_profile":{"borough":"BK","situation":"family 5 eviction imminent income 28k","household_size":5},"identified_needs":[{"category":"housing","priority":1},{"category":"benefits","priority":2},{"category":"school","priority":3}],"resource_searches":[{"resource_types":["shelter"],"filters":{"borough":"BK"},"limit":5},{"resource_types":["benefits_center"],"filters":{"borough":"BK"},"limit":3},{"resource_types":["school"],"filters":{"borough":"BK"},"limit":3}]}

Q: Someone broke into my apartment. I don't feel safe going back. I have a 6 year old.
A: {"intent":"needs_assessment","client_profile":{"borough":null,"situation":"crime victim unsafe at home child 6","household_size":2},"identified_needs":[{"category":"safety","priority":1},{"category":"housing","priority":2}],"resource_searches":[{"resource_types":["domestic_violence","shelter"],"filters":{},"limit":5},{"resource_types":["benefits_center"],"filters":{},"limit":3}]}

Q: I just arrived from Haiti with my two children. We speak Haitian Creole and need shelter tonight near Flatbush.
A: {"intent":"needs_assessment","client_profile":{"borough":"BK","situation":"migrant Haiti 2 children Haitian Creole","household_size":3},"identified_needs":[{"category":"housing","priority":1},{"category":"food","priority":2}],"resource_searches":[{"resource_types":["shelter"],"filters":{"borough":"BK"},"limit":5},{"resource_types":["food_bank","community_center"],"filters":{"borough":"BK"},"limit":3}]}

Q: What shelters in Brooklyn have available beds right now?
A: {"intent":"lookup","resource_types":["shelter"],"filters":{"borough":"BK"},"limit":5}

Q: Find wheelchair accessible hospitals near Jamaica Queens that accept Medicaid
A: {"intent":"lookup","resource_types":["hospital"],"filters":{"borough":"QN","ada_accessible":true},"limit":5}

Q: How many food banks are open in the Bronx?
A: {"intent":"lookup","resource_types":["food_bank"],"filters":{"borough":"BX"},"limit":10}

Q: A cold emergency is declared. 3 Brooklyn shelters just hit capacity. 200 people are still outside. It's 15F.
A: {"intent":"simulate","scenario":"cold_emergency","params":{"borough":"BK","people_displaced":200,"temperature_f":15}}

Q: Which NYC boroughs are most underserved by social services?
A: {"intent":"simulate","scenario":"resource_gap","params":{}}

Q: A migrant bus just arrived with 80 people who speak Spanish and Mandarin. They need shelter food and schools.
A: {"intent":"simulate","scenario":"migrant_allocation","params":{"people":80,"languages":["Spanish","Mandarin"],"needs":["shelter","food_bank","school"]}}

Q: Why is the Bronx the most underserved borough?
A: {"intent":"explain","question":"why_underserved","target":"BX"}

Q: How confident are you in this cold emergency plan for Brooklyn?
A: {"intent":"explain","question":"confidence_emergency","target":"BK"}

Q: Why are you recommending this shelter? How safe is it?
A: {"intent":"explain","question":"why_recommend","target":""}"""


def _extract_json(text: str) -> dict:
    """Extract the outermost JSON object from model output (handles thinking-model output)."""
    # Strip <think>...</think> if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # Strip markdown code fences
    text = re.sub(r'```(?:json)?\s*', '', text).strip()
    text = re.sub(r'```\s*$', '', text).strip()

    # Find the first '{' and extract the balanced outermost JSON object
    start = text.find('{')
    if start == -1:
        raise ValueError(f"No JSON found in model output:\n{text[:200]}")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON: {e}\n{text[start:i + 1][:300]}")

    raise ValueError(f"Unterminated JSON in model output:\n{text[start:start + 300]}")


BOROUGH_KEYWORDS = {
    "manhattan": "MN", "bronx": "BX", "brooklyn": "BK", "flatbush": "BK",
    "bedstuy": "BK", "bed-stuy": "BK", "queens": "QN", "jamaica": "QN",
    "flushing": "QN", "staten island": "SI", "harlem": "MN", "bronx": "BX",
    "astoria": "QN", "bushwick": "BK", "williamsburg": "BK", "the bronx": "BX",
}

def _rule_based_plan(query: str) -> dict:
    """Keyword-based fallback planner — always works, no LLM needed."""
    q = query.lower()

    # Detect borough
    borough = None
    for kw, code in BOROUGH_KEYWORDS.items():
        if kw in q:
            borough = code
            break

    # Detect simulation intent
    if any(w in q for w in ["cold emergency", "hit capacity", "people outside",
                              "people displaced", "overflow", "resource gap",
                              "underserved", "most underserved"]):
        if "resource gap" in q or "underserved" in q:
            return {"intent": "simulate", "scenario": "resource_gap", "params": {}}
        people = 200
        import re
        m = re.search(r'(\d+)\s*people', q)
        if m:
            people = int(m.group(1))
        temp = 15
        m2 = re.search(r'(\d+)\s*[°f]', q)
        if m2:
            temp = int(m2.group(1))
        return {"intent": "simulate", "scenario": "cold_emergency",
                "params": {"borough": borough or "BK", "people_displaced": people, "temperature_f": temp}}

    # Detect needs_assessment (person describing situation)
    situation_keywords = ["i'm", "i am", "my ", "we are", "we're", "losing",
                          "evicted", "homeless", "fleeing", "arrived", "just got out",
                          "can't afford", "need help", "elderly mother", "disabled"]
    if any(w in q for w in situation_keywords):
        needs = []
        searches = []
        if any(w in q for w in ["housing", "shelter", "evict", "losing", "homeless", "apartment"]):
            needs.append({"category": "housing", "priority": 1, "reasoning": "housing instability mentioned"})
            searches.append({"resource_types": ["shelter"], "filters": {"borough": borough}, "limit": 5})
        if any(w in q for w in ["food", "hungry", "meal", "pantry"]):
            needs.append({"category": "food", "priority": 2, "reasoning": "food need mentioned"})
            searches.append({"resource_types": ["food_bank"], "filters": {"borough": borough}, "limit": 3})
        if any(w in q for w in ["income", "snap", "medicaid", "benefit", "cash", "assistance", "28k", "$"]):
            needs.append({"category": "benefits", "priority": 2, "reasoning": "income/benefits mentioned"})
            searches.append({"resource_types": ["benefits_center"], "filters": {"borough": borough}, "limit": 3})
        if any(w in q for w in ["kid", "child", "school", "son", "daughter", "age"]):
            needs.append({"category": "school", "priority": 3, "reasoning": "children mentioned"})
            searches.append({"resource_types": ["school", "childcare"], "filters": {"borough": borough}, "limit": 3})
        if any(w in q for w in ["elderly", "mother", "father", "senior", "alone", "can't live"]):
            needs.append({"category": "senior_services", "priority": 1, "reasoning": "elderly care needed"})
            searches.append({"resource_types": ["senior_services"], "filters": {"borough": borough}, "limit": 5})
        if any(w in q for w in ["violence", "abuse", "fleeing", "dv", "domestic"]):
            needs.append({"category": "safety", "priority": 1, "reasoning": "safety concern"})
            searches.append({"resource_types": ["domestic_violence"], "filters": {"borough": borough}, "limit": 5})
        if any(w in q for w in ["medical", "hospital", "health", "doctor", "clinic", "pregnant"]):
            needs.append({"category": "medical", "priority": 2, "reasoning": "medical need mentioned"})
            searches.append({"resource_types": ["hospital", "clinic"], "filters": {"borough": borough}, "limit": 3})

        if not needs:
            needs.append({"category": "housing", "priority": 1, "reasoning": "general assistance needed"})
            searches.append({"resource_types": ["shelter"], "filters": {"borough": borough}, "limit": 5})

        import re as _re
        size_m = _re.search(r'(\d+)\s*(kid|child|people|person|family)', q)
        size = int(size_m.group(1)) + 1 if size_m else None

        return {
            "intent": "needs_assessment",
            "client_profile": {"borough": borough, "household_size": size, "situation": query[:80]},
            "identified_needs": needs[:3],
            "resource_searches": searches[:3],
        }

    # Simple lookup
    resource_types = []
    if any(w in q for w in ["shelter", "bed", "housing", "sleep", "overnight"]):
        resource_types.append("shelter")
    if any(w in q for w in ["food", "pantry", "meal", "hungry", "soup kitchen"]):
        resource_types.append("food_bank")
    if any(w in q for w in ["hospital", "medical", "health", "doctor", "clinic", "er", "emergency room"]):
        resource_types += ["hospital", "clinic"]
    if any(w in q for w in ["school", "education"]):
        resource_types.append("school")
    if any(w in q for w in ["benefit", "snap", "medicaid", "cash assistance", "hra"]):
        resource_types.append("benefits_center")
    if any(w in q for w in ["senior", "elderly", "aging"]):
        resource_types.append("senior_services")
    if any(w in q for w in ["domestic violence", "dv shelter", "abuse"]):
        resource_types.append("domestic_violence")
    if any(w in q for w in ["childcare", "daycare", "day care"]):
        resource_types.append("childcare")

    ada = True if any(w in q for w in ["wheelchair", "accessible", "ada", "disability"]) else None

    return {
        "intent": "lookup",
        "resource_types": resource_types or ["shelter", "food_bank", "hospital"],
        "filters": {"borough": borough, "ada_accessible": ada},
        "limit": 5,
    }


def generate_plan(nl_query: str) -> dict:
    """
    Convert a natural language query to a JSON plan.
    Tries LLM first, falls back to rule-based classifier.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Convert this to JSON: {nl_query} /no_think"},
    ]
    raw = plan_chat(messages)
    if not raw:
        raw = ""
    try:
        plan = _extract_json(raw)
        if plan.get("intent") in ("lookup", "needs_assessment", "simulate", "explain"):
            return plan
        raise ValueError("Invalid intent")
    except (ValueError, Exception):
        return {
            "intent": "lookup",
            "resource_types": ["shelter", "food_bank", "hospital"],
            "filters": {},
            "limit": 5,
            "_parse_error": raw[:200],
        }
