"""
pipeline/clarify.py — Generate a follow-up clarifying question after the first answer.

The caseworker asks ONE targeted question to unlock better resource matching:
- Missing info that would change the results (ID, MetroCard, language, disability, kids ages)
- Eligibility-gating info (income already known → skip; immigration status matters for some benefits)
- Safety constraints (DV situation → don't ask address)

Returns a single question string, or empty string if nothing useful to ask.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm.client import synth_chat

CLARIFY_PROMPT = """You are an NYC DSS caseworker AI. You just gave a client a resource referral.
Decide: is there ONE question whose answer would add a NEW resource type to the recommendations?

Say DONE if ANY of these are true:
- The query already has: borough + household size + income + situation type
- This is a simple lookup (not a personal situation)
- This is a simulation query
- You already asked 1 question and have enough to act
- The answer would only change the wording, not the actual resources shown

Ask ONE question (max 20 words) ONLY if the answer would unlock a new resource search:
- No borough → ask borough (completely changes results)
- No medical info + kids/elderly mentioned → ask about disabilities/chronic conditions
- No ID mentioned + benefits needed → ask if they have ID (affects eligibility)
- No language mentioned + non-English name/context → ask primary language
- Safety unclear + possible DV → ask if they are safe (affects shelter type)

Output format: just the question, nothing else. Or exactly: DONE

Examples:
Family losing housing, borough unknown → "What part of NYC are you in — Brooklyn, Manhattan, Queens, Bronx, or Staten Island?"
Family with kids, no medical info → "Do any of your children have disabilities or chronic medical conditions?"
Migrant, no ID mentioned → "Do you have any ID or immigration documents with you?"
Simple lookup "shelters in Brooklyn" → DONE
Cold emergency simulation → DONE
Already asked borough, now have full picture → DONE
"""


def get_clarifying_question(original_query: str, answer: str, turn: int = 0) -> str:
    """
    Return a follow-up question, or empty string if no useful question remains.
    turn=0 means first follow-up, turn=1 means second, etc.
    After turn 1 we stop asking.
    """
    if turn >= 2:
        return ""

    messages = [
        {"role": "system", "content": CLARIFY_PROMPT},
        {"role": "user", "content": (
            f"Original query: {original_query}\n\n"
            f"Answer given: {answer[:400]}\n\n"
            f"Turn: {turn + 1}/2. What's the single most important follow-up question?"
        )},
    ]
    response = synth_chat(messages)
    if not response:
        return ""
    response = response.strip()
    if response.upper().startswith("DONE") or len(response) < 5:
        return ""
    # Strip any accidental numbering or prefixes
    for prefix in ["Question:", "Q:", "Follow-up:", "Ask:"]:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    return response


def merge_query(original: str, question: str, answer: str) -> str:
    """Merge original query + clarification into an enriched query for re-planning."""
    return f"{original.rstrip('.')}. Additional info: {question} — {answer}"
