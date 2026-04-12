"""
NeMo Guardrails integration for NYC Social Services Engine.

Hybrid approach:
  1. Fast regex pre-filter (< 1ms) — instant block for obvious PII/crisis
  2. NeMo Guardrails self-check (~500ms) — semantic topic/jailbreak detection

Usage (sync):      check_safety(user_input)
Usage (async):     await check_safety_async(user_input)
"""
import asyncio
import os
from pathlib import Path

from .actions import (
    apply_guardrails,
    detect_pii_regex,
    detect_crisis,
    detect_jailbreak,
    detect_harmful,
    detect_off_topic,
)

_GUARDRAILS_DIR = Path(__file__).resolve().parent
_rails = None
_nemo_available = False

try:
    from nemoguardrails import LLMRails, RailsConfig
    _nemo_available = True
except ImportError:
    pass


def _init_rails():
    """Lazy init — only load NeMo Guardrails on first use."""
    global _rails
    if _rails is not None or not _nemo_available:
        return _rails
    try:
        config = RailsConfig.from_path(str(_GUARDRAILS_DIR))
        _rails = LLMRails(config)
        return _rails
    except Exception as e:
        print(f"[guardrails] NeMo Guardrails init failed: {e}")
        return None


async def check_safety_async(user_input: str, use_llm_fallback: bool = True) -> dict:
    """Async version — use inside FastAPI endpoints."""
    # Layer 1: fast regex
    regex_result = apply_guardrails(user_input)
    if not regex_result["allow"]:
        regex_result["checked_by"] = "regex"
        return regex_result

    # Layer 2: NeMo Guardrails (async)
    if use_llm_fallback and _nemo_available:
        rails = _init_rails()
        if rails is not None:
            try:
                response = await rails.generate_async(messages=[{
                    "role": "user", "content": user_input
                }])
                if isinstance(response, dict):
                    content = response.get("content", "")
                else:
                    content = str(response)

                content_lower = content.lower()
                is_block = any(phrase in content_lower for phrase in [
                    "i can't", "i cannot", "i'm not able", "i am not able",
                    "cannot help", "can't help", "cannot answer", "can't answer",
                    "cannot provide", "can't provide", "unable to", "not able to",
                    "not related to", "off-topic", "off topic",
                    "outside my", "beyond my", "not my job", "not what i",
                    "i apologize", "i'm sorry but", "i am sorry but",
                    "my purpose is", "my role is",
                ])
                if is_block:
                    return {
                        "allow": False,
                        "reason": "nemo_guardrails_block",
                        "crisis_type": None,
                        "replacement_response": content or (
                            "I'm a NYC Social Services assistant — I help with shelter, "
                            "food, healthcare, benefits, and safety. Can I help you with any of those?"
                        ),
                        "checked_by": "nemo_guardrails",
                    }
            except Exception as e:
                print(f"[guardrails] NeMo async check failed: {e}")

    regex_result["checked_by"] = "regex"
    return regex_result


def check_safety(user_input: str, use_llm_fallback: bool = True) -> dict:
    """
    Two-layer safety check for NYC Social Services queries.

    Layer 1 — Fast regex (sub-millisecond):
      - PII (SSN, credit card)
      - Crisis keywords (suicide, abuse)
      - Known jailbreak patterns
      - Obvious off-topic markers

    Layer 2 — NeMo Guardrails self-check (~500ms, LLM-based):
      - Semantic topic classification
      - Subtle jailbreak attempts
      - Novel off-topic queries

    Returns dict: {
      allow: bool,
      reason: str,
      replacement_response: str,
      crisis_type: str | None,
      checked_by: "regex" | "nemo_guardrails"
    }
    """
    # Layer 1: fast regex
    regex_result = apply_guardrails(user_input)
    if not regex_result["allow"]:
        regex_result["checked_by"] = "regex"
        return regex_result

    # Layer 2: NeMo Guardrails (only for borderline cases)
    if use_llm_fallback and _nemo_available:
        rails = _init_rails()
        if rails is not None:
            try:
                # Use the self_check_input prompt via direct generation
                response = rails.generate(messages=[{
                    "role": "user", "content": user_input
                }])
                # rails.generate returns a dict or str depending on version
                if isinstance(response, dict):
                    content = response.get("content", "")
                else:
                    content = str(response)

                # If NeMo blocked it, response contains a refusal / bot redirect
                content_lower = content.lower()
                is_block = any(phrase in content_lower for phrase in [
                    "i can't", "i cannot", "i'm not able", "i am not able",
                    "cannot help", "can't help", "cannot answer", "can't answer",
                    "cannot provide", "can't provide", "unable to", "not able to",
                    "not related to", "off-topic", "off topic",
                    "outside my", "beyond my", "not my job", "not what i",
                    "i apologize", "i'm sorry but", "i am sorry but",
                    "my purpose is", "my role is",
                ])
                if is_block:
                    return {
                        "allow": False,
                        "reason": "nemo_guardrails_block",
                        "crisis_type": None,
                        "replacement_response": content or (
                            "I'm a NYC Social Services assistant — I help with shelter, "
                            "food, healthcare, benefits, and safety. "
                            "Can I help you with any of those?"
                        ),
                        "checked_by": "nemo_guardrails",
                    }
            except Exception as e:
                # If NeMo fails, fall through to regex result (which allowed it)
                print(f"[guardrails] NeMo check failed: {e}, using regex result")

    regex_result["checked_by"] = "regex"
    return regex_result


__all__ = [
    "check_safety",
    "check_safety_async",
    "apply_guardrails",
    "detect_pii_regex",
    "detect_crisis",
    "detect_jailbreak",
    "detect_harmful",
    "detect_off_topic",
]
