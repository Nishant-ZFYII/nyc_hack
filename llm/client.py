"""
llm/client.py — LLM provider fallback ladder

Tries providers in order:
  1. NIM container on DGX Spark (official NVIDIA, best Spark story)
  2. Raw vLLM on DGX Spark (same model, fast to deploy)
  3. llama.cpp GGUF server (bulletproof, ~20GB Q4 quant)
  4. Claude Haiku (local dev / pre-hackathon testing — reliable JSON)
  5. OpenAI GPT-4o-mini (fallback if Claude key not set)
  6. OpenRouter Nemotron (last resort)

Set env vars:
  ANTHROPIC_API_KEY   — for Claude Haiku (recommended for local dev)
  OPENAI_API_KEY      — for GPT-4o-mini
  OPENROUTER_API_KEY  — for OpenRouter

Usage:
    from llm.client import chat, get_active_provider
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    probe_timeout: int = 3


PROVIDERS: list[Provider] = [
    # Laptop-friendly Ollama models tried first (fit in 6 GB VRAM).
    Provider(
        name="Ollama llama3 (local laptop)",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="llama3",
    ),
    Provider(
        name="Ollama llama3:8b (local laptop)",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="llama3:8b",
    ),
    Provider(
        name="Ollama phi3:mini (small laptop fallback)",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="phi3:mini",
    ),
    # DGX Spark models kept for reproducibility on bigger hardware.
    Provider(
        name="Ollama Nemotron-3-Nano 30B (DGX Spark)",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="nemotron-3-nano",
    ),
    Provider(
        name="Ollama Nemotron Mini (DGX Spark)",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="nemotron-mini",
    ),
    Provider(
        name="Ollama Nemotron 70B (DGX Spark)",
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama",
        model="nemotron",
    ),
    Provider(
        name="NIM (DGX Spark)",
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model=MODEL_NAME,
    ),
    Provider(
        name="Claude Haiku (local dev)",
        base_url="https://api.anthropic.com/v1",
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        model="claude-haiku-4-5-20251001",
        probe_timeout=10,
    ),
    Provider(
        name="GPT-4o-mini (local dev)",
        base_url="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="gpt-4o-mini",
        probe_timeout=10,
    ),
    Provider(
        name="vLLM (DGX Spark)",
        base_url="http://127.0.0.1:8001/v1",  # different port so we can try both
        api_key="EMPTY",
        model=MODEL_NAME,
    ),
    Provider(
        name="llama.cpp (local GGUF)",
        base_url="http://127.0.0.1:8080/v1",
        api_key="EMPTY",
        model="local",
    ),
    Provider(
        name="OpenRouter (fallback)",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        probe_timeout=10,
    ),
]

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_active_provider: Provider | None = None
_active_client: OpenAI | None = None


def _probe_provider(provider: Provider) -> bool:
    """Return True if the provider is reachable and has a valid API key."""
    if not provider.api_key or provider.api_key == "EMPTY":
        # Local providers (NIM/vLLM/llama.cpp) — probe endpoint directly
        if "127.0.0.1" not in provider.base_url:
            return False  # remote provider with no key
    try:
        if "anthropic" in provider.base_url:
            # Anthropic uses x-api-key header, probe via /v1/models
            r = requests.get(
                f"{provider.base_url}/models",
                timeout=provider.probe_timeout,
                headers={"x-api-key": provider.api_key, "anthropic-version": "2023-06-01"},
            )
            return r.status_code == 200
        else:
            r = requests.get(
                f"{provider.base_url}/models",
                timeout=provider.probe_timeout,
                headers={"Authorization": f"Bearer {provider.api_key}"} if provider.api_key != "EMPTY" else {},
            )
            return r.status_code == 200
    except Exception:
        return False


def _detect_provider() -> tuple[Provider, OpenAI]:
    """Probe providers in order and return the first responsive one."""
    for p in PROVIDERS:
        logger.info("Probing provider: %s at %s", p.name, p.base_url)
        if _probe_provider(p):
            logger.info("Using provider: %s", p.name)
            if "anthropic" in p.base_url:
                # Anthropic is OpenAI-compatible via their base_url
                client = OpenAI(
                    base_url="https://api.anthropic.com/v1",
                    api_key=p.api_key,
                    default_headers={"anthropic-version": "2023-06-01"},
                )
            else:
                client = OpenAI(base_url=p.base_url, api_key=p.api_key)
            return p, client
    raise RuntimeError(
        "No LLM provider available. Check:\n"
        "  1. vLLM server started:  python -m vllm.entrypoints.openai.api_server ...\n"
        "  2. NIM container running: docker run --gpus all nvcr.io/nim/...\n"
        "  3. OPENROUTER_API_KEY env var set for remote fallback"
    )


def get_client() -> tuple[Provider, OpenAI]:
    """Return the cached (provider, client), detecting on first call."""
    global _active_provider, _active_client
    if _active_provider is None:
        _active_provider, _active_client = _detect_provider()
    return _active_provider, _active_client


def get_active_provider() -> str:
    """Return the name of the currently active provider (for UI display)."""
    try:
        p, _ = get_client()
        return p.name
    except Exception:
        return "none"


def reset_provider():
    """Force re-detection on next call (useful after restarting a service)."""
    global _active_provider, _active_client
    _active_provider = None
    _active_client   = None


# ---------------------------------------------------------------------------
# Main chat interface
# ---------------------------------------------------------------------------

def chat(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
    retries: int = 3,
) -> str:
    """
    Send a chat request to the active LLM provider.

    Parameters
    ----------
    messages    : OpenAI-format messages list
    temperature : 0.0 for planning (deterministic), 0.3 for synthesis
    max_tokens  : max response tokens
    retries     : number of retry attempts before giving up

    Returns
    -------
    str — the model's response content
    """
    for attempt in range(retries):
        try:
            provider, client = get_client()

            # Anthropic SDK path — more reliable than OpenAI compat layer
            if "anthropic" in provider.base_url:
                import anthropic
                ac = anthropic.Anthropic(api_key=provider.api_key)
                system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                user_msgs  = [m for m in messages if m["role"] != "system"]

                # Prefill "{" forces Claude to output raw JSON with no markdown fences
                is_json_call = system_msg and ("JSON" in system_msg or "json" in system_msg)
                if is_json_call:
                    user_msgs = list(user_msgs) + [{"role": "assistant", "content": "{"}]

                resp = ac.messages.create(
                    model=provider.model,
                    system=system_msg,
                    messages=user_msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                result = resp.content[0].text
                # Re-attach the prefilled "{" since Anthropic returns only the continuation
                if is_json_call:
                    result = "{" + result
                return result

            # OpenAI-compatible path (vLLM, NIM, OpenRouter, GPT, Ollama)
            resp = client.chat.completions.create(
                model=provider.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = resp.choices[0].message.content
            # Strip Qwen3 thinking blocks if present
            import re as _re
            result = _re.sub(r'<think>.*?</think>', '', result or '', flags=_re.DOTALL).strip()
            return result

        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                reset_provider()  # try next provider
                time.sleep(1)

    return "Error: LLM unavailable after all retries."


# ---------------------------------------------------------------------------
# Convenience wrappers used by planner and synthesizer
# ---------------------------------------------------------------------------

def plan_chat(messages: list[dict]) -> str:
    """Call the LLM for query planning (temp=0, deterministic)."""
    return chat(messages, temperature=0.0, max_tokens=2048)


def synth_chat(messages: list[dict]) -> str:
    """Call the LLM for answer synthesis (temp=0.3, a bit creative)."""
    return chat(messages, temperature=0.3, max_tokens=600)


# ---------------------------------------------------------------------------
# Quick ping for health check / checkpoint validation
# ---------------------------------------------------------------------------

def ping() -> bool:
    """Return True if the LLM responds to a trivial prompt."""
    try:
        resp = chat([{"role": "user", "content": "Reply with just the word: OK"}], max_tokens=10)
        return "OK" in resp.upper() or len(resp) > 0
    except Exception:
        return False
