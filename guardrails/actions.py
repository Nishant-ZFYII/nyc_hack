"""
guardrails/actions.py — Python actions called by Colang flows.

These implement the detection logic for PII, crisis, jailbreak, and hallucination.
Registered with NeMo Guardrails via LLMRails.register_action().
"""
import re


# ── PII Detection ────────────────────────────────────────────────────────────
PII_PATTERNS = [
    # SSN
    (r'\b\d{3}-\d{2}-\d{4}\b', 'ssn'),
    (r'\b\d{9}\b', 'ssn_unformatted'),
    # Credit card (13-19 digits, possibly with separators)
    (r'\b(?:\d[ -]*?){13,19}\b', 'credit_card'),
    # US passport (9 letters/digits)
    (r'\b[A-Z]\d{8}\b', 'passport'),
    # Phone patterns that look like account numbers (10+ digits continuous)
    # skip — too many false positives
]

SAFE_PII_EXCEPTIONS = [
    # Allow mentioning that you don't have these
    "no ssn", "no social security", "lost my ssn", "no id", "no documents",
    "don't have an ssn", "don't have social security",
]


def detect_pii_regex(text: str) -> bool:
    """Return True if text contains PII that shouldn't be shared here."""
    t = (text or "").lower()

    # Allow people to say "I don't have SSN" without blocking
    if any(excp in t for excp in SAFE_PII_EXCEPTIONS):
        return False

    for pattern, kind in PII_PATTERNS:
        if re.search(pattern, text or ""):
            return True
    return False


# ── Crisis Detection ──────────────────────────────────────────────────────────
SUICIDE_KEYWORDS = [
    "kill myself", "end my life", "suicide", "suicidal",
    "don't want to live", "want to die", "end it all",
    "hurt myself", "self harm", "self-harm",
]

ABUSE_KEYWORDS = [
    "beating me", "hitting me", "hits me", "hit me",
    "threatens to kill", "going to kill me", "abusing me",
    "afraid of my husband", "afraid of my boyfriend",
    "afraid of my partner", "he hurts", "she hurts",
    "domestic violence", "dv situation",
]


def detect_crisis(text: str) -> str:
    """Return 'suicide', 'abuse', or '' if no crisis detected."""
    t = (text or "").lower()

    if any(k in t for k in SUICIDE_KEYWORDS):
        return "suicide"
    if any(k in t for k in ABUSE_KEYWORDS):
        return "abuse"
    return ""


# ── Jailbreak Detection ──────────────────────────────────────────────────────
JAILBREAK_PATTERNS = [
    r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions',
    r'forget\s+(?:your|the)\s+instructions',
    r'you\s+are\s+(?:now|actually)\s+(?:a|an)\s+\w+\s+(?:ai|assistant|bot)',
    r'(?:dan|developer)\s+mode',
    r'pretend\s+(?:you|to)\s+(?:are|be)\s+not',
    r'disregard\s+(?:all|your)',
    r'override\s+(?:your|the)\s+(?:instructions|rules|system)',
    r'reveal\s+(?:your|the)\s+(?:system\s+)?prompt',
    r'what\s+are\s+your\s+instructions',
]


def detect_jailbreak(text: str) -> bool:
    """Return True if text looks like a jailbreak attempt."""
    t = (text or "").lower()
    return any(re.search(p, t) for p in JAILBREAK_PATTERNS)


# ── Harmful Output Detection ─────────────────────────────────────────────────
HARMFUL_OUTPUT_KEYWORDS = [
    "commit fraud", "evade", "hide from authorities",
    "fake documents", "forge", "bribe",
    "hack into", "illegally obtain",
]


def detect_harmful(text: str) -> bool:
    """Check if bot output contains harmful advice."""
    t = (text or "").lower()
    return any(k in t for k in HARMFUL_OUTPUT_KEYWORDS)


# ── Hallucination Detection (for resource names) ─────────────────────────────
# This is a soft check — if the bot outputs resource names, we verify them
# against the known mart names in the caller code (verify.py already does this).
# Here we just flag overly generic claims.
VAGUE_RESOURCE_PHRASES = [
    "generic shelter",
    "any shelter",
    "a nearby hospital",  # without naming it
]


def detect_invented_resource(text: str) -> bool:
    """Check if bot output contains vague or invented resource claims.

    This is intentionally conservative — we rely on the real verifier
    in pipeline/verify.py for actual fact-checking.
    """
    # For now, return False — the main verification happens in verify.py
    # This hook is here for future expansion.
    return False


# ── Registration helper ──────────────────────────────────────────────────────
def register_all(rails):
    """Register all actions with a NeMo Guardrails LLMRails instance.

    Usage:
        from nemoguardrails import LLMRails, RailsConfig
        config = RailsConfig.from_path("guardrails/")
        rails = LLMRails(config)
        from guardrails.actions import register_all
        register_all(rails)
    """
    rails.register_action(detect_pii_regex, name="detect_pii_regex")
    rails.register_action(detect_crisis, name="detect_crisis")
    rails.register_action(detect_jailbreak, name="detect_jailbreak")
    rails.register_action(detect_harmful, name="detect_harmful")
    rails.register_action(detect_invented_resource, name="detect_invented_resource")


# ── Standalone lightweight filter (works without NeMo Guardrails installed) ───
def apply_guardrails(user_input: str, bot_output: str = None) -> dict:
    """
    Lightweight guardrails check without full NeMo Guardrails dependency.
    Returns dict with: {allow, reason, replacement_response, crisis_type}.

    Use this as a quick inline check before calling the LLM.
    """
    # Input checks
    if detect_pii_regex(user_input):
        return {
            "allow": False,
            "reason": "pii_detected",
            "crisis_type": None,
            "replacement_response": (
                "For your safety, please don't share sensitive info like SSN or credit card numbers. "
                "Just tell me what you need help with — we can verify identity through official channels later."
            ),
        }

    crisis = detect_crisis(user_input)
    if crisis == "suicide":
        return {
            "allow": False,
            "reason": "crisis_suicide",
            "crisis_type": "suicide",
            "replacement_response": (
                "I hear you, and you're not alone. Please call or text 988 — the Suicide & Crisis Lifeline. "
                "It's free, 24/7, and confidential. You matter, and help is available right now. "
                "I can also help you find a mental health clinic nearby — would you like that?"
            ),
        }
    if crisis == "abuse":
        return {
            "allow": False,
            "reason": "crisis_abuse",
            "crisis_type": "abuse",
            "replacement_response": (
                "If you're in danger right now, call 911. For help leaving an abusive situation safely, "
                "call the NYC DV Hotline: 1-800-621-HOPE (4673). It's 24/7, free, anonymous. "
                "They have interpreters if English is hard. "
                "Would you like me to find a confidential DV shelter near you?"
            ),
        }

    if detect_jailbreak(user_input):
        return {
            "allow": False,
            "reason": "jailbreak_attempt",
            "crisis_type": None,
            "replacement_response": (
                "I'm here to help you find NYC social services. "
                "I can't change my role. What do you need help with?"
            ),
        }

    # Output checks (if bot_output provided)
    if bot_output:
        if detect_harmful(bot_output):
            return {
                "allow": False,
                "reason": "harmful_output",
                "crisis_type": None,
                "replacement_response": (
                    "I can only suggest legal, official resources. "
                    "Let me find you the right help through proper channels."
                ),
            }

    return {"allow": True, "reason": None, "crisis_type": None, "replacement_response": None}
