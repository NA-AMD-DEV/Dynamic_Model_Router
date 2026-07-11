"""Infer a task's category from its prompt. Costs zero tokens.

The prompt never names its category. A classifier call would spend tokens to
save tokens, so this is a keyword heuristic instead.

Categories match R2's CATEGORY_CONFIGS in agent/config.py exactly — the 8
official hackathon categories, not placeholders. If you add a category here,
add its config too, or classify() will route it to a real config_for()
fallback anyway but you'll have lost the routing signal.

Patterns are tried in PRIORITY order and the first match wins, so the order is
load-bearing: "summarise what this function does" must hit code_debugging
before summarisation. Put the more specific / higher-signal category first.
"""

import re

DEFAULT_CATEGORY = "factual_knowledge"

# First match wins. Ordered most-specific to least.
PRIORITY: list[tuple[str, list[str]]] = [
    ("code_debugging", [
        r"\bbug\b", r"\bdebug\b", r"\bfix\b.*\b(function|code|bug)\b",
        r"\bwhat'?s wrong with\b", r"\bcompiles?\b", r"\bsyntax error\b",
        r"\bdoesn'?t work\b", r"\braises?\b.*\berror\b",
    ]),
    ("code_generation", [
        r"```",
        r"\bfunction\b", r"\bdef\b", r"\bclass\b",
        r"\bwrite (a |the )?(python|java|c\+\+|javascript|sql|rust|go)\b",
        r"\bimplement\b.*\b(function|method|algorithm)\b",
    ]),
    ("logical_reasoning", [
        r"\bpuzzle\b", r"\bconstraint\b", r"\beither .* or\b",
        r"\bif .* then\b", r"\btrue or false\b",
        r"\bwho (is|owns|lives|likes)\b",  # classic logic-grid phrasing
    ]),
    ("math_reasoning", [
        r"\bcalculate\b", r"\bcompute\b", r"\bsolve\b",
        r"\bhow (much|many)\b",
        r"\b(sum|product|remainder|quotient|derivative|integral)\b",
        r"\bequation\b", r"\bpercent(age)?\b",
        r"\d+\s*[+\-*/^]\s*\d+",
    ]),
    ("named_entity_recognition", [
        r"\bnamed entit", r"\bentit(y|ies)\b",
        r"\bextract .*\b(name|person|place|organi[sz]ation|location|date)s?\b",
        r"\b(person|organi[sz]ation|location)s? mentioned\b",
    ]),
    ("sentiment_classification", [
        r"\bsentiment\b", r"\bpositive or negative\b",
        r"\b(positive|negative|neutral)\b.*\b(tone|review|opinion)\b",
        r"\bhow does .* feel\b",
    ]),
    ("summarisation", [
        r"\bsummari[sz]e\b", r"\bsummary\b", r"\btl;?dr\b",
        r"\bin one sentence\b", r"\bcondense\b", r"\bkey points?\b",
    ]),
]

# Compiled once at import. Prompts can be long; recompiling per task wastes time
# against the 10-minute run limit.
_COMPILED: list[tuple[str, list[re.Pattern[str]]]] = [
    (name, [re.compile(p, re.IGNORECASE) for p in pats])
    for name, pats in PRIORITY
]


def classify(prompt: str) -> str:
    """Return the category for `prompt`, or DEFAULT_CATEGORY if nothing matches.

    Never raises: an unclassifiable prompt routes to factual_knowledge, the
    most general of the 8 configs (direct question, concise answer).
    """
    if not prompt:
        return DEFAULT_CATEGORY
    for name, patterns in _COMPILED:
        if any(p.search(prompt) for p in patterns):
            return name
    return DEFAULT_CATEGORY
