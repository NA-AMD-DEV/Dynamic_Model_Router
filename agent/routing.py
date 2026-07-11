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

from agent.config import DEFAULT_CATEGORY

# First match wins. Ordered most-specific to least.
PRIORITY: list[tuple[str, list[str]]] = [
    # Sentiment before code/math: "positive or negative" and "feedback" are
    # unambiguous, and words like "product" (as in "the product broke") must not
    # leak into math. A clear sentiment ask wins outright.
    ("sentiment_classification", [
        r"\bsentiment\b", r"\bpositive or negative\b", r"\bnegative or positive\b",
        r"\bpositive,?\s*negative,?\s*(or\s*)?neutral\b",  # list form
        r"\b(positive|negative|neutral)\b.*\b(tone|review|opinion|feedback|sentiment)\b",
        r"\b(review|feedback|opinion|tone|tweet)\b.*\b(positive|negative|neutral)\b",
        r"\b(classify|what'?s|describe)\b.*\btone\b",
        r"\bhow does .* feel\b",
    ]),
    # Debugging before generation: repair language ("fix", "crashes", "bug")
    # means we're mending existing code, even when the word "function" appears.
    ("code_debugging", [
        r"\bbug\b", r"\bdebug\b",
        r"\bfix\b", r"\bcrash(es|ed|ing)?\b", r"\bwhat'?s wrong with\b",
        r"\bcompiles?\b", r"\bsyntax error\b", r"\bdoesn'?t work\b",
        r"\braises?\b.*\berror\b", r"\bshould (return|do|be)\b.*\bbut\b",
    ]),
    ("code_generation", [
        r"```",
        r"\bfunction\b", r"\bdef\b", r"\bclass\b",
        r"\bwrite (a |the )?(python|java|c\+\+|javascript|sql|rust|go)\b",
        r"\bimplement\b.*\b(function|method|algorithm)\b",
    ]),
    # Logic before math, but its patterns are phrasing-specific so numeric word
    # problems still fall through to math_reasoning below.
    ("logical_reasoning", [
        r"\bpuzzle\b", r"\bconstraint\b", r"\beither .* or\b",
        r"\bif .* then\b", r"\btrue or false\b",
        r"\bwho (is|owns|lives|likes|came|finished)\b",  # logic-grid phrasing
        r"\bwho (is|came|finished|ranks?|placed)\b.*\b(tallest|shortest|oldest|youngest|furthest|second|first|last)\b",
        r"\bwhich\b.*\b(is|comes?|ranks?)\b.*\b(tallest|shortest|oldest|youngest|furthest north|furthest)\b",
        r"\b\w+ beat \w+\b",  # transitive ordering: "Dana beat Evan"
        r"\bcan we conclude\b", r"\bdoes (it|this) follow\b",
        r"\ball \w+ are\b.*\bsome\b",  # syllogism
        r"\bnorth of\b|\bsouth of\b|\beast of\b|\bwest of\b",  # spatial ordering
    ]),
    ("math_reasoning", [
        r"\bcalculate\b", r"\bcompute\b", r"\bsolve\b",
        r"\bhow (much|many)\b",
        r"\b(remainder|quotient|derivative|integral)\b",  # dropped bare 'product'/'sum'
        r"\bequation\b", r"\bpercent(age)?\b",
        r"%",  # any percent sign signals arithmetic (discount, of, off)
        r"\bwhat is\b.*\d+.*\b(of|times|plus|minus|divided|multiplied)\b",
        r"\b(discount(ed)?|sale price|split|evenly|to the power of)\b",
        r"\baverage (speed|of)\b",
        r"\$\s*\d+",  # money amounts: "$40", "$87"
        r"\d+\s*[+\-*/^]\s*\d+",
    ]),
    ("named_entity_recognition", [
        r"\bnamed entit", r"\bentit(y|ies)\b",
        r"\b(extract|identify|pull out|list)\b.*\b(name|person|people|place|organi[sz]ation|location|date)s?\b",
        r"\b(which|what)\b.*\b(people|person|organi[sz]ations?|places?|locations?)\b.*\bmentioned\b",
        r"\b(people|person|organi[sz]ation|location)s? mentioned\b",
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
