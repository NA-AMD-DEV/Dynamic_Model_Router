"""Infer a task's category from its prompt. Costs zero tokens.

The prompt never names its category. A classifier call would spend tokens to
save tokens, so this is a keyword heuristic instead.

Patterns are tried in PRIORITY order and the first match wins, so the order is
load-bearing: "summarize what this function does" is a code task, not a
summarization task. Put the more specific category first.
"""

import re

DEFAULT_CATEGORY = "general"

# First match wins. Ordered most-specific to least.
PRIORITY: list[tuple[str, list[str]]] = [
    ("code", [
        r"```",
        r"\bfunction\b", r"\bdef\b", r"\bclass\b",
        r"\bbug\b", r"\bdebug\b", r"\brefactor\b",
        r"\bcompiles?\b", r"\bsyntax error\b",
        r"\bwrite (a |the )?(python|java|c\+\+|javascript|sql|rust|go)\b",
        r"\bimplement\b.*\b(function|method|algorithm)\b",
    ]),
    ("math", [
        r"\bcalculate\b", r"\bcompute\b", r"\bsolve\b",
        r"\bhow (much|many)\b",
        r"\b(sum|product|remainder|quotient|derivative|integral)\b",
        r"\bequation\b", r"\bpercent(age)?\b",
        r"\d+\s*[+\-*/^]\s*\d+",
    ]),
    ("ner", [
        r"\bnamed entit", r"\bentit(y|ies)\b",
        r"\bextract .*\b(name|person|place|organi[sz]ation|location|date)s?\b",
        r"\b(person|organi[sz]ation|location)s? mentioned\b",
    ]),
    ("sentiment", [
        r"\bsentiment\b", r"\bpositive or negative\b",
        r"\b(positive|negative|neutral)\b.*\b(tone|review|opinion)\b",
        r"\bhow does .* feel\b",
    ]),
    ("translation", [
        r"\btranslate\b",
        r"\b(into|to|from) (english|spanish|french|german|hindi|chinese|japanese|arabic)\b",
        r"\bin (spanish|french|german|hindi|chinese|japanese|arabic)\b",
    ]),
    ("summarization", [
        r"\bsummari[sz]e\b", r"\bsummary\b", r"\btl;?dr\b",
        r"\bin one sentence\b", r"\bcondense\b", r"\bkey points?\b",
    ]),
    ("extraction", [
        r"\bextract\b", r"\bpull out\b", r"\blist all\b",
        r"\bfind all\b", r"\bwhat are the\b.*\bin the (text|passage|article)\b",
    ]),
    ("classification", [
        r"\bclassify\b", r"\bcategori[sz]e\b", r"\blabel\b",
        r"\bwhich (category|class|type)\b",
        r"\bis this (a|an)\b.*\bor\b",
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

    Never raises: an unclassifiable prompt is routed to the general config,
    which is deliberately the most permissive one.
    """
    if not prompt:
        return DEFAULT_CATEGORY
    for name, patterns in _COMPILED:
        if any(p.search(prompt) for p in patterns):
            return name
    return DEFAULT_CATEGORY
