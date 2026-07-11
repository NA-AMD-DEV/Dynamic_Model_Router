"""Per-category model, prompt, and token budget.

R2 owns the defaults in _DEFAULTS. Two ways to change them, in precedence order:

  1. Env override, no rebuild:  ROUTER_<CATEGORY>_MODEL, ROUTER_<CATEGORY>_MAX_TOKENS
  2. Edit _DEFAULTS and rerun.

Nothing is resolved at import. `config_for` reads the environment on every call,
so the harness can inject ALLOWED_MODELS after this module loads, and R2 can
change a model between runs without touching code or rebuilding the image.

Model IDs come from ALLOWED_MODELS at runtime, never from a literal here.
"""

import os
import re
import sys
from dataclasses import dataclass

DEFAULT_CATEGORY = "factual_knowledge"


@dataclass(frozen=True)
class Config:
    model: str
    system: str
    max_tokens: int
    reasoning: str | None = None  # per-category reasoning_effort; None = global default


@dataclass(frozen=True)
class _Default:
    """A config before the environment is consulted.

    `model_index` selects from ALLOWED_MODELS rather than naming a model, since
    the real IDs aren't published until launch day. 0 is the cheapest/first.

    `tier` ("small" | "medium" | "large" | "") ranks ALLOWED_MODELS by
    capability inferred from the ids themselves (see `pick_capability`) --
    no vendor/model name is ever hardcoded, so this adapts to whatever family
    is actually injected on judging day.
    """
    model_index: int
    system: str
    max_tokens: int
    tier: str = ""
    reasoning: str | None = None  # None -> global REASONING_EFFORT (usually "none")


def allowed_models() -> list[str]:
    """Parse ALLOWED_MODELS. Accepts comma-, semicolon-, or newline-separated."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    parts = [p.strip() for p in raw.replace(";", ",").replace("\n", ",").split(",")]
    return [p for p in parts if p]


def pick_model(preference: int = 0) -> str:
    """Choose from ALLOWED_MODELS by index, clamped to the last entry.

    Returns "" when ALLOWED_MODELS is unset (local dev with the stub). R2's
    real client must treat "" as a hard error rather than guess an ID.
    """
    models = allowed_models()
    if not models:
        return ""
    return models[min(max(preference, 0), len(models) - 1)]


# Parameter count embedded in a model id, e.g. "qwen3-30b-a3b" -> 30,
# "llama-3.3-70b-instruct" -> 70. Deliberately has ZERO vendor/model-name
# knowledge -- the exact models on judging day aren't knowable in advance, so
# capability is inferred from whatever ALLOWED_MODELS actually contains, not
# from a guess about which family will be injected.
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def _param_billions(model_id: str) -> float | None:
    """Best-effort parameter count in billions parsed from the id, or None
    when no size token is present -- common on flagship/commercial ids that
    don't encode it (e.g. 'kimi-k2p6', 'deepseek-v4-pro', 'minimax-m2').

    A MoE id often lists both an active and a total count ('30b-a3b'): take
    the max, since total parameter count tracks capability more closely than
    active count, and API-billed tokens don't care about the difference.
    """
    matches = _SIZE_RE.findall(model_id)
    return max((float(m) for m in matches), default=None) if matches else None


def _rank_by_capability(models: list[str]) -> list[str]:
    """Order `models` least- to most-capable using ONLY signal in the ids --
    no hardcoded vendor/model names, so this adapts to whatever ALLOWED_MODELS
    contains on judging day, whichever families those turn out to be.

    Sized ids (an 'NNb' token present) sort by that count. Unsized ids are
    placed after every sized one, in their given order: an unsized flagship
    name is more likely to be a substantial production model than a small
    distilled one, so this errs toward not under-ranking it into 'small'.
    This is a heuristic, not a guarantee -- see pick_capability.
    """
    sized = [(m, _param_billions(m)) for m in models]
    # key=... is essential: sorting bare (m, b) tuples would order by the
    # model-id STRING first (alphabetically) and only fall back to `b` on a
    # name tie, silently ignoring the parameter count entirely.
    known = sorted(((m, b) for m, b in sized if b is not None), key=lambda pair: pair[1])
    unknown = [m for m, b in sized if b is None]
    return [m for m, _ in known] + unknown


def pick_capability(tier: str) -> str:
    """small | medium | large, chosen from ALLOWED_MODELS purely by relative
    capability signal in the ids themselves. Degrades gracefully as the list
    shrinks: with 2 models 'medium' collapses to the same choice as 'large';
    with 1 model every tier resolves to it. "" when ALLOWED_MODELS is unset.

    This offline ranking is a starting point for local testing before the
    real ALLOWED_MODELS is known, not a guarantee of correctness -- pure id
    inspection can't fully order arbitrary flagship names (see the module
    docstring). Once the real list is revealed, MEASURE with
    `python -m eval.score` and pin the winner via ROUTER_<CATEGORY>_MODEL.
    """
    models = allowed_models()
    if not models:
        return ""
    ranked = _rank_by_capability(models)
    n = len(ranked)
    if tier == "small":
        return ranked[0]
    if tier == "large":
        return ranked[-1]
    return ranked[n // 2]  # medium: biases toward 'large' as the list shrinks


def _env_key(category: str, field: str) -> str:
    return f"ROUTER_{category.upper()}_{field}"


def _override_int(category: str, field: str, fallback: int) -> int:
    """Read an int override, ignoring anything unparseable.

    A typo'd override must not crash the run: the harness gives us one shot,
    and a bad env var should degrade to the tuned default, loudly.
    """
    raw = os.environ.get(_env_key(category, field))
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        print(
            f"ignoring {_env_key(category, field)}={raw!r}: not an integer",
            file=sys.stderr,
        )
        return fallback


# R2: tune these. The 8 keys are the official hackathon categories — keep them
# in sync with agent/routing.py's PRIORITY list.
#
# `tier` is the INITIAL routing guess to measure against, not the final
# answer -- small/medium/large is resolved from ALLOWED_MODELS at call time
# by pick_capability (no vendor name is ever hardcoded). Rough split:
#   small  -- trivial asks: sentiment, NER, plain factual recall
#   medium -- moderate generation/summarisation work
#   large  -- code correctness, and the math/logic residue agent/solvers.py
#             couldn't answer at 0 tokens (that residue is BY DEFINITION the
#             hard/ambiguous case -- it deserves the strongest model, not a
#             mid-tier one, even though "logical reasoning" in the abstract
#             might otherwise be a medium-tier task)
# Once MEASURED (via `python -m eval.score` against the real injected list),
# bake the fewest-tokens-that-passes model per category here or via
# ROUTER_<CATEGORY>_MODEL=<exact id> / ROUTER_<CATEGORY>_MODEL_INDEX=n.
#
# Token budgets DO differ by category (cheap asks get tiny caps; code and
# multi-step reasoning need headroom) — that's a safe, model-independent lever.
# System prompts are billed as prompt tokens on EVERY call: every word here
# must either raise accuracy or shorten the output. Constraint kept, filler cut.
_DEFAULTS: dict[str, _Default] = {
    "factual_knowledge": _Default(
        0,
        "Reply with only the answer, concisely. No preamble.",
        150,
        tier="small",
    ),
    "math_reasoning": _Default(
        0,
        "Solve internally; output ONLY the final answer. No working shown.",
        200,
        tier="large",  # solver handles most; fallback residue is the hard case
    ),
    "sentiment_classification": _Default(
        0,
        # Label only: the grader scores the label; a 'reason' sentence is
        # ~10-15 billed tokens buying nothing.
        "Reply with one word: positive, negative, or neutral. Nothing else.",
        10,
        tier="small",
    ),
    "summarisation": _Default(
        0,
        "Match the prompt's length/format constraint exactly. "
        "Output ONLY the summary.",
        150,
        tier="medium",
    ),
    "named_entity_recognition": _Default(
        0,
        "Reply ONLY with label: value pairs, one entity per line "
        "(e.g. PERSON: John Smith).",
        120,
        tier="small",
    ),
    "code_debugging": _Default(
        0,
        "Return ONLY the corrected, complete function in one code block. No prose.",
        400,
        tier="medium",
    ),
    "logical_reasoning": _Default(
        0,
        "Reason internally; output ONLY the final answer. No reasoning shown.",
        200,
        tier="large",  # solver handles most; fallback residue is the hard case
    ),
    "code_generation": _Default(
        0,
        "Return ONLY one code block with the complete implementation. No explanation.",
        400,
        tier="medium",
    ),
}

CATEGORIES: tuple[str, ...] = tuple(_DEFAULTS)


def config_for(category: str) -> Config:
    """Resolve a category's config against the current environment.

    Unknown categories fall back to the general default, which is deliberately
    the most permissive one.
    """
    spec = _DEFAULTS.get(category)
    if spec is None:
        category = DEFAULT_CATEGORY
        spec = _DEFAULTS[DEFAULT_CATEGORY]

    # Model precedence, most specific first:
    #   1. ROUTER_<CAT>_MODEL       -- an explicit pinned id
    #   2. ROUTER_<CAT>_MODEL_INDEX -- an explicit index into ALLOWED_MODELS
    #   3. the category's tier      -- capability-ranked, else model_index
    #   4. model_index default
    model = os.environ.get(_env_key(category, "MODEL"))
    if not model:
        if os.environ.get(_env_key(category, "MODEL_INDEX")) is not None:
            model = pick_model(_override_int(category, "MODEL_INDEX", spec.model_index))
        elif spec.tier:
            model = pick_capability(spec.tier)
        else:
            model = pick_model(spec.model_index)

    # None (unset) keeps the global default; setting it (even to "") overrides.
    reasoning = os.environ.get(_env_key(category, "REASONING_EFFORT"), spec.reasoning)

    return Config(
        model=model,
        system=os.environ.get(_env_key(category, "SYSTEM"), spec.system),
        max_tokens=_override_int(category, "MAX_TOKENS", spec.max_tokens),
        reasoning=reasoning,
    )
