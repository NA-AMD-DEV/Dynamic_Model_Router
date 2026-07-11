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
import sys
from dataclasses import dataclass

DEFAULT_CATEGORY = "factual_knowledge"


@dataclass(frozen=True)
class Config:
    model: str
    system: str
    max_tokens: int


@dataclass(frozen=True)
class _Default:
    """A config before the environment is consulted.

    `model_index` selects from ALLOWED_MODELS rather than naming a model, since
    the real IDs aren't published until launch day. 0 is the cheapest/first.
    """
    model_index: int
    system: str
    max_tokens: int


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
# in sync with agent/routing.py's PRIORITY list. Cheap categories get tiny
# budgets; code and multi-step reasoning need headroom. model_index is an
# offset into ALLOWED_MODELS (0 = cheapest, not an ID).
_DEFAULTS: dict[str, _Default] = {
    "factual_knowledge": _Default(
        0,
        "Answer the question directly and concisely. "
        "No preamble, no restating the question. Answer only.",
        150,
    ),
    "math_reasoning": _Default(
        1,
        "Solve the problem step by step internally, but output ONLY the final "
        "numeric or short answer. Do not show your work. Answer only.",
        200,
    ),
    "sentiment_classification": _Default(
        0,
        "Classify the sentiment as positive, negative, or neutral. "
        "Reply in the format: label: <sentiment> | reason: <one short sentence>. "
        "Nothing else.",
        60,
    ),
    "summarisation": _Default(
        0,
        "Summarise the given text to exactly match the requested length/format "
        "constraint in the prompt. Output ONLY the summary, nothing else.",
        150,
    ),
    "named_entity_recognition": _Default(
        0,
        "Extract named entities. Reply ONLY as label:value pairs, one per line "
        "(e.g. PERSON: John Smith). No prose, no explanation.",
        120,
    ),
    "code_debugging": _Default(
        1,
        "Find the bug and return ONLY the corrected, complete function in a single "
        "code block. No explanation, no prose before or after.",
        400,
    ),
    "logical_reasoning": _Default(
        1,
        "Solve the constraint puzzle. Verify all conditions are satisfied internally, "
        "but output ONLY the final answer. No reasoning shown.",
        200,
    ),
    "code_generation": _Default(
        1,
        "Write the function exactly as specified. Return ONLY a single code block "
        "with the complete, correct implementation. No explanation.",
        400,
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

    # An explicit model ID beats the index. R2 uses this to pin one category to
    # a specific model mid-experiment without disturbing the others.
    model = os.environ.get(_env_key(category, "MODEL"))
    if not model:
        model = pick_model(_override_int(category, "MODEL_INDEX", spec.model_index))

    return Config(
        model=model,
        system=os.environ.get(_env_key(category, "SYSTEM"), spec.system),
        max_tokens=_override_int(category, "MAX_TOKENS", spec.max_tokens),
    )
