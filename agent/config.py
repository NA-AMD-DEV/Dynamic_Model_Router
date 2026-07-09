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

DEFAULT_CATEGORY = "general"


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


# Answer-only instruction shared by every category. Every token here is billed
# on every call, so it stays one line.
_BASE = "Answer only. No preamble, no explanation, no restating the question."

# R2: tune these. Cheap categories get tiny budgets; code and multi-step math
# need headroom. model_index is an offset into ALLOWED_MODELS, not an ID.
_DEFAULTS: dict[str, _Default] = {
    "sentiment": _Default(0, f"{_BASE} Reply with one word.", 5),
    "ner": _Default(0, f"{_BASE} One 'label: value' per line.", 64),
    "classification": _Default(0, f"{_BASE} Reply with the label only.", 10),
    "extraction": _Default(0, f"{_BASE} Output the extracted values only.", 96),
    "summarization": _Default(0, f"{_BASE} Honour any requested length.", 128),
    "translation": _Default(0, f"{_BASE} Output the translation only.", 256),
    "math": _Default(1, f"{_BASE} Give the final answer only.", 256),
    "code": _Default(1, f"{_BASE} One code block. No prose.", 512),
    DEFAULT_CATEGORY: _Default(1, _BASE, 512),
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
