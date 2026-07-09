"""Per-category model, prompt, and token budget.

R2 owns the values in CONFIGS. Edit them freely; the container reads this table
and never hardcodes a model or a prompt.

Model IDs come from ALLOWED_MODELS at runtime, never from a literal here. The
env var is a separator-delimited list published on launch day.
"""

import os
from dataclasses import dataclass

DEFAULT_CATEGORY = "general"


@dataclass(frozen=True)
class Config:
    model: str
    system: str
    max_tokens: int


def allowed_models() -> list[str]:
    """Parse ALLOWED_MODELS. Accepts comma-, semicolon-, or whitespace-separated."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    parts = [p.strip() for p in raw.replace(";", ",").replace("\n", ",").split(",")]
    return [p for p in parts if p]


def pick_model(preference: int = 0) -> str:
    """Choose from ALLOWED_MODELS by index, clamped.

    Index 0 is whatever the organisers list first. R2 should tune these indices
    per category once the real list is published; the clamp means an
    out-of-range preference degrades to the last model rather than crashing.
    """
    models = allowed_models()
    if not models:
        # ALLOWED_MODELS unset (local dev). The stub never calls a model, and
        # R2's real client should surface this loudly rather than guess an ID.
        return ""
    return models[min(preference, len(models) - 1)]


# Answer-only instruction shared by every category. Every token here is billed
# on every call, so it stays one line.
_BASE = "Answer only. No preamble, no explanation, no restating the question."

# R2: tune `system`, `max_tokens`, and the pick_model index per category.
# Cheap categories get tiny budgets; code and multi-step math need headroom.
CONFIGS: dict[str, Config] = {
    "sentiment": Config(pick_model(0), f"{_BASE} Reply with one word.", 5),
    "ner": Config(pick_model(0), f"{_BASE} One 'label: value' per line.", 64),
    "classification": Config(pick_model(0), f"{_BASE} Reply with the label only.", 10),
    "extraction": Config(pick_model(0), f"{_BASE} Output the extracted values only.", 96),
    "summarization": Config(pick_model(0), f"{_BASE} Honour any requested length.", 128),
    "translation": Config(pick_model(0), f"{_BASE} Output the translation only.", 256),
    "math": Config(pick_model(1), f"{_BASE} Give the final answer only.", 256),
    "code": Config(pick_model(1), f"{_BASE} One code block. No prose.", 512),
    DEFAULT_CATEGORY: Config(pick_model(1), _BASE, 512),
}


def config_for(category: str) -> Config:
    """Look up a category's config, falling back to the general default."""
    return CONFIGS.get(category, CONFIGS[DEFAULT_CATEGORY])
