"""Local LLM-judge — R3 owns this. Mirrors the real accuracy gate so we can
estimate whether we clear it BEFORE submitting.

The judge scores each answer against its expected_intent, not against an exact
string, because the real gate scores intent. It routes through the same
Fireworks client the agent uses.

Judge tokens are LOCAL-ONLY -- they never count toward the competition ranking
(only the agent's calls through the real harness do). So the judge is allowed
to reason: reasoning models (e.g. deepseek-v4-pro) cannot emit a verdict under
a tiny max_tokens, so we give it room and read the LAST 0/1 it produces (the
verdict comes after any reasoning). A cramped judge silently scores everything
0 -- that was the original bug.

R3: the real gate's threshold and rubric are unknown. Treat this as a proxy,
keep a safety margin, and re-check with fresh rewordings. A locally 'passing'
score is necessary, not sufficient.
"""

import os
import re
import sys

from agent.config import allowed_models
from agent.fireworks_client import call_model

# Judge with a cheap model by default; override if R3 wants a stronger judge.
JUDGE_MODEL_ENV = "JUDGE_MODEL"


def _int_env(name: str, fallback: int) -> int:
    """A typo'd override degrades to the default, loudly -- same policy as
    agent/config.py: never let a bad env var crash a run."""
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        print(f"ignoring {name}={raw!r}: not an integer", file=sys.stderr)
        return fallback


# Enough room for a reasoning model to think and then emit its verdict. Override
# via JUDGE_MAX_TOKENS. Judge cost is local-only, so err generous, not stingy.
JUDGE_MAX_TOKENS = _int_env("JUDGE_MAX_TOKENS", 2000)

_JUDGE_SYSTEM = (
    "You are a strict grader. You are given a QUESTION, the EXPECTED INTENT of a "
    "correct answer, and a CANDIDATE answer. Decide whether the candidate "
    "satisfies the expected intent. The expected intent may state HARD "
    "requirements -- exact sentence or bullet counts, word limits, required "
    "entity labels, reasons that must acknowledge both positive and negative "
    "aspects, multiple sub-answers. Enforce every stated requirement literally: "
    "if the candidate violates any one of them, it does not pass. You may reason "
    "briefly, but your reply must END with a single line containing only "
    "'1' (satisfies) or '0' (does not)."
)


def _judge_model() -> str:
    override = os.environ.get(JUDGE_MODEL_ENV)
    if override:
        return override
    models = allowed_models()
    return models[0] if models else ""


def score_one(prompt: str, expected_intent: str, candidate: str) -> dict:
    """Return {"score": 0 or 1, "error": str | None}.

    An empty candidate scores 0 without spending a judge call -- a non-answer
    never satisfies intent, and we shouldn't pay tokens to confirm it.
    """
    if not candidate.strip():
        return {"score": 0, "error": None}

    user = (
        f"QUESTION:\n{prompt}\n\n"
        f"EXPECTED INTENT:\n{expected_intent}\n\n"
        f"CANDIDATE:\n{candidate}\n\n"
        "Does the candidate satisfy the expected intent? "
        "End with a line containing only 1 or 0."
    )
    result = call_model(
        prompt=user,
        system_prompt=_JUDGE_SYSTEM,
        model=_judge_model(),
        max_tokens=JUDGE_MAX_TOKENS,
    )
    if result["error"]:
        return {"score": 0, "error": result["error"]}

    # The verdict is the LAST 0/1 the model emits -- a reasoning model produces
    # the digit after its thinking, and stray 0/1 may appear mid-reasoning.
    digits = re.findall(r"[01]", result["answer"])
    if not digits:
        # Model answered but never produced a verdict digit (e.g. ran out of
        # room mid-reasoning). Surface it rather than silently scoring 0.
        return {"score": 0, "error": f"no verdict digit in judge output: {result['answer'][:120]!r}"}
    return {"score": int(digits[-1]), "error": None}
