"""Local LLM-judge — R3 owns this. Mirrors the real accuracy gate so we can
estimate whether we clear it BEFORE submitting.

The judge scores each answer against its expected_intent, not against an exact
string, because the real gate scores intent. It routes through the same
Fireworks client the agent uses, so judging costs tokens too -- keep the judge
model cheap and its output tiny (a single 0/1).

R3: the real gate's threshold and rubric are unknown. Treat this as a proxy,
keep a safety margin, and re-check with fresh rewordings. A locally 'passing'
score is necessary, not sufficient.
"""

import os
import re

from agent.config import allowed_models
from agent.fireworks_client import call_model

# Judge with a cheap model by default; override if R3 wants a stronger judge.
JUDGE_MODEL_ENV = "JUDGE_MODEL"

_JUDGE_SYSTEM = (
    "You are a strict grader. You are given a QUESTION, the EXPECTED INTENT of a "
    "correct answer, and a CANDIDATE answer. Reply with exactly '1' if the "
    "candidate satisfies the expected intent, or '0' if it does not. Reply with "
    "only the single digit, nothing else."
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
        "Does the candidate satisfy the expected intent? Reply 1 or 0."
    )
    result = call_model(
        prompt=user,
        system_prompt=_JUDGE_SYSTEM,
        model=_judge_model(),
        max_tokens=1,
    )
    if result["error"]:
        return {"score": 0, "error": result["error"]}

    # Be forgiving about the model wrapping the digit in stray text.
    m = re.search(r"[01]", result["answer"])
    score = int(m.group()) if m else 0
    return {"score": score, "error": None}
