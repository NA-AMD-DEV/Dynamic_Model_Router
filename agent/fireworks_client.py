"""Fireworks AI client — R2 owns this file. The only place that talks to Fireworks.

agent/core.py calls `call_model` once per task with that category's config.
"""

import os
import time

from openai import OpenAI

from agent.config import allowed_models

_client: OpenAI | None = None

# The harness requires each request to complete under 30s. The OpenAI SDK's
# default timeout is 600s, which would let one hung call blow both that limit
# and eat into the 10-minute total run budget (main.py only checks its budget
# between tasks, not during one). A few seconds of margin under the hard limit
# for network overhead on top of the call itself.
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "25"))


def _get_client() -> OpenAI:
    """Constructed on first use, not at import — so importing this module in a
    test or a REPL doesn't require FIREWORKS_API_KEY to already be set.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("FIREWORKS_API_KEY")
        base_url = os.environ.get("FIREWORKS_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("FIREWORKS_API_KEY and FIREWORKS_BASE_URL must be set")
        _client = OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_S)
    return _client


# Reasoning models (e.g. deepseek-v4-pro) narrate "We are asked:..." before
# answering, which (a) truncates the real answer under a tight max_tokens and
# (b) bills tokens that buy nothing -- the exact metric we're ranked on.
# reasoning_effort=none suppressed it: 299 -> 60 tokens, same answer.
#
# But the harness injects an UNKNOWN model list on launch day, and a model that
# doesn't recognise the param returns 400. So we send it by default, and if the
# call is rejected for it, transparently retry WITHOUT it -- never let this
# optimisation cost a submission. Set REASONING_EFFORT="" to disable entirely.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "none")


def _looks_like_param_rejection(exc: Exception) -> bool:
    """True if the error is the server rejecting an unknown/invalid parameter
    (HTTP 400), as opposed to a transient failure worth retrying as-is.

    Checks status_code (the OpenAI SDK sets it on APIStatusError) before the
    message text: a substring like "400" would also match "try again in 400ms"
    or "4000 tokens", misclassifying transients.
    """
    if getattr(exc, "status_code", None) == 400:
        return True
    msg = str(exc).lower()
    return "invalid_request" in msg or "extra inputs" in msg


def call_model(prompt: str, system_prompt: str, model: str, max_tokens: int) -> dict:
    """Single call to Fireworks. Retries once on transient failure, then
    degrades to a best-effort empty answer so the task still returns.

    Returns {"answer": str, "tokens": int, "prompt_tokens": int,
             "completion_tokens": int, "truncated": bool, "error": str | None}.

    `tokens` is the total; the prompt/completion split exists because the
    official ranking may count only completion tokens (the leaderboard numbers
    are too small to include prompts), and tuning the wrong lever wastes work.
    `truncated` means the answer hit max_tokens (finish_reason == "length"):
    those tokens were billed for an answer that will likely be judged wrong.
    """
    models = allowed_models()
    if models and model not in models:
        model = models[0]  # safety net: never call a disallowed model

    # Missing credentials is a config error, not a transient one: fail fast
    # rather than sleeping through a pointless retry. Still returns a dict --
    # call_model never raises, so callers without their own guard stay safe.
    try:
        client = _get_client()
    except RuntimeError as exc:
        return _failure(str(exc))

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    # extra_body carries reasoning_effort; dropped if the model rejects it.
    extra_body = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else None

    # Two independent retry budgets, so a param-drop retry never consumes the
    # transient one (a range(2) loop here once fell off the end and returned
    # None when a transient error preceded a 400). Each branch is bounded --
    # extra_body can only be dropped once, transient_left only decrements --
    # so the loop makes at most 3 requests and always returns a dict.
    transient_left = 1
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,  # low temp = more consistent, deterministic-ish answers
                extra_body=extra_body,
            )
            choice = response.choices[0]
            answer = (choice.message.content or "").strip()
            usage = response.usage
            return {
                "answer": answer,
                "tokens": usage.total_tokens if usage else 0,
                "prompt_tokens": (usage.prompt_tokens if usage else 0) or 0,
                "completion_tokens": (usage.completion_tokens if usage else 0) or 0,
                "truncated": choice.finish_reason == "length",
                "error": None,
            }
        except Exception as exc:
            # If the model rejected reasoning_effort, drop it and retry once
            # without it -- the param is an optimisation, never a requirement.
            if extra_body is not None and _looks_like_param_rejection(exc):
                extra_body = None
                continue
            if transient_left:
                transient_left -= 1
                time.sleep(1)  # brief pause before retry
                continue
            return _failure(str(exc))


def _failure(error: str) -> dict:
    """The degraded shape: same keys as a success, so no caller branches."""
    return {
        "answer": "",
        "tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "truncated": False,
        "error": error,
    }
