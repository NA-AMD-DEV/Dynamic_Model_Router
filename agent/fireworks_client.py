"""Fireworks AI client — R2 owns this file. The only place that talks to Fireworks.

agent/core.py calls `call_model` once per task with that category's config.
"""

import os
import re
import sys
import time
import threading

from openai import OpenAI

from agent.config import available_models, mark_unavailable

_client: OpenAI | None = None
_client_lock = threading.Lock()

# The harness requires each request to complete under 30s. The OpenAI SDK's
# default timeout is 600s, which would let one hung call blow both that limit
# and eat into the 10-minute total run budget (main.py only checks its budget
# between tasks, not during one). A few seconds of margin under the hard limit
# for network overhead on top of the call itself.
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "25"))

# 0.2 sampled a genuinely different completion on every run -- harmless for
# most categories (any correct phrasing passes), but summarisation's rubric is
# exact-format (precise sentence/bullet/word counts), so the SAME prompt could
# land at 2 sentences one run and 3 the next: a pass flipping to a fail purely
# from sampling, not quality. The judge shares this call path, so grading of
# an identical answer could also vary run to run. 0.0 minimizes (does not
# guarantee -- some providers retain minor nondeterminism even at 0) that.
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))


def _get_client() -> OpenAI:
    """Constructed on first use, not at import — so importing this module in a
    test or a REPL doesn't require FIREWORKS_API_KEY to already be set.

    Thread-safe: under ROUTER_CONCURRENCY>1, multiple workers may call this
    simultaneously.  Double-checked locking ensures exactly one OpenAI instance
    is constructed.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is not None:          # double-check after acquiring lock
            return _client
        api_key = os.environ.get("FIREWORKS_API_KEY")
        base_url = os.environ.get("FIREWORKS_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("FIREWORKS_API_KEY and FIREWORKS_BASE_URL must be set")
        # max_retries=0: call_model already implements its own retry budgets
        # (transient, param-rejection, model-failover). The SDK's default
        # (2 internal retries, each with its own fresh REQUEST_TIMEOUT_S and
        # backoff) stacks invisibly UNDER our retry loop -- one call_model
        # attempt could balloon to ~3x REQUEST_TIMEOUT_S before our own retry
        # even sees an exception, and our retry then does it again. Measured:
        # a single task hit 154s (5x the 30s hard per-request limit) this way.
        _client = OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=REQUEST_TIMEOUT_S, max_retries=0,
        )
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


def _looks_like_model_unavailable(exc: Exception) -> bool:
    """True if the proxy reports the model isn't deployed / doesn't exist (404
    NOT_FOUND). ALLOWED_MODELS can list ids that aren't actually served on the
    serverless instance; routing to one must fail over to a live model, not
    fail the task."""
    if getattr(exc, "status_code", None) == 404:
        return True
    msg = str(exc).lower()
    return "not_found" in msg or "not deployed" in msg or "inaccessible" in msg


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


# Some reasoning models inline their thinking as <think>...</think> in content
# even at low effort. Those tokens are already billed, but shipping them in the
# answer breaks the answer-only invariant and fails intent judging.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    # An unclosed <think> means the output is reasoning that never finished --
    # everything from the tag on is thinking, not answer.
    idx = text.lower().find("<think>")
    if idx != -1:
        text = text[:idx]
    return text.strip()


def call_model(prompt: str, system_prompt: str, model: str, max_tokens: int,
               reasoning_effort: str | None = None,
               allow_failover: bool = True) -> dict:
    """Single call to Fireworks. Retries once on transient failure, then
    degrades to a best-effort empty answer so the task still returns.

    Returns {"answer": str, "tokens": int, "prompt_tokens": int,
             "completion_tokens": int, "truncated": bool,
             "actual_model": str, "error": str | None}.

    `allow_failover=False` is probe mode (used by config.calibrate_lean): the
    requested model is called exactly as given -- no substitution on entry, no
    failover to another model on 404, and NO mark_unavailable from inside this
    client. A probe must never attribute one model's usage to another, and all
    availability bookkeeping stays with the (single-threaded) caller.

    `tokens` is the total; the prompt/completion split exists because the
    official ranking may count only completion tokens (the leaderboard numbers
    are too small to include prompts), and tuning the wrong lever wastes work.
    `truncated` means the answer hit max_tokens (finish_reason == "length"):
    those tokens were billed for an answer that will likely be judged wrong.

    `reasoning_effort` overrides the global default for this one call. The
    agent path passes a per-category value; the judge (eval/judge.py) passes
    nothing, so it keeps the generous global default it needs to emit a verdict.
    """
    models = available_models()
    if allow_failover and models and model not in models:
        model = models[0]  # safety net: never call a disallowed/dead model

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
    # An explicit per-call value wins; None means "use the global default".
    effort = reasoning_effort if reasoning_effort is not None else REASONING_EFFORT
    extra_body = {"reasoning_effort": effort} if effort else None

    # Independent retry budgets, so one kind of retry never consumes another's
    # (a range(2) loop here once fell off the end and returned None). Each is
    # bounded -- extra_body drops at most once, transient_left decrements once,
    # model fallback walks a finite list -- so the loop always returns a dict.
    tried_models: set[str] = set()
    transient_left = 1
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=TEMPERATURE,
                extra_body=extra_body,
            )
            choice = response.choices[0]
            answer = _strip_think(choice.message.content or "")
            usage = response.usage
            return {
                "answer": answer,
                "tokens": usage.total_tokens if usage else 0,
                "prompt_tokens": (usage.prompt_tokens if usage else 0) or 0,
                "completion_tokens": (usage.completion_tokens if usage else 0) or 0,
                "truncated": choice.finish_reason == "length",
                # `model` here is whatever was actually called (it mutates on
                # failover); prefer the server's echo when it exposes one.
                "actual_model": getattr(response, "model", None) or model,
                "error": None,
            }
        except Exception as exc:
            # Model not deployed: blocklist it so routing stops picking it, and
            # fail this task over to another live model rather than losing it.
            # In probe mode (allow_failover=False) do neither -- report the
            # error and let the caller own all availability bookkeeping.
            if _looks_like_model_unavailable(exc):
                if not allow_failover:
                    return _failure(str(exc))
                mark_unavailable(model)
                tried_models.add(model)
                nxt = next((m for m in available_models() if m not in tried_models), None)
                if nxt:
                    print(f"call_model: {model} unavailable, failing over to {nxt}", file=sys.stderr)
                    model = nxt
                    continue
                print(f"call_model: all models exhausted after {model} unavailable", file=sys.stderr)
                return _failure(str(exc))
            # If the model rejected reasoning_effort, drop it and retry once
            # without it -- the param is an optimisation, never a requirement.
            if extra_body is not None and _looks_like_param_rejection(exc):
                extra_body = None
                continue
            if transient_left:
                transient_left -= 1
                time.sleep(1)  # brief pause before retry
                continue
            # All retries for THIS model are exhausted. In failover mode, try
            # the remaining live models before giving up — maximises the chance
            # of a non-empty answer in an unfamiliar judging environment where
            # different models may accept different params or have different
            # availability windows.
            if allow_failover:
                tried_models.add(model)
                nxt = next((m for m in available_models() if m not in tried_models), None)
                if nxt:
                    print(f"call_model: {model} hard-failed ({str(exc)[:80]}), "
                          f"trying {nxt}", file=sys.stderr)
                    model = nxt
                    transient_left = 1  # fresh retry budget for the new model
                    # Re-enable reasoning_effort for the new model — it may
                    # support what the old one rejected.
                    effort = reasoning_effort if reasoning_effort is not None else REASONING_EFFORT
                    extra_body = {"reasoning_effort": effort} if effort else None
                    continue
            print(f"call_model failed: model={model} error={str(exc)[:200]}", file=sys.stderr)
            return _failure(str(exc))


def _failure(error: str) -> dict:
    """The degraded shape: same keys as a success, so no caller branches."""
    return {
        "answer": "",
        "tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "truncated": False,
        "actual_model": "",
        "error": error,
    }
