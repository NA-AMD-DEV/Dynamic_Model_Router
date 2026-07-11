"""Fireworks AI client — R2 owns this file. The only place that talks to Fireworks.

agent/core.py calls `call_model` once per task with that category's config.
"""

import os
import time

from openai import OpenAI

from agent.config import allowed_models

_client: OpenAI | None = None


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
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def call_model(prompt: str, system_prompt: str, model: str, max_tokens: int) -> dict:
    """Single call to Fireworks. Retries once on transient failure, then
    degrades to a best-effort empty answer so the task still returns.

    Returns {"answer": str, "tokens": int, "error": str | None}.
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
        return {"answer": "", "tokens": 0, "error": str(exc)}

    for attempt in range(2):  # 1 retry on transient error
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,  # low temp = more consistent, deterministic-ish answers
            )
            answer = (response.choices[0].message.content or "").strip()
            tokens_used = response.usage.total_tokens if response.usage else 0
            return {"answer": answer, "tokens": tokens_used, "error": None}
        except Exception as exc:
            if attempt == 0:
                time.sleep(1)  # brief pause before retry
                continue
            return {"answer": "", "tokens": 0, "error": str(exc)}
