"""
Fireworks AI client wrapper — R2 owns this file entirely.
This is the ONLY place in the whole project that talks to Fireworks.
R1 (container/harness) will import `call_model` and use it via config_for().
"""

import os
import time
from openai import OpenAI

# --- Load config from environment (NEVER hardcode these) ---
# NOTE: we do NOT raise at import time anymore. A missing key should
# degrade gracefully (every task gets a fallback answer), not crash the
# whole harness before R1's fallback-seeding/finally-flush can even run.
API_KEY = os.environ.get("FIREWORKS_API_KEY")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL")
ALLOWED_MODELS = os.environ.get("ALLOWED_MODELS", "").split(",")
ALLOWED_MODELS = [m.strip() for m in ALLOWED_MODELS if m.strip()]

DEFAULT_MODEL = ALLOWED_MODELS[0] if ALLOWED_MODELS else None

_client = None
if API_KEY and BASE_URL:
    _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_model(prompt: str, system_prompt: str, model: str, max_tokens: int) -> dict:
    """
    Makes a single call to Fireworks. Returns dict with answer + token usage.
    Retries once on transient failure. Never raises — always returns something,
    even if FIREWORKS_API_KEY/BASE_URL were never set.
    """
    if _client is None:
        return {
            "answer": "",
            "tokens": 0,
            "error": "FIREWORKS_API_KEY and/or FIREWORKS_BASE_URL not set in environment",
        }

    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL  # safety net — never call a disallowed model

    for attempt in range(2):  # 1 retry on transient error
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0
            return {"answer": answer, "tokens": tokens_used, "error": None}
        except Exception as e:
            if attempt == 0:
                time.sleep(1)
                continue
            return {"answer": "", "tokens": 0, "error": str(e)}