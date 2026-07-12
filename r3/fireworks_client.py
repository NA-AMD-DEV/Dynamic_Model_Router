"""
Fireworks AI client wrapper — R2 owns this file entirely.
This is the ONLY place in the whole project that talks to Fireworks.
R1 (container/harness) will import `answer_task` and call it per task.
"""

import os
import time
from openai import OpenAI

# --- Load config from environment (NEVER hardcode these) ---
API_KEY = os.environ.get("FIREWORKS_API_KEY")
BASE_URL = os.environ.get("FIREWORKS_BASE_URL")
ALLOWED_MODELS = os.environ.get("ALLOWED_MODELS", "").split(",")
ALLOWED_MODELS = [m.strip() for m in ALLOWED_MODELS if m.strip()]

if not API_KEY or not BASE_URL:
    raise RuntimeError("FIREWORKS_API_KEY and FIREWORKS_BASE_URL must be set")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Fallback model if something goes wrong picking a category-specific one
DEFAULT_MODEL = ALLOWED_MODELS[0] if ALLOWED_MODELS else None


def call_model(prompt: str, system_prompt: str, model: str, max_tokens: int) -> dict:
    """
    Makes a single call to Fireworks. Returns dict with answer + token usage.
    Retries once on transient failure. Never raises — always returns something.
    """
    if model not in ALLOWED_MODELS:
        model = DEFAULT_MODEL  # safety net — never call a disallowed model

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
            answer = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens if response.usage else 0
            return {"answer": answer, "tokens": tokens_used, "error": None}
        except Exception as e:
            if attempt == 0:
                time.sleep(1)  # brief pause before retry
                continue
            # Both attempts failed — best-effort fallback so the task still returns
            return {"answer": "", "tokens": 0, "error": str(e)}
