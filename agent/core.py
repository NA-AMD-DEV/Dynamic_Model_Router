"""The shared seam. R1 calls it, R2 implements it, R3 tests it.

Frozen signature and both invariants below; the container and the eval
harness depend on them.
"""

from agent.config import config_for
from agent.fireworks_client import call_model
from agent.routing import classify


def answer_task(task: dict) -> str:
    """Answer one task.

    Invariants (the container relies on both):
      - Always returns a `str`. Never returns None, never raises. On any
        internal failure, return a best-effort string instead.
      - Returns the answer only: no preamble, no restating the prompt.

    `task` is `{"task_id": "t1", "prompt": "..."}`. The prompt does not name
    its category; `classify` infers it and `config_for` supplies that
    category's model, system prompt, and max_tokens.
    """
    return answer_task_detailed(task)["answer"]


def answer_task_detailed(task: dict) -> dict:
    """Same work as `answer_task`, but returns the full record R3 needs to
    total the ranking metric:

        {"answer": str, "category": str, "tokens": int, "prompt_tokens": int,
         "completion_tokens": int, "truncated": bool, "error": str | None}

    `answer_task` is the frozen container-facing seam (str only); this is the
    eval-facing one. Both go through the exact same code path so the tokens
    R3 measures are the tokens the container actually spends.
    """
    prompt = task.get("prompt", "")
    category = classify(prompt)
    cfg = config_for(category)

    result = call_model(
        prompt=prompt,
        system_prompt=cfg.system,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
    )
    return {
        "answer": result["answer"],
        "category": category,
        "tokens": result["tokens"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "truncated": result["truncated"],
        "error": result["error"],
    }
