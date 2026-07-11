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
    prompt = task.get("prompt", "")
    category = classify(prompt)
    cfg = config_for(category)

    result = call_model(
        prompt=prompt,
        system_prompt=cfg.system,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
    )
    return result["answer"]
