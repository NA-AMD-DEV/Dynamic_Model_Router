"""The shared seam. R1 calls it, R2 implements it, R3 tests it.

R2: replace the body of `answer_task` with the real Fireworks call. Keep the
signature and the two invariants below; the container and the eval harness both
depend on them.
"""

from agent.routing import classify
from agent.config import config_for


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

    # R2: this is where the Fireworks call goes. `cfg` gives you
    # cfg.model, cfg.system, cfg.max_tokens for this category.
    return f"[stub:{category}] {prompt[:60]}"
