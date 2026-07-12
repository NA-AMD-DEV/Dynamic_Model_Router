"""The shared seam. R1 calls it, R2 implements it, R3 tests it.

Frozen signature and both invariants below; the container and the eval
harness depend on them.
"""

from agent.config import DEFAULT_CATEGORY, config_for
from agent.fireworks_client import call_model
from agent.local_client import local_enabled, local_generate
from agent.routing import classify
from agent.solvers import solve_logic, solve_math

# A task we can answer in Python costs zero tokens and can't be got wrong by a
# weak model. Solvers are precision-first: they return None the moment anything
# is ambiguous, and we fall back to the model. Keyed by category.
_SOLVERS = {
    "math_reasoning": solve_math,
    "logical_reasoning": solve_logic,
}


def _zero_token(answer: str, category: str) -> dict:
    """The detailed record for an answer produced without a model call."""
    return {
        "answer": answer, "category": category, "tokens": 0,
        "prompt_tokens": 0, "completion_tokens": 0, "truncated": False,
        "error": None,
    }


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
    if not isinstance(prompt, str) or not prompt.strip():
        # A blank prompt has no answer; don't pay a model call to learn that.
        # (The isinstance guard also keeps classify() from raising on junk.)
        return _zero_token("", DEFAULT_CATEGORY)
    category = classify(prompt)

    # Deterministic 0-token shortcut for math/logic. On None (anything the
    # solver can't answer with confidence) we fall through to the model, which
    # for these categories is the STRONG tier -- the fallback is the hard
    # residue, exactly where the best model is worth its tokens.
    solver = _SOLVERS.get(category)
    if solver is not None:
        shortcut = solver(prompt)
        if shortcut is not None:
            return _zero_token(shortcut, category)

    cfg = config_for(category)

    # LOCAL_MODEL_PATH set -> run the feasibility probe's bundled model (0
    # Fireworks tokens). Unset -> the normal Fireworks path. The judge stays on
    # Fireworks regardless: it calls call_model directly, not through here.
    if local_enabled():
        result = local_generate(prompt, cfg.system, cfg.max_tokens)
    else:
        result = call_model(
            prompt=prompt,
            system_prompt=cfg.system,
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            reasoning_effort=cfg.reasoning,
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
