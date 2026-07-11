"""Optional LOCAL inference backend — feasibility probe only.

Competition rule: inference run INSIDE the container on a bundled model counts
as ZERO Fireworks tokens (only calls through FIREWORKS_BASE_URL are scored).
This backend lets us MEASURE, before committing to a local-first rearchitecture,
whether a small CPU model can clear the accuracy gate on the 8 categories
within the 30s-per-request / 10-min-total time budget.

Enabled only when `LOCAL_MODEL_PATH` points at a GGUF file. Otherwise nothing
here runs and the agent uses Fireworks exactly as before. `llama-cpp-python` is
imported LAZILY inside `_get_llm`, so it is a probe-only dependency: the
Fireworks path, the container image, and the test suite never need it.

Returns the SAME dict shape as agent.fireworks_client.call_model, so the whole
pipeline (routing, solvers, prompts, eval.score, the judge) works unchanged --
only where the tokens are spent differs.
"""

import os

from agent.fireworks_client import _strip_think

_llm = None


def local_enabled() -> bool:
    return bool(os.environ.get("LOCAL_MODEL_PATH"))


def _get_llm():
    """Construct the llama.cpp model once. Threads default to the box's CPU
    count (the judging VM is CPU-only); n_batch is high to speed prompt eval,
    the trick that gave the reference local-inference teams their throughput.
    """
    global _llm
    if _llm is None:
        from llama_cpp import Llama  # lazy: only the probe needs this dependency

        _llm = Llama(
            model_path=os.environ["LOCAL_MODEL_PATH"],
            n_ctx=int(os.environ.get("LOCAL_N_CTX", "2048")),
            n_threads=int(os.environ.get("LOCAL_N_THREADS", str(os.cpu_count() or 2))),
            n_batch=int(os.environ.get("LOCAL_N_BATCH", "512")),
            verbose=False,
        )
    return _llm


def _failure(error: str) -> dict:
    return {
        "answer": "", "tokens": 0, "prompt_tokens": 0,
        "completion_tokens": 0, "truncated": False, "error": error,
    }


def local_generate(prompt: str, system_prompt: str, max_tokens: int) -> dict:
    """Run one chat completion on the bundled local model. Never raises --
    returns the standard record with an `error` string on any failure, so the
    caller's contract (answer_task never raises) is preserved."""
    try:
        llm = _get_llm()
    except Exception as exc:  # missing dep, bad path, OOM at load
        return _failure(f"local model load failed: {exc}")

    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
    except Exception as exc:
        return _failure(f"local generation failed: {exc}")

    choice = resp["choices"][0]
    answer = _strip_think(choice["message"].get("content") or "")
    usage = resp.get("usage") or {}
    # Local tokens count as ZERO for the competition; we still report them so
    # the probe can gauge output verbosity and prompt-eval cost.
    return {
        "answer": answer,
        "tokens": usage.get("total_tokens", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "truncated": choice.get("finish_reason") == "length",
        "error": None,
    }
