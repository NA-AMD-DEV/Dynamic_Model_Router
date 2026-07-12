"""Tests for the R3 eval slice: the token-accounting seam, the judge, and the
scorer. These use a mocked Fireworks client so they need no key and no network.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import agent.fireworks_client as fc
from agent.core import answer_task, answer_task_detailed


def _mock_completion(content: str, tokens: int, prompt_tokens: int = 0,
                     completion_tokens: int = 0, finish_reason: str = "stop"):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = finish_reason
    resp.usage.total_tokens = tokens
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    fake = MagicMock()
    monkeypatch.setattr(fc, "_get_client", lambda: fake)
    return fake


def test_detailed_seam_exposes_tokens_and_category(mock_client):
    mock_client.chat.completions.create.return_value = _mock_completion("  hi  ", 37)
    # A math prompt the solver deliberately defers on (multi-step phrasing), so
    # this still exercises the model path and its token accounting.
    detail = answer_task_detailed({"task_id": "t1", "prompt": "Calculate 17 * 23 and then subtract 40."})
    assert detail["answer"] == "hi"          # stripped
    assert detail["tokens"] == 37            # the ranking metric, not dropped
    assert detail["category"] == "math_reasoning"
    assert detail["error"] is None


def test_detailed_seam_exposes_token_split_and_truncation(mock_client):
    mock_client.chat.completions.create.return_value = _mock_completion(
        "partial", 50, prompt_tokens=30, completion_tokens=20, finish_reason="length"
    )
    detail = answer_task_detailed({"task_id": "t1", "prompt": "hello"})
    assert detail["prompt_tokens"] == 30
    assert detail["completion_tokens"] == 20
    assert detail["truncated"] is True   # hit max_tokens: cap-tuning signal


def test_local_backend_is_used_when_LOCAL_MODEL_PATH_set(monkeypatch):
    # With LOCAL_MODEL_PATH set, agent inference routes to the local backend
    # and never touches Fireworks (0 counted tokens).
    import agent.core as core
    monkeypatch.setenv("LOCAL_MODEL_PATH", "/models/whatever.gguf")

    captured = {}

    def fake_local(prompt, system, max_tokens):
        captured["called"] = True
        return {"answer": "iron", "tokens": 12, "prompt_tokens": 9,
                "completion_tokens": 3, "truncated": False, "error": None}

    def boom(*a, **k):
        raise AssertionError("Fireworks must not be called in local mode")

    monkeypatch.setattr(core, "local_generate", fake_local)
    monkeypatch.setattr(core, "call_model", boom)

    detail = core.answer_task_detailed({"task_id": "f", "prompt": "Which element has symbol Fe?"})
    assert captured.get("called") and detail["answer"] == "iron"


def test_local_generate_returns_error_dict_when_dep_missing(monkeypatch):
    # llama-cpp-python is a probe-only dependency and isn't installed for the
    # test suite; local_generate must degrade to an error dict, never raise.
    import agent.local_client as lc
    monkeypatch.setenv("LOCAL_MODEL_PATH", "/models/none.gguf")
    monkeypatch.setattr(lc, "_llm", None)
    result = lc.local_generate("p", "s", 50)
    assert result["answer"] == "" and result["tokens"] == 0
    assert result["error"] is not None  # surfaced, not swallowed


def test_empty_prompt_short_circuits_without_a_call(mock_client):
    for prompt in ("", "   ", None):
        detail = answer_task_detailed({"task_id": "t9", "prompt": prompt})
        assert detail["answer"] == "" and detail["tokens"] == 0
        assert detail["error"] is None
    mock_client.chat.completions.create.assert_not_called()


def test_inline_think_blocks_are_stripped(mock_client):
    mock_client.chat.completions.create.return_value = _mock_completion(
        "<think>6 times 7... classic</think>42", 20
    )
    assert answer_task({"task_id": "t1", "prompt": "What is 6*7?"}) == "42"


def test_unclosed_think_block_yields_empty_answer(mock_client):
    # Truncated mid-reasoning: everything after <think> is thinking, not answer.
    mock_client.chat.completions.create.return_value = _mock_completion(
        "<think>let me consider the", 20, finish_reason="length"
    )
    assert answer_task({"task_id": "t1", "prompt": "hard question"}) == ""


def test_frozen_seam_still_returns_bare_str(mock_client):
    mock_client.chat.completions.create.return_value = _mock_completion("answer", 5)
    out = answer_task({"task_id": "t1", "prompt": "hello"})
    assert isinstance(out, str) and out == "answer"


def test_detailed_and_frozen_agree_on_answer(mock_client):
    mock_client.chat.completions.create.return_value = _mock_completion("same", 9)
    task = {"task_id": "t1", "prompt": "hello"}
    assert answer_task(task) == answer_task_detailed(task)["answer"]


def test_reasoning_param_rejection_falls_back(monkeypatch):
    # A model that rejects reasoning_effort (400) must not fail the call: drop
    # the param and retry. This is the compliance safety net for launch day,
    # when the injected model list is unknown.
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setattr(fc, "REASONING_EFFORT", "none")
    seen = []

    def create(**kw):
        seen.append(kw.get("extra_body"))
        if kw.get("extra_body") is not None:
            raise Exception("Error code: 400 - invalid_request_error: Extra inputs are not permitted")
        return _mock_completion("ok", 10)

    fake = MagicMock()
    fake.chat.completions.create = create
    monkeypatch.setattr(fc, "_get_client", lambda: fake)

    result = fc.call_model("p", "s", "test-model", 100)
    assert result["answer"] == "ok" and result["error"] is None
    assert seen == [{"reasoning_effort": "none"}, None]  # tried with, then without


def test_per_category_reasoning_effort_overrides_global(monkeypatch):
    # A per-category effort passed by the agent path wins over the global; the
    # judge, which passes nothing, is unaffected.
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setattr(fc, "REASONING_EFFORT", "none")
    seen = []

    def create(**kw):
        seen.append(kw.get("extra_body"))
        return _mock_completion("ok", 10)

    fake = MagicMock()
    fake.chat.completions.create = create
    monkeypatch.setattr(fc, "_get_client", lambda: fake)

    fc.call_model("p", "s", "test-model", 100, reasoning_effort="low")
    assert seen == [{"reasoning_effort": "low"}]           # explicit per-call value
    seen.clear()
    fc.call_model("p", "s", "test-model", 100)             # judge-style: no arg
    assert seen == [{"reasoning_effort": "none"}]          # falls to global default


def test_config_for_threads_reasoning_effort(monkeypatch):
    from agent.config import config_for
    monkeypatch.setenv("ROUTER_CODE_DEBUGGING_REASONING_EFFORT", "low")
    assert config_for("code_debugging").reasoning == "low"
    assert config_for("factual_knowledge").reasoning is None  # default: global


def test_reasoning_param_sent_when_enabled(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setattr(fc, "REASONING_EFFORT", "none")
    seen = []

    def create(**kw):
        seen.append(kw.get("extra_body"))
        return _mock_completion("ok", 10)

    fake = MagicMock()
    fake.chat.completions.create = create
    monkeypatch.setattr(fc, "_get_client", lambda: fake)

    fc.call_model("p", "s", "test-model", 100)
    assert seen == [{"reasoning_effort": "none"}]  # sent on the first try


def test_undeployed_model_fails_over_to_a_live_one(monkeypatch):
    # ALLOWED_MODELS lists a model that isn't served (404). The task must not
    # fail: blocklist the dead model and retry on the next live one.
    import agent.config as cfg
    monkeypatch.setattr(cfg, "_UNAVAILABLE", set())
    monkeypatch.setenv("ALLOWED_MODELS", "dead-model, live-model")
    monkeypatch.setattr(fc, "REASONING_EFFORT", "")
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    seen = []

    def create(**kw):
        seen.append(kw["model"])
        if kw["model"] == "dead-model":
            exc = Exception("Error code: 404 - Model not found, inaccessible, and/or not deployed")
            exc.status_code = 404
            raise exc
        return _mock_completion("ok", 8)

    fake = MagicMock()
    fake.chat.completions.create = create
    monkeypatch.setattr(fc, "_get_client", lambda: fake)

    result = fc.call_model("p", "s", "dead-model", 50)
    assert result["answer"] == "ok" and result["error"] is None
    assert seen == ["dead-model", "live-model"]
    assert "dead-model" in cfg._UNAVAILABLE  # remembered, so routing skips it next time


def test_marked_unavailable_model_is_dropped_from_routing(monkeypatch):
    import agent.config as cfg
    from agent.config import available_models, config_for
    monkeypatch.setattr(cfg, "_UNAVAILABLE", set())
    monkeypatch.setenv("ALLOWED_MODELS", "gemma-4-31b-it, minimax-m3, kimi-k2p7-code")
    # gemma ranks 'largest' by parsed size, but once it 404s it's out.
    cfg.mark_unavailable("gemma-4-31b-it")
    assert "gemma-4-31b-it" not in available_models()
    # No category should resolve to the dead model any more.
    assert all(config_for(c).model != "gemma-4-31b-it" for c in cfg.CATEGORIES)


def test_transient_then_param_rejection_still_returns_dict(monkeypatch):
    # Regression: a transient failure followed by a 400 used to exhaust the
    # retry loop and fall off the end, returning None -- which broke the
    # "never raises" invariant one frame up. Must drop the param and try again.
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setattr(fc, "REASONING_EFFORT", "none")
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    calls = []

    def create(**kw):
        calls.append(kw.get("extra_body"))
        if len(calls) == 1:
            raise Exception("connection timed out")
        if len(calls) == 2:
            raise Exception("Error code: 400 - invalid_request_error: Extra inputs are not permitted")
        return _mock_completion("ok", 7)

    fake = MagicMock()
    fake.chat.completions.create = create
    monkeypatch.setattr(fc, "_get_client", lambda: fake)

    result = fc.call_model("p", "s", "test-model", 100)
    assert result is not None and result["answer"] == "ok" and result["error"] is None
    assert calls == [{"reasoning_effort": "none"}, {"reasoning_effort": "none"}, None]


def test_transient_mentioning_400ms_keeps_reasoning_param(monkeypatch):
    # "try again in 400ms" is a transient, not a param rejection: the retry
    # must keep extra_body rather than throwing the optimisation away.
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    monkeypatch.setattr(fc, "REASONING_EFFORT", "none")
    monkeypatch.setattr(fc.time, "sleep", lambda s: None)
    calls = []

    def create(**kw):
        calls.append(kw.get("extra_body"))
        if len(calls) == 1:
            raise Exception("Error code: 429 - rate limited, try again in 400ms")
        return _mock_completion("ok", 7)

    fake = MagicMock()
    fake.chat.completions.create = create
    monkeypatch.setattr(fc, "_get_client", lambda: fake)

    result = fc.call_model("p", "s", "test-model", 100)
    assert result["answer"] == "ok"
    assert calls == [{"reasoning_effort": "none"}, {"reasoning_effort": "none"}]


def test_param_rejection_detected_by_status_code():
    exc = Exception("Bad request")
    exc.status_code = 400
    assert fc._looks_like_param_rejection(exc)
    # Substring "400"/"4000" alone is not evidence of a param rejection.
    assert not fc._looks_like_param_rejection(Exception("maximum context is 4000 tokens"))


def test_call_model_never_raises_without_credentials(monkeypatch):
    # No key set: must return an error dict, not raise -- eval tooling has no
    # per-call guard of its own.
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    monkeypatch.setattr(fc, "_client", None)
    result = fc.call_model("p", "s", "m", 10)
    assert result["answer"] == "" and result["tokens"] == 0
    assert result["error"] is not None


def test_judge_scores_empty_answer_zero_without_a_call(monkeypatch):
    from eval.judge import score_one

    # An empty candidate must not cost a judge call.
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        return {"answer": "1", "tokens": 1, "error": None}

    monkeypatch.setattr("eval.judge.call_model", spy)
    verdict = score_one("q", "intent", "   ")
    assert verdict["score"] == 0
    assert called["n"] == 0  # no tokens spent on a non-answer


def test_judge_parses_digit_from_noisy_output(monkeypatch):
    from eval.judge import score_one

    monkeypatch.setattr(
        "eval.judge.call_model",
        lambda **k: {"answer": "Score: 1", "tokens": 2, "error": None},
    )
    assert score_one("q", "intent", "a real answer")["score"] == 1


def test_judge_takes_last_digit_after_reasoning(monkeypatch):
    # A reasoning model thinks first (with stray digits) then gives its verdict
    # last. This is the case that made the whole eval score 0% -- the verdict is
    # the FINAL digit, not the first.
    from eval.judge import score_one

    reasoning = (
        "Let me check. The candidate says 30. There are 3 friends and the "
        "expected value is 0 remainder... actually the candidate is correct.\n1"
    )
    monkeypatch.setattr(
        "eval.judge.call_model",
        lambda **k: {"answer": reasoning, "tokens": 40, "error": None},
    )
    assert score_one("q", "intent", "30")["score"] == 1


def test_judge_reports_error_when_no_verdict_digit(monkeypatch):
    # Reasoning ran out of room before emitting a verdict -- surface it rather
    # than silently scoring 0 (which hides a broken judge as a failing answer).
    from eval.judge import score_one

    monkeypatch.setattr(
        "eval.judge.call_model",
        lambda **k: {"answer": "Let me think about whether this", "tokens": 5, "error": None},
    )
    verdict = score_one("q", "intent", "some answer")
    assert verdict["score"] == 0
    assert verdict["error"] is not None and "no verdict digit" in verdict["error"]


def test_eval_set_is_wellformed_and_covers_all_categories():
    from agent.config import CATEGORIES

    data = json.loads(
        (__import__("pathlib").Path("eval/eval_set.json")).read_text(encoding="utf-8")
    )
    tasks = data["tasks"]
    for t in tasks:
        assert {"task_id", "prompt", "category", "expected_intent"} <= t.keys()
        assert t["category"] in CATEGORIES, f"{t['task_id']} uses unknown category"
    covered = {t["category"] for t in tasks}
    assert covered == set(CATEGORIES), (
        f"eval set missing categories: {set(CATEGORIES) - covered}"
    )
    ids = [t["task_id"] for t in tasks]
    assert len(ids) == len(set(ids)), "duplicate task_id in eval set"
