"""Tests for the R3 eval slice: the token-accounting seam, the judge, and the
scorer. These use a mocked Fireworks client so they need no key and no network.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import agent.fireworks_client as fc
from agent.core import answer_task, answer_task_detailed


def _mock_completion(content: str, tokens: int):
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.total_tokens = tokens
    return resp


@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    fake = MagicMock()
    monkeypatch.setattr(fc, "_get_client", lambda: fake)
    return fake


def test_detailed_seam_exposes_tokens_and_category(mock_client):
    mock_client.chat.completions.create.return_value = _mock_completion("  hi  ", 37)
    detail = answer_task_detailed({"task_id": "t1", "prompt": "What is 15% of 240?"})
    assert detail["answer"] == "hi"          # stripped
    assert detail["tokens"] == 37            # the ranking metric, not dropped
    assert detail["category"] == "math_reasoning"
    assert detail["error"] is None


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
