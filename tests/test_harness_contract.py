"""The invariants that decide whether we score at all.

Run: python -m pytest tests/ -q     (pytest is a dev dep, not in the image)

These assert the contract, not the answers. Accuracy is R3's eval set; this is
the container guaranteeing that whatever R2 produces reaches /output intact.
"""

import json
import time

import pytest

import agent.main as m
from agent.config import CATEGORIES, config_for
from agent.routing import classify

TASKS = [{"task_id": f"t{i}", "prompt": f"prompt {i}"} for i in range(1, 11)]


def _run(tmp_path, monkeypatch, answer_fn, concurrency=1, budget=60.0):
    out = tmp_path / "results.json"
    monkeypatch.setattr(m, "answer_task", answer_fn)
    monkeypatch.setattr(m, "OUTPUT_PATH", out)
    monkeypatch.setattr(m, "CONCURRENCY", concurrency)
    monkeypatch.setattr(m, "RUN_BUDGET_S", budget)
    m.run(list(TASKS))
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("concurrency", [1, 4])
def test_every_task_id_appears_once_in_input_order(tmp_path, monkeypatch, concurrency):
    res = _run(tmp_path, monkeypatch, lambda t: "ok:" + t["task_id"], concurrency)
    assert [r["task_id"] for r in res] == [t["task_id"] for t in TASKS]


@pytest.mark.parametrize("concurrency", [1, 4])
def test_a_raising_task_still_yields_a_string_answer(tmp_path, monkeypatch, concurrency):
    def flaky(task):
        if task["task_id"] in {"t3", "t7"}:
            raise RuntimeError("simulated API failure")
        return "ok"

    res = _run(tmp_path, monkeypatch, flaky, concurrency)
    assert len(res) == len(TASKS)
    assert all(isinstance(r["answer"], str) for r in res)
    failed = {r["task_id"] for r in res if r["answer"] == ""}
    assert failed == {"t3", "t7"}


@pytest.mark.parametrize("concurrency", [1, 4])
def test_non_string_answers_are_coerced(tmp_path, monkeypatch, concurrency):
    res = _run(tmp_path, monkeypatch, lambda t: 42, concurrency)
    assert all(isinstance(r["answer"], str) for r in res)


@pytest.mark.parametrize("concurrency", [1, 4])
def test_exhausted_budget_still_writes_every_task_id(tmp_path, monkeypatch, concurrency):
    res = _run(tmp_path, monkeypatch, lambda t: "ok", concurrency, budget=-1.0)
    assert [r["task_id"] for r in res] == [t["task_id"] for t in TASKS]
    assert all(r["answer"] == "" for r in res)


def test_budget_stops_dispatch_but_drains_in_flight(tmp_path, monkeypatch):
    def slow(task):
        time.sleep(0.3)
        return "ok"

    # Two workers, budget dies during the first batch: t1/t2 finish, rest don't start.
    res = _run(tmp_path, monkeypatch, slow, concurrency=2, budget=0.25)
    answered = [r["task_id"] for r in res if r["answer"]]
    assert answered == ["t1", "t2"]
    assert len(res) == len(TASKS)


def test_missing_task_id_gets_a_synthetic_key(tmp_path, monkeypatch):
    out = tmp_path / "results.json"
    monkeypatch.setattr(m, "answer_task", lambda t: "ok")
    monkeypatch.setattr(m, "OUTPUT_PATH", out)
    monkeypatch.setattr(m, "CONCURRENCY", 1)
    m.run([{"prompt": "no id here"}])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert len(res) == 1 and res[0]["task_id"] == "__missing_0"


def test_unreadable_input_still_leaves_valid_json(tmp_path, monkeypatch):
    out = tmp_path / "results.json"
    monkeypatch.setattr(m, "INPUT_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(m, "OUTPUT_PATH", out)
    assert m.main() == 1
    assert json.loads(out.read_text(encoding="utf-8")) == []


def test_malformed_input_still_leaves_valid_json(tmp_path, monkeypatch):
    bad = tmp_path / "tasks.json"
    bad.write_text('{"task_id": "t1"}', encoding="utf-8")
    out = tmp_path / "results.json"
    monkeypatch.setattr(m, "INPUT_PATH", bad)
    monkeypatch.setattr(m, "OUTPUT_PATH", out)
    assert m.main() == 1
    assert json.loads(out.read_text(encoding="utf-8")) == []


# --- routing: priority order is load-bearing -------------------------------

@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Summarize what this function does: def f(x): return x", "code"),
        ("Summarise in one sentence: the fox jumped.", "summarization"),
        ("Write a Python function that reverses a list.", "code"),
        ("What is the sentiment of this review: 'loved it'", "sentiment"),
        ("Extract the named entities from: Ada met Charles.", "ner"),
        ("Translate into Spanish: where is the station?", "translation"),
        ("Calculate 17 * 23 and subtract 40.", "math"),
        ("Classify this email as spam or not spam: 'free cruise'", "classification"),
        ("", "general"),
        ("Ponder the nature of a Tuesday afternoon.", "general"),
    ],
)
def test_classify(prompt, expected):
    assert classify(prompt) == expected


# --- config: resolved per call, not frozen at import -----------------------

def test_allowed_models_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "tiny, mid, big")
    assert config_for("sentiment").model == "tiny"
    monkeypatch.setenv("ALLOWED_MODELS", "other")
    assert config_for("sentiment").model == "other"


def test_explicit_model_override_beats_index(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "tiny, mid")
    monkeypatch.setenv("ROUTER_CODE_MODEL", "pinned")
    assert config_for("code").model == "pinned"


def test_model_index_clamps_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "tiny, mid")
    monkeypatch.setenv("ROUTER_CODE_MODEL_INDEX", "99")
    assert config_for("code").model == "mid"


def test_unparseable_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ROUTER_CODE_MAX_TOKENS", "not-a-number")
    assert config_for("code").max_tokens == 512


def test_max_tokens_override_applies(monkeypatch):
    monkeypatch.setenv("ROUTER_SENTIMENT_MAX_TOKENS", "3")
    assert config_for("sentiment").max_tokens == 3


def test_unknown_category_falls_back_to_general():
    assert config_for("nonsense") == config_for("general")


def test_every_routed_category_has_a_config():
    from agent.routing import PRIORITY

    for name, _ in PRIORITY:
        assert name in CATEGORIES, f"router emits {name!r} with no config"
