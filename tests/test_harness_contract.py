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
        ("Summarize what this function does: def f(x): return x", "code_generation"),
        ("Summarise in one sentence: the fox jumped.", "summarisation"),
        ("Write a Python function that reverses a list.", "code_generation"),
        ("What's wrong with this function, it raises an error: def f(x): return x/0", "code_debugging"),
        ("What is the sentiment of this review: 'loved it'", "sentiment_classification"),
        ("Extract the named entities from: Ada met Charles.", "named_entity_recognition"),
        ("If Alice is taller than Bob, and Bob is taller than Carol, who is shortest?", "logical_reasoning"),
        ("Calculate 17 * 23 and subtract 40.", "math_reasoning"),
        ("", "factual_knowledge"),
        ("What is the capital of Australia?", "factual_knowledge"),
        # Regressions the eval-set routing check caught (was misrouting):
        ("What is 15% of 240?", "math_reasoning"),
        ("A train travels 60 km in 45 minutes. What is its average speed in km/h?", "math_reasoning"),
        ("Is this feedback positive or negative: 'the product broke after two days.'", "sentiment_classification"),
        ("This function should return the average but crashes on an empty list. Fix it: def avg(xs): return sum(xs)/len(xs)", "code_debugging"),
        ("List the named entities: Apple announced it in Cupertino.", "named_entity_recognition"),
        # Harder rewordings the expanded eval set caught (was misrouting to factual):
        ("If a shirt costs $40 and is discounted by 25%, what is the sale price?", "math_reasoning"),
        ("Classify the tone of this tweet: 'ten out of ten.'", "sentiment_classification"),
        ("Positive, negative, or neutral: 'worst service ever.'", "sentiment_classification"),
        ("Which people, organisations, and places are mentioned here: Satya met the mayor.", "named_entity_recognition"),
        ("Four runners finished. Dana beat Evan. Who came last?", "logical_reasoning"),
        ("A is north of B. C is south of B. Which is furthest north?", "logical_reasoning"),
        # Must NOT be stolen by the superlative logic pattern:
        ("In which year did the first human land on the Moon?", "factual_knowledge"),
    ],
)
def test_classify(prompt, expected):
    assert classify(prompt) == expected


# --- config: resolved per call, not frozen at import -----------------------

def test_allowed_models_is_read_at_call_time(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "tiny, mid, big")
    assert config_for("factual_knowledge").model == "tiny"
    monkeypatch.setenv("ALLOWED_MODELS", "other")
    assert config_for("factual_knowledge").model == "other"


def test_explicit_model_override_beats_index(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "tiny, mid")
    monkeypatch.setenv("ROUTER_CODE_GENERATION_MODEL", "pinned")
    assert config_for("code_generation").model == "pinned"


def test_model_index_clamps_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "tiny, mid")
    monkeypatch.setenv("ROUTER_CODE_GENERATION_MODEL_INDEX", "99")
    assert config_for("code_generation").model == "mid"


def test_unparseable_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ROUTER_CODE_GENERATION_MAX_TOKENS", "not-a-number")
    assert config_for("code_generation").max_tokens == 400


def test_max_tokens_override_applies(monkeypatch):
    monkeypatch.setenv("ROUTER_SENTIMENT_CLASSIFICATION_MAX_TOKENS", "3")
    assert config_for("sentiment_classification").max_tokens == 3


def test_unknown_category_falls_back_to_default():
    assert config_for("nonsense") == config_for("factual_knowledge")


def test_every_routed_category_has_a_config():
    from agent.routing import PRIORITY

    for name, _ in PRIORITY:
        assert name in CATEGORIES, f"router emits {name!r} with no config"


def test_every_config_category_is_reachable_by_routing():
    """The inverse check: a config nobody routes to never gets exercised."""
    from agent.routing import DEFAULT_CATEGORY as ROUTING_DEFAULT
    from agent.routing import PRIORITY

    routed = {name for name, _ in PRIORITY} | {ROUTING_DEFAULT}
    assert routed == set(CATEGORIES)
