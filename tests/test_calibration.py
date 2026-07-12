"""Tests for the startup lean calibration: measure each live model's template
cost once, route general categories to the cheapest.

Calibration runs single-threaded at startup (no locks to test); what matters:
ranking by MEASURED prompt_tokens, probe mode never contaminating results via
failover, dead models excluded not misattributed, probe spend accounted, and
graceful fallback to capability ranking when probing is impossible.
"""

from unittest.mock import MagicMock

import pytest

import agent.config as cfg
import agent.fireworks_client as fc
from agent.config import calibrate_lean, calibration_tokens, config_for, pick_lean


def _probe_result(prompt_tokens: int, total: int | None = None, error: str | None = None):
    if error is not None:
        return {"answer": "", "tokens": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "truncated": False,
                "actual_model": "", "error": error}
    return {"answer": "x", "tokens": total if total is not None else prompt_tokens + 1,
            "prompt_tokens": prompt_tokens, "completion_tokens": 1,
            "truncated": False, "actual_model": "whoever", "error": None}


@pytest.fixture(autouse=True)
def _reset_calibration(monkeypatch):
    monkeypatch.setattr(cfg, "_lean_ranking", None)
    monkeypatch.setattr(cfg, "_calibration_tokens", 0)
    monkeypatch.setattr(cfg, "_UNAVAILABLE", set())
    monkeypatch.delenv("LOCAL_MODEL_PATH", raising=False)


def test_calibration_ranks_by_measured_prompt_tokens(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "verbose-model, lean-model")
    costs = {"verbose-model": 140, "lean-model": 45}

    def fake(prompt, system_prompt, model, max_tokens, **kw):
        assert kw.get("allow_failover") is False  # probes must not fail over
        return _probe_result(costs[model])

    monkeypatch.setattr(fc, "call_model", fake)
    calibrate_lean()
    assert pick_lean() == "lean-model"
    # General categories route to the measured-leanest model...
    assert config_for("summarisation").model == "lean-model"
    assert config_for("factual_knowledge").model == "lean-model"


def test_lean_can_select_a_code_specialist_for_general_categories(monkeypatch):
    # The whole point on the real list: the code model IS the leanest, and the
    # specialist exclusion in capability ranking must not block the lean pick.
    monkeypatch.setenv("ALLOWED_MODELS", "minimax-m3, kimi-k2p7-code")
    costs = {"minimax-m3": 137, "kimi-k2p7-code": 42}
    monkeypatch.setattr(fc, "call_model",
                        lambda p, s, model, m, **kw: _probe_result(costs[model]))
    calibrate_lean()
    assert config_for("sentiment_classification").model == "kimi-k2p7-code"
    assert config_for("summarisation").model == "kimi-k2p7-code"
    # Code categories were already on the specialist -- unchanged.
    assert config_for("code_generation").model == "kimi-k2p7-code"


def test_calibration_skipped_with_fewer_than_two_live_models(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "only-model")
    called = []
    monkeypatch.setattr(fc, "call_model", lambda *a, **k: called.append(1) or _probe_result(10))
    calibrate_lean()
    assert called == []            # no probe fired: nothing to choose between
    assert pick_lean() == ""       # tier resolution falls back to capability
    assert calibration_tokens() == 0


def test_calibration_skipped_in_local_mode(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "a-model, b-model")
    monkeypatch.setenv("LOCAL_MODEL_PATH", "/models/x.gguf")
    called = []
    monkeypatch.setattr(fc, "call_model", lambda *a, **k: called.append(1) or _probe_result(10))
    calibrate_lean()
    assert called == [] and pick_lean() == ""


def test_probe_failure_falls_back_to_capability_ranking(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "model-8b, model-70b")
    monkeypatch.setattr(fc, "call_model",
                        lambda *a, **k: _probe_result(0, error="boom: connection refused"))
    calibrate_lean()
    assert pick_lean() == ""  # nothing probeable
    # Routing still works via capability ranking (8b = small).
    assert config_for("sentiment_classification").model == "model-8b"


def test_dead_model_probe_is_excluded_not_misattributed(monkeypatch):
    # Mixed real-world list: gemmas 404 at probe time. They must be marked
    # unavailable and excluded; the ranking is built only from live models.
    monkeypatch.setenv(
        "ALLOWED_MODELS",
        "minimax-m3, kimi-k2p7-code, gemma-4-31b-it, gemma-4-26b-a4b-it",
    )
    costs = {"minimax-m3": 137, "kimi-k2p7-code": 42}

    def fake(prompt, system_prompt, model, max_tokens, **kw):
        if "gemma" in model:
            return _probe_result(0, error="Error code: 404 - Model not found, inaccessible, and/or not deployed")
        return _probe_result(costs[model])

    monkeypatch.setattr(fc, "call_model", fake)
    calibrate_lean()
    assert pick_lean() == "kimi-k2p7-code"           # lowest-overhead LIVE model
    assert "gemma-4-31b-it" in cfg._UNAVAILABLE      # dead ones blocklisted
    assert "gemma-4-26b-a4b-it" in cfg._UNAVAILABLE
    picks = {c: config_for(c).model for c in cfg.CATEGORIES}
    assert set(picks.values()) == {"kimi-k2p7-code"}  # everything on the lean pick


def test_probe_tokens_are_accumulated(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "a-model, b-model")
    monkeypatch.setattr(fc, "call_model",
                        lambda p, s, model, m, **kw: _probe_result(50, total=51))
    calibrate_lean()
    assert calibration_tokens() == 102  # two probes, real spend counted


def test_calibration_runs_once_per_process(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "a-model, b-model")
    calls = []
    monkeypatch.setattr(fc, "call_model",
                        lambda p, s, model, m, **kw: calls.append(model) or _probe_result(10))
    calibrate_lean()
    calibrate_lean()  # second call is a no-op: cached
    assert len(calls) == 2


# --- probe mode in the client itself ----------------------------------------

def test_probe_mode_does_not_fail_over_or_mark_unavailable(monkeypatch):
    monkeypatch.setattr(cfg, "_UNAVAILABLE", set())
    monkeypatch.setenv("ALLOWED_MODELS", "dead-model, live-model")
    monkeypatch.setattr(fc, "REASONING_EFFORT", "")
    seen = []

    def create(**kw):
        seen.append(kw["model"])
        exc = Exception("Error code: 404 - Model not found, inaccessible, and/or not deployed")
        exc.status_code = 404
        raise exc

    fake = MagicMock()
    fake.chat.completions.create = create
    monkeypatch.setattr(fc, "_get_client", lambda: fake)

    result = fc.call_model("p", "s", "dead-model", 1, allow_failover=False)
    assert result["error"] is not None
    assert seen == ["dead-model"]                    # no second model tried
    assert "dead-model" not in cfg._UNAVAILABLE      # bookkeeping left to caller


def test_success_reports_actual_model(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "m1")
    monkeypatch.setattr(fc, "REASONING_EFFORT", "")
    resp = MagicMock()
    resp.choices[0].message.content = "hi"
    resp.choices[0].finish_reason = "stop"
    resp.usage.total_tokens = 5
    resp.usage.prompt_tokens = 4
    resp.usage.completion_tokens = 1
    resp.model = "m1-echoed-by-server"
    fake = MagicMock()
    fake.chat.completions.create.return_value = resp
    monkeypatch.setattr(fc, "_get_client", lambda: fake)

    result = fc.call_model("p", "s", "m1", 5)
    assert result["actual_model"] == "m1-echoed-by-server"
