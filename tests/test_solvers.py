"""Tests for the deterministic 0-token solvers.

Two things matter here and they pull in opposite directions:
  1. Coverage  -- the solvers must actually answer the eval-set math/logic
     tasks (and paraphrases of them), or they buy no tokens.
  2. Precision -- they must return None, never a wrong answer, on anything
     ambiguous or adversarial. A confident-but-wrong 0-token answer costs the
     accuracy gate, which is worse than paying the model.

So every solvable case asserts the exact answer, and a whole block asserts
None on inputs a solver must refuse.
"""

from unittest.mock import MagicMock

import pytest

import agent.fireworks_client as fc
from agent.core import answer_task_detailed
from agent.solvers import solve_logic, solve_math

# --- math: the eval-set cases ----------------------------------------------

@pytest.mark.parametrize("prompt,expected", [
    ("What is 15% of 240?", "36"),
    ("A train travels 60 km in 45 minutes. What is its average speed in km/h?", "80 km/h"),
    ("If a shirt costs $40 and is discounted by 25%, what is the sale price?", "$30"),
    ("Three friends split a bill of $87 evenly. How much does each pay, rounded to the nearest cent?", "$29.00"),
    ("What is the value of 2 to the power of 10?", "1024"),
])
def test_solve_math_eval_cases(prompt, expected):
    assert solve_math(prompt) == expected


# --- math: paraphrases / fuzz (generalisation, not memorisation) -----------

@pytest.mark.parametrize("prompt,expected", [
    ("Calculate 20% of 50.", "10"),
    ("What is 8 to the power of 3?", "512"),
    ("A car covers 150 miles in 2 hours. What is its average speed?", "75 mph"),
    ("A jacket costs $80 and is discounted by 10%. Sale price?", "$72"),
    ("Four people split a bill of $100 evenly, rounded to the nearest cent.", "$25.00"),
    ("Compute 12 * 12.", "144"),
    ("What is 7 + 6 * 2?", "19"),   # operator precedence
])
def test_solve_math_paraphrases(prompt, expected):
    assert solve_math(prompt) == expected


# --- math: must defer (return None) ----------------------------------------

@pytest.mark.parametrize("prompt", [
    "Calculate 17 * 23 and then subtract 40.",     # multi-step phrasing
    "What is the derivative of x^2?",               # not arithmetic
    "How many prime numbers are below 100?",        # needs reasoning
    "What is 2 to the power of 9999?",              # exponent cap -> refuse
    "__import__('os').system('rm -rf /')",         # not an arithmetic expr
    "Estimate the population of France.",           # no computation
])
def test_solve_math_defers(prompt):
    assert solve_math(prompt) is None


# --- math: OFFICIAL sample tasks are multi-step -- must defer, never misfire.
# T02b once misfired: the bare-expression matcher grabbed "3/4" and answered
# "0.75" (correct answer: $4.50). A confidently-wrong 0-token answer is the
# worst possible outcome for the accuracy gate.

@pytest.mark.parametrize("prompt", [
    # T02 (warehouse, multi-step inventory)
    "A warehouse starts with 2,400 units. In Q1 it sells 37% of stock. "
    "In Q2 it restocks 800 units. In Q3 it sells 640 units. "
    "How many units remain at the end of Q3?",
    # T02b (recipe scaling THEN pricing -- the "3/4" -> "0.75" misfire)
    "A recipe requires 3/4 cup of sugar for 12 cookies. How much sugar is "
    "needed for 30 cookies? If sugar costs $2.40 per cup, what is the total "
    "cost of sugar for 30 cookies?",
    # percent-of embedded in longer prose: one step of a bigger problem
    "The stock index rose 15% of 240 points in the morning and then fell back.",
    # two questions = two computations
    "What is 20% of 50? And what is 30% of 90?",
    # mixed-unit duration: grabbing just "2 hours" would answer 90, not 80
    "A car travels 180 km in 2 hours 15 minutes. What is its average speed?",
    # markup THEN discount: grabbing one percent answered $36 (correct: $63)
    "A store buys a jacket for $60 and marks the price up by 40%. During a "
    "sale, the marked price is discounted by 25%. What is the sale price?",
])
def test_solve_math_defers_on_official_multistep(prompt):
    assert solve_math(prompt) is None


# --- logic: the eval-set cases ---------------------------------------------

@pytest.mark.parametrize("prompt,expected", [
    ("If Alice is taller than Bob, and Bob is taller than Carol, who is the shortest?", "Carol"),
    ("Tom is older than Sara. Sara is older than Rob. Rob is older than Kim. Who is the second oldest?", "Sara"),
    ("Four runners finished a race. Dana beat Evan. Evan beat Faye. Faye beat Gus. Who came last?", "Gus"),
    ("A is north of B. C is south of B. Which is furthest north?", "A"),
])
def test_solve_logic_eval_cases(prompt, expected):
    assert solve_logic(prompt) == expected


# --- logic: paraphrases ----------------------------------------------------

@pytest.mark.parametrize("prompt,expected", [
    ("Mia is faster than Noah. Noah is faster than Owen. Who is the fastest?", "Mia"),
    ("Rick is younger than Sam. Sam is younger than Tia. Who is the oldest?", "Tia"),
    ("P beat Q. Q beat R. Who won?", "P"),
    ("X is above Y. Y is above Z. Which is lowest?", "Z"),
])
def test_solve_logic_paraphrases(prompt, expected):
    assert solve_logic(prompt) == expected


# --- logic: must defer ------------------------------------------------------

@pytest.mark.parametrize("prompt", [
    "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly?",
    "Alice is taller than Bob. Carol is taller than Dave. Who is the tallest?",  # two chains: ambiguous
    "Alice is taller than Bob. Bob is taller than Alice. Who is tallest?",       # contradiction
    "Who is the tallest?",                                                        # no premises
    "Alice is taller than Bob. Who is faster?",                                   # no superlative match target... 'faster'? none
    "If it rains, the ground is wet. It is raining. Is the ground wet?",          # not an ordering
    # Negation would add a WRONG edge if the comparative regex fired:
    "Alice is not taller than Bob. Bob is taller than Carol. Who is the tallest?",
    "Alice isn't taller than Bob. Who is the shortest?",
    # Equality/ties break a strict ordering:
    "Alice and Bob are equally tall. Bob is taller than Carol. Who is the tallest?",
    "Everyone except Dana beat Evan. Who came first?",
])
def test_solve_logic_defers(prompt):
    assert solve_logic(prompt) is None


# --- integration through core: solver hit spends no model call -------------

@pytest.fixture
def spy_client(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "test-model")
    fake = MagicMock()
    monkeypatch.setattr(fc, "_get_client", lambda: fake)
    return fake


def test_solvable_math_never_calls_the_model(spy_client):
    detail = answer_task_detailed({"task_id": "m", "prompt": "What is 15% of 240?"})
    assert detail["answer"] == "36"
    assert detail["tokens"] == 0
    assert detail["category"] == "math_reasoning"
    spy_client.chat.completions.create.assert_not_called()


def test_solvable_logic_never_calls_the_model(spy_client):
    detail = answer_task_detailed(
        {"task_id": "l", "prompt": "If Alice is taller than Bob, and Bob is taller than Carol, who is the shortest?"}
    )
    assert detail["answer"] == "Carol"
    assert detail["tokens"] == 0
    spy_client.chat.completions.create.assert_not_called()


def test_syllogism_falls_back_to_the_model(spy_client):
    resp = MagicMock()
    resp.choices[0].message.content = "no"
    resp.choices[0].finish_reason = "stop"
    resp.usage.total_tokens = 12
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 2
    spy_client.chat.completions.create.return_value = resp

    detail = answer_task_detailed(
        {"task_id": "s", "prompt": "All roses are flowers. Some flowers fade quickly. Can we conclude that some roses fade quickly? Answer yes or no."}
    )
    assert detail["category"] == "logical_reasoning"
    assert detail["tokens"] == 12            # the model was used
    assert detail["answer"] == "no"
    spy_client.chat.completions.create.assert_called_once()
