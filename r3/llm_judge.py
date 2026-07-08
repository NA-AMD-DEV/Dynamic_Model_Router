#!/usr/bin/env python3
"""
llm_judge.py — R3's accuracy judge.

Phase 1 (today): a heuristic backend that works completely offline — no API
key needed — so you can start measuring accuracy before R2's Fireworks
client even exists. Good enough to sanity-check the harness and eval set.

Phase 2 (once R2's Fireworks wrapper exists): flip --backend to "llm" to use
a real LLM-as-judge call, matching what the actual accuracy gate does
(score each answer 0-1 against expected intent). This is the team's Phase 2
task ("Wire the LLM-judge... mirrors the real gate").

Usage:
    # Phase 1 — offline heuristic judge
    python llm_judge.py --eval eval_set.json --results results.json

    # Phase 2 — real LLM judge (needs FIREWORKS_API_KEY / FIREWORKS_BASE_URL
    # / a JUDGE_MODEL env var set, separately from your submission's
    # ALLOWED_MODELS — this is *your* dev-time tool, not the submitted agent)
    python llm_judge.py --eval eval_set.json --results results.json --backend llm
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def extract_code(answer: str) -> str:
    """Pull code out of a markdown fenced block if present, else assume the
    whole answer is code (agents sometimes skip the fences)."""
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", answer, re.DOTALL)
    if fenced:
        return "\n".join(fenced)
    return answer


def run_code_tests(answer: str, test_cases: list, timeout: int = 5) -> float:
    """Actually executes the agent's code against real test cases in a
    subprocess (cross-platform, works the same on Windows/Mac/Linux), rather
    than guessing correctness from keyword overlap. Returns 1.0 only if
    EVERY test case passes -- partial credit isn't meaningful here since a
    half-working function still fails the real accuracy gate.
    """
    code = extract_code(answer)
    tests_json_str = json.dumps(json.dumps(test_cases))  # double-encode: safe to embed as a Python string literal
    harness = f"""
{code}

import json
_tests = json.loads({tests_json_str})
_out = []
for t in _tests:
    try:
        actual = eval(t["call"])
        passed = actual == t["expected"]
    except Exception as e:
        actual = f"ERROR: {{e}}"
        passed = False
    _out.append({{"call": t["call"], "passed": passed}})
print(json.dumps(_out))
"""
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(harness)
            path = f.name
        result = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return 0.0  # code didn't even run -- syntax error, crash, etc.
        outcomes = json.loads(result.stdout.strip().splitlines()[-1])
        return 1.0 if all(o["passed"] for o in outcomes) else 0.0
    except Exception:
        return 0.0
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def score_ner(task, answer: str) -> float:
    """Checks each required entity individually instead of one blended
    keyword-overlap score, so a partial extraction is scored honestly rather
    than hidden behind a single threshold."""
    entities = task.get("expected_entities", [])
    if not entities:
        return heuristic_score_generic(task, answer)
    norm_answer = normalize(answer)
    found = sum(1 for e in entities if normalize(e) in norm_answer)
    return 1.0 if found / len(entities) >= 0.75 else 0.0


def score_summary(task, answer: str) -> float:
    """Checks the stated format constraint (sentence/word count) actually
    holds, then falls back to keyword overlap for content coverage. A
    summary that ignores 'exactly one sentence' fails regardless of content
    quality, matching how a real judge would treat an instruction-following
    failure.
    """
    constraints = task.get("constraints", {})
    text = answer.strip()

    if "max_words" in constraints:
        word_count = len(text.split())
        if word_count > constraints["max_words"]:
            return 0.0

    if "sentence_count" in constraints:
        # rough sentence split -- good enough for a Day-1 offline heuristic
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sentences) != constraints["sentence_count"]:
            return 0.0

    return heuristic_score_generic(task, answer)


def heuristic_score_generic(task, answer: str) -> float:
    """The original crude fallback: exact-match for tasks with a single
    right answer, keyword overlap for open-ended tasks (factual knowledge,
    sentiment justification, summarization content). Still rough -- this is
    what --backend llm is for -- but no longer the only check for code/NER/
    summary format, which now have real deterministic checks above.
    """
    norm_answer = normalize(answer)

    if task.get("expected_answer"):
        norm_expected = normalize(str(task["expected_answer"]))
        return 1.0 if norm_expected and norm_expected in norm_answer else 0.0

    intent_words = set(normalize(task.get("expected_intent", "")).split())
    answer_words = set(norm_answer.split())
    if not intent_words:
        return 0.0
    overlap = len(intent_words & answer_words) / len(intent_words)
    return 1.0 if overlap >= 0.35 else 0.0


def heuristic_score(task, answer: str) -> float:
    """Dispatches to a category-specific scorer where a real deterministic
    check exists, falls back to generic keyword/exact-match otherwise."""
    category = task.get("category")
    if category in ("code_debugging", "code_generation") and task.get("test_cases"):
        return run_code_tests(answer, task["test_cases"])
    if category == "named_entity_recognition":
        return score_ner(task, answer)
    if category == "text_summarisation":
        return score_summary(task, answer)
    return heuristic_score_generic(task, answer)


def llm_score(task, answer: str, client, judge_model: str) -> float:
    """Real LLM-as-judge call. Requires the `openai` package and Fireworks
    (or any OpenAI-compatible) credentials in the environment.
    """
    prompt = f"""You are grading an AI agent's answer for a hackathon accuracy gate.

Task prompt given to the agent:
{task['prompt']}

What a correct answer must contain / satisfy:
{task.get('expected_intent', '(no rubric provided)')}

The agent's answer:
{answer}

Score this answer as either 1 (meets the expected intent) or 0 (does not).
Reply with ONLY the single digit 1 or 0, nothing else."""

    resp = client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    return 1.0 if raw.startswith("1") else 0.0


def make_llm_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: `openai` package not installed. Run: pip install openai --break-system-packages", file=sys.stderr)
        sys.exit(1)

    base_url = os.environ.get("FIREWORKS_BASE_URL")
    api_key = os.environ.get("FIREWORKS_API_KEY")
    judge_model = os.environ.get("JUDGE_MODEL")

    if not base_url or not api_key:
        print("ERROR: FIREWORKS_BASE_URL / FIREWORKS_API_KEY not set in environment.", file=sys.stderr)
        print("These will be injected by the real harness on submission day, but for your own", file=sys.stderr)
        print("local judging runs before then, use a dev key from your team + set JUDGE_MODEL.", file=sys.stderr)
        sys.exit(1)
    if not judge_model:
        print("ERROR: set JUDGE_MODEL to a model ID to use as the judge (any allowed model works fine as a judge).", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, judge_model


def main():
    parser = argparse.ArgumentParser(description="R3 accuracy judge.")
    parser.add_argument("--eval", default="eval_set.json", help="Path to eval_set.json (has expected_intent/expected_answer).")
    parser.add_argument("--results", default="results.json", help="Path to results.json produced by local_harness.py.")
    parser.add_argument("--backend", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--out", default="score_report.json", help="Where to write the per-category report.")
    args = parser.parse_args()

    with open(args.eval, "r") as f:
        eval_tasks = {t["task_id"]: t for t in json.load(f)}
    with open(args.results, "r") as f:
        results = {r["task_id"]: r["answer"] for r in json.load(f)}

    client, judge_model = (None, None)
    if args.backend == "llm":
        client, judge_model = make_llm_client()

    per_category = defaultdict(lambda: {"n": 0, "passed": 0, "failures": []})

    for task_id, task in eval_tasks.items():
        answer = results.get(task_id)
        category = task["category"]
        per_category[category]["n"] += 1

        if answer is None:
            per_category[category]["failures"].append({"task_id": task_id, "reason": "missing from results.json"})
            continue

        if args.backend == "heuristic":
            score = heuristic_score(task, answer)
        else:
            score = llm_score(task, answer, client, judge_model)

        if score >= 1.0:
            per_category[category]["passed"] += 1
        else:
            per_category[category]["failures"].append({"task_id": task_id, "answer": answer[:200]})

    report = {}
    total_n, total_passed = 0, 0
    for category, stats in sorted(per_category.items()):
        acc = stats["passed"] / stats["n"] if stats["n"] else 0.0
        report[category] = {
            "n": stats["n"],
            "passed": stats["passed"],
            "accuracy": round(acc, 3),
            "failures": stats["failures"],
        }
        total_n += stats["n"]
        total_passed += stats["passed"]

    overall_acc = round(total_passed / total_n, 3) if total_n else 0.0

    with open(args.out, "w") as f:
        json.dump({"overall_accuracy": overall_acc, "per_category": report}, f, indent=2)

    print(f"\n=== Accuracy report ({args.backend} backend) ===")
    for category, stats in report.items():
        flag = "  <-- WEAK" if stats["accuracy"] < 0.7 else ""
        print(f"  {category:28s}  {stats['passed']:2d}/{stats['n']:2d}  ({stats['accuracy']*100:5.1f}%){flag}")
    print(f"  {'OVERALL':28s}  {total_passed:2d}/{total_n:2d}  ({overall_acc*100:5.1f}%)")
    print(f"\nFull report written to {args.out}")

    if args.backend == "heuristic":
        print("\nNOTE: this is the offline heuristic backend — a rough Day-1 stand-in.")
        print("Switch to --backend llm once R2's Fireworks client exists, to mirror the real gate.")


if __name__ == "__main__":
    main()
