"""`python -m eval.score` -- R3's go/no-go command.

Runs the agent over eval/eval_set.json, judges every answer against its
expected intent, and prints:
  - per-category accuracy and token totals
  - the routing check (did each task land in its expected category?)
  - the two numbers that decide the competition: overall accuracy (the gate)
    and TOTAL tokens (the ranking metric)

Needs a live Fireworks key (real answers + real judging). With no key it
still runs, but every call fails and the numbers are all zero -- which is
itself a useful smoke test of the plumbing.

R3 owns the pass/fail call. This tool reports; it does not decide.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from agent.config import calibrate_lean, calibration_tokens
from agent.core import answer_task_detailed
from eval.judge import score_one

EVAL_SET = Path(__file__).parent / "eval_set.json"

# The judging VM's hard limits. Reported so the local-inference probe can see
# at a glance whether a candidate model fits the time budget on this box.
PER_REQUEST_LIMIT_S = 30.0
TOTAL_RUN_LIMIT_S = 10 * 60.0


def load_eval_set(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tasks"]


def run(tasks: list[dict], gate: float) -> int:
    # Same startup step the container performs: probe template costs once so
    # routing prefers the leanest model. Probe tokens are REAL judged spend --
    # counted into the reported total below, never hidden.
    try:
        calibrate_lean()
    except Exception as exc:
        print(f"lean calibration skipped: {exc!r}", file=sys.stderr)

    per_cat_correct: dict[str, int] = defaultdict(int)
    per_cat_total: dict[str, int] = defaultdict(int)
    per_cat_tokens: dict[str, int] = defaultdict(int)
    per_cat_prompt: dict[str, int] = defaultdict(int)
    per_cat_compl: dict[str, int] = defaultdict(int)
    per_cat_trunc: dict[str, int] = defaultdict(int)
    per_cat_latency: dict[str, float] = defaultdict(float)
    routing_misses: list[tuple[str, str, str]] = []
    total_tokens = 0
    total_correct = 0
    total_latency = 0.0
    slowest = (0.0, "")  # (seconds, task_id) -- the per-request budget check
    errors: list[tuple[str, str]] = []

    for t in tasks:
        expected_cat = t["category"]
        t0 = time.monotonic()
        detail = answer_task_detailed(t)
        dt = time.monotonic() - t0
        per_cat_latency[expected_cat] += dt
        total_latency += dt
        if dt > slowest[0]:
            slowest = (dt, t["task_id"])
        got_cat = detail["category"]
        if got_cat != expected_cat:
            routing_misses.append((t["task_id"], expected_cat, got_cat))

        verdict = score_one(t["prompt"], t["expected_intent"], detail["answer"])
        correct = verdict["score"] == 1

        per_cat_total[expected_cat] += 1
        per_cat_correct[expected_cat] += int(correct)
        per_cat_tokens[expected_cat] += detail["tokens"]
        per_cat_prompt[expected_cat] += detail["prompt_tokens"]
        per_cat_compl[expected_cat] += detail["completion_tokens"]
        per_cat_trunc[expected_cat] += int(detail["truncated"])
        total_tokens += detail["tokens"]
        total_correct += int(correct)

        if detail["truncated"]:
            errors.append((t["task_id"], f"truncated at max_tokens ({expected_cat})"))
        if detail["error"]:
            errors.append((t["task_id"], f"agent: {detail['error']}"))
        if verdict["error"]:
            errors.append((t["task_id"], f"judge: {verdict['error']}"))

    _print_report(
        per_cat_correct, per_cat_total, per_cat_tokens,
        per_cat_prompt, per_cat_compl, per_cat_trunc, per_cat_latency,
        routing_misses, errors, total_correct, len(tasks), total_tokens, gate,
        total_latency, slowest,
    )

    overall_acc = total_correct / len(tasks) if tasks else 0.0
    # Exit non-zero if below the proxy gate, so this can gate CI / a pre-push hook.
    return 0 if overall_acc >= gate else 1


def _print_report(correct, total, tokens, prompt_toks, compl_toks, trunc, latency,
                  routing_misses, errors, total_correct, n, total_tokens, gate,
                  total_latency, slowest) -> None:
    print("\n=== per-category ===")
    print(f"{'category':<28} {'acc':>8} {'n':>4} {'tokens':>9} {'prompt':>8} "
          f"{'compl':>8} {'trunc':>6} {'s/task':>7}")
    print("-" * 86)
    for cat in sorted(total):
        acc = correct[cat] / total[cat] if total[cat] else 0.0
        avg_s = latency[cat] / total[cat] if total[cat] else 0.0
        print(f"{cat:<28} {acc:>7.0%} {total[cat]:>4} {tokens[cat]:>9,}"
              f" {prompt_toks[cat]:>8,} {compl_toks[cat]:>8,} {trunc[cat]:>6} {avg_s:>7.1f}")

    print("\n=== routing ===")
    if routing_misses:
        print(f"{len(routing_misses)} task(s) routed to an unexpected category:")
        for tid, exp, got in routing_misses:
            print(f"  {tid}: expected {exp!r}, got {got!r}")
    else:
        print("all tasks routed to their expected category")

    if errors:
        print("\n=== errors (agent/judge failures) ===")
        for tid, msg in errors:
            print(f"  {tid}: {msg}")

    overall_acc = total_correct / n if n else 0.0
    total_prompt = sum(prompt_toks.values())
    total_compl = sum(compl_toks.values())
    total_trunc = sum(trunc.values())
    probe = calibration_tokens()
    print("\n=== summary ===")
    print(f"overall accuracy : {overall_acc:.0%}  ({total_correct}/{n})   gate proxy >= {gate:.0%}")
    print(f"TOTAL tokens     : {total_tokens + probe:,}   (ranking metric -- drive this down)")
    print(f"  tasks          : {total_tokens:,}  (prompt {total_prompt:,} + completion {total_compl:,})")
    print(f"  calibration    : {probe:,}  (startup template probes -- real judged spend)")
    if total_trunc:
        print(f"truncated        : {total_trunc} answer(s) hit max_tokens -- billed tokens "
              "buying likely-wrong answers; raise those caps")

    # Timing -- the binding constraint for the local-inference probe. Harmless
    # on the Fireworks path (calls are fast); decisive on CPU-only local models.
    print(f"\nwall time        : {total_latency:.1f}s total   (limit {TOTAL_RUN_LIMIT_S:.0f}s"
          f" for the whole run -- concurrency can parallelise this)")
    print(f"slowest task     : {slowest[0]:.1f}s ({slowest[1]})   "
          f"(limit {PER_REQUEST_LIMIT_S:.0f}s PER request -- a hard per-call ceiling)")
    if slowest[0] > PER_REQUEST_LIMIT_S:
        print("  !! a task exceeded the 30s per-request limit -- this model is too slow "
              "for that category as configured")
    if total_latency > TOTAL_RUN_LIMIT_S:
        print("  !! total exceeds the 10-min run budget at concurrency=1 -- needs "
              "parallelism or a faster/smaller model")

    verdict = "PASS (proxy)" if overall_acc >= gate else "BELOW GATE"
    print(f"verdict          : {verdict}")
    if errors:
        print("NOTE: failures above (likely a missing/invalid Fireworks key) make "
              "these numbers a plumbing smoke test, not a real score.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the agent over the eval set.")
    ap.add_argument("--eval-set", type=Path, default=EVAL_SET)
    ap.add_argument("--gate", type=float, default=0.9,
                    help="proxy accuracy threshold for a non-zero exit (default 0.9: "
                         "the target gate is 85%% and this proxy needs margin)")
    args = ap.parse_args()

    tasks = load_eval_set(args.eval_set)
    return run(tasks, args.gate)


if __name__ == "__main__":
    sys.exit(main())
