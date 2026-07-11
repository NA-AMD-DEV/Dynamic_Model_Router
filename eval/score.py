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
from collections import defaultdict
from pathlib import Path

from agent.core import answer_task_detailed
from eval.judge import score_one

EVAL_SET = Path(__file__).parent / "eval_set.json"


def load_eval_set(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tasks"]


def run(tasks: list[dict], gate: float) -> int:
    per_cat_correct: dict[str, int] = defaultdict(int)
    per_cat_total: dict[str, int] = defaultdict(int)
    per_cat_tokens: dict[str, int] = defaultdict(int)
    routing_misses: list[tuple[str, str, str]] = []
    total_tokens = 0
    total_correct = 0
    errors: list[tuple[str, str]] = []

    for t in tasks:
        expected_cat = t["category"]
        detail = answer_task_detailed(t)
        got_cat = detail["category"]
        if got_cat != expected_cat:
            routing_misses.append((t["task_id"], expected_cat, got_cat))

        verdict = score_one(t["prompt"], t["expected_intent"], detail["answer"])
        correct = verdict["score"] == 1

        per_cat_total[expected_cat] += 1
        per_cat_correct[expected_cat] += int(correct)
        per_cat_tokens[expected_cat] += detail["tokens"]
        total_tokens += detail["tokens"]
        total_correct += int(correct)

        if detail["error"]:
            errors.append((t["task_id"], f"agent: {detail['error']}"))
        if verdict["error"]:
            errors.append((t["task_id"], f"judge: {verdict['error']}"))

    _print_report(
        per_cat_correct, per_cat_total, per_cat_tokens,
        routing_misses, errors, total_correct, len(tasks), total_tokens, gate,
    )

    overall_acc = total_correct / len(tasks) if tasks else 0.0
    # Exit non-zero if below the proxy gate, so this can gate CI / a pre-push hook.
    return 0 if overall_acc >= gate else 1


def _print_report(correct, total, tokens, routing_misses, errors,
                  total_correct, n, total_tokens, gate) -> None:
    print("\n=== per-category ===")
    print(f"{'category':<28} {'acc':>8} {'n':>4} {'tokens':>9}")
    print("-" * 52)
    for cat in sorted(total):
        acc = correct[cat] / total[cat] if total[cat] else 0.0
        print(f"{cat:<28} {acc:>7.0%} {total[cat]:>4} {tokens[cat]:>9,}")

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
    print("\n=== summary ===")
    print(f"overall accuracy : {overall_acc:.0%}  ({total_correct}/{n})   gate proxy >= {gate:.0%}")
    print(f"TOTAL tokens     : {total_tokens:,}   (ranking metric -- drive this down)")
    verdict = "PASS (proxy)" if overall_acc >= gate else "BELOW GATE"
    print(f"verdict          : {verdict}")
    if errors:
        print("NOTE: failures above (likely a missing/invalid Fireworks key) make "
              "these numbers a plumbing smoke test, not a real score.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Score the agent over the eval set.")
    ap.add_argument("--eval-set", type=Path, default=EVAL_SET)
    ap.add_argument("--gate", type=float, default=0.8,
                    help="proxy accuracy threshold for a non-zero exit (default 0.8)")
    args = ap.parse_args()

    tasks = load_eval_set(args.eval_set)
    return run(tasks, args.gate)


if __name__ == "__main__":
    sys.exit(main())
