#!/usr/bin/env python3
# token_report.py -- R3's token accounting.
#
# The real submission is scored on total tokens recorded by the judging proxy,
# after the accuracy gate is passed. This script sums whatever usage
# local_harness.py logged, so the team can track token cost per category and
# watch it trend down as R2 tunes prompts/routing (Phase 4: "Token squeeze").
#
# IMPORTANT: this only sums usage the agent module actually reports. Local
# model calls are free / not counted in the real scoring, and don't need to be
# logged here. Only Fireworks-routed calls matter for the real leaderboard --
# make sure whatever R2 builds attaches accurate usage numbers per call.
#
# Usage:
#     python token_report.py --usage usage_log.json --eval eval_set.json

import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description="R3 token accounting report.")
    parser.add_argument("--usage", default="usage_log.json", help="Path to usage_log.json from local_harness.py.")
    parser.add_argument("--eval", default="eval_set.json", help="Path to eval_set.json, used to group tokens by category.")
    parser.add_argument("--out", default="token_report.json", help="Where to write the report.")
    args = parser.parse_args()

    with open(args.usage, "r") as f:
        usage_log = json.load(f)
    with open(args.eval, "r") as f:
        eval_tasks = {t["task_id"]: t for t in json.load(f)}

    per_category = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
    total_prompt, total_completion, total_calls = 0, 0, 0

    for task_id, usage in usage_log.items():
        category = eval_tasks.get(task_id, {}).get("category", "unknown")
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        per_category[category]["prompt_tokens"] += pt
        per_category[category]["completion_tokens"] += ct
        per_category[category]["calls"] += 1
        total_prompt += pt
        total_completion += ct
        total_calls += 1

    report = {
        "total_calls": total_calls,
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "per_category": {
            cat: {**stats, "total_tokens": stats["prompt_tokens"] + stats["completion_tokens"]}
            for cat, stats in sorted(per_category.items())
        },
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("=== Token report ===")
    for cat, stats in report["per_category"].items():
        print(f"  {cat:28s}  {stats['calls']:2d} calls   {stats['total_tokens']:5d} tokens")
    print(f"  {'TOTAL':28s}  {total_calls:2d} calls   {report['total_tokens']:5d} tokens")
    print(f"\nNote: tasks with zero logged calls used a local model (free) or the agent")
    print(f"module didn't report usage -- {len(eval_tasks) - total_calls} of {len(eval_tasks)} tasks have no usage entry.")
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
