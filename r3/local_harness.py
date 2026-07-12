#!/usr/bin/env python3
"""
local_harness.py — R3's local mirror of the real judging harness.

What the real harness does (per the participant guide):
  1. Reads tasks from /input/tasks.json
  2. Calls the agent's answer_task() for each task
  3. Writes /output/results.json before exiting
  4. A failed task must still emit a best-effort answer — never drop a task_id
  5. Whole run must finish inside a 10-minute wall-clock budget

This script reproduces that loop locally so R3 can test accuracy + timing
before anyone pushes a real Docker image.

Usage:
    python local_harness.py \
        --tasks eval_set.json \
        --out results.json \
        --agent-module stub_agent

Once R1/R2 ship a real module (e.g. core.agent) that exposes answer_task,
swap --agent-module stub_agent -> --agent-module core.agent — nothing else
in this script should need to change. That's the point of freezing the
answer_task(task) -> str contract early.
"""

import argparse
import importlib
import json
import sys
import time

MAX_RUNTIME_SECONDS = 10 * 60  # mirrors the real 10-minute cap


def load_tasks(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    # eval_set.json has extra fields (category, expected_intent); the real
    # tasks.json only has task_id + prompt. Strip down to the real contract
    # so this harness behaves exactly like the actual one.
    tasks = [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in data]
    return tasks


def run(tasks, agent_module_name: str):
    agent = importlib.import_module(agent_module_name)
    if not hasattr(agent, "answer_task"):
        print(f"ERROR: module '{agent_module_name}' has no answer_task(task) -> str", file=sys.stderr)
        sys.exit(1)

    results = []
    usage_log = {}
    start = time.time()

    for task in tasks:
        elapsed = time.time() - start
        if elapsed > MAX_RUNTIME_SECONDS:
            print(f"WARNING: runtime guard hit at {elapsed:.1f}s — flushing remaining tasks with a fallback answer.", file=sys.stderr)
            results.append({"task_id": task["task_id"], "answer": ""})
            continue

        try:
            answer = agent.answer_task(task)
            if not isinstance(answer, str):
                answer = str(answer)
        except Exception as e:
            # A failed task must still emit something — never drop a task_id.
            print(f"WARNING: task {task['task_id']} raised {e!r} — emitting fallback answer.", file=sys.stderr)
            answer = ""

        results.append({"task_id": task["task_id"], "answer": answer})

        # Optional: pick up per-call token usage if the agent module exposes it,
        # so token_report.py has real numbers instead of estimates.
        usage = getattr(agent.answer_task, "last_usage", None)
        if usage:
            usage_log[task["task_id"]] = usage

    total_time = time.time() - start
    return results, usage_log, total_time


def main():
    parser = argparse.ArgumentParser(description="R3 local harness — mirrors the real /input -> /output contract.")
    parser.add_argument("--tasks", default="eval_set.json", help="Path to a tasks file (eval_set.json or a real tasks.json).")
    parser.add_argument("--out", default="results.json", help="Where to write results.json.")
    parser.add_argument("--usage-out", default="usage_log.json", help="Where to write per-task token usage (if available).")
    parser.add_argument("--agent-module", default="stub_agent", help="Python module exposing answer_task(task) -> str.")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    print(f"Loaded {len(tasks)} tasks from {args.tasks}")

    results, usage_log, total_time = run(tasks, args.agent_module)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    with open(args.usage_out, "w") as f:
        json.dump(usage_log, f, indent=2)

    # sanity checks mirroring the real invariants
    task_ids_in = {t["task_id"] for t in tasks}
    task_ids_out = {r["task_id"] for r in results}
    missing = task_ids_in - task_ids_out
    dupes = len(results) != len(task_ids_out)

    print(f"Wrote {len(results)} results to {args.out} in {total_time:.2f}s")
    if missing:
        print(f"CONTRACT VIOLATION: missing task_ids in output: {missing}", file=sys.stderr)
        sys.exit(1)
    if dupes:
        print("CONTRACT VIOLATION: duplicate task_ids in output.", file=sys.stderr)
        sys.exit(1)
    if total_time > MAX_RUNTIME_SECONDS:
        print(f"TIMING VIOLATION: {total_time:.1f}s exceeds the 10-minute cap.", file=sys.stderr)
        sys.exit(1)

    print("Contract check passed: every task_id present exactly once, within time budget.")


if __name__ == "__main__":
    main()
