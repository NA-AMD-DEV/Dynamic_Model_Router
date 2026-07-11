"""`python -m eval.run_local <tasks.json> <results.json>` -- R3's harness mirror.

Runs the real container entrypoint (agent.main.run) against a local tasks file
and writes results.json exactly as the judging harness would. Use it to dry-run
the pipeline without Docker, or to feed edge-case inputs (empty prompt, huge
batch) and confirm the output contract holds.

This deliberately calls the SAME code the container runs -- it is a mirror, not
a reimplementation, so anything that passes here reflects real behaviour.
"""

import argparse
import json
import sys
from pathlib import Path

import agent.main as m


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the agent locally, harness-style.")
    ap.add_argument("tasks", type=Path, help="input tasks.json")
    ap.add_argument("results", type=Path, help="output results.json to write")
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        print(f"expected a JSON list, got {type(tasks).__name__}", file=sys.stderr)
        return 1

    # Point the entrypoint's output at the requested path, then reuse its loop.
    m.OUTPUT_PATH = args.results
    m.run(tasks)

    out = json.loads(args.results.read_text(encoding="utf-8"))
    ids_out = [r["task_id"] for r in out]
    # The output contract: every DISTINCT input id appears exactly once, and
    # every answer is a string. Duplicate input ids collapsing to one row is
    # correct (dedup), not a violation -- so compare against the distinct set.
    distinct_in = {t.get("task_id") for t in tasks}
    all_strings = all(isinstance(r["answer"], str) for r in out)
    no_dupes = len(ids_out) == len(set(ids_out))
    covers_input = distinct_in.issubset(set(ids_out)) or not tasks
    ok = all_strings and no_dupes and covers_input

    print(f"wrote {len(out)} results to {args.results}")
    print(f"contract: {'OK' if ok else 'VIOLATED'} "
          f"(distinct-in={len(distinct_in)}, out={len(ids_out)}, "
          f"unique-out={no_dupes}, all-string={all_strings})")
    if len(tasks) != len(distinct_in):
        print(f"note: input had {len(tasks) - len(distinct_in)} duplicate task_id(s); "
              "deduped in output (correct)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
