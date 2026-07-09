"""Container entrypoint: read /input/tasks.json, answer each, write /output/results.json.

Two hard rules from the harness drive everything here:
  - Every task_id in the input appears exactly once in the output.
  - results.json is valid JSON and exists on exit, even when tasks fail.

So results are seeded with a fallback answer for every task before any work
starts, and flushed to disk in a finally block. A crashed task, a hung API
call, or the 10-minute wall each degrade to a scored-but-wrong answer rather
than a missing key.
"""

import json
import os
import sys
import time
from pathlib import Path

from agent.core import answer_task

INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))

# The harness kills the run at 10 minutes. Stop starting new tasks with enough
# slack left to serialise whatever we have.
RUN_BUDGET_S = float(os.environ.get("RUN_BUDGET_S", 9 * 60 + 15))

# Anything we couldn't answer. An empty string is still valid JSON and still
# gets judged; a missing task_id is not.
FALLBACK = ""


def load_tasks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError(f"expected a JSON list of tasks, got {type(tasks).__name__}")
    return tasks


def write_results(path: Path, results: dict[str, str]) -> None:
    """Serialise atomically: a partial file is worse than an old one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"task_id": tid, "answer": ans} for tid, ans in results.items()]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    tmp.replace(path)


def run(tasks: list[dict]) -> dict[str, str]:
    # Seed every task_id up front. From here on, results is always complete;
    # the loop only ever improves an answer, never adds or drops a key.
    results: dict[str, str] = {}
    for i, task in enumerate(tasks):
        tid = task.get("task_id") or f"__missing_{i}"
        results[tid] = FALLBACK

    started = time.monotonic()
    try:
        for i, task in enumerate(tasks):
            tid = task.get("task_id") or f"__missing_{i}"

            elapsed = time.monotonic() - started
            if elapsed > RUN_BUDGET_S:
                print(
                    f"run budget exhausted after {i}/{len(tasks)} tasks "
                    f"({elapsed:.0f}s); remaining tasks keep the fallback answer",
                    file=sys.stderr,
                )
                break

            try:
                answer = answer_task(task)
            except Exception as exc:  # one bad task must not sink the batch
                print(f"task {tid} failed: {exc!r}", file=sys.stderr)
                continue

            results[tid] = answer if isinstance(answer, str) else str(answer)
    finally:
        # Runs on a normal return, an unexpected raise, and on the
        # KeyboardInterrupt/SystemExit paths.
        write_results(OUTPUT_PATH, results)

    return results


def main() -> int:
    try:
        tasks = load_tasks(INPUT_PATH)
    except Exception as exc:
        # No tasks means no task_ids, so there is nothing valid to write beyond
        # an empty list. Say why, loudly, and still leave valid JSON behind.
        print(f"could not read {INPUT_PATH}: {exc!r}", file=sys.stderr)
        write_results(OUTPUT_PATH, {})
        return 1

    run(tasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
