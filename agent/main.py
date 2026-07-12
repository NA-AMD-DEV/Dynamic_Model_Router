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
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from agent.core import answer_task

INPUT_PATH = Path(os.environ.get("INPUT_PATH", "/input/tasks.json"))
OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "/output/results.json"))

# The harness kills the run at 10 minutes. Stop starting new tasks with enough
# slack left to serialise whatever we have.
RUN_BUDGET_S = float(os.environ.get("RUN_BUDGET_S", 9 * 60 + 15))

# Sequential by default: deterministic for R3's evals, and it keeps per-task
# latency measurable for R2. Raise it only once real timings show the 10-minute
# limit is in reach. Each worker is one in-flight Fireworks call, so this is
# also the rate-limit blast radius.
CONCURRENCY = max(1, int(os.environ.get("ROUTER_CONCURRENCY", "1")))

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

    # Measure each live model's template cost ONCE, single-threaded, before
    # any worker exists -- routing then prefers the leanest model. Also doubles
    # as a warmup (absorbs cold starts) and self-heal (dead models 404 here,
    # not on a scored task). Must never sink the run.
    try:
        from agent.config import calibrate_lean
        calibrate_lean()
    except Exception as exc:
        print(f"lean calibration skipped: {exc!r}", file=sys.stderr)

    started = time.monotonic()
    try:
        if CONCURRENCY > 1:
            _run_concurrent(tasks, results, started)
        else:
            _run_sequential(tasks, results, started)
    finally:
        # Runs on a normal return, an unexpected raise, and on the
        # KeyboardInterrupt/SystemExit paths.
        write_results(OUTPUT_PATH, results)

    return results


def _answer_one(task: dict, tid: str) -> str | None:
    """Call the core for one task. Returns None if it failed."""
    try:
        answer = answer_task(task)
    except Exception as exc:  # one bad task must not sink the batch
        print(f"task {tid} failed: {exc!r}", file=sys.stderr)
        return None
    return answer if isinstance(answer, str) else str(answer)


def _run_sequential(tasks: list[dict], results: dict[str, str], started: float) -> None:
    for i, task in enumerate(tasks):
        tid = task.get("task_id") or f"__missing_{i}"

        elapsed = time.monotonic() - started
        if elapsed > RUN_BUDGET_S:
            print(
                f"run budget exhausted after {i}/{len(tasks)} tasks "
                f"({elapsed:.0f}s); remaining tasks keep the fallback answer",
                file=sys.stderr,
            )
            return

        answer = _answer_one(task, tid)
        if answer is not None:
            results[tid] = answer


def _run_concurrent(tasks: list[dict], results: dict[str, str], started: float) -> None:
    """Answer tasks in a thread pool. Ordering is safe: `results` was seeded in
    input order and dicts preserve insertion order, so workers overwrite values
    in place and never change the key sequence.

    Threads (not processes) because the work is a blocking HTTPS call, and
    CPython releases the GIL while waiting on the socket.
    """
    remaining = deque(
        (task.get("task_id") or f"__missing_{i}", task)
        for i, task in enumerate(tasks)
    )
    attempted = 0
    out_of_time = False

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        pending: dict = {}
        while remaining or pending:
            # Top the pool up, unless the clock has run out. Tasks left in
            # `remaining` keep their seeded fallback answer.
            while remaining and len(pending) < CONCURRENCY:
                if time.monotonic() - started > RUN_BUDGET_S:
                    out_of_time = True
                    break
                tid, task = remaining.popleft()
                pending[pool.submit(_answer_one, task, tid)] = tid
                attempted += 1

            if out_of_time:
                # Drain in-flight work rather than abandoning it: those tokens
                # are already spent, and the answers are still worth having.
                remaining.clear()

            if not pending:
                break

            # Wait on whichever worker finishes first, so the budget check
            # above re-runs promptly rather than after the whole batch.
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                tid = pending.pop(fut)
                answer = fut.result()  # _answer_one swallowed any exception
                if answer is not None:
                    results[tid] = answer

    # Only the clock abandons tasks. A task that was attempted and raised is
    # already reported by _answer_one; don't blame the budget for it.
    if out_of_time:
        print(
            f"run budget exhausted after dispatching {attempted}/{len(tasks)} tasks "
            f"({time.monotonic() - started:.0f}s); "
            "undispatched tasks keep the fallback answer",
            file=sys.stderr,
        )


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
