"""
TEMPORARY STUB — replace with the real import once R1/R2 ship answer_task.

This exists so R3 (eval/QA) can build and test the harness + judge pipeline
on Day 1 without waiting on R1's container or R2's Fireworks wrapper.

Contract (frozen in the team guide):
    def answer_task(task: dict) -> str
    task = {"task_id": "t1", "prompt": "..."}
    returns the answer string only — nothing else.

Once R1/R2 have a real module (e.g. core/agent.py exposing answer_task),
point local_harness.py at that module instead of this stub via --agent-module.
"""

import random
import time


def answer_task(task: dict) -> str:
    """Fake answer_task. Deliberately dumb — just proves the plumbing works.

    Returns a placeholder string and simulates a tiny bit of latency and a
    fake token usage dict, so token_report.py has something to sum.
    """
    time.sleep(0.01)  # simulate latency, keep this tiny
    prompt = task.get("prompt", "")
    fake_answer = f"[STUB ANSWER for task {task.get('task_id')}] " + prompt[:40]

    # Fake usage numbers, attached via a side-channel the harness will pick up.
    # Real R2 client should return something similar so token_report.py can sum it.
    answer_task.last_usage = {
        "prompt_tokens": max(10, len(prompt.split())),
        "completion_tokens": random.randint(5, 20),
    }
    return fake_answer
