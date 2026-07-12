#!/usr/bin/env python3
# run_eval.py -- one command to run the whole R3 pipeline:
#   1. local_harness.py   (tasks -> results, contract check)
#   2. llm_judge.py        (results -> accuracy per category)
#   3. token_report.py     (usage -> tokens per category)
#
# This is what you run before every submission (Phase 5/6 dress rehearsal),
# and periodically during development to catch regressions early.
#
# Usage:
#     python run_eval.py --agent-module stub_agent --judge-backend heuristic
#
# Once R1/R2 have a real agent module and Fireworks creds are available:
#     python run_eval.py --agent-module core.agent --judge-backend llm

import argparse
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nSTOPPED: '{' '.join(cmd)}' exited with code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the full R3 eval pipeline.")
    parser.add_argument("--eval", default="eval_set.json")
    parser.add_argument("--agent-module", default="stub_agent")
    parser.add_argument("--judge-backend", choices=["heuristic", "llm"], default="heuristic")
    args = parser.parse_args()

    run([sys.executable, "local_harness.py",
         "--tasks", args.eval, "--out", "results.json",
         "--usage-out", "usage_log.json", "--agent-module", args.agent_module])

    run([sys.executable, "llm_judge.py",
         "--eval", args.eval, "--results", "results.json",
         "--backend", args.judge_backend, "--out", "score_report.json"])

    run([sys.executable, "token_report.py",
         "--usage", "usage_log.json", "--eval", args.eval,
         "--out", "token_report.json"])

    print("\n=== Pipeline complete ===")
    print("See score_report.json for accuracy, token_report.json for token usage.")


if __name__ == "__main__":
    main()
