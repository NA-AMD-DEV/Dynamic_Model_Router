#!/usr/bin/env python3
# preflight_check.py -- automates as much of PRESUBMIT_CHECKLIST.md as
# realistically can be automated. Anything it can't verify (e.g. an actual
# `docker pull` from a clean machine) it prints as a manual reminder instead
# of silently skipping, so nothing falls through the cracks.
#
# Usage:
#   python preflight_check.py --agent-module stub_agent
#   python preflight_check.py --agent-module core.agent --dockerfile ../Dockerfile
#
# Exit code 0 if no FAILs (WARNs are still worth reading). Exit code 1 if
# any check FAILs -- treat that as "do not submit yet."

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
results = []  # (status, check_name, detail)


def record(status, name, detail=""):
    results.append((status, name, detail))


def check_contract(agent_module_name, eval_path):
    """Runs the harness against the eval set and checks the I/O contract."""
    try:
        with open(eval_path) as f:
            eval_tasks = json.load(f)
    except Exception as e:
        record(FAIL, "load eval set", str(e))
        return

    tasks = [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in eval_tasks]

    try:
        agent = importlib.import_module(agent_module_name)
    except Exception as e:
        record(FAIL, "import agent module", f"could not import '{agent_module_name}': {e}")
        return

    if not hasattr(agent, "answer_task"):
        record(FAIL, "agent contract", f"'{agent_module_name}' has no answer_task(task) -> str")
        return

    start = time.time()
    outputs = []
    errors = 0
    for t in tasks:
        try:
            ans = agent.answer_task(t)
            if not isinstance(ans, str):
                record(WARN, f"answer type ({t['task_id']})", "answer_task did not return a str -- will be coerced")
                ans = str(ans)
        except Exception as e:
            errors += 1
            ans = ""
        outputs.append({"task_id": t["task_id"], "answer": ans})
    elapsed = time.time() - start

    if errors:
        record(WARN, "agent exceptions", f"{errors} task(s) raised an exception and got a fallback empty answer")

    ids_in = {t["task_id"] for t in tasks}
    ids_out = [o["task_id"] for o in outputs]
    missing = ids_in - set(ids_out)
    dupes = len(ids_out) != len(set(ids_out))

    if missing:
        record(FAIL, "no missing task_ids", f"missing: {missing}")
    else:
        record(PASS, "no missing task_ids")

    if dupes:
        record(FAIL, "no duplicate task_ids", "duplicate task_ids found in output")
    else:
        record(PASS, "no duplicate task_ids")

    schema_ok = all("task_id" in o and "answer" in o for o in outputs)
    record(PASS if schema_ok else FAIL, "result objects have task_id + answer")

    # write it out and confirm it's parseable JSON round-trip
    tmp_path = "preflight_results.json"
    with open(tmp_path, "w") as f:
        json.dump(outputs, f)
    try:
        with open(tmp_path) as f:
            json.load(f)
        record(PASS, "results.json is valid JSON")
    except Exception as e:
        record(FAIL, "results.json is valid JSON", str(e))
    os.remove(tmp_path)

    if elapsed > 600:
        record(FAIL, "runtime under 10 minutes", f"{elapsed:.1f}s")
    elif elapsed > 300:
        record(WARN, "runtime under 10 minutes", f"{elapsed:.1f}s -- getting close to the cap, watch this")
    else:
        record(PASS, "runtime under 10 minutes", f"{elapsed:.1f}s")


def check_env_var_usage(agent_module_name):
    """Static scan of the agent module's source for hardcoded secrets/models
    and for direct (non-env-var) references to Fireworks endpoints."""
    try:
        mod = importlib.import_module(agent_module_name)
        src_path = getattr(mod, "__file__", None)
    except Exception:
        src_path = None

    if not src_path or not os.path.exists(src_path):
        record(WARN, "static source scan", "could not locate agent module source file to scan -- check by hand")
        return

    with open(src_path) as f:
        src = f.read()

    if re.search(r'os\.environ(\.get)?\(["\']FIREWORKS_API_KEY', src) is None and "FIREWORKS_API_KEY" in src:
        record(WARN, "FIREWORKS_API_KEY read from env", "mentioned but not obviously read via os.environ -- check by hand")
    elif "FIREWORKS_API_KEY" in src:
        record(PASS, "FIREWORKS_API_KEY read from env")
    else:
        record(WARN, "FIREWORKS_API_KEY read from env", "no reference found in this module -- fine if it lives elsewhere, otherwise check")

    if re.search(r'["\']sk-[a-zA-Z0-9]{10,}["\']', src) or re.search(r'["\']fw_[a-zA-Z0-9]{10,}["\']', src):
        record(FAIL, "no hardcoded API keys", "found a string that looks like a hardcoded API key")
    else:
        record(PASS, "no hardcoded API keys")

    if re.search(r'\.env["\']', src) and "load_dotenv" in src:
        record(WARN, "no bundled .env dependency", ".env loading found -- fine for local dev, MUST NOT be bundled in the submitted image")
    else:
        record(PASS, "no bundled .env dependency detected in source")

    if re.search(r'https?://api\.fireworks\.ai', src):
        record(FAIL, "no direct Fireworks endpoint bypass", "found a hardcoded api.fireworks.ai URL -- must route through FIREWORKS_BASE_URL")
    else:
        record(PASS, "no direct Fireworks endpoint bypass")


def check_dockerfile(dockerfile_path):
    if not dockerfile_path:
        record(WARN, "Dockerfile checks", "no --dockerfile path given, skipped -- run this again once R1 has one")
        return
    if not os.path.exists(dockerfile_path):
        record(WARN, "Dockerfile checks", f"'{dockerfile_path}' not found -- skipped")
        return

    with open(dockerfile_path) as f:
        content = f.read()

    if re.search(r'COPY\s+.*\.env', content):
        record(FAIL, "Dockerfile does not COPY a .env file", "found a COPY line referencing .env")
    else:
        record(PASS, "Dockerfile does not COPY a .env file")

    if re.search(r'ENV\s+FIREWORKS_API_KEY', content):
        record(FAIL, "Dockerfile does not hardcode FIREWORKS_API_KEY", "found an ENV line setting it directly")
    else:
        record(PASS, "Dockerfile does not hardcode FIREWORKS_API_KEY")

    record(WARN, "linux/amd64 build flag", "not checkable from Dockerfile alone -- confirm the BUILD COMMAND used --platform linux/amd64")


def check_image_size(image_name):
    if not image_name:
        record(WARN, "image size under 10GB", "no --image given, skipped -- run again once an image is built, e.g. --image your-image:latest")
        return
    try:
        out = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.Size}}", image_name],
                              capture_output=True, text=True, timeout=10)
        if not out.stdout.strip():
            record(WARN, "image size under 10GB", f"docker has no local image matching '{image_name}' -- build it first")
            return
        record(WARN, "image size under 10GB", f"local size: {out.stdout.strip()} (this is uncompressed local size, not the compressed registry size the judge checks -- treat as a rough signal only)")
    except FileNotFoundError:
        record(WARN, "image size under 10GB", "docker CLI not available in this environment -- check manually on a machine with Docker")
    except Exception as e:
        record(WARN, "image size under 10GB", str(e))


def check_category_coverage(score_report_path):
    if not os.path.exists(score_report_path):
        record(WARN, "all 8 categories exercised", f"'{score_report_path}' not found -- run run_eval.py first")
        return
    with open(score_report_path) as f:
        report = json.load(f)
    per_cat = report.get("per_category", {})
    zero_cats = [c for c, s in per_cat.items() if s.get("accuracy", 0) == 0]
    if len(per_cat) < 8:
        record(WARN, "all 8 categories exercised", f"only {len(per_cat)} categories present in eval set -- guide requires coverage of 8")
    if zero_cats:
        record(FAIL if len(zero_cats) > 2 else WARN, "no category stuck at 0%", f"0% accuracy in: {zero_cats}")
    else:
        record(PASS, "no category stuck at 0%")


def main():
    parser = argparse.ArgumentParser(description="R3 automated pre-submission checks.")
    parser.add_argument("--agent-module", default="stub_agent")
    parser.add_argument("--eval", default="eval_set.json")
    parser.add_argument("--dockerfile", default=None, help="Path to the Dockerfile, once R1 has one.")
    parser.add_argument("--image", default=None, help="Local docker image tag, once one is built.")
    parser.add_argument("--score-report", default="score_report.json", help="Output of llm_judge.py, for category coverage check.")
    args = parser.parse_args()

    print(f"Running preflight checks against agent module '{args.agent_module}'...\n")

    check_contract(args.agent_module, args.eval)
    check_env_var_usage(args.agent_module)
    check_dockerfile(args.dockerfile)
    check_image_size(args.image)
    check_category_coverage(args.score_report)

    print(f"{'STATUS':6s} CHECK")
    print("-" * 70)
    fails = 0
    for status, name, detail in results:
        if status == FAIL:
            fails += 1
        line = f"{status:6s} {name}"
        if detail:
            line += f"  -- {detail}"
        print(line)

    print("\n" + ("READY (no FAILs -- still review WARNs and the manual items in PRESUBMIT_CHECKLIST.md)"
                   if fails == 0 else f"NOT READY -- {fails} check(s) FAILED, fix before submitting"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
