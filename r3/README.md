# R3 — Eval, Data & QA

This folder is the team's local mirror of the real judging harness, plus the
eval set and scoring tools. Nothing here is submitted directly — it's the
tooling that tells the team whether what R1/R2 build is actually good enough,
before we burn one of our 10-per-hour submission slots on it.

## Organizer clarifications (received Day 2)

Straight from the hackathon organizers, relevant to R3's tooling and the whole team:

- **Accuracy gate confirmed at 80%.** Below that, no leaderboard placement regardless of token count. `llm_judge.py` now checks this explicitly (`--gate 0.80` by default) and prints a clear CLEARS/BELOW line.
- **The real hidden eval set is exactly 19 fixed tasks** — every real score is n/19. Our local 48-task practice set is intentionally separate and won't produce matching percentages; treat local scores as directional, not predictive of the exact real number.
- **The real LLM judge isn't perfectly deterministic run-to-run** (organizer-confirmed). If a submission is close to the 80% line, don't trust one run — build in a safety margin.
- **Fireworks credits are per-team**, sent as a coupon to the team's registered email within 2-3 business days of registration, redeemed at fire-pass. If R2 hasn't sent creds yet, this is likely why — worth checking whether the team's coupon has arrived yet.
- **No separate "AMD cloud credits"** — compute is the team's Jupyter instance at notebooks.amd.com/hackathon (8h/day).
- **Gemma models cost ~$7/hour even when idle** once deployed. If anyone (including R3, for local `JUDGE_MODEL` testing) deploys a Gemma model, undeploy it immediately after testing — it's shared team budget ($50 total) and idle time burns it fast.
- **Registry pull counters** (GitHub Packages / Docker Hub download count) show whether the organizers have actually pulled your image yet — useful during grading backlogs, added to the checklist.

## Files

| File | What it does |
|---|---|
| `eval_set.json` | 32 hand-written tasks, 4 per capability category, each with an `expected_intent` (and `expected_answer` where there's a single correct value). This is **our own practice set** — not the hidden grading set. |
| `local_harness.py` | Mirrors the real container contract: reads a tasks file, calls `answer_task(task) -> str`, writes `results.json`. Enforces the same invariants the real harness does (every `task_id` present exactly once, 10-minute cap). |
| `stub_agent.py` | Temporary placeholder `answer_task`. Lets the harness/judge/token pipeline run today, before R1's container or R2's Fireworks wrapper exist. **Delete this once a real agent module exists.** |
| `llm_judge.py` | Scores `results.json` against `eval_set.json`. `heuristic` backend (default, offline, no API key) now uses **real checks per category**: code tasks are actually executed against test cases, NER checks each required entity individually, summaries are checked against their stated format constraint (sentence/word count). Factual knowledge and sentiment justification still fall back to crude keyword overlap — no cheap deterministic check exists for open-ended prose, which is exactly what the `llm` backend (real LLM-as-judge, mirrors the actual gate) is for once Fireworks creds exist. |
| `good_agent_for_testing.py` | Test fixture only, not a deliverable — a fake agent with genuinely correct answers to every eval task. Use it to sanity-check the judge itself after any changes: `python run_eval.py --agent-module good_agent_for_testing` should score high. If it doesn't, the judge broke, not the agent. |
| `token_report.py` | Sums per-task token usage (if the agent module reports it) into a per-category and total token count, so we can watch token cost trend down as R2 tunes things. |
| `run_eval.py` | Runs all three of the above in sequence. This is the one command to run before every real submission. |
| `PRESUBMIT_CHECKLIST.md` | Human-readable checklist mapping every guide rule to a check, organized by I/O contract / runtime / Fireworks rules / anti-hardcoding / submission process. |
| `preflight_check.py` | Automates as much of the checklist as a script can: contract validation, JSON schema, runtime, a static source scan for hardcoded keys/models/endpoint bypasses, Dockerfile checks (once one exists), and category coverage. Exits non-zero if anything FAILs — treat that as "do not submit yet." |

## How to run it right now

```bash
cd r3
python run_eval.py --agent-module stub_agent --judge-backend heuristic
```

This uses the stub agent and the offline heuristic judge — no API keys
needed. You should see per-category accuracy (expect it to be bad, the stub
is fake) and a token report.

## How this connects to R1 / R2

- The contract is frozen: `answer_task(task: dict) -> str`, `task = {"task_id", "prompt"}`.
- The moment R1/R2 have a real module exposing that function (e.g.
  `core/agent.py`), swap it in:
  ```bash
  python run_eval.py --agent-module core.agent --judge-backend llm
  ```
  (the `llm` backend needs `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and a
  `JUDGE_MODEL` env var set — ask R2 for a dev key/model to use for judging;
  this is separate from the team's real `ALLOWED_MODELS` list)
- `local_harness.py` is intentionally dumb about *how* `answer_task` works —
  it doesn't care if it's local-model-only, Fireworks-only, or routed. That
  logic lives entirely in R1/R2's code.
- If the agent module wants token accounting to work, it should set
  `answer_task.last_usage = {"prompt_tokens": ..., "completion_tokens": ...}`
  after each call — see `stub_agent.py` for the pattern.

## Status

- [x] Phase 1: local harness built, 32-task eval set drafted, scoring
      skeleton (heuristic backend) working end-to-end
- [x] Pre-submission checklist + automated preflight script drafted early
      (normally Phase 5/6, pulled forward since it only depends on the
      published rules, not on R1/R2's code)
- [x] Phase 3: expanded eval set to 48 tasks (6 per category) with harder/
      reworded variants -- sarcasm detection, compounding math traps,
      multi-constraint summaries, over-extraction traps in NER, subtle bugs
      (mutable default args, off-by-one), a 5-entity logic puzzle, and
      recursive/nested code generation. Verified with the good/bad agent
      fixtures: 91.7% vs 10.4%, and the two riskiest new tasks (the mutable-
      default-arg test and the 5-runner logic puzzle) were independently
      solved/brute-forced to confirm correctness before trusting them.
- [ ] Phase 2: wire the real LLM-judge backend once Fireworks creds exist
- [ ] Phase 4: token leaderboard tracking as R2 tunes prompts
- [ ] Phase 5: dress rehearsal — run `preflight_check.py` against the actual
      built Docker image + Dockerfile (`--dockerfile`, `--image` flags), not
      just the Python module
- [ ] Phase 6: final pre-submission checklist sign-off — go through
      `PRESUBMIT_CHECKLIST.md` by hand for the items the script can't check
      (e.g. a clean-machine `docker pull` test)
