# R3 — Eval, Data & QA

This folder is the team's local mirror of the real judging harness, plus the
eval set and scoring tools. Nothing here is submitted directly — it's the
tooling that tells the team whether what R1/R2 build is actually good enough,
before we burn one of our 10-per-hour submission slots on it.

## Files

| File | What it does |
|---|---|
| `eval_set.json` | 32 hand-written tasks, 4 per capability category, each with an `expected_intent` (and `expected_answer` where there's a single correct value). This is **our own practice set** — not the hidden grading set. |
| `local_harness.py` | Mirrors the real container contract: reads a tasks file, calls `answer_task(task) -> str`, writes `results.json`. Enforces the same invariants the real harness does (every `task_id` present exactly once, 10-minute cap). |
| `stub_agent.py` | Temporary placeholder `answer_task`. Lets the harness/judge/token pipeline run today, before R1's container or R2's Fireworks wrapper exist. **Delete this once a real agent module exists.** |
| `llm_judge.py` | Scores `results.json` against `eval_set.json`. Two backends: `heuristic` (offline, no API key, rough — today's default) and `llm` (real LLM-as-judge call, mirrors the actual accuracy gate — switch to this once Fireworks creds exist). |
| `token_report.py` | Sums per-task token usage (if the agent module reports it) into a per-category and total token count, so we can watch token cost trend down as R2 tunes things. |
| `run_eval.py` | Runs all three of the above in sequence. This is the one command to run before every real submission. |

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
- [ ] Phase 2: wire the real LLM-judge backend once Fireworks creds exist
- [ ] Phase 3: expand eval set with harder/reworded variants per category
- [ ] Phase 4: token leaderboard tracking as R2 tunes prompts
- [ ] Phase 5: dress rehearsal — run this against the actual built Docker
      image, not just a Python module
- [ ] Phase 6: final pre-submission checklist sign-off
