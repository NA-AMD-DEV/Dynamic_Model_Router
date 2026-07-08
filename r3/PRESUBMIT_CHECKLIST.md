# Pre-Submission Checklist (R3 sign-off)

Run through this in full before EVERY real submission (we only get 10/hour,
so a wasted slot on a preventable failure is costly). Each item maps to a
specific rule or failure status from the participant guide, so if something
fails here, the "why" column tells you exactly which real-world error it
would have produced.

Nothing on this list should be marked done from memory — actually run the
check. `preflight_check.py` automates the ones that can be automated;
everything else is a manual look.

## 1. I/O contract

- [ ] Container reads tasks from `/input/tasks.json` on startup (not a hardcoded local path)
- [ ] Container writes to `/output/results.json` before exiting
- [ ] Every `task_id` from the input appears exactly once in the output — none missing, none duplicated
  - Why: missing task_ids → that task scores zero. Duplicates are a schema violation.
- [ ] Every result object has both a `task_id` and an `answer` field
  - Why: anything else → `INVALID_RESULTS_SCHEMA`
- [ ] `results.json` is valid, parseable JSON — test with `python -m json.tool results.json`
  - Why: malformed JSON → entire submission scores zero, not just one task
- [ ] Container still writes a (possibly empty) answer for a task that errors internally — never silently drops a task_id

## 2. Runtime & platform

- [ ] Container starts and is ready within 60 seconds
- [ ] Each individual request resolves in under 30 seconds
- [ ] Full run (all tasks) finishes inside the 10-minute cap
  - Why: exceeding this → `TIMEOUT`
- [ ] Container exits with code 0 on success, non-zero on failure
  - Why: silent bad exit → `RUNTIME_ERROR`
- [ ] Image was built with `--platform linux/amd64` (mandatory if built on Apple Silicon)
  - Why: missing amd64 manifest → `PULL_ERROR`, image never even runs
- [ ] Image compressed size is under 10GB — check with `docker images` after build
  - Why: over the cap → `IMAGE_TOO_LARGE`, rejected before pulling

## 3. Fireworks / env var rules (Track 1 specific)

- [ ] `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, `ALLOWED_MODELS` are read from `os.environ` at runtime — not hardcoded, not loaded from a bundled `.env`
- [ ] No `.env` file is copied into the Docker image (`grep -i "\.env" Dockerfile` should show nothing being COPYed in)
- [ ] All Fireworks calls are routed through `FIREWORKS_BASE_URL` — no direct calls to `api.fireworks.ai` or any other endpoint bypassing it
  - Why: bypassing → calls aren't recorded, submission scores zero tokens
- [ ] No model ID is hardcoded anywhere in the agent code — always read from `ALLOWED_MODELS` at runtime
  - Why: calling a model outside the list → `MODEL_VIOLATION`, whole submission invalidated
- [ ] Local/open-source models used inside the container are fine and expected — just confirm they're not accidentally being counted as Fireworks calls in your own token log

## 4. Correctness / anti-hardcoding

- [ ] No answer is hardcoded or cached against specific known inputs (including the practice tasks from the guide)
  - Why: evaluation uses unseen prompt variants — hardcoded practice answers will not transfer and looks like cheating if detected
- [ ] Agent has been tested against reworded/paraphrased versions of practice tasks, not just the originals verbatim
- [ ] All 8 capability categories have been exercised at least once in the local eval set, and none are sitting at 0% right before submission

## 5. Submission process

- [ ] Docker image is pushed to a **public** registry and confirmed pullable (test with `docker pull <image>` from a clean/different machine or after `docker rmi` locally)
- [ ] Staying within the 10-submissions-per-hour rate limit — don't burn slots on untested changes
- [ ] Run the full `run_eval.py` pipeline one final time against the actual built image (not just the Python module) before submitting — see Phase 5 in the team README

## Quick automated pass

From `r3/`, run:

```bash
python preflight_check.py --dockerfile ../Dockerfile --agent-module core.agent
```

This automates what it can (JSON schema, contract, env var usage scan, timing,
Dockerfile checks) and prints a clear PASS/WARN/FAIL per item. Anything it
can't check (e.g. actual `docker pull` from a clean host) is called out
explicitly so a human still checks it by hand.
