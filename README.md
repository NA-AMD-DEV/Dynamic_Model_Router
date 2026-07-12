# Dynamic Model Router

A general-purpose AI agent for the **AMD Developer Hackathon · Act II · Track 1**. It reads a batch of tasks, routes each one to the cheapest model and prompt configuration that still answers it correctly, and writes the answers back out.

## How Track 1 is scored

Scoring happens in two stages, and they are not equal:

1. **Accuracy gate (pass/fail).** An LLM judge scores every answer against expected intent. Fall below the threshold and you are excluded from the leaderboard — your token count is never read. Organizers confirmed the real gate is **80%**.
2. **Token efficiency (the ranking).** Every submission that clears the gate is ranked ascending by total tokens recorded through the judging proxy. Fewer tokens wins.

So the whole design is: reach "good enough" accuracy on all 8 task categories, then cut every token that isn't buying accuracy. Submission also requires a public Docker image, a public GitHub repo, and a live demo app — see [Submitting](#submitting) below.

## The contract

The judging harness mounts an input file and expects an output file. Nothing else about the container is negotiable.

**Input** — `/input/tasks.json`

```json
[
  { "task_id": "t1", "prompt": "Summarise in one sentence: ..." },
  { "task_id": "t2", "prompt": "..." }
]
```

**Output** — `/output/results.json`

```json
[
  { "task_id": "t1", "answer": "..." },
  { "task_id": "t2", "answer": "..." }
]
```

The internal seam every component agrees on:

```python
def answer_task(task: dict) -> str:
    # task = {"task_id": "t1", "prompt": "..."}
    # the prompt does NOT name its category — infer it
    # returns the answer string only, nothing else
```

Invariants:

- Every `task_id` in the input appears exactly once in the output. A failed task still emits a best-effort string so the JSON stays valid.
- All inference routes through `FIREWORKS_BASE_URL` using `FIREWORKS_API_KEY`. Bypass it and tokens aren't recorded, so you can't rank.
- Only models listed in `ALLOWED_MODELS`. Read and split it at runtime; never hardcode IDs and never bundle a `.env`.

## Correctness first, then tokens

The official rubric (public validation samples, retired scoring cases) grades fuller answers than a token-minimal instinct produces: sentiment wants a label **plus a one-sentence reason acknowledging both sides** of a mixed review; factual tasks may demand a brief explanation; summaries must match the requested format **exactly** (sentence/bullet counts, word limits, each unit on its own line); NER wants every entity with exact labels; math is multi-step. Prompts and caps in `agent/config.py` are sized to that rubric — an answer that saves tokens but fails the gate scores zero.

Token levers, applied only after correctness:

| Lever | What it means |
|---|---|
| **Deterministic solvers** | Trivial single-step math/logic answered in Python at **zero tokens** — see below |
| **Measured model routing** | No model or vendor name is ever hardcoded — see below |
| **Self-healing** | A model listed in `ALLOWED_MODELS` but not actually deployed gets blocklisted and routed around, mid-run |
| `max_tokens` cap | Ceiling per task type, sized so the *required* output is never truncated |
| `temperature=0` | Same prompt → same completion, run to run — required for exact-format rubrics (see below) |
| No printed reasoning | Reasoning tokens count — `reasoning_effort=none`, tunable per category |
| Single call per task | No self-critique loops unless a category needs them |
| Category routing | Keyword heuristic, not a model call — a classifier would spend tokens to save tokens |

**Zero-token solvers (`agent/solvers.py`).** An answer computed in Python never calls the model — 0 tokens *and* can't be got wrong by a weak model. `solve_math` handles short, direct single-step asks (percent-of, single discount, even splits, powers, single-unit average speed, bare arithmetic via an AST whitelist — never `eval`); `solve_logic` handles clean transitive/spatial ordering. Both are **ultra-conservative**: any multi-step signal, second question, negation, tie, mixed-unit duration, or markup-then-discount pricing defers to the model — real tasks are mostly multi-step, so solvers are a safe bonus, not the main path. Every near-miss misfire found during development (a fraction inside a recipe problem, a mixed-unit duration, a markup+discount) is locked in as a regression test.

**Measured model routing (`agent/config.py`).** The exact models on judging day aren't knowable in advance, so nothing is hardcoded. Resolution order per category:

1. `ROUTER_<CATEGORY>_MODEL` — an explicit pin (env override, no rebuild).
2. **Specialist match** — a category tagged `specialist="code"` claims any live model whose id contains `code`/`coder`, if one exists.
3. **Measured-leanest (`calibrate_lean`)** — once per run, before any task is scored, every live model gets one minimal probe recording its real prompt-template overhead (this varies up to ~3× between model families). The category then routes to the cheapest model that measurement has shown clears its rubric (`lean_ok=True`, the default). Categories where the lean model measurably *failed* the rubric (currently `summarisation`, and the math/logic solver-defer residue — a case hard enough to reach the model at all deserves the strongest one, not the cheapest) are pinned `lean_ok=False` and skip straight to step 4.
4. **Capability ranking** — a parameter count parsed from the id itself (`qwen3-30b-a3b` → 30, `llama-3.3-70b-instruct` → 70) ranks whatever's left; ids with no parseable size sort after every sized one.

The calibration probe also doubles as a warmup (absorbs cold-start latency before any scored task) and as the mechanism that discovers undeployed models (a 404 there blocklists the model before it can fail a real task). Probe tokens are real spend and are added to `eval.score`'s reported total, never hidden. Once the real list is known, `python -m eval.score` measures actual per-category results and any correction gets baked into `_DEFAULTS` or pinned via `ROUTER_<CATEGORY>_MODEL`.

**Determinism (`TEMPERATURE`).** Fireworks calls default to `temperature=0`. At the previous `0.2`, the same prompt could sample a different completion each run — invisible for most categories, but exact-format rubrics (precise sentence/bullet counts) would flip pass/fail purely from sampling. The client also sets `max_retries=0` on the OpenAI SDK: its own default (2 hidden internal retries, each with a fresh timeout) was stacking invisibly under the app's own retry logic and could balloon a single task to 5× the 30-second per-request limit.

## Runtime limits

The judging VM enforces these; violating any one scores zero.

- Image built for `linux/amd64`, public, pulls with no login, ≤ 10 GB compressed
- Ready in under 60 s; each request under 30 s (`agent/fireworks_client.py` sets a 25 s client timeout via `REQUEST_TIMEOUT_S`, with `max_retries=0` so the SDK can't silently multiply that); whole run under 10 min
- Exit 0 on success; valid `results.json` even when tasks fail
- All responses in English

## Layout

| Area | Files | What it does |
|---|---|---|
| **Container & harness** | `agent/main.py`, `agent/routing.py`, `Dockerfile` | Reads `/input`, routes each task to a category, writes `/output`, guarantees the contract even on failure. Zero-token keyword routing. |
| **Model & prompt** | `agent/fireworks_client.py`, `agent/config.py`, `agent/core.py`, `agent/solvers.py` | The only code that calls Fireworks. Deterministic 0-token solvers, measured model routing (calibration, self-heal, specialist/capability fallback), per-category prompts and `max_tokens`, all overridable by env var without a rebuild. |
| **Local inference (unused)** | `agent/local_client.py` | A feasibility probe for 0-token local (in-container) inference. Off by default (`LOCAL_MODEL_PATH` unset) — measured CPU latency on a small model made the 10-minute budget too risky on a hidden set of unknown size. See `LOCAL_PROBE.md`. |
| **Eval & QA** | `eval/` | Local harness mirror, a 64-task eval set (8/category) modeled on the organizers' own retired public samples, a rubric-strict local LLM judge, and the `score` go/no-go command. |
| **Demo (not judged)** | `demo/app.py` | A Streamlit UI calling the exact same `answer_task_detailed()` seam as the Docker image, for the submission's separately-required live demo URL. Isolated `demo/requirements.txt`; excluded from the image via `.dockerignore`. |

The 8 categories are `factual_knowledge`, `math_reasoning`, `sentiment_classification`, `summarisation`, `named_entity_recognition`, `code_debugging`, `logical_reasoning`, `code_generation`. `agent/routing.py` and `agent/config.py` must agree on this set — a test enforces it.

## Running it

Set credentials first (never commit them — see `.env.example`):

```
export FIREWORKS_API_KEY=...  FIREWORKS_BASE_URL=...  ALLOWED_MODELS=model-a,model-b
```

**Locally, harness-style** (no Docker):

```
python -m eval.run_local fixtures/tasks.json out/results.json
```

**Score against the eval set** (accuracy gate proxy + total tokens — the ranking metric):

```
python -m eval.score            # exits non-zero if below the proxy gate
```

Prints per-category accuracy/tokens/truncations, a routing check, and — for any task that judged wrong without an error — the actual answer with real line breaks preserved and a per-line word count, so a rubric miss is diagnosable from the output instead of guessed at.

**In the container:**

```
docker build --platform linux/amd64 -t dynamic-model-router:dev .
docker run --rm -v "$PWD/fixtures:/input:ro" -v "$PWD/out:/output" \
  -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  dynamic-model-router:dev
```

**Tests** (no key or network needed — the Fireworks client is mocked):

```
python -m pytest tests/ -q
```

**Demo app** (optional, not part of the judged pipeline):

```
streamlit run demo/app.py
```

Needs `FIREWORKS_API_KEY` / `FIREWORKS_BASE_URL` / `ALLOWED_MODELS` in the environment (locally) or under **Settings → Secrets** (Streamlit Community Cloud). Without credentials it still runs and shows the graceful-degradation path (empty answers, no crash) rather than failing.

### Token-efficiency knobs (per category, no rebuild)

`ROUTER_<CATEGORY>_MODEL=<exact id>` (or `ROUTER_<CATEGORY>_MODEL_INDEX=n`) pins a category past the measured-routing chain entirely. Other per-category knobs: `ROUTER_<CATEGORY>_MAX_TOKENS`, `ROUTER_<CATEGORY>_SYSTEM`, `ROUTER_<CATEGORY>_REASONING_EFFORT`. Global: `TEMPERATURE` (default `0`), `REASONING_EFFORT` (default `none`), `ROUTER_CONCURRENCY` (the Dockerfile bakes `3`; each worker is one in-flight call, so it's also the rate-limit blast radius). `lean_ok` (whether a category is eligible for the measured-cheapest model) is a `_DEFAULTS` flag in code, not currently env-overridable — flip it in `agent/config.py` if a fresh measurement changes the call.

All of `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` are read from the environment at runtime — nothing is hardcoded or baked into the image, and the model list is re-parsed on every call so whatever the harness injects is what gets used.

> **Before submitting:** the judging harness injects **only** `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` — no `ROUTER_*` variables exist on judging day. Any tuning proven out via env vars must be baked into `_DEFAULTS` in `agent/config.py` (or an `ENV` line in the Dockerfile) before the image is built, or it silently won't apply.

## Results

Measured on the internal 64-task eval (8/category, modeled on the organizers' retired public samples), real Fireworks models, live LLM judge:

- **98% accuracy** (63/64), zero agent/judge errors, zero timeouts, deterministic run to run (`temperature=0`) — clears the confirmed 80% gate with margin.
- **136 unit tests** passing (`tests/`), fully mocked — no live key needed to verify routing, solvers, calibration, self-heal, or failover logic in isolation.
- Docker image built for `linux/amd64`, pushed public, and verified end-to-end: logged-out anonymous pull succeeds, a fresh run of the pulled image produces valid `results.json` for every task, both with real credentials and with none (graceful degradation, no crash).

## Submitting

See `SUBMIT.md` for the exact push/pull/verify sequence used to confirm the image is genuinely public and pullable before submission (the FAQ's #1 failure mode is `PULL_ERROR`). See `LOCAL_PROBE.md` for why local (0-token) inference was investigated and not shipped.
