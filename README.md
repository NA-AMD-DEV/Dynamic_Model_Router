# Dynamic Model Router

A general-purpose AI agent for the **AMD Developer Hackathon · Act II · Track 1**. It reads a batch of tasks, routes each one to the cheapest model and prompt configuration that still answers it correctly, and writes the answers back out.

## How Track 1 is scored

Scoring happens in two stages, and they are not equal:

1. **Accuracy gate (pass/fail).** An LLM judge scores every answer against expected intent. Fall below the threshold and you are excluded from the leaderboard — your token count is never read.
2. **Token efficiency (the ranking).** Every submission that clears the gate is ranked ascending by total tokens recorded through the judging proxy. Fewer tokens wins.

So the whole design is: reach "good enough" accuracy on all 8 task categories, then cut every token that isn't buying accuracy.

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
- Only models listed in `ALLOWED_MODELS` (published on launch day). Read and split it at runtime; never hardcode IDs and never bundle a `.env`.

## Where the tokens go

Levers, roughly in order of impact:

| Lever | What it means |
|---|---|
| **Deterministic solvers** | Math and logic answered in Python at **zero tokens** — see below |
| Model tiering | Hard categories → the largest capable model; cheap ones (sentiment, NER) → the smallest — see below |
| `max_tokens` cap | Hard output ceiling per task type |
| System prompt length | One ruthlessly short shared prefix, billed on every call |
| No printed reasoning | Reasoning tokens count — `reasoning_effort=none`, tunable per category |
| Answer-only output | No preamble, no restating the question; inline `<think>` blocks stripped |
| Single call per task | No self-critique loops unless a category needs them |
| Category routing | Keyword heuristic, not a model call — a classifier would spend tokens to save tokens |

**Zero-token solvers (`agent/solvers.py`).** The ranking metric is total proxy tokens, and an answer computed in Python never calls the model — so it costs 0 tokens *and* can't be got wrong by a weak model. `solve_math` handles percent-of, discounts, even splits, powers, average speed, and clean arithmetic (via an AST whitelist, never `eval`). `solve_logic` handles transitive comparison and spatial-ordering puzzles (a "greater-than" graph + unique topological sort). Both are **precision-first**: any ambiguity returns `None` and falls back to the model (routed to the `large` tier, since the residue is the hard case). A confident-but-wrong 0-token answer would cost the accuracy gate, which is worse than paying tokens.

**Adaptive model tiering (`agent/config.py`).** The exact models on judging day aren't knowable in advance, so no model or vendor name is ever hardcoded. `pick_capability` ranks whatever `ALLOWED_MODELS` contains, purely from a parameter count parsed out of each id (`qwen3-30b-a3b` → 30, `llama-3.3-70b-instruct` → 70); each category's `_Default.tier` (`small` / `medium` / `large`) then resolves to a real model id at call time. Ids with no parseable size (common on flagship/commercial names like `kimi-k2p6`, `deepseek-v4-pro`) fall in after every sized model — an imperfect heuristic, not a guarantee. Once the real list is known, `python -m eval.score` measures the actual fewest-tokens-that-passes model per category, and that gets pinned via `ROUTER_<CATEGORY>_MODEL`.

## Runtime limits

The judging VM enforces these; violating any one scores zero.

- Image built for `linux/amd64`, public, pulls with no login, ≤ 10 GB compressed
- Ready in under 60 s; each request under 30 s (enforced: `agent/fireworks_client.py` sets a 25 s client timeout, `REQUEST_TIMEOUT_S` to override); whole run under 10 min
- Exit 0 on success; valid `results.json` even when tasks fail
- All responses in English

## Layout

Three vertical slices, one owner each, meeting only at `answer_task`:

| Slice | Files | What it does |
|---|---|---|
| **Container & harness** | `agent/main.py`, `agent/routing.py`, `Dockerfile` | Reads `/input`, routes each task to a category, writes `/output`, guarantees the contract even on failure. Zero-token keyword routing. |
| **Model & prompt** | `agent/fireworks_client.py`, `agent/config.py`, `agent/core.py`, `agent/solvers.py` | The only code that calls Fireworks. Deterministic 0-token solvers for math/logic; per-category prompts, model tier, `max_tokens`, and `reasoning_effort`, all overridable by env var without a rebuild. |
| **Eval & QA** | `eval/` | Local harness mirror, a 64-task eval set (8/category) across the 8 categories, a local LLM judge, and the `score` go/no-go command. |

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

### Token-efficiency knobs (per category, no rebuild)

Every category defaults to `ALLOWED_MODELS[0]` — the agent makes **no assumption** about the injected list's order or which model is "stronger", since the harness doesn't promise one. R2 points a category at a specific model only after measuring, via `ROUTER_<CATEGORY>_MODEL=<exact id>` (or `ROUTER_<CATEGORY>_MODEL_INDEX=n`). Other per-category knobs: `ROUTER_<CATEGORY>_MAX_TOKENS`, `ROUTER_<CATEGORY>_SYSTEM`. Example: `ROUTER_CODE_GENERATION_MAX_TOKENS=256`.

All three of `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` are read from the environment at runtime — nothing is hardcoded or baked into the image, and the model list is re-parsed on every call so whatever the harness injects is what gets used. Set `ROUTER_CONCURRENCY=N` to parallelise (the Dockerfile bakes 3; each worker is one in-flight call, so it's also the rate-limit blast radius).

> **Before submitting:** the judging harness injects **only** `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` — no `ROUTER_*` variables exist on judging day. Any tuning proven out via env vars must be baked into `_DEFAULTS` in `agent/config.py` (or an `ENV` line in the Dockerfile) before the image is built, or it silently won't apply.

## Status

Container, routing, model/prompt wiring, and the eval harness all run and are covered by `tests/` (42 passing). Still open, and requiring a live Fireworks key + human judgment (not something the code can self-certify):

- **R2** — measure real accuracy and token cost per category via `python -m eval.score`, then drive tokens down (the Phase 4 squeeze). The prompts and budgets in `agent/config.py` are reasonable starting points, not tuned numbers.
- **R3** — expand `eval/eval_set.json` beyond the 2–3 starter tasks per category with harder rewordings, then own the go/no-go sign-off.
- **Docker build** — unverified locally (no daemon available when this was built); run the build command above on a clean machine before submitting.

See `AMD_Track1_Interactive_Guide.html` for the full phase-by-phase plan and the pre-submission checklist.
