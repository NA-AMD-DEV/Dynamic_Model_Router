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

Seven levers, roughly in order of impact:

| Lever | What it means |
|---|---|
| Model choice | Smallest allowed model that still passes each category |
| `max_tokens` cap | Hard output ceiling per task type |
| System prompt length | One ruthlessly short shared prefix, billed on every call |
| No printed reasoning | Reasoning tokens count — get correctness without printing the working |
| Answer-only output | No preamble, no restating the question |
| Single call per task | No self-critique loops unless a category needs them |
| Category routing | Cheap categories (sentiment, NER) get tiny configs; hard ones get headroom |

Category routing uses a keyword heuristic rather than a model call — a classifier call would spend tokens to save tokens.

## Runtime limits

The judging VM enforces these; violating any one scores zero.

- Image built for `linux/amd64`, public, pulls with no login, ≤ 10 GB compressed
- Ready in under 60 s; each request under 30 s; whole run under 10 min
- Exit 0 on success; valid `results.json` even when tasks fail
- All responses in English

## Layout

Three vertical slices, one owner each, meeting only at `answer_task`:

- **Container & harness** — Dockerfile, entrypoint, the `/input` → `/output` loop, error handling, routing
- **Model & prompt** — the Fireworks client, per-category prompts and caps, the tuning loop
- **Eval & QA** — a local harness mirroring the real one, a hand-built eval set across all 8 categories, a local LLM judge, and final go/no-go

## Status

Scaffolding not yet committed. See `AMD_Track1_Interactive_Guide.html` for the full phase-by-phase plan, per-role task breakdown, and the pre-submission checklist.
