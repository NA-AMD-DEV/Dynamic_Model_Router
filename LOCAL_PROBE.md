# Local-inference feasibility probe

**Question this answers:** can a model bundled *inside the container* (0 Fireworks
tokens — local inference counts as zero) clear the 85% accuracy gate on the 8
categories **within the time budget on the judging hardware**?

Judging box: **2 vCPU, 4 GB RAM**, image ≤ 10 GB compressed. Per request ≤ 30 s,
whole run ≤ 10 min. On CPU, local inference is the binding constraint — a bigger
model is more accurate but risks TIMEOUT. This probe measures both accuracy and
per-task latency so the decision is data, not a guess.

## Why this is even worth testing

The leaderboard #1 scores **0 tokens at 94.7%** — only possible by running a
local model and calling Fireworks ~never. Our solvers already do this for
math/logic (0 tokens). A local LLM extends it to the other six categories. The
open question is purely whether a model small enough for 2 vCPU / 4 GB is
*accurate enough* and *fast enough*.

## RAM math (why the model must be small)

4 GB total − OS/Python (~0.8 GB) − llama.cpp KV cache at n_ctx 2048 (~0.3 GB)
leaves **~2.5–3 GB for weights**. That caps you at roughly a **3B model at
q4_k_m** (~2 GB). 7B+ will OOM. Start small.

## Candidate models (GGUF, q4_k_m)

| Model | ~size | Notes |
|---|---|---|
| Qwen2.5-1.5B-Instruct | ~1.0 GB | Fastest; the reference 0-token teams used ~this. Accuracy risk on code. |
| **Qwen2.5-3B-Instruct** | ~2.0 GB | Best speed/accuracy balance to start with. |
| Llama-3.2-3B-Instruct | ~2.0 GB | Strong instruct, non-reasoning (no thinking tokens). |
| Qwen2.5-Coder-3B-Instruct | ~2.0 GB | If code categories fail on a general model. |
| Gemma-2-2B-it | ~1.7 GB | Small, solid language quality. |

Solvers already zero math/logic, so the local model only needs
factual/sentiment/summarisation/NER (easy) + code (the hard part). If code fails
locally, the fallback is a **hybrid**: local for the easy five, Fireworks
(`kimi-k2p7-code`) for code only.

## Run it

```bash
pip install llama-cpp-python            # probe-only dep; NOT in the image
# download a GGUF, e.g. via huggingface-cli, to ./models/qwen2.5-3b-instruct-q4_k_m.gguf

export LOCAL_MODEL_PATH="./models/qwen2.5-3b-instruct-q4_k_m.gguf"
export LOCAL_N_THREADS=2                 # SIMULATE the 2-vCPU judging box
export LOCAL_N_CTX=2048
export JUDGE_MODEL="accounts/fireworks/models/minimax-m3"   # judge still uses Fireworks (free)

python -m eval.score
```

`eval.score` runs the whole pipeline (routing + solvers + prompts) on the local
model and now prints an `s/task` column plus **wall time** and **slowest task**
against the 30 s / 10 min limits.

## Pass / fail bars

- **Accuracy** ≥ 90% proxy (85% gate + margin), and no category far below.
- **Slowest task** < 30 s (hard per-request limit).
- **Total wall time** comfortably < 10 min *at the real task count* — and remember
  your dev CPU is likely faster per core than the judging box, so leave headroom.

If it clears all three → local-first is viable (target ~0 tokens). If it clears
accuracy but not time → smaller model or hybrid. If it misses accuracy on code
only → hybrid (local easy + Fireworks code). If it misses accuracy broadly →
stay Fireworks-tiered.

> The probe is **off by default** (`LOCAL_MODEL_PATH` unset). The production image
> stays Fireworks-only and tiny — `llama-cpp-python` is lazy-imported and never
> installed unless you run this probe.
