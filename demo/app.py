"""Streamlit demo for Dynamic Model Router (AMD Hackathon Act II, Track 1).

This is a DEMO/PRESENTATION wrapper only -- it is not part of the judged
container. It imports the exact same `answer_task_detailed` seam the real
Docker entrypoint (agent/main.py) uses, so what you see here is genuinely
the same routing/solving/calling pipeline, not a reimplementation.

Deploy: Streamlit Community Cloud, main file path = demo/app.py. Add
FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS (and optionally
JUDGE_MODEL) under the app's Settings -> Secrets, TOML format:

    FIREWORKS_API_KEY = "..."
    FIREWORKS_BASE_URL = "..."
    ALLOWED_MODELS = "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code"
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

# Make the repo root importable (this file lives in demo/, the `agent`
# package lives at the repo root) without needing an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Streamlit secrets live in st.secrets, but every agent/* module reads
# credentials from os.environ (that's the real container's contract too) --
# bridge them BEFORE importing agent.core, since config resolution happens
# at call time but the client needs the key/base_url present either way.
#
# os.environ only accepts strings. A secret entered as a TOML array (e.g.
# ALLOWED_MODELS = ["model-a", "model-b"]) comes back from st.secrets as a
# list, and assigning a list into os.environ crashes the whole app with an
# opaque TypeError inside os.encode. Coerce defensively: join sequences with
# a comma (our own parser already accepts comma-separated), str() anything
# else, so a formatting slip in Secrets degrades, never crashes the page.
for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS", "JUDGE_MODEL"):
    if key in st.secrets and key not in os.environ:
        value = st.secrets[key]
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        os.environ[key] = str(value)

from agent.config import calibrate_lean, CATEGORIES  # noqa: E402
from agent.core import answer_task_detailed  # noqa: E402

st.set_page_config(page_title="Dynamic Model Router", page_icon="🔀", layout="centered")

EXAMPLES = {
    "Factual (explains)": "What is the capital of Australia, and briefly explain its history as the capital.",
    "Math (multi-step)": "A store buys a jacket for $60 and marks the price up by 40%. During a sale, the marked price is discounted by 25%. What is the sale price?",
    "Sentiment (mixed)": "Classify the sentiment of this review as Positive, Negative, or Neutral and give a one-sentence reason: 'The product arrived two days late, but the item worked perfectly and support was great.'",
    "Summarisation (exact format)": "Summarize the following in exactly two sentences: 'Remote learning expanded access to education for isolated students. At the same time, educators reported lower engagement, and students without reliable internet fell behind.'",
    "Named entities": "Extract all named entities from the following text and label each as PERSON, ORGANIZATION, LOCATION, or DATE: 'On March 15 2023, Sundar Pichai announced Google would open a lab in Zurich.'",
    "Code debugging": "This function should return the average but crashes on an empty list. Fix it: def avg(xs): return sum(xs)/len(xs)",
    "Logical reasoning": "If Alice is taller than Bob, and Bob is taller than Carol, who is the shortest?",
    "Code generation": "Write a Python function that returns True if a string is a palindrome.",
}

st.title("🔀 Dynamic Model Router")
st.caption(
    "AMD Developer Hackathon · Act II · Track 1 — a token-efficient routing agent. "
    "Every answer below runs through the exact same code as the judged Docker container."
)

with st.expander("How this works", expanded=False):
    st.markdown(
        "1. **Classify** — a zero-token keyword heuristic infers the task category.\n"
        "2. **Solve free** — trivial single-step math/logic is answered deterministically "
        "in Python at 0 tokens; anything ambiguous defers to the model.\n"
        "3. **Route lean** — at startup, every live Fireworks model is probed once to "
        "measure its real per-request overhead, and each category routes to the "
        "cheapest model that still clears the accuracy gate (measured, never guessed).\n"
        "4. **Self-heal** — a model listed as allowed but not actually deployed is "
        "detected and routed around automatically, mid-run."
    )

if "FIREWORKS_API_KEY" not in os.environ or "ALLOWED_MODELS" not in os.environ:
    st.warning(
        "No Fireworks credentials configured for this demo instance -- answers will "
        "come back empty (this is the same graceful-degradation path the real "
        "container uses when misconfigured, not a crash). Add FIREWORKS_API_KEY, "
        "FIREWORKS_BASE_URL, and ALLOWED_MODELS under Settings → Secrets to enable "
        "real answers.",
        icon="⚠️",
    )
else:
    with st.spinner("Warming up: probing live models for per-request cost…"):
        try:
            calibrate_lean()
        except Exception as exc:  # calibration must never block the demo
            st.info(f"Calibration skipped: {exc!r}")

st.subheader("Try it")
cols = st.columns(4)
picked = None
for i, (label, prompt) in enumerate(EXAMPLES.items()):
    if cols[i % 4].button(label, use_container_width=True):
        picked = prompt

prompt = st.text_area(
    "Prompt (or click an example above)",
    value=picked or st.session_state.get("prompt", ""),
    height=100,
    key="prompt",
)

if st.button("Route & answer", type="primary", disabled=not prompt.strip()):
    t0 = time.monotonic()
    with st.spinner("Routing…"):
        detail = answer_task_detailed({"task_id": "demo", "prompt": prompt})
    elapsed = time.monotonic() - t0

    c1, c2, c3 = st.columns(3)
    c1.metric("Category", detail["category"])
    c2.metric("Tokens", detail["tokens"])
    c3.metric("Latency", f"{elapsed:.1f}s")

    model = detail.get("model") or "(0-token solver — no model called)"
    st.caption(f"Model: `{model}`")

    st.text_area("Answer", value=detail["answer"], height=140, disabled=True)

    if detail["error"]:
        st.error(f"Agent error: {detail['error']}")

st.divider()
st.caption(
    f"{len(CATEGORIES)} categories · full source, eval harness, and Docker image: "
    "see the GitHub repo linked in this submission."
)
