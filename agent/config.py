"""Per-category model, prompt, and token budget.

R2 owns the defaults in _DEFAULTS. Two ways to change them, in precedence order:

  1. Env override, no rebuild:  ROUTER_<CATEGORY>_MODEL, ROUTER_<CATEGORY>_MAX_TOKENS
  2. Edit _DEFAULTS and rerun.

Nothing is resolved at import. `config_for` reads the environment on every call,
so the harness can inject ALLOWED_MODELS after this module loads, and R2 can
change a model between runs without touching code or rebuilding the image.

Model IDs come from ALLOWED_MODELS at runtime, never from a literal here.
"""

import os
import re
import sys
import threading
from dataclasses import dataclass

DEFAULT_CATEGORY = "factual_knowledge"


@dataclass(frozen=True)
class Config:
    model: str
    system: str
    max_tokens: int
    reasoning: str | None = None  # per-category reasoning_effort; None = global default


@dataclass(frozen=True)
class _Default:
    """A config before the environment is consulted.

    `model_index` selects from ALLOWED_MODELS rather than naming a model, since
    the real IDs aren't published until launch day. 0 is the cheapest/first.

    `tier` ("small" | "medium" | "large" | "") ranks ALLOWED_MODELS by
    capability inferred from the ids themselves (see `pick_capability`) --
    no vendor/model name is ever hardcoded, so this adapts to whatever family
    is actually injected on judging day.
    """
    model_index: int
    system: str
    max_tokens: int
    tier: str = ""
    reasoning: str | None = None  # None -> global REASONING_EFFORT (usually "none")
    specialist: str = ""          # e.g. "code": prefer a code-tuned model if one exists
    lean_ok: bool = True          # False: measurement showed the lean model fails
    #                               this category's rubric -- stay on the
    #                               capability-ranked pick instead


def allowed_models() -> list[str]:
    """Parse ALLOWED_MODELS. Accepts comma-, semicolon-, or newline-separated."""
    raw = os.environ.get("ALLOWED_MODELS", "")
    parts = [p.strip() for p in raw.replace(";", ",").replace("\n", ",").split(",")]
    return [p for p in parts if p]


# Models the proxy reported as not-found / not-deployed THIS RUN. ALLOWED_MODELS
# may list ids that aren't actually served on the serverless instance (e.g. a
# model that needs a dedicated deployment); calling one 404s and fails the task.
# call_model records them here on a 404 so routing never picks them again --
# a runtime-learned filter, no model name ever hardcoded.
#
# Thread-safety: workers read this via available_models() while another worker's
# mark_unavailable() mutates it.  A bare set iteration + mutation = RuntimeError
# under concurrency.  The lock serialises both; available_models snapshots under
# the lock so iteration is always over a private copy.
_UNAVAILABLE: set[str] = set()
_UNAVAILABLE_LOCK = threading.Lock()


def mark_unavailable(model: str) -> None:
    if model:
        with _UNAVAILABLE_LOCK:
            _UNAVAILABLE.add(model)


def available_models() -> list[str]:
    """ALLOWED_MODELS minus any learned to be undeployed this run. Falls back to
    the full list if every model has been marked (better to retry a maybe-flaky
    one than route to nothing)."""
    with _UNAVAILABLE_LOCK:
        unavail = set(_UNAVAILABLE)          # snapshot under lock
    live = [m for m in allowed_models() if m not in unavail]
    return live or allowed_models()


def pick_model(preference: int = 0) -> str:
    """Choose from ALLOWED_MODELS by index, clamped to the last entry.

    Returns "" when ALLOWED_MODELS is unset (local dev with the stub). R2's
    real client must treat "" as a hard error rather than guess an ID.
    """
    models = available_models()
    if not models:
        return ""
    return models[min(max(preference, 0), len(models) - 1)]


# Parameter count embedded in a model id, e.g. "qwen3-30b-a3b" -> 30,
# "llama-3.3-70b-instruct" -> 70. Deliberately has ZERO vendor/model-name
# knowledge -- the exact models on judging day aren't knowable in advance, so
# capability is inferred from whatever ALLOWED_MODELS actually contains, not
# from a guess about which family will be injected.
_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def _param_billions(model_id: str) -> float | None:
    """Best-effort parameter count in billions parsed from the id, or None
    when no size token is present -- common on flagship/commercial ids that
    don't encode it (e.g. 'kimi-k2p6', 'deepseek-v4-pro', 'minimax-m2').

    A MoE id often lists both an active and a total count ('30b-a3b'): take
    the max, since total parameter count tracks capability more closely than
    active count, and API-billed tokens don't care about the difference.
    """
    matches = _SIZE_RE.findall(model_id)
    return max((float(m) for m in matches), default=None) if matches else None


def _rank_by_capability(models: list[str]) -> list[str]:
    """Order `models` least- to most-capable using ONLY signal in the ids --
    no hardcoded vendor/model names, so this adapts to whatever ALLOWED_MODELS
    contains on judging day, whichever families those turn out to be.

    Sized ids (an 'NNb' token present) sort by that count. Unsized ids are
    placed after every sized one, in their given order: an unsized flagship
    name is more likely to be a substantial production model than a small
    distilled one, so this errs toward not under-ranking it into 'small'.
    This is a heuristic, not a guarantee -- see pick_capability.
    """
    sized = [(m, _param_billions(m)) for m in models]
    # key=... is essential: sorting bare (m, b) tuples would order by the
    # model-id STRING first (alphabetically) and only fall back to `b` on a
    # name tie, silently ignoring the parameter count entirely.
    known = sorted(((m, b) for m, b in sized if b is not None), key=lambda pair: pair[1])
    unknown = [m for m, b in sized if b is None]
    return [m for m, _ in known] + unknown


# A task-specialised model advertises its niche in its id (e.g. "...-coder...",
# "kimi-k2p7-code"). Detected by substring only -- no specific model hardcoded.
# Such a model is RESERVED for its category: a code model shouldn't be the
# general pick for summarisation just because it happens to be the largest.
_SPECIALISMS: dict[str, tuple[str, ...]] = {
    "code": ("code", "coder"),
}


def _specialism_of(model_id: str) -> str | None:
    low = model_id.lower()
    for tag, keys in _SPECIALISMS.items():
        if any(k in low for k in keys):
            return tag
    return None


def pick_specialist(tag: str) -> str:
    """First available model specialised for `tag` (e.g. a code model), or "" if
    none. Lets code categories claim a code-tuned model without naming it."""
    for model in available_models():
        if _specialism_of(model) == tag:
            return model
    return ""


# --- lean calibration -------------------------------------------------------
# The chat-template overhead differs WILDLY between models (measured: ~130-150
# prompt tokens/task on one family vs ~40-56 on another, ~3x) and prompt tokens
# dominate the bill. Which model is leanest can't be known from names, so we
# MEASURE it: once at startup, one minimal probe per live model records its
# fixed template cost, and general categories then route to the cheapest.
#
# Runs ONCE, single-threaded, from the entrypoint BEFORE any worker exists --
# workers only ever read the result, so no lock is needed. Probe tokens are
# real judged tokens: they accumulate in _calibration_tokens and eval.score
# adds them to the reported total (never optimize against a fake number).
_lean_ranking: list[str] | None = None
_calibration_tokens = 0


def calibration_tokens() -> int:
    return _calibration_tokens


def _error_says_unavailable(error: str) -> bool:
    low = error.lower()
    return "404" in error or "not_found" in low or "not deployed" in low or "inaccessible" in low


def calibrate_lean() -> None:
    """Measure each live model's fixed prompt-template cost and cache the
    ranking (cheapest first). Call once at startup, before any worker thread.

    Skips itself when there's no choice to make (<2 live models) or when local
    inference is enabled (Fireworks probes would be irrelevant spend). A probe
    failure never breaks routing: unprobeable models are just left out, and if
    nothing was probed the lean ranking stays empty and tier resolution falls
    back to capability ranking.
    """
    global _lean_ranking, _calibration_tokens
    if _lean_ranking is not None:  # already calibrated this run
        return
    from agent.local_client import local_enabled  # lazy: avoid import cycle
    if local_enabled():
        return
    models = available_models()
    if len(models) < 2:
        return

    from agent.fireworks_client import call_model  # lazy: avoid import cycle
    costs: list[tuple[int, str]] = []
    for m in list(models):
        # Probe mode: call exactly this model, no failover, no client-side
        # bookkeeping -- a dead model must not get another model's numbers.
        # reasoning_effort is left at the production default deliberately.
        result = call_model("hi", "", m, 1, allow_failover=False)
        _calibration_tokens += result["tokens"]
        if result["error"] is not None:
            if _error_says_unavailable(result["error"]):
                mark_unavailable(m)  # startup is single-threaded: safe here
            continue  # unprobeable: excluded from lean, capability still has it
        costs.append((result["prompt_tokens"], m))
    if costs:
        costs.sort()
        _lean_ranking = [m for _, m in costs]


def pick_lean() -> str:
    """The measured-leanest live model, or "" when calibration hasn't run /
    found nothing (caller falls back to capability ranking)."""
    if _lean_ranking:
        with _UNAVAILABLE_LOCK:
            unavail = set(_UNAVAILABLE)
        for m in _lean_ranking:
            if m not in unavail:
                return m
    return ""


def pick_capability(tier: str) -> str:
    """small | medium | large, chosen from ALLOWED_MODELS purely by relative
    capability signal in the ids themselves. Degrades gracefully as the list
    shrinks: with 2 models 'medium' collapses to the same choice as 'large';
    with 1 model every tier resolves to it. "" when ALLOWED_MODELS is unset.

    Specialist models (e.g. a code model) are excluded from the GENERAL ranking
    unless nothing else is left -- so a general category never lands on a
    code-tuned model just because it parses as the biggest.

    This offline ranking is a starting point for local testing before the
    real ALLOWED_MODELS is known, not a guarantee of correctness -- pure id
    inspection can't fully order arbitrary flagship names (see the module
    docstring). Once the real list is revealed, MEASURE with
    `python -m eval.score` and pin the winner via ROUTER_<CATEGORY>_MODEL.
    """
    models = available_models()
    if not models:
        return ""
    general = [m for m in models if _specialism_of(m) is None] or models
    ranked = _rank_by_capability(general)
    n = len(ranked)
    if tier == "small":
        return ranked[0]
    if tier == "large":
        return ranked[-1]
    return ranked[n // 2]  # medium: biases toward 'large' as the list shrinks


def _env_key(category: str, field: str) -> str:
    return f"ROUTER_{category.upper()}_{field}"


def _override_int(category: str, field: str, fallback: int) -> int:
    """Read an int override, ignoring anything unparseable.

    A typo'd override must not crash the run: the harness gives us one shot,
    and a bad env var should degrade to the tuned default, loudly.
    """
    raw = os.environ.get(_env_key(category, field))
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError:
        print(
            f"ignoring {_env_key(category, field)}={raw!r}: not an integer",
            file=sys.stderr,
        )
        return fallback


# R2: tune these. The 8 keys are the official hackathon categories — keep them
# in sync with agent/routing.py's PRIORITY list.
#
# `tier` is the INITIAL routing guess to measure against, not the final
# answer -- small/medium/large is resolved from ALLOWED_MODELS at call time
# by pick_capability (no vendor name is ever hardcoded). Rough split:
#   small  -- trivial asks: sentiment, NER, plain factual recall
#   medium -- moderate generation/summarisation work
#   large  -- code correctness, and the math/logic residue agent/solvers.py
#             couldn't answer at 0 tokens (that residue is BY DEFINITION the
#             hard/ambiguous case -- it deserves the strongest model, not a
#             mid-tier one, even though "logical reasoning" in the abstract
#             might otherwise be a medium-tier task)
# Once MEASURED (via `python -m eval.score` against the real injected list),
# bake the fewest-tokens-that-passes model per category here or via
# ROUTER_<CATEGORY>_MODEL=<exact id> / ROUTER_<CATEGORY>_MODEL_INDEX=n.
#
# Token budgets DO differ by category — but the official sample tasks proved
# the real answers are FULLER than our first guesses (a two-sided sentiment
# reason, explanations when asked, exact summary formats). The judge scores
# correctness first; a truncated or generic answer fails regardless of tokens.
# Caps are sized so the REQUIRED output is never truncated (the eval's
# truncation counter validates this); prompts demand exactly what the official
# rubric grades, nothing extra.
_DEFAULTS: dict[str, _Default] = {
    "factual_knowledge": _Default(
        0,
        # Real tasks may ask for an explanation ("...and briefly explain why").
        # "Answer only, no preamble" produced the generic answers the official
        # FAQ warns about -- but unbounded explanations ran long and truncated
        # at the cap (4/8 on the eval). The rubric says "briefly": bound it.
        "Answer the question directly and completely, including any explanation "
        "it asks for, in at most 4 sentences. No filler.",
        300,
        tier="small",
    ),
    "math_reasoning": _Default(
        0,
        # Official samples are multi-step ("minor arithmetic shown or implied"
        # is acceptable). Brief working improves correctness on weak models;
        # verbose derivations waste tokens.
        "Solve carefully step by step. Show only brief working, then state the "
        "final answer(s) clearly.",
        300,
        tier="large",  # solver handles the trivial residue; the rest is hard
        # lean_ok defaults True, which would route this residue to the
        # CHEAPEST model -- backwards: a solver defer means the case was hard
        # enough to need the model at all, so it deserves the strongest one,
        # not the leanest. Pin off lean explicitly.
        lean_ok=False,
    ),
    "sentiment_classification": _Default(
        0,
        # Official rubric: label + one-sentence reason; when the text has both
        # good and bad aspects, the reason MUST acknowledge both, and a bare
        # label fails. (The earlier label-only cap-10 config scored 0% on this.)
        "Give the sentiment label the prompt asks for, then a one-sentence "
        "reason. If the text mixes positive and negative aspects, acknowledge "
        "both in the reason and prefer 'mixed' or 'neutral' over one-sided labels.",
        80,
        tier="small",
    ),
    "summarisation": _Default(
        0,
        # Official rubric fails any deviation from the requested format
        # ("exactly two sentences", "exactly three bullets, each <=15 words").
        # MEASURED failure mode (2 tasks, reproducible at temperature=0): the
        # model satisfies the COUNT by cramming units onto one line/sentence
        # instead of genuinely separating them -- "- bullet one. - bullet two."
        # on a single line, or two ideas merged into one sentence with "but".
        # "Obey exactly" alone doesn't tell it HOW; spell out the mechanics.
        "Obey the prompt's format constraint EXACTLY. For bullet points: put "
        "EACH bullet on its own line starting with '- ', one bullet per "
        "requested point, within the word limit. For a sentence count: write "
        "that many separate, complete sentences, each ending in its own "
        "period -- never merge two ideas into one sentence with 'and', 'but', "
        "or a semicolon. Cover the key points from all sides of the text "
        "within that structure. Output ONLY the summary.",
        # Cap 200 truncated verbose summaries mid-generation (measured: 6/8
        # truncated -> 25%). A truncated summary is definitely wrong. 800 gives
        # headroom; unused cap costs nothing (the model stops when done).
        800,
        tier="medium",
        # MEASURED: the lean (code-tuned) model scored 62% here -- it misses
        # exact-format constraints -- while the capability pick scored 100%.
        # +~800 tokens buys back the gate margin. Correctness first.
        lean_ok=False,
    ),
    "named_entity_recognition": _Default(
        0,
        # Official rubric: every entity present, labels exact.
        "Extract ALL named entities. Label each as PERSON, ORGANIZATION, "
        "LOCATION, or DATE (or the labels the prompt requests), one "
        "'LABEL: value' per line. Miss nothing; no prose.",
        150,
        tier="small",
    ),
    "code_debugging": _Default(
        0,
        # No official sample exists for code categories (blind spot): keep the
        # corrected code mandatory, allow a brief why only when asked.
        "Fix the code. Return the corrected, complete code in one code block; "
        "add a brief explanation only if the prompt asks why.",
        400,
        tier="medium",
        specialist="code",
    ),
    "logical_reasoning": _Default(
        0,
        "Reason carefully internally. State the final answer clearly; add a "
        "brief justification only if the prompt asks for one.",
        200,
        tier="large",  # solver handles the trivial residue; the rest is hard
        lean_ok=False,  # see math_reasoning: solver-defer residue is the hard
        # case by construction -- route it to the strongest model, not leanest.
    ),
    "code_generation": _Default(
        0,
        "Write the requested code. Return the complete implementation in one "
        "code block; add a brief explanation only if the prompt asks for one.",
        400,
        tier="medium",
        specialist="code",
    ),
}

CATEGORIES: tuple[str, ...] = tuple(_DEFAULTS)


def config_for(category: str) -> Config:
    """Resolve a category's config against the current environment.

    Unknown categories fall back to the general default, which is deliberately
    the most permissive one.
    """
    spec = _DEFAULTS.get(category)
    if spec is None:
        category = DEFAULT_CATEGORY
        spec = _DEFAULTS[DEFAULT_CATEGORY]

    # Model precedence, most specific first:
    #   1. ROUTER_<CAT>_MODEL       -- an explicit pinned id
    #   2. ROUTER_<CAT>_MODEL_INDEX -- an explicit index into ALLOWED_MODELS
    #   3. the category's specialist -- a code-tuned model for code categories,
    #                                   if the injected list actually has one
    #   4. measured-leanest model   -- when startup calibration ran (D2 showed
    #                                   the leanest model also passes the gate
    #                                   everywhere; specialists are eligible)
    #   5. the category's tier      -- capability-ranked, else model_index
    #   6. model_index default
    model = os.environ.get(_env_key(category, "MODEL"))
    if not model:
        if os.environ.get(_env_key(category, "MODEL_INDEX")) is not None:
            model = pick_model(_override_int(category, "MODEL_INDEX", spec.model_index))
        elif spec.specialist and pick_specialist(spec.specialist):
            model = pick_specialist(spec.specialist)
        elif spec.tier:
            lean = pick_lean() if spec.lean_ok else ""
            model = lean or pick_capability(spec.tier)
        else:
            model = pick_model(spec.model_index)

    # None (unset) keeps the global default; setting it (even to "") overrides.
    reasoning = os.environ.get(_env_key(category, "REASONING_EFFORT"), spec.reasoning)

    return Config(
        model=model,
        system=os.environ.get(_env_key(category, "SYSTEM"), spec.system),
        max_tokens=_override_int(category, "MAX_TOKENS", spec.max_tokens),
        reasoning=reasoning,
    )
