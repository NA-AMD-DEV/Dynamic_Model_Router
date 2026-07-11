"""
Per-category configuration: system prompt, max_tokens, model choice.
R1's harness classifies a task into one of these 8 keys, then calls
answer_task(prompt, category) from task_router.py.

NOTE: model names below are placeholders. Swap in real ALLOWED_MODELS
values once revealed on launch day. For now use whatever you can test with.
"""

from fireworks_client import ALLOWED_MODELS

# Convenience handles — adjust indices once real models are known.
# Assume index 0 = smaller/cheaper model, last index = biggest/strongest model.
CHEAP_MODEL = ALLOWED_MODELS[0] if ALLOWED_MODELS else None
STRONG_MODEL = ALLOWED_MODELS[-1] if ALLOWED_MODELS else None

CATEGORY_CONFIGS = {
    "factual_knowledge": {
        "system_prompt": (
            "Answer the question directly and concisely. "
            "No preamble, no restating the question. Answer only."
        ),
        "model": CHEAP_MODEL,
        "max_tokens": 150,
    },
    "math_reasoning": {
        "system_prompt": (
            "Solve the problem step by step internally, but output ONLY the final "
            "numeric or short answer. Do not show your work. Answer only."
        ),
        "model": STRONG_MODEL,
        "max_tokens": 200,
    },
    "sentiment_classification": {
        "system_prompt": (
            "Classify the sentiment as positive, negative, or neutral. "
            "Reply in the format: label: <sentiment> | reason: <one short sentence>. "
            "Nothing else."
        ),
        "model": CHEAP_MODEL,
        "max_tokens": 100,
    },
    "summarisation": {
        "system_prompt": (
            "Summarise the given text to exactly match the requested length/format "
            "constraint in the prompt. Output ONLY the summary, nothing else."
        ),
        "model": CHEAP_MODEL,
        "max_tokens": 150,
    },
    "named_entity_recognition": {
        "system_prompt": (
            "Extract named entities. Reply ONLY as label:value pairs, one per line "
            "(e.g. PERSON: John Smith). No prose, no explanation."
        ),
        "model": CHEAP_MODEL,
        "max_tokens": 120,
    },
    "code_debugging": {
        "system_prompt": (
            "Find the bug and return ONLY the corrected, complete function in a single "
            "code block. No explanation, no prose before or after."
        ),
        "model": STRONG_MODEL,
        "max_tokens": 400,
    },
    "logical_reasoning": {
        "system_prompt": (
            "Solve the constraint puzzle. Verify all conditions are satisfied internally, "
            "but output ONLY the final answer. No reasoning shown."
        ),
        "model": STRONG_MODEL,
        "max_tokens": 200,
    },
    "code_generation": {
        "system_prompt": (
            "Write the function exactly as specified. Return ONLY a single code block "
            "with the complete, correct implementation. No explanation."
        ),
        "model": STRONG_MODEL,
        "max_tokens": 400,
    },
}

# Safety default for anything unclassified (R1 routes here if unsure)
DEFAULT_CONFIG = {
    "system_prompt": "Answer the prompt directly and concisely. Answer only.",
    "model": CHEAP_MODEL,
    "max_tokens": 200,
}
