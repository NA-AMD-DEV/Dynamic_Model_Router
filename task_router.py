"""
Frozen interface between R1 (container/harness) and R2 (model/prompt logic).
R1 calls answer_task() for every task. This is the ONLY function R1 needs to know about.
"""

from fireworks_client import call_model
from category_configs import CATEGORY_CONFIGS, DEFAULT_CONFIG


def answer_task(task_id: str, prompt: str, category: str = None) -> dict:
    """
    category: one of the 8 category keys in CATEGORY_CONFIGS, or None/unknown.
    Returns: {"task_id": ..., "answer": ..., "tokens": ..., "error": ...}
    """
    config = CATEGORY_CONFIGS.get(category, DEFAULT_CONFIG)

    result = call_model(
        prompt=prompt,
        system_prompt=config["system_prompt"],
        model=config["model"],
        max_tokens=config["max_tokens"],
    )

    return {
        "task_id": task_id,
        "answer": result["answer"],
        "tokens": result["tokens"],
        "error": result["error"],
    }


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity check the wiring
    test = answer_task(
    task_id="t1",
    prompt="I love this product, it works great!",
    category="sentiment_classification",
)
    print(test)
