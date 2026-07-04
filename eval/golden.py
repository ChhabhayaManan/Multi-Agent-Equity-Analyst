"""Golden-set loader for the chatbot RAGAS evaluation."""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenItem:
    type: str
    question: str
    ground_truth: str


def load_golden_set(path: str = "data/golden_set.json"):
    """Return (ticker, [GoldenItem, ...]). Raises ValueError on bad data."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    ticker = data.get("ticker")
    questions = data.get("questions")
    if not ticker or not isinstance(questions, list) or not questions:
        raise ValueError("golden set needs a ticker and a non-empty questions list")
    items = []
    for q in questions:
        try:
            items.append(GoldenItem(q["type"], q["question"], q["ground_truth"]))
        except (KeyError, TypeError) as e:
            raise ValueError(f"malformed golden question: {q!r}") from e
    return ticker, items
