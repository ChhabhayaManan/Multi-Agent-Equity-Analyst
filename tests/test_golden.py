# tests/test_golden.py
import json

import pytest

from eval.golden import GoldenItem, load_golden_set


def _write(tmp_path, obj):
    p = tmp_path / "golden.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


def test_load_valid(tmp_path):
    path = _write(tmp_path, {
        "ticker": "HDFCBANK.NS",
        "questions": [
            {"type": "direct-factual", "question": "Q?", "ground_truth": "A."}]})
    ticker, items = load_golden_set(path)
    assert ticker == "HDFCBANK.NS"
    assert items == [GoldenItem("direct-factual", "Q?", "A.")]


def test_load_rejects_empty_questions(tmp_path):
    path = _write(tmp_path, {"ticker": "HDFCBANK.NS", "questions": []})
    with pytest.raises(ValueError):
        load_golden_set(path)


def test_load_rejects_missing_fields(tmp_path):
    path = _write(tmp_path, {
        "ticker": "HDFCBANK.NS",
        "questions": [{"question": "Q?"}]})
    with pytest.raises(ValueError):
        load_golden_set(path)
