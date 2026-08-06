import json
import math

import numpy as np
import pytest

from utils.helpers import scrub_nan


def test_scrub_nan_replaces_nan_and_inf_with_none():
    assert scrub_nan(float("nan")) is None
    assert scrub_nan(float("inf")) is None
    assert scrub_nan(float("-inf")) is None
    assert scrub_nan(19.2) == 19.2


def test_scrub_nan_handles_numpy_floats():
    """Price/ratio data arrives from pandas as numpy scalars."""
    assert scrub_nan(np.float64("nan")) is None
    assert scrub_nan(np.float32("nan")) is None
    assert scrub_nan(np.float64(12.5)) == 12.5


def test_scrub_nan_leaves_non_numbers_alone():
    assert scrub_nan(None) is None
    assert scrub_nan("NaN") == "NaN"
    assert scrub_nan(True) is True
    assert scrub_nan(7) == 7


def test_scrub_nan_recurses_through_dicts_and_lists():
    dirty = {"snap": {"price": float("nan"), "high": 3737.7},
             "rows": [{"pe": float("nan")}, {"pe": 20.3}]}
    assert scrub_nan(dirty) == {"snap": {"price": None, "high": 3737.7},
                                "rows": [{"pe": None}, {"pe": 20.3}]}


def test_scrubbed_payload_is_strict_json():
    """json.dumps writes float('nan') as a bare `NaN` token, which is not
    valid JSON. An LLM copies that into its tool-call arguments and the
    provider rejects the whole generation (Groq: 400 tool_use_failed)."""
    dirty = {"price": float("nan")}
    assert "NaN" in json.dumps(dirty)                      # the bug, unscrubbed
    with pytest.raises(ValueError):
        json.dumps(dirty, allow_nan=False)
    assert json.dumps(scrub_nan(dirty), allow_nan=False) == '{"price": null}'


def test_scrub_nan_does_not_mutate_input():
    original = {"price": float("nan")}
    scrub_nan(original)
    assert math.isnan(original["price"])
