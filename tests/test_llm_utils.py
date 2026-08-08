"""Ladder behaviour for _BackoffLLM.

The ladder is Groq 70b -> Groq gpt-oss-120b -> Gemini. Every test here pins a
failure mode that previously ended the ladder early and lost the Gemini
fallback.
"""

import threading

import pytest

from utils.llm import GROQ_MODELS


class FakeAPIError(Exception):
    """Stands in for groq.APIStatusError / NotFoundError (has status_code)."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def rate_limit_error():
    return FakeAPIError(
        "Error code: 429 - {'error': {'message': 'Rate limit reached', "
        "'code': 'rate_limit_exceeded'}}", 429)


def too_large_error():
    """A 413. Note the body says rate_limit_exceeded - that is what made the
    old classifier mistake it for a plain 429 and step down a tier."""
    return FakeAPIError(
        "Error code: 413 - {'error': {'message': 'Request too large for model "
        "`llama-3.3-70b-versatile` on tokens per minute (TPM): Limit 12000, "
        "Requested 17212, please reduce your message size and try again.', "
        "'type': 'tokens', 'code': 'rate_limit_exceeded'}}", 413)


def model_not_found_error(model):
    return FakeAPIError(
        f"Error code: 404 - {{'error': {{'message': 'The model `{model}` does "
        "not exist or you do not have access to it.', "
        "'code': 'model_not_found'}}}}", 404)


def tool_use_failed_error():
    return FakeAPIError(
        "Error code: 400 - {'error': {'message': 'Failed to call a function. "
        "Please adjust your prompt. See failed_generation for more details.', "
        "'code': 'tool_use_failed'}}", 400)


class FakeModel:
    """Stands in for ChatGroq / ChatGoogleGenerativeAI."""

    def __init__(self, name, error=None, error_times=0):
        self.name = name
        self.error = error
        self.error_times = error_times
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, _input, **kwargs):
        self.calls += 1
        if self.error is not None and self.calls <= self.error_times:
            raise self.error
        return f"{self.name}-answer"


@pytest.fixture
def ladder(monkeypatch):
    """Returns (tier1, tier2, gemini) fakes wired into utils.llm."""
    tier1 = FakeModel("groq-70b")
    tier2 = FakeModel("groq-120b")
    gemini = FakeModel("gemini")
    registry = {GROQ_MODELS[0]: tier1, GROQ_MODELS[1]: tier2}
    monkeypatch.setattr("utils.llm._groq",
                        lambda model=GROQ_MODELS[0]: registry[model])
    monkeypatch.setattr("utils.llm._gemini", lambda: gemini)
    return tier1, tier2, gemini


def test_groq_models_are_ids_this_key_can_reach():
    """llama-4-scout 404s on the free tier - a dead tier-2 id used to raise
    that 404 as a hard error and Gemini was never reached."""
    assert "meta-llama/llama-4-scout-17b-16e-instruct" not in GROQ_MODELS
    assert GROQ_MODELS[0] == "llama-3.3-70b-versatile"
    # 8b-instant has the smallest bucket (6k TPM) and is already spent on the
    # guardrail judge and chat memory; it must not be an agent tier.
    assert "llama-3.1-8b-instant" not in GROQ_MODELS


def test_happy_path_uses_first_groq_tier(ladder):
    from utils.llm import get_llm
    tier1, tier2, gemini = ladder
    assert get_llm().invoke("hi") == "groq-70b-answer"
    assert (tier2.calls, gemini.calls) == (0, 0)


def test_rate_limit_steps_to_second_groq_tier(ladder):
    from utils.llm import get_llm
    tier1, tier2, gemini = ladder
    tier1.error, tier1.error_times = rate_limit_error(), 1
    assert get_llm().invoke("hi") == "groq-120b-answer"
    assert gemini.calls == 0


def test_dead_tier_does_not_end_the_ladder(ladder):
    """Regression: a 404 on tier 2 is not a rate limit and not a tool-use
    failure, so the old code broke out and raised it. Gemini never ran."""
    from utils.llm import get_llm
    tier1, tier2, gemini = ladder
    tier1.error, tier1.error_times = rate_limit_error(), 1
    tier2.error, tier2.error_times = model_not_found_error(GROQ_MODELS[1]), 1
    assert get_llm().invoke("hi") == "gemini-answer"


def test_oversized_request_skips_groq_and_goes_straight_to_gemini(ladder):
    """Regression: a 413 means the prompt cannot fit ANY Groq bucket (12k on
    70b, 8k on gpt-oss-120b), so stepping down a tier is guaranteed to fail
    again. Only Gemini has the context window."""
    from utils.llm import get_llm
    tier1, tier2, gemini = ladder
    tier1.error, tier1.error_times = too_large_error(), 1
    assert get_llm().invoke("hi") == "gemini-answer"
    assert tier2.calls == 0


def test_tool_use_failed_retries_once_then_changes_model(ladder):
    """temperature=0 means an identical retry gives an identical failure, so
    one retry is the cap before switching models."""
    from utils.llm import get_llm
    tier1, tier2, gemini = ladder
    tier1.error, tier1.error_times = tool_use_failed_error(), 99
    assert get_llm().invoke("hi") == "groq-120b-answer"
    assert tier1.calls == 2


def test_local_error_raises_without_burning_the_ladder(ladder):
    """A schema/validation error is not provider-side - it fails identically
    everywhere, so retrying it wastes quota."""
    from utils.llm import get_llm
    tier1, tier2, gemini = ladder

    def boom(_input, **kwargs):
        tier1.calls += 1
        raise ValueError("schema mismatch")

    tier1.invoke = boom
    with pytest.raises(ValueError):
        get_llm().invoke("hi")
    assert (tier1.calls, tier2.calls, gemini.calls) == (1, 0, 0)


def test_raises_last_error_when_every_tier_fails(ladder):
    from utils.llm import get_llm
    tier1, tier2, gemini = ladder
    for m in (tier1, tier2, gemini):
        m.error, m.error_times = rate_limit_error(), 99
    with pytest.raises(FakeAPIError):
        get_llm().invoke("hi")
    assert gemini.calls == 1


def test_threads_share_no_state(monkeypatch):
    """One branch's fallback must not downgrade another branch's provider."""
    from utils.llm import get_llm
    gemini = FakeModel("gemini")
    monkeypatch.setattr("utils.llm._gemini", lambda: gemini)
    results = {}

    def call(name, tier1, tier2):
        registry = {GROQ_MODELS[0]: tier1, GROQ_MODELS[1]: tier2}
        monkeypatch.setattr("utils.llm._groq",
                            lambda model=GROQ_MODELS[0]: registry[model])
        results[name] = get_llm().invoke("hi")

    limited = FakeModel("groq-limited", rate_limit_error(), 99)
    call("limited", limited, FakeModel("groq-120b", rate_limit_error(), 99))
    call("healthy", FakeModel("groq-healthy"), FakeModel("groq-120b"))
    assert results["limited"] == "gemini-answer"
    assert results["healthy"] == "groq-healthy-answer"

    t = threading.Thread(target=call,
                         args=("threaded", FakeModel("groq-t"), FakeModel("t2")))
    t.start()
    t.join()
    assert results["threaded"] == "groq-t-answer"
