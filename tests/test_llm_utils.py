"""Ladder behaviour for _BackoffLLM.

The ladder is Groq gpt-oss-20b -> Gemini. Every test here pins a failure mode
that previously ended the ladder early and lost the Gemini fallback.
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
    """Returns (groq, gemini) fakes wired into utils.llm."""
    groq = FakeModel("groq-20b")
    gemini = FakeModel("gemini")
    monkeypatch.setattr("utils.llm._groq", lambda model=GROQ_MODELS[0]: groq)
    monkeypatch.setattr("utils.llm._gemini", lambda: gemini)
    return groq, gemini


def test_groq_models_are_ids_this_key_can_reach():
    """Every id here must be live on the free tier - a dead id used to raise
    its 404 as a hard error and Gemini was never reached."""
    assert GROQ_MODELS == ("openai/gpt-oss-20b",)
    # 120b answered structured requests in prose instead of calling the
    # forced tool and 400d every time; it must not come back as a tier.
    assert "openai/gpt-oss-120b" not in GROQ_MODELS


def test_happy_path_uses_the_groq_tier(ladder):
    from utils.llm import get_llm
    groq, gemini = ladder
    assert get_llm().invoke("hi") == "groq-20b-answer"
    assert gemini.calls == 0


def test_rate_limit_steps_to_gemini(ladder):
    from utils.llm import get_llm
    groq, gemini = ladder
    groq.error, groq.error_times = rate_limit_error(), 1
    assert get_llm().invoke("hi") == "gemini-answer"


def test_dead_tier_does_not_end_the_ladder(ladder):
    """Regression: a 404 is not a rate limit and not a tool-use failure, so
    the old code broke out and raised it. Gemini never ran."""
    from utils.llm import get_llm
    groq, gemini = ladder
    groq.error, groq.error_times = model_not_found_error(GROQ_MODELS[0]), 1
    assert get_llm().invoke("hi") == "gemini-answer"


def test_oversized_request_skips_groq_and_goes_straight_to_gemini(ladder):
    """Regression: a 413 means the prompt cannot fit the 8k Groq bucket, and
    only Gemini has the context window."""
    from utils.llm import get_llm
    groq, gemini = ladder
    groq.error, groq.error_times = too_large_error(), 1
    assert get_llm().invoke("hi") == "gemini-answer"
    assert groq.calls == 1


def test_tool_use_failed_retries_once_then_changes_model(ladder):
    """temperature=0 means an identical retry gives an identical failure, so
    one retry is the cap before switching models."""
    from utils.llm import get_llm
    groq, gemini = ladder
    groq.error, groq.error_times = tool_use_failed_error(), 99
    assert get_llm().invoke("hi") == "gemini-answer"
    assert groq.calls == 2


def test_local_error_raises_without_burning_the_ladder(ladder):
    """A schema/validation error is not provider-side - it fails identically
    everywhere, so retrying it wastes quota."""
    from utils.llm import get_llm
    groq, gemini = ladder

    def boom(_input, **kwargs):
        groq.calls += 1
        raise ValueError("schema mismatch")

    groq.invoke = boom
    with pytest.raises(ValueError):
        get_llm().invoke("hi")
    assert (groq.calls, gemini.calls) == (1, 0)


def test_raises_last_error_when_every_tier_fails(ladder):
    from utils.llm import get_llm
    groq, gemini = ladder
    for m in (groq, gemini):
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

    def call(name, groq):
        monkeypatch.setattr("utils.llm._groq", lambda model=GROQ_MODELS[0]: groq)
        results[name] = get_llm().invoke("hi")

    call("limited", FakeModel("groq-limited", rate_limit_error(), 99))
    call("healthy", FakeModel("groq-healthy"))
    assert results["limited"] == "gemini-answer"
    assert results["healthy"] == "groq-healthy-answer"

    t = threading.Thread(target=call, args=("threaded", FakeModel("groq-t")))
    t.start()
    t.join()
    assert results["threaded"] == "groq-t-answer"
