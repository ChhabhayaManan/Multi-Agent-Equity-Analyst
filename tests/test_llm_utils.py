import threading

import pytest


class RateLimitError(Exception):
    def __init__(self):
        super().__init__("429 rate limit exceeded")


class FakeModel:
    """Stands in for ChatGroq/ChatGoogleGenerativeAI."""

    def __init__(self, name, fail_times=0):
        self.name = name
        self.fail_times = fail_times
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, _input, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RateLimitError()
        return f"{self.name}-answer"


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr("utils.llm.time.sleep", lambda s: sleeps.append(s))
    return sleeps


def _patch_models(monkeypatch, groq, gemini):
    monkeypatch.setattr("utils.llm._groq", lambda: groq)
    monkeypatch.setattr("utils.llm._gemini", lambda: gemini)


def test_happy_path_uses_groq(monkeypatch, no_sleep):
    from utils.llm import get_llm
    groq, gemini = FakeModel("groq"), FakeModel("gemini")
    _patch_models(monkeypatch, groq, gemini)
    assert get_llm().invoke("hi") == "groq-answer"
    assert gemini.calls == 0 and no_sleep == []


def test_backoff_schedule_then_recovery(monkeypatch, no_sleep):
    from utils.llm import get_llm
    groq, gemini = FakeModel("groq", fail_times=2), FakeModel("gemini")
    _patch_models(monkeypatch, groq, gemini)
    assert get_llm().invoke("hi") == "groq-answer"
    assert len(no_sleep) == 2
    assert 8 <= no_sleep[0] <= 12    # 10s +/- 2 jitter
    assert 13 <= no_sleep[1] <= 17   # 15s +/- 2 jitter
    assert gemini.calls == 0


def test_falls_back_to_gemini_after_exhaustion(monkeypatch, no_sleep):
    from utils.llm import get_llm
    groq, gemini = FakeModel("groq", fail_times=99), FakeModel("gemini")
    _patch_models(monkeypatch, groq, gemini)
    assert get_llm().invoke("hi") == "gemini-answer"
    assert groq.calls == 4          # initial + 3 backoff retries
    assert len(no_sleep) == 3


def test_non_rate_limit_error_raises(monkeypatch, no_sleep):
    from utils.llm import get_llm

    class Boom(FakeModel):
        def invoke(self, _input, **kwargs):
            raise ValueError("schema mismatch")

    _patch_models(monkeypatch, Boom("groq"), FakeModel("gemini"))
    with pytest.raises(ValueError):
        get_llm().invoke("hi")
    assert no_sleep == []


def test_threads_share_no_state(monkeypatch, no_sleep):
    """One thread's fallback to Gemini must not switch another thread's provider."""
    from utils.llm import get_llm
    rate_limited = FakeModel("groq-limited", fail_times=99)
    healthy = FakeModel("groq-healthy")
    gemini = FakeModel("gemini")
    monkeypatch.setattr("utils.llm._gemini", lambda: gemini)
    results = {}

    def call(name, model):
        monkeypatch.setattr("utils.llm._groq", lambda: model)
        results[name] = get_llm().invoke("hi")

    # Sequential calls (deterministic), then threaded (smoke): either way,
    # provider choice is local to the call.
    call("limited", rate_limited)
    call("healthy", healthy)
    assert results["limited"] == "gemini-answer"
    assert results["healthy"] == "groq-healthy-answer"

    t = threading.Thread(target=call, args=("threaded", FakeModel("groq-t")))
    t.start(); t.join()
    assert results["threaded"] == "groq-t-answer"
