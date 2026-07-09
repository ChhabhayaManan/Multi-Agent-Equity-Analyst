import pytest

from templates.schemas.outputs import CompetitorOutput, PeerComparison


def _canned(metrics_first=None):
    def peer(tk, metrics):
        return PeerComparison(ticker=tk, name=tk.split(".")[0].title(),
                              reason_for_inclusion="overlapping retail banking business",
                              competition_intensity="STRONG",
                              target_standing="INLINE", metrics=metrics)

    full = {"pe": 17.0, "roe": 15.0, "ret_1m": 2.0}
    return CompetitorOutput(
        peers=[peer("ICICIBANK.NS", metrics_first if metrics_first is not None else full),
               peer("AXISBANK.NS", full), peer("KOTAKBANK.NS", full)],
        comparison_summary="The target trades at a premium to all three peers "
                           "while matching their recent returns.",
        overall_standing="AHEAD")


def _patch(monkeypatch, output):
    from agents import competitor_intelligence_agent as mod
    calls = {"system": None, "agent_input": None, "stored": None}

    class FakeAgent:
        def invoke(self, payload):
            calls["agent_input"] = payload
            return {"messages": [], "structured_response": output}

    def fake_build(system):
        calls["system"] = system
        return FakeAgent()

    monkeypatch.setattr(mod, "_build_agent", fake_build)
    monkeypatch.setattr(mod, "store_to_pinecone",
                        lambda ticker, docs, st, meta=None: calls.__setitem__("stored", st))
    return mod, calls


@pytest.fixture
def patched(monkeypatch):
    return _patch(monkeypatch, _canned())


def test_run_produces_structured_output(patched):
    mod, calls = patched
    out, fetch_count = mod.run("HDFCBANK.NS", "HDFC Bank Ltd")
    assert isinstance(out, CompetitorOutput)
    assert fetch_count == 3
    assert calls["stored"] == "competitor"
    assert "HDFCBANK.NS" in calls["system"] + str(calls["agent_input"])


def test_fetch_count_ignores_metricless_peers(monkeypatch):
    mod, _ = _patch(monkeypatch, _canned(metrics_first={"pe": None, "roe": None}))
    _, fetch_count = mod.run("HDFCBANK.NS", "HDFC Bank Ltd")
    assert fetch_count == 2


def test_retry_feedback_reaches_agent_task(patched):
    mod, calls = patched
    mod.run("HDFCBANK.NS", "HDFC Bank Ltd", retry_feedback="verify peer market caps")
    assert "verify peer market caps" in str(calls["agent_input"])
