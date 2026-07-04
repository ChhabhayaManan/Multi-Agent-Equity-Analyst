"""Graph behavior tests with all agents faked. No network, no LLM."""

import pytest

from templates.schemas.outputs import (
    CompetitorOutput, DocsOutput, EventOutput, FundamentalsOutput,
    GuidanceItem, NewsItem, NewsOutput, PeerComparison, ReportOutput,
    RiskItem, TimelineEvent)

GOOD = {
    "fundamentals": (FundamentalsOutput(
        company_profile={}, valuation={"pe": 1.0, "pb": 1.0, "roe": 1.0,
                                       "debt_equity": 1.0},
        price_snapshot={"price": 100.0}, shareholding={},
        summary="A perfectly fine plain-language summary of the company and "
                "stock state."), 4),
    "competitor": (CompetitorOutput(
        peers=[PeerComparison(ticker=f"P{i}.NS", name=f"P{i}",
                              reason_for_inclusion="overlap",
                              competition_intensity="STRONG",
                              target_standing="INLINE",
                              metrics={"pe": 1.0, "roe": 1.0, "ret_1m": 1.0})
               for i in range(3)],
        comparison_summary="Target compares closely with peers across "
                           "valuation and recent returns overall.",
        overall_standing="INLINE"), 3),
    "news": (NewsOutput(
        items=[NewsItem(title="Quarterly numbers released today", date="2026-07-01",
                        source_url="https://x/a", summary="s", impact_on_stock="i",
                        sector_impact="sec", sentiment="NEUTRAL", sentiment_score=0.0)],
        narrative="Quarterly numbers released today led coverage.",
        overall_sentiment="NEUTRAL"), 1),
    "events": (EventOutput(
        events=[TimelineEvent(date="2026-06-01", type="other", significance="LOW",
                              summary="s", what_it_meant="m", how_it_affected="a",
                              price_move_1d=None, price_move_5d=None,
                              filing_ref=None)],
        highlights=["one"]), 1),
    "docs": (DocsOutput(
        guidance=[GuidanceItem(metric="m", value="v", period="p", source="s",
                               quote="a sufficiently long supporting quote here")],
        risks=[], strategy_highlights=[], management_tone="confident",
        tone_trend="t", narrative="n"), 1),
}

REPORT = ReportOutput(exec_summary="e", sections={k: "x" for k in GOOD},
                      sources=[], missing_sections=[])


@pytest.fixture
def gr(monkeypatch):
    from workflow import nodes
    monkeypatch.setattr(nodes, "namespace_exists", lambda t: True)
    deleted = []
    monkeypatch.setattr(nodes, "delete_namespace", lambda t: deleted.append(t))
    monkeypatch.setattr(nodes, "store_to_pinecone", lambda *a, **k: None)
    for name, result in GOOD.items():
        monkeypatch.setattr(nodes, "AGENT_RUNNERS",
                            {**nodes.AGENT_RUNNERS, name: (lambda r: lambda t, c, f="": r)(result)})
    monkeypatch.setattr(nodes, "run_synthesis", lambda state, feedback="": REPORT)

    from workflow.graph import build_graph
    return build_graph(), nodes, deleted


def _invoke(graph):
    return graph.invoke({"ticker": "T.NS", "company_name": "T Ltd"})


def test_happy_path_all_pass(gr):
    graph, nodes, deleted = gr
    state = _invoke(graph)
    assert deleted == ["T.NS"]                       # stale namespace wiped
    assert state["report"].exec_summary == "e"
    for name in GOOD:
        assert state["runs"][name]["status"] == "passed"
        assert state["runs"][name]["attempts"] == 1


def test_retry_then_pass_injects_feedback(gr, monkeypatch):
    graph, nodes, _ = gr
    calls = []

    def flaky(t, c, feedback=""):
        calls.append(feedback)
        if len(calls) == 1:
            bad = GOOD["news"][0].model_copy(update={"narrative": "short"})
            return bad, 1                            # fails title-mention rule
        return GOOD["news"]

    monkeypatch.setattr(nodes, "AGENT_RUNNERS",
                        {**nodes.AGENT_RUNNERS, "news": flaky})
    from workflow.graph import build_graph
    state = build_graph().invoke({"ticker": "T.NS", "company_name": "T Ltd"})
    assert state["runs"]["news"]["status"] == "passed"
    assert state["runs"]["news"]["attempts"] == 2
    assert "narrative" in calls[1]                   # validator reason reached retry


def test_empty_data_no_retry(gr, monkeypatch):
    graph, nodes, _ = gr
    calls = []

    def empty(t, c, feedback=""):
        calls.append(1)
        return (NewsOutput(items=[], narrative="No relevant recent news found "
                                                "for the company today.",
                           overall_sentiment="NEUTRAL"), 0)

    monkeypatch.setattr(nodes, "AGENT_RUNNERS",
                        {**nodes.AGENT_RUNNERS, "news": empty})
    from workflow.graph import build_graph
    state = build_graph().invoke({"ticker": "T.NS", "company_name": "T Ltd"})
    assert state["runs"]["news"]["status"] == "failed_partial"
    assert len(calls) == 1                           # no retry on empty source


def test_exception_retries_then_partial(gr, monkeypatch):
    graph, nodes, _ = gr
    calls = []

    def boom(t, c, feedback=""):
        calls.append(1)
        raise RuntimeError("network down")

    monkeypatch.setattr(nodes, "AGENT_RUNNERS",
                        {**nodes.AGENT_RUNNERS, "docs": boom})
    from workflow.graph import build_graph
    state = build_graph().invoke({"ticker": "T.NS", "company_name": "T Ltd"})
    assert state["runs"]["docs"]["status"] == "failed_partial"
    assert len(calls) == 3                           # 3 attempts, then partial
    assert state["report"] is not None               # report still ships


def test_synthesis_raises_twice_report_is_none(gr, monkeypatch):
    graph, nodes, _ = gr
    calls = []

    def boom(state, feedback=""):
        calls.append(1)
        raise RuntimeError("provider down")

    monkeypatch.setattr(nodes, "run_synthesis", boom)
    state = graph.invoke({"ticker": "T.NS", "company_name": "T Ltd"})
    assert len(calls) == 2                           # 1 initial + 1 retry
    assert state["report"] is None                   # graph completes, no crash


def test_synthesis_raises_once_then_succeeds(gr, monkeypatch):
    graph, nodes, _ = gr
    calls = []

    def flaky(state, feedback=""):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient provider error")
        return REPORT

    monkeypatch.setattr(nodes, "run_synthesis", flaky)
    state = graph.invoke({"ticker": "T.NS", "company_name": "T Ltd"})
    assert len(calls) == 2
    assert state["report"].exec_summary == "e"        # report produced on retry


def test_stream_report_yields_updates_and_final_report(monkeypatch):
    import types as _types
    updates = [
        {"run_fundamentals": {"runs": {"fundamentals": {"status": "passed"}}}},
        {"synthesis": {"runs": {"fundamentals": {"status": "passed"}},
                       "report": REPORT}},
    ]

    class FakeGraph:
        def __init__(self):
            self.stream_kwargs = None

        def stream(self, inputs, **kwargs):
            self.stream_kwargs = kwargs
            return iter(updates)

    fake = FakeGraph()
    import workflow.graph as g
    monkeypatch.setattr(g, "build_graph", lambda: fake)
    yielded = list(g.stream_report("T.NS", "T Ltd"))

    assert yielded[0]["node"] == "run_fundamentals"
    assert yielded[0]["done"] is False
    assert yielded[-1]["done"] is True
    assert yielded[-1]["report"] is REPORT
    assert yielded[-1]["runs"]["fundamentals"]["status"] == "passed"
    assert fake.stream_kwargs.get("stream_mode") == "updates"
