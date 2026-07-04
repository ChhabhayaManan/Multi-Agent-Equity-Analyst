import pytest

from templates.schemas.outputs import (
    FundamentalsOutput, NewsItem, NewsOutput, ReportOutput)

FUND = FundamentalsOutput(
    company_profile={"sector": "Banks"}, valuation={"pe": 19.2},
    price_snapshot={"price": 1712.5}, shareholding={"promoter": 25.0},
    summary="Large private bank trading near its 52-week high after a run.")

NEWS = NewsOutput(
    items=[NewsItem(title="Profit rises", date="2026-07-01",
                    source_url="https://example.com/a", summary="s",
                    impact_on_stock="i", sector_impact="sec",
                    sentiment="POSITIVE", sentiment_score=0.5)],
    narrative="Profit rises set the tone.", overall_sentiment="POSITIVE")

CANNED = ReportOutput(
    exec_summary="A steady quarter with profit growth.",
    sections={"fundamentals": "## F", "competitors": "## C", "events": "## E",
              "news": "## N", "docs": "## D"},
    sources=["llm-invented-source"],       # overwritten in code
    missing_sections=["llm-wrong"])        # overwritten in code


def _state(**over):
    runs = {n: {"status": "passed", "attempts": 1, "failure_reasons": [],
                "fetch_count": 1}
            for n in ("fundamentals", "competitor", "news", "events", "docs")}
    runs["docs"] = {"status": "failed_partial", "attempts": 3,
                    "failure_reasons": ["x"], "fetch_count": 0}
    base = dict(ticker="HDFCBANK.NS", company_name="HDFC Bank Ltd",
                fundamentals=FUND, competitor=None, news=NEWS, events=None,
                docs=None, runs=runs)
    base.update(over)
    return base


@pytest.fixture
def patched(monkeypatch):
    from agents import synthesis_agent as mod
    captured = {}

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt.to_string()
            return CANNED

    monkeypatch.setattr(mod, "get_llm", lambda schema=None: FakeLLM())
    return mod, captured


def test_missing_sections_and_sources_computed_in_code(patched):
    mod, captured = patched
    report = mod.run(_state())
    assert report.missing_sections == ["docs"]            # from runs, not the LLM
    assert "https://example.com/a" in report.sources      # from news item
    assert "llm-invented-source" not in report.sources
    assert "MISSING" in captured["prompt"]                # docs marked missing in context
    assert "Profit rises" in captured["prompt"]           # news output serialized in
