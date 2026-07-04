import pytest
from pydantic import ValidationError


def _peer(**over):
    base = dict(
        ticker="ICICIBANK.NS", name="ICICI Bank Ltd",
        reason_for_inclusion="Competes directly in retail lending and deposits",
        competition_intensity="FIERCE", target_standing="INLINE",
        metrics={"pe": 17.8, "roe": 18.4, "ret_1m": 2.1},
    )
    base.update(over)
    return base


def test_competitor_output_roundtrip():
    from templates.schemas.outputs import CompetitorOutput
    out = CompetitorOutput(
        peers=[_peer(), _peer(ticker="AXISBANK.NS"), _peer(ticker="KOTAKBANK.NS")],
        comparison_summary="Target trades at a premium to peers on P/E.",
        overall_standing="AHEAD",
    )
    assert out.peers[0].competition_intensity == "FIERCE"


def test_competitor_rejects_bad_enum_and_too_few_peers():
    from templates.schemas.outputs import CompetitorOutput
    with pytest.raises(ValidationError):
        CompetitorOutput(peers=[_peer(competition_intensity="EXTREME")] * 3,
                         comparison_summary="x", overall_standing="AHEAD")
    with pytest.raises(ValidationError):
        CompetitorOutput(peers=[_peer(), _peer()],  # min 3
                         comparison_summary="x", overall_standing="AHEAD")


def test_news_item_score_bounds():
    from templates.schemas.outputs import NewsItem
    ok = dict(title="T", date="2026-07-01", source_url="https://x",
              summary="s", impact_on_stock="i", sector_impact="sec",
              sentiment="NEUTRAL", sentiment_score=0.1)
    NewsItem(**ok)
    with pytest.raises(ValidationError):
        NewsItem(**{**ok, "sentiment_score": 1.5})


def test_event_and_docs_and_fundamentals_and_report():
    from templates.schemas.outputs import (
        EventOutput, TimelineEvent, DocsOutput, GuidanceItem, RiskItem,
        FundamentalsOutput, ReportOutput)
    ev = TimelineEvent(date="2026-05-14", type="dividend", significance="HIGH",
                       summary="s", what_it_meant="m", how_it_affected="a",
                       price_move_1d=1.8, price_move_5d=None, filing_ref=None)
    EventOutput(events=[ev], highlights=["one"])
    DocsOutput(
        guidance=[GuidanceItem(metric="credit growth", value="17-18%", period="FY27",
                               source="Q4 FY26 concall", quote="x" * 25)],
        risks=[RiskItem(risk="NIM compression", source="AR FY26", quote="y" * 25)],
        strategy_highlights=["branch expansion"], management_tone="confident",
        tone_trend="steadier since FY25", narrative="n")
    FundamentalsOutput(company_profile={"sector": "Banks"},
                       valuation={"pe": 19.2}, price_snapshot={"price": 1712.5},
                       shareholding={"promoter": 0.0}, summary="s" * 60)
    ReportOutput(exec_summary="e", sections={"fundamentals": "## F"},
                 sources=["src"], missing_sections=[])


def test_quote_min_length():
    from templates.schemas.outputs import GuidanceItem
    with pytest.raises(ValidationError):
        GuidanceItem(metric="m", value="v", period="p", source="s", quote="short")


def test_agent_input():
    from templates.schemas.inputs import AgentInput
    a = AgentInput(ticker="HDFCBANK.NS", company_name="HDFC Bank Ltd")
    assert a.ticker.endswith(".NS")


def test_chatbot_response_defaults():
    from templates.schemas.outputs import ChatbotResponse

    r = ChatbotResponse(answer="HDFC Bank's P/E is 19.2 [live: yfinance].")
    assert r.answer.startswith("HDFC")
    assert r.sources_used == []
    assert r.charts == []


def test_chatbot_response_full():
    from templates.schemas.outputs import ChatbotResponse

    r = ChatbotResponse(
        answer="See chart.",
        sources_used=["get_live_price", "plot_price_chart"],
        charts=["data/charts/HDFCBANK.NS_6mo_1719999999.html"],
    )
    assert "plot_price_chart" in r.sources_used
    assert len(r.charts) == 1
