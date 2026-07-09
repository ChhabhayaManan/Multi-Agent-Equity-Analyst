from templates.schemas.outputs import (
    CompetitorOutput, DocsOutput, EventOutput, FundamentalsOutput,
    GuidanceItem, NewsItem, NewsOutput, PeerComparison, RiskItem,
    TimelineEvent)


def _run(fetch_count=5, attempts=1):
    return {"status": "running", "attempts": attempts,
            "failure_reasons": [], "fetch_count": fetch_count}


def _fundamentals(**over):
    base = dict(
        company_profile={"sector": "Banks"},
        valuation={"pe": 19.2, "pb": 2.8, "roe": 16.9, "roce": None,
                   "debt_equity": 1.1, "dividend_yield": 1.1},
        price_snapshot={"price": 1712.5, "ret_1y": 18.2},
        shareholding={"promoter": 25.5},
        summary="A large private bank trading near the top of its 52-week "
                "range after a strong one-year run.")
    base.update(over)
    return FundamentalsOutput(**base)


def _news(narrative=None):
    item = NewsItem(title="Quarterly profit rises on strong margins",
                    date="2026-07-01", source_url="https://example.com/a",
                    summary="Profit rose.", impact_on_stock="Supportive.",
                    sector_impact="Neutral.", sentiment="POSITIVE",
                    sentiment_score=0.5)
    return NewsOutput(items=[item],
                      narrative=narrative or "Quarterly profit rises on "
                                             "strong margins set the tone.",
                      overall_sentiment="POSITIVE")


def test_fundamentals_pass_and_fail():
    from workflow.validator import validate
    assert validate("fundamentals", _fundamentals(), _run()).passed
    thin = _fundamentals(valuation={"pe": 19.2, "pb": None, "roe": None,
                                    "roce": None, "debt_equity": None,
                                    "dividend_yield": None})
    result = validate("fundamentals", thin, _run())
    assert not result.passed and result.reasons


def test_advice_word_fails_but_quotes_exempt():
    from workflow.validator import validate
    bad = _fundamentals(summary="This stock is a strong buy at current "
                                "levels for long-term holders of quality.")
    result = validate("fundamentals", bad, _run())
    assert not result.passed
    assert any("buy" in r for r in result.reasons)

    docs = DocsOutput(
        guidance=[GuidanceItem(metric="capex", value="Rs 500cr", period="FY27",
                               source="Q4 concall",
                               quote="we will invest five hundred crores in new branches")],
        risks=[RiskItem(risk="margin", source="AR", quote="margins may compress further")],
        strategy_highlights=["expansion"], management_tone="confident",
        tone_trend="steady", narrative="Management outlined branch expansion plans.")
    assert validate("docs", docs, _run()).passed  # "invest" inside quote is exempt


def test_advice_word_boundary():
    from workflow.validator import scan_advice
    ok = _fundamentals(summary="Investor sentiment and shareholding shifted "
                               "toward institutions during the quarter here.")
    assert scan_advice(ok) == []  # "investor" must not trip "invest"


def test_news_rules():
    from workflow.validator import validate
    assert validate("news", _news(), _run()).passed
    off_topic = _news(narrative="Something completely unrelated happened elsewhere entirely.")
    assert not validate("news", off_topic, _run()).passed


def test_events_chronology():
    from workflow.validator import validate

    def ev(date):
        return TimelineEvent(date=date, type="other", significance="LOW",
                             summary="s", what_it_meant="m", how_it_affected="a",
                             price_move_1d=None, price_move_5d=None, filing_ref=None)

    good = EventOutput(events=[ev("2026-05-01"), ev("2026-06-01")], highlights=["x"])
    assert validate("events", good, _run()).passed
    bad = EventOutput(events=[ev("2026-06-01"), ev("2026-05-01")], highlights=["x"])
    assert not validate("events", bad, _run()).passed


def test_competitor_rules():
    from workflow.validator import validate

    def peer(n_metrics):
        metrics = {"pe": 17.0, "roe": 15.0, "ret_1m": 1.0, "ret_3m": None}
        metrics = dict(list(metrics.items())[:n_metrics]) if n_metrics < 4 else metrics
        return PeerComparison(ticker="X.NS", name="X",
                              reason_for_inclusion="overlapping lending business",
                              competition_intensity="STRONG",
                              target_standing="INLINE", metrics=metrics)

    good = CompetitorOutput(peers=[peer(4)] * 3,
                            comparison_summary="Target leads peers on returns "
                                               "but trades at a premium valuation.",
                            overall_standing="AHEAD")
    assert validate("competitor", good, _run()).passed
    thin = CompetitorOutput(peers=[peer(2)] * 3,
                            comparison_summary="Target leads peers on returns "
                                               "but trades at a premium valuation.",
                            overall_standing="AHEAD")
    assert not validate("competitor", thin, _run()).passed


def test_empty_data_detection():
    from workflow.validator import validate
    empty_news = NewsOutput(items=[], narrative="No relevant recent news was "
                                                "found for the company.",
                            overall_sentiment="NEUTRAL")
    result = validate("news", empty_news, _run(fetch_count=0))
    assert not result.passed and result.empty_data
    # fetch_count=-1 (agent exception sentinel) must NOT flag empty_data
    result = validate("news", None, _run(fetch_count=-1))
    assert not result.passed and not result.empty_data


def test_none_output_fails():
    from workflow.validator import validate
    result = validate("fundamentals", None, _run())
    assert not result.passed and result.reasons and not result.empty_data


def test_merge_runs_reducer():
    from workflow.state import merge_runs
    left = {"news": {"attempts": 1}}
    right = {"docs": {"attempts": 1}, "news": {"attempts": 2}}
    merged = merge_runs(left, right)
    assert merged["news"]["attempts"] == 2 and "docs" in merged
