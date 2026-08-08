import pytest

from templates.schemas.outputs import (
    CompetitorOutput, DocsOutput, EventOutput, FundamentalsOutput, GuidanceItem,
    NewsItem, NewsOutput, PeerComparison, ReportOutput, RiskItem, TimelineEvent)

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
    exec_summary="A steady quarter with profit growth across the franchise.",
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


def test_report_output_has_no_sections_field():
    assert "sections" not in ReportOutput.model_fields


def test_build_index_text_serializes_structured_outputs():
    from agents.synthesis_agent import build_index_text
    report = ReportOutput(exec_summary="Exec text.", sources=[],
                          missing_sections=[])
    text = build_index_text(_state(), report)
    assert "Exec text." in text
    assert "Profit rises" in text            # news item title
    assert "1712.5" in text                  # fundamentals price passed through
    assert "MISSING" not in text             # absent agents are simply skipped


def test_missing_sections_and_sources_computed_in_code(patched):
    mod, captured = patched
    report = mod.run(_state())
    # every specialist absent from state is missing, whatever runs says
    assert report.missing_sections == ["competitor", "events", "docs"]
    assert "https://example.com/a" in report.sources      # from news item
    assert "llm-invented-source" not in report.sources
    assert "MISSING" in captured["prompt"]                # docs marked missing in context
    assert "Profit rises" in captured["prompt"]           # news output serialized in


def _fat_state():
    """A realistic full run: five specialists, every list at full length."""
    long_text = "Deposit repricing weighs on margins through the first half. " * 6
    peers = [PeerComparison(
        ticker=f"PEER{i}.NS", name=f"Peer {i} Ltd",
        reason_for_inclusion=long_text, competition_intensity="STRONG",
        target_standing="INLINE",
        metrics={"pe": 17.0 + i, "roe": 15.0, "debt_equity": None,
                 "mktcap_cr": 895000.0, "ret_1m": 2.1, "ret_3m": 8.9,
                 "ret_6m": 12.4}) for i in range(12)]
    items = [NewsItem(
        title=f"Headline number {i}", date="2026-07-01",
        source_url=f"https://example.com/{i}", summary=long_text,
        impact_on_stock=long_text, sector_impact=long_text,
        sentiment="POSITIVE", sentiment_score=0.5) for i in range(30)]
    events = [TimelineEvent(
        date="2026-05-14", type="dividend", significance="HIGH",
        summary=f"Event {i}", what_it_meant=long_text,
        how_it_affected=long_text, price_move_1d=1.8, price_move_5d=3.2,
        filing_ref=f"BSE Ann. {i}") for i in range(30)]
    docs = DocsOutput(
        guidance=[GuidanceItem(metric="credit growth", value="17-18% YoY",
                               period="FY27", source="Q4 FY26 concall",
                               quote=long_text) for _ in range(20)],
        risks=[RiskItem(risk="NIM compression", source="Annual Report FY26",
                        quote=long_text) for _ in range(20)],
        strategy_highlights=[f"Priority {i}" for i in range(20)],
        management_tone="confident", tone_trend="Improving since FY25.",
        narrative=long_text)
    return _state(
        competitor=CompetitorOutput(peers=peers, comparison_summary=long_text,
                                    overall_standing="AHEAD"),
        news=NewsOutput(items=items, narrative=long_text,
                        overall_sentiment="POSITIVE"),
        events=EventOutput(events=events, highlights=["Dividend declared"]),
        docs=docs,
        runs={n: {"status": "passed", "attempts": 1, "failure_reasons": [],
                  "fetch_count": 1}
              for n in ("fundamentals", "competitor", "news", "events", "docs")})


def test_payload_fits_the_groq_bucket(patched):
    """A full run used to serialize to ~17k tokens and every Groq model 413s:
    the largest free-tier bucket is 12k TPM."""
    mod, captured = patched
    mod.run(_fat_state())
    assert len(captured["prompt"]) <= mod.MAX_PAYLOAD_CHARS + 4000


def test_payload_drops_fields_the_exec_summary_cannot_use(patched):
    """Synthesis writes only exec_summary. Verbatim quotes, source URLs and
    per-peer metric dicts are rendered from the specialist outputs directly."""
    mod, captured = patched
    mod.run(_fat_state())
    prompt = captured["prompt"]
    assert "https://example.com/0" not in prompt          # sources computed in code
    assert "mktcap_cr" not in prompt                      # peer metric dicts
    assert "quote" not in prompt                          # verbatim doc quotes


def test_payload_keeps_what_the_exec_summary_needs(patched):
    mod, captured = patched
    mod.run(_fat_state())
    prompt = captured["prompt"]
    assert "Headline number 0" in prompt                  # news titles
    assert "Peer 0 Ltd" in prompt and "AHEAD" in prompt   # peer standing
    assert "credit growth" in prompt and "17-18% YoY" in prompt
    assert "confident" in prompt                          # management tone
    assert "Large private bank" in prompt                 # fundamentals summary


def test_oversized_lists_are_capped_not_dropped(patched):
    mod, captured = patched
    mod.run(_fat_state())
    prompt = captured["prompt"]
    assert "Headline number 0" in prompt
    assert "Headline number 29" not in prompt             # tail trimmed


def test_missing_sections_survive_compaction(patched):
    mod, captured = patched
    report = mod.run(_state())
    assert report.missing_sections == ["competitor", "events", "docs"]
    assert "MISSING" in captured["prompt"]
    assert "Large private bank" in captured["prompt"]     # present ones still sent
