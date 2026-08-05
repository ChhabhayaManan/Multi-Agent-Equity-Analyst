from app.ui_helpers import (DASH, badge_html, chart_iframe_html, event_rows,
                            fmt_market_cap, fmt_num, group_sources,
                            guidance_rows, kv_rows, move_chip_html, news_rows,
                            peer_rows, section_note, status_icon)
from templates.schemas.outputs import (CompetitorOutput, DocsOutput,
                                       EventOutput, FundamentalsOutput,
                                       GuidanceItem, NewsItem, NewsOutput,
                                       PeerComparison, RiskItem, TimelineEvent)

FUND = FundamentalsOutput(
    company_profile={"sector": "Banks", "industry": "Private Banks",
                     "description": "A bank."},
    valuation={"pe": 19.2, "pb": 2.8, "roe": 16.9, "roce": None,
               "debt_equity": None, "dividend_yield": 1.1},
    price_snapshot={"price": 1712.5, "high_52w": 1794.0, "low_52w": 1363.5,
                    "ret_1m": 2.4, "ret_6m": 9.8, "ret_1y": 18.2,
                    "mktcap_cr": 1305000.0},
    shareholding={"promoter": 0.0, "fii": 47.8, "dii": 35.2, "public": 17.0},
    summary="A large private bank.")

COMP = CompetitorOutput(
    peers=[PeerComparison(ticker="ICICIBANK.NS", name="ICICI Bank",
                          reason_for_inclusion="overlapping retail book",
                          competition_intensity="FIERCE",
                          target_standing="INLINE",
                          metrics={"pe": 17.8, "roe": 18.4, "debt_equity": None,
                                   "mktcap_cr": 895000.0, "ret_1m": 2.1,
                                   "ret_3m": 8.9, "ret_6m": 12.4})],
    comparison_summary="Premium to the peer set.", overall_standing="AHEAD")

NEWS = NewsOutput(
    items=[NewsItem(title="Profit rises", date="2026-07-01",
                    source_url="https://example.com/a", summary="s",
                    impact_on_stock="i", sector_impact="sec",
                    sentiment="POSITIVE", sentiment_score=0.5)],
    narrative="Coverage centred on the print.", overall_sentiment="POSITIVE")

EVENTS = EventOutput(
    events=[TimelineEvent(date="2026-05-14", type="dividend",
                          significance="HIGH", summary="Dividend declared",
                          what_it_meant="confidence", how_it_affected="rose",
                          price_move_1d=1.8, price_move_5d=None,
                          filing_ref="BSE Ann. 2026-05-14")],
    highlights=["Rs 22/share dividend"])

DOCS = DocsOutput(
    guidance=[GuidanceItem(metric="credit growth", value="17-18% YoY",
                           period="FY27", source="Q4 FY26 concall, 2026-04-19",
                           quote="we remain confident of 17 to 18 percent")],
    risks=[RiskItem(risk="deposit repricing", source="Annual Report FY26",
                    quote="continued upward repricing of the deposit base")],
    strategy_highlights=["branch expansion"], management_tone="confident",
    tone_trend="steadier since FY25", narrative="Direction is stable.")


def test_badge_html_colors_known_values():
    assert "AHEAD" in badge_html("AHEAD", "standing")
    assert "background" in badge_html("POSITIVE", "sentiment").lower()
    assert "MYSTERY" in badge_html("MYSTERY", "nope")


def test_chart_iframe_reads_file(tmp_path):
    f = tmp_path / "c.html"
    f.write_text("<div>CHART</div>", encoding="utf-8")
    assert "CHART" in chart_iframe_html(str(f))


def test_chart_iframe_missing_returns_none():
    assert chart_iframe_html("does/not/exist.html") is None


def test_section_note():
    assert section_note(["news"], "news") is not None
    assert section_note(["news"], "docs") is None


def test_fmt_num_and_market_cap_handle_none():
    assert fmt_num(None) == DASH
    assert fmt_num(1712.5, "₹") == "₹1,712.50"
    assert fmt_market_cap(None) == DASH
    assert "L Cr" in fmt_market_cap(1.305e13)


def test_move_chip_html():
    assert move_chip_html("1D", 1.8).count("+1.80%") == 1
    assert "-2.00%" in move_chip_html("5D", -2.0)
    assert move_chip_html("1D", None) == ""


def test_status_icon_covers_every_state():
    assert status_icon("passed") == "✅"
    assert status_icon("no_data") == "📭"
    assert status_icon("failed_partial") == "⚠️"
    assert status_icon(None) == "⏳"
    assert status_icon("nonsense") == "⏳"


def test_kv_rows_formats_units_and_none():
    rows = kv_rows(FUND.valuation,
                   [("pe", "P/E", ""), ("roce", "ROCE", "%")])
    assert rows == [{"Metric": "P/E", "Value": "19.20"},
                    {"Metric": "ROCE", "Value": DASH}]


def test_peer_rows_pins_target_first_with_blank_3m():
    rows = peer_rows(COMP, FUND, "HDFCBANK.NS", "HDFC Bank Ltd")
    assert rows[0]["Company"] == "HDFC Bank Ltd (target)"
    assert rows[0]["P/E"] == 19.2
    assert rows[0]["3M %"] is None            # fundamentals has no 3-month return
    assert rows[0]["Intensity"] == ""
    assert rows[1]["Company"] == "ICICI Bank"
    assert rows[1]["3M %"] == 8.9
    assert rows[1]["D/E"] is None             # absent metric stays None, not 0


def test_peer_rows_without_fundamentals_has_no_target_row():
    rows = peer_rows(COMP, None, "HDFCBANK.NS", "HDFC Bank Ltd")
    assert len(rows) == 1
    assert rows[0]["Company"] == "ICICI Bank"


def test_news_rows_carry_score_and_url():
    rows = news_rows(NEWS)
    assert rows[0]["Title"] == "Profit rises"
    assert rows[0]["Score"] == 0.5
    assert rows[0]["Link"] == "https://example.com/a"


def test_event_rows_keep_none_moves():
    rows = event_rows(EVENTS)
    assert rows[0]["1D %"] == 1.8
    assert rows[0]["5D %"] is None
    assert rows[0]["Type"] == "dividend"


def test_guidance_rows():
    rows = guidance_rows(DOCS)
    assert rows[0] == {"Metric": "credit growth", "Value": "17-18% YoY",
                       "Period": "FY27",
                       "Source": "Q4 FY26 concall, 2026-04-19"}


def test_group_sources_groups_and_skips_empty():
    groups = dict(group_sources({"news": NEWS, "docs": DOCS, "events": EVENTS,
                                 "fundamentals": FUND, "competitor": COMP}))
    assert groups["News articles"] == ["https://example.com/a"]
    assert "Q4 FY26 concall, 2026-04-19" in groups["Documents"]
    assert "Annual Report FY26" in groups["Documents"]
    assert groups["Event filings"] == ["BSE Ann. 2026-05-14"]


def test_group_sources_empty_input_returns_no_groups():
    assert group_sources({"news": None, "docs": None, "events": None}) == []
