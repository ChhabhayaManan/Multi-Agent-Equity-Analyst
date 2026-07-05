from app.ui_helpers import (badge_html, chart_iframe_html, extract_citations,
                            move_chip_html, parse_price_moves, prose_html,
                            section_note, split_bullets, split_labeled,
                            strip_move_sentence)


def test_badge_html_colors_known_values():
    assert "AHEAD" in badge_html("AHEAD", "standing")
    assert "background" in badge_html("POSITIVE", "sentiment").lower()
    # unknown value still renders a neutral chip, never raises
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


def test_extract_citations_marks_and_dedupes():
    text = ("PE 4.63 (Valuation) near 52w low (Price Snapshot); "
            "again (Valuation).")
    clean, cites = extract_citations(text)
    assert cites == ["Valuation", "Price Snapshot"]      # deduped
    assert "@@1@@" in clean and "@@2@@" in clean
    assert clean.count("@@1@@") == 2                      # reused marker


def test_extract_citations_leaves_non_citations():
    clean, cites = extract_citations("announced under Regulation 30 (LODR).")
    assert cites == []
    assert "(LODR)" in clean


def test_extract_citations_handles_nested_parens():
    text = "filed (BSE Ann. 2026-07-03: Reg. 74 (5) of SEBI (DP) Regulations)."
    clean, cites = extract_citations(text)
    assert len(cites) == 1
    assert "(5)" in cites[0] and "(DP)" in cites[0]
    assert "@@1@@" in clean


def test_split_bullets_keeps_hyphenated_names_intact():
    text = ("Competitors include Blue Star and Johnson Controls - Hitachi "
            "Air Conditioning. Voltas trades at a premium P/E.")
    bullets = split_bullets(text)
    assert any("Johnson Controls - Hitachi" in b for b in bullets)
    assert len(bullets) == 2                             # split on sentence only


def test_split_bullets_does_not_split_decimals_or_rs():
    bullets = split_bullets("Market cap of Rs. 1,99,900.25 crore is large.")
    assert bullets == ["Market cap of Rs. 1,99,900.25 crore is large."]


def test_split_labeled_events():
    text = ("Events: - **June 5, 2026**: A change happened. - "
            "**June 30, 2026**: Dividend fixed.")
    intro, groups = split_labeled(text)
    assert intro == "Events"
    assert [g[0] for g in groups] == ["June 5, 2026", "June 30, 2026"]


def test_parse_and_strip_price_moves():
    body = ("A change happened. The stock price moved -0.5% the next day "
            "and 1.43% over the next five trading days.")
    assert parse_price_moves(body) == (-0.5, 1.43)
    stripped = strip_move_sentence(body)
    assert "moved" not in stripped
    assert stripped.startswith("A change happened")


def test_prose_html_and_move_chip():
    html = prose_html("PE low @@1@@ here", ["Valuation, 2025"])
    assert "<sup" in html and "Valuation, 2025" in html
    assert move_chip_html("1D", 1.8).count("+1.80%") == 1
    assert "-2.00%" in move_chip_html("5D", -2.0)
    assert move_chip_html("1D", None) == ""
