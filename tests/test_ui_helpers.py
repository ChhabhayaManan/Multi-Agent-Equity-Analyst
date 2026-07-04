from app.ui_helpers import badge_html, chart_iframe_html, section_note


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
