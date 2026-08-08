from app.pdf_builder import build_html, build_pdf
from templates.schemas.outputs import ReportOutput
from tests.test_ui_helpers import COMP, DOCS, EVENTS, FUND, NEWS


def _stored(**over):
    stored = {
        "ticker": "HDFCBANK", "company_name": "HDFC Bank Ltd",
        "generated_at": "2026-07-04T10:00:00",
        "report": ReportOutput(exec_summary="Exec summary text.",
                               sources=["Q4 FY26 concall, 2026-04-19"],
                               missing_sections=["news"]),
        "specialists": {"fundamentals": FUND, "competitor": COMP,
                        "events": EVENTS, "news": NEWS, "docs": DOCS},
        "runs": {},
    }
    stored.update(over)
    return stored


def test_build_html_contains_every_block_in_order():
    html = build_html(_stored())
    titles = ["Executive Summary", "Company &amp; Fundamentals",
              "Competitive Landscape", "Event Timeline", "News Analysis",
              "Financial Documents", "Sources"]
    positions = [html.find(t) for t in titles]
    assert all(p != -1 for p in positions)
    assert positions == sorted(positions)


def test_build_html_renders_structured_detail():
    html = build_html(_stored())
    assert "Exec summary text." in html
    assert "ICICI Bank" in html                  # peer row
    assert "HDFC Bank Ltd (target)" in html      # target pinned into peer table
    assert "Dividend declared" in html           # event row
    assert "Profit rises" in html                # news row
    assert "credit growth" in html               # guidance row
    assert "deposit repricing" in html           # risk
    assert "Q4 FY26 concall, 2026-04-19" in html  # grouped source


def test_build_html_flags_missing_specialist():
    html = build_html(_stored(specialists={"fundamentals": FUND,
                                           "competitor": None, "events": None,
                                           "news": None, "docs": None}))
    assert html.lower().count("data unavailable") >= 4


def test_build_pdf_returns_pdf_bytes():
    pdf = build_pdf(_stored())
    assert isinstance(pdf, (bytes, bytearray))
    assert bytes(pdf[:4]) == b"%PDF"
    assert len(pdf) > 500
