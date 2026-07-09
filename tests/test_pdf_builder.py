from templates.schemas.outputs import ReportOutput
from app.pdf_builder import build_html, build_pdf

SECTION_ORDER = ["fundamentals", "competitors", "events", "news", "docs"]


def _stored():
    return {
        "ticker": "HDFCBANK", "company_name": "HDFC Bank Ltd",
        "generated_at": "2026-07-04T10:00:00",
        "report": ReportOutput(
            exec_summary="Exec summary text.",
            sections={k: f"## {k.title()}\nBody for {k}." for k in SECTION_ORDER},
            sources=["Q4 FY26 concall, 2026-04-19"],
            missing_sections=["news"]),
    }


def test_build_html_contains_all_sections_in_order():
    html = build_html(_stored())
    assert "Exec summary text." in html
    positions = [html.find(f"Body for {k}") for k in SECTION_ORDER]
    assert all(p != -1 for p in positions)
    assert positions == sorted(positions)          # fixed order preserved
    assert "Q4 FY26 concall" in html               # sources rendered
    assert "HDFC Bank Ltd" in html


def test_build_html_flags_missing_section():
    html = build_html(_stored())
    assert "unavailable" in html.lower()            # news is missing


def test_build_pdf_returns_pdf_bytes():
    pdf = build_pdf(_stored())
    assert isinstance(pdf, (bytes, bytearray))
    assert bytes(pdf[:4]) == b"%PDF"
    assert len(pdf) > 500
