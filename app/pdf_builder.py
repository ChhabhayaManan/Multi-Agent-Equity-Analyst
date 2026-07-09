"""Render a stored report to a self-contained HTML doc, then to PDF bytes."""
import io

import markdown as _md
from xhtml2pdf import pisa

SECTION_ORDER = ["fundamentals", "competitors", "events", "news", "docs"]
SECTION_TITLES = {
    "fundamentals": "Company & Fundamentals", "competitors": "Competitive Landscape",
    "events": "Event Timeline", "news": "News Analysis", "docs": "Financial Documents"}

_CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 11px; color: #1a1a1a; }
h1 { font-size: 20px; } h2 { font-size: 15px; margin-top: 16px; }
.meta { color: #666; font-size: 10px; margin-bottom: 12px; }
.missing { color: #b00020; font-style: italic; }
.sources { margin-top: 16px; font-size: 10px; color: #444; }
"""


def build_html(stored: dict) -> str:
    report = stored["report"]
    missing = set(report.missing_sections)
    parts = [
        f"<h1>{stored['company_name']} ({stored['ticker']})</h1>",
        f"<div class='meta'>Generated {stored['generated_at']}</div>",
        "<h2>Executive Summary</h2>",
        _md.markdown(report.exec_summary),
    ]
    for key in SECTION_ORDER:
        parts.append(f"<h2>{SECTION_TITLES[key]}</h2>")
        if key in missing:
            parts.append("<p class='missing'>Data unavailable for this section.</p>")
        parts.append(_md.markdown(report.sections.get(key, "")))
    if report.sources:
        items = "".join(f"<li>{s}</li>" for s in report.sources)
        parts.append(f"<div class='sources'><h2>Sources</h2><ul>{items}</ul></div>")
    body = "\n".join(parts)
    return f"<html><head><style>{_CSS}</style></head><body>{body}</body></html>"


def build_pdf(stored: dict) -> bytes:
    html = build_html(stored)
    buf = io.BytesIO()
    pisa.CreatePDF(src=html, dest=buf, encoding="utf-8")
    return buf.getvalue()
