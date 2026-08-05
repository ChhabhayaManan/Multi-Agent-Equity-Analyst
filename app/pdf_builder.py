"""Render a stored report to a self-contained HTML doc, then to PDF bytes.
Tables only — xhtml2pdf has no flex/grid support. Row shaping is shared with
the Streamlit page via app.ui_helpers so the two formats cannot drift."""
import io
from html import escape

from xhtml2pdf import pisa

from app.ui_helpers import (DASH, event_rows, fmt_num, group_sources,
                            guidance_rows, kv_rows, news_rows, peer_rows)

_VALUATION_LABELS = [("pe", "P/E", ""), ("pb", "P/B", ""), ("roe", "ROE", "%"),
                     ("roce", "ROCE", "%"), ("debt_equity", "Debt / Equity", ""),
                     ("dividend_yield", "Dividend yield", "%")]
_PRICE_LABELS = [("price", "Price", "₹"), ("high_52w", "52-week high", "₹"),
                 ("low_52w", "52-week low", "₹"), ("ret_1m", "1-month return", "%"),
                 ("ret_6m", "6-month return", "%"), ("ret_1y", "1-year return", "%"),
                 ("mktcap_cr", "Market cap", "cr")]
_HOLDING_LABELS = [("promoter", "Promoter", "%"), ("fii", "FII", "%"),
                   ("dii", "DII", "%"), ("public", "Public", "%")]

_CSS = """
body { font-family: Helvetica, Arial, sans-serif; font-size: 10px; color: #1a1a1a; }
h1 { font-size: 20px; } h2 { font-size: 15px; margin-top: 16px; }
h3 { font-size: 12px; margin: 10px 0 4px; }
.meta { color: #666; font-size: 9px; margin-bottom: 12px; }
.missing { color: #b00020; font-style: italic; }
.sources { margin-top: 16px; font-size: 9px; color: #444; }
table { width: 100%; border-collapse: collapse; margin: 4px 0 8px; }
th { background: #eef1f6; text-align: left; padding: 3px 5px; font-size: 9px; }
td { padding: 3px 5px; border-bottom: 1px solid #e2e6ee; font-size: 9px; }
.quote { color: #333; font-style: italic; margin: 2px 0 6px 8px; }
.label { color: #666; }
"""


def _cell(value) -> str:
    if value is None or value == "":
        return DASH
    if isinstance(value, float):
        return fmt_num(value)
    return escape(str(value))


def _table(rows: list) -> str:
    """list[dict] -> HTML table. Empty rows render nothing."""
    if not rows:
        return ""
    headers = list(rows[0].keys())
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell(r.get(h))}</td>" for h in headers) + "</tr>"
        for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _missing(title: str) -> str:
    return (f"<h2>{title}</h2>"
            "<p class='missing'>Data unavailable for this section.</p>")


def _fundamentals(f, generated_at: str) -> str:
    if f is None:
        return _missing("Company &amp; Fundamentals")
    profile = f.company_profile or {}
    out = ["<h2>Company &amp; Fundamentals</h2>",
           f"<p>{escape(f.summary)}</p>"]
    meta = " · ".join(escape(str(profile[k])) for k in ("sector", "industry")
                      if profile.get(k))
    if meta:
        out.append(f"<p class='label'>{meta}</p>")
    if profile.get("description"):
        out.append(f"<p>{escape(str(profile['description']))}</p>")
    out.append("<h3>Valuation</h3>" + _table(kv_rows(f.valuation, _VALUATION_LABELS)))
    out.append(f"<h3>Price snapshot (as of {escape(generated_at)})</h3>"
               + _table(kv_rows(f.price_snapshot, _PRICE_LABELS)))
    out.append("<h3>Shareholding</h3>"
               + _table(kv_rows(f.shareholding, _HOLDING_LABELS)))
    return "".join(out)


def _competitors(c, fundamentals, ticker: str, company_name: str) -> str:
    if c is None:
        return _missing("Competitive Landscape")
    out = ["<h2>Competitive Landscape</h2>",
           f"<p><b>Overall standing:</b> {escape(c.overall_standing)}</p>",
           f"<p>{escape(c.comparison_summary)}</p>",
           _table(peer_rows(c, fundamentals, ticker, company_name))]
    for p in c.peers:
        out.append(f"<p><b>{escape(p.name)}</b> ({escape(p.ticker)}) — "
                   f"{escape(p.competition_intensity)} competition, target "
                   f"{escape(p.target_standing)}.<br/>"
                   f"{escape(p.reason_for_inclusion)}</p>")
    return "".join(out)


def _events(e) -> str:
    if e is None:
        return _missing("Event Timeline")
    out = ["<h2>Event Timeline</h2>"]
    if e.highlights:
        items = "".join(f"<li>{escape(h)}</li>" for h in e.highlights)
        out.append(f"<h3>Highlights</h3><ul>{items}</ul>")
    out.append(_table(event_rows(e)))
    for ev in e.events:
        ref = f" ({escape(ev.filing_ref)})" if ev.filing_ref else ""
        out.append(f"<p><b>{escape(ev.date)} — {escape(ev.summary)}</b>{ref}<br/>"
                   f"{escape(ev.what_it_meant)}<br/>{escape(ev.how_it_affected)}</p>")
    return "".join(out)


def _news(n) -> str:
    if n is None:
        return _missing("News Analysis")
    out = ["<h2>News Analysis</h2>",
           f"<p><b>Overall sentiment:</b> {escape(n.overall_sentiment)}</p>",
           f"<p>{escape(n.narrative)}</p>", _table(news_rows(n))]
    for it in n.items:
        out.append(f"<p><b>{escape(it.title)}</b> ({escape(it.date)})<br/>"
                   f"{escape(it.summary)}<br/>"
                   f"<span class='label'>Stock:</span> {escape(it.impact_on_stock)}<br/>"
                   f"<span class='label'>Sector:</span> {escape(it.sector_impact)}</p>")
    return "".join(out)


def _docs(d) -> str:
    if d is None:
        return _missing("Financial Documents")
    out = ["<h2>Financial Documents</h2>",
           f"<p><b>Management tone:</b> {escape(d.management_tone)} — "
           f"{escape(d.tone_trend)}</p>", f"<p>{escape(d.narrative)}</p>",
           "<h3>Guidance</h3>", _table(guidance_rows(d))]
    for g in d.guidance:
        out.append(f"<p class='quote'>“{escape(g.quote)}” — {escape(g.source)}</p>")
    out.append("<h3>Risks</h3>")
    for r in d.risks:
        out.append(f"<p><b>{escape(r.risk)}</b> ({escape(r.source)})<br/>"
                   f"<span class='quote'>“{escape(r.quote)}”</span></p>")
    if d.strategy_highlights:
        items = "".join(f"<li>{escape(s)}</li>" for s in d.strategy_highlights)
        out.append(f"<h3>Strategy</h3><ul>{items}</ul>")
    return "".join(out)


def _sources(specialists: dict) -> str:
    groups = group_sources(specialists)
    if not groups:
        return ""
    blocks = []
    for title, refs in groups:
        items = "".join(f"<li>{escape(r)}</li>" for r in refs)
        blocks.append(f"<h3>{escape(title)}</h3><ul>{items}</ul>")
    return f"<div class='sources'><h2>Sources</h2>{''.join(blocks)}</div>"


def build_html(stored: dict) -> str:
    report = stored["report"]
    s = stored.get("specialists") or {}
    parts = [
        f"<h1>{escape(stored['company_name'])} ({escape(stored['ticker'])})</h1>",
        f"<div class='meta'>Generated {escape(stored['generated_at'])}</div>",
        "<h2>Executive Summary</h2>",
        f"<p>{escape(report.exec_summary)}</p>",
        _fundamentals(s.get("fundamentals"), stored["generated_at"]),
        _competitors(s.get("competitor"), s.get("fundamentals"),
                     stored["ticker"], stored["company_name"]),
        _events(s.get("events")),
        _news(s.get("news")),
        _docs(s.get("docs")),
        _sources(s),
    ]
    body = "\n".join(parts)
    return f"<html><head><style>{_CSS}</style></head><body>{body}</body></html>"


def build_pdf(stored: dict) -> bytes:
    html = build_html(stored)
    buf = io.BytesIO()
    pisa.CreatePDF(src=html, dest=buf, encoding="utf-8")
    return buf.getvalue()
