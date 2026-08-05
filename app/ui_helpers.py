"""Pure (Streamlit-free) presentation helpers so they can be unit-tested.
Row builders shape specialist outputs into table rows shared by the
Streamlit page and the PDF builder, so the two formats cannot drift."""
from pathlib import Path
from typing import Optional

DASH = "—"

_COLORS = {
    "AHEAD": "#1b7f3b", "INLINE": "#8a6d00", "BEHIND": "#b00020",
    "POSITIVE": "#1b7f3b", "NEUTRAL": "#555", "NEGATIVE": "#b00020",
    "HIGH": "#b00020", "MEDIUM": "#8a6d00", "LOW": "#1b7f3b",
    "confident": "#1b7f3b", "cautious": "#8a6d00", "defensive": "#b00020",
    "FIERCE": "#b00020", "STRONG": "#8a6d00", "MODERATE": "#4c6ef5",
    "MILD": "#1b7f3b",
}
_NEUTRAL = "#555"

_STATUS_ICONS = {"passed": "✅", "no_data": "📭", "failed_partial": "⚠️",
                 "running": "🔄", "pending": "⏳"}


def badge_html(label: str, kind: str) -> str:
    color = _COLORS.get(label, _NEUTRAL)
    return (f"<span style='background:{color};color:#fff;border-radius:10px;"
            f"padding:2px 10px;font-size:12px;font-weight:600'>{label}</span>")


def chart_iframe_html(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def section_note(missing: list, key: str) -> Optional[str]:
    if key in missing:
        return "_Data unavailable for this section._"
    return None


def status_icon(status: Optional[str]) -> str:
    """Agent run status -> tab icon. Unknown or None reads as pending."""
    return _STATUS_ICONS.get(status or "", "⏳")


def fmt_market_cap(value: Optional[float]) -> str:
    """INR absolute -> Indian crore/lakh-crore string."""
    if not value:
        return DASH
    cr = value / 1e7
    if cr >= 1e5:
        return f"₹{cr / 1e5:.2f} L Cr"
    return f"₹{cr:,.0f} Cr"


def fmt_num(value: Optional[float], prefix: str = "", suffix: str = "",
            dp: int = 2) -> str:
    if value is None:
        return DASH
    return f"{prefix}{value:,.{dp}f}{suffix}"


def move_chip_html(label: str, pct: Optional[float]) -> str:
    """Green/red % pill for an event price move; '' when the move is unknown."""
    if pct is None:
        return ""
    color = "#3fb950" if pct >= 0 else "#f0616d"
    sign = "+" if pct >= 0 else ""
    return (f"<span class='movechip' style='background:{color}1f;color:{color};"
            f"border:1px solid {color}55'>{label} {sign}{pct:.2f}%</span>")


def kv_rows(mapping: dict, labels: list) -> list:
    """[(key, display_label, unit)] + a metric dict -> Metric/Value rows.
    Units: '' plain, '%' percent, '₹' rupee, 'cr' Indian crore string."""
    rows = []
    for key, label, unit in labels:
        value = (mapping or {}).get(key)
        if unit == "cr":
            text = fmt_market_cap(value * 1e7 if value is not None else None)
        elif unit == "%":
            text = fmt_num(value, suffix="%")
        elif unit == "₹":
            text = fmt_num(value, prefix="₹")
        else:
            text = fmt_num(value)
        rows.append({"Metric": label, "Value": text})
    return rows


def peer_rows(competitor, fundamentals, ticker: str,
              company_name: str) -> list:
    """Peer table rows with the target pinned first. Values stay numeric so
    st.dataframe can sort them; missing numbers stay None (blank cell)."""
    rows = []
    if fundamentals is not None:
        val = fundamentals.valuation or {}
        snap = fundamentals.price_snapshot or {}
        rows.append({
            "Company": f"{company_name} (target)", "Ticker": ticker,
            "P/E": val.get("pe"), "ROE %": val.get("roe"),
            "D/E": val.get("debt_equity"), "MCap (Cr)": snap.get("mktcap_cr"),
            "1M %": snap.get("ret_1m"), "3M %": None,
            "6M %": snap.get("ret_6m"), "Intensity": "", "Standing": ""})
    for p in (competitor.peers if competitor else []):
        m = p.metrics or {}
        rows.append({
            "Company": p.name, "Ticker": p.ticker,
            "P/E": m.get("pe"), "ROE %": m.get("roe"),
            "D/E": m.get("debt_equity"), "MCap (Cr)": m.get("mktcap_cr"),
            "1M %": m.get("ret_1m"), "3M %": m.get("ret_3m"),
            "6M %": m.get("ret_6m"),
            "Intensity": p.competition_intensity,
            "Standing": p.target_standing})
    return rows


def news_rows(news) -> list:
    return [{"Date": it.date, "Title": it.title, "Sentiment": it.sentiment,
             "Score": it.sentiment_score, "Link": it.source_url}
            for it in (news.items if news else [])]


def event_rows(events) -> list:
    return [{"Date": e.date, "Type": e.type, "Significance": e.significance,
             "1D %": e.price_move_1d, "5D %": e.price_move_5d,
             "Summary": e.summary}
            for e in (events.events if events else [])]


def guidance_rows(docs) -> list:
    return [{"Metric": g.metric, "Value": g.value, "Period": g.period,
             "Source": g.source}
            for g in (docs.guidance if docs else [])]


def group_sources(specialists: dict) -> list:
    """[(group_title, [ref])] built from the stored specialist objects.
    Deduped per group, empty groups omitted."""
    news = specialists.get("news")
    docs = specialists.get("docs")
    events = specialists.get("events")
    raw = [
        ("News articles",
         [it.source_url for it in (news.items if news else []) if it.source_url]),
        ("Documents",
         [g.source for g in (docs.guidance if docs else [])]
         + [r.source for r in (docs.risks if docs else [])]),
        ("Event filings",
         [e.filing_ref for e in (events.events if events else []) if e.filing_ref]),
    ]
    out = []
    for title, refs in raw:
        seen, unique = set(), []
        for r in refs:
            if r and r not in seen:
                seen.add(r)
                unique.append(r)
        if unique:
            out.append((title, unique))
    return out


def css_block() -> str:
    """Shared 'fintech-pro' polish. Injected once per page via st.markdown.
    Light-touch: styles native widgets (metrics/tabs/containers), no layout
    hacks, so a Streamlit internals change degrades gracefully to plain."""
    return """<style>
.block-container {padding-top: 2.6rem; max-width: 1120px;}
[data-testid="stMetric"] {
  background: #161b26; border: 1px solid #26304a;
  border-radius: 12px; padding: 14px 16px;
}
[data-testid="stMetricLabel"] p {opacity: .65; font-size: .78rem;
  letter-spacing: .03em; text-transform: uppercase;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 12px;}
div[data-baseweb="tab-list"] {gap: 6px; border-bottom: 1px solid #26304a;}
div[data-baseweb="tab-border"] {display: none;}
div[data-baseweb="tab-highlight"] {background: transparent;}
button[role="tab"] {
  background: #161b26; border: 1px solid #26304a; border-bottom: none;
  border-radius: 10px 10px 0 0; padding: 9px 18px; margin-bottom: -1px;
}
button[role="tab"] p {font-weight: 600; font-size: 0.95rem;}
button[role="tab"][aria-selected="true"] {
  background: #1f2a44; border-color: #3a4a6b;
}
button[role="tab"][aria-selected="true"] p {color: #8fb4ff;}
[data-testid="stExpander"] details {border-radius: 10px; border-color: #26304a;}
hr {border-color: #26304a;}
.pill {display:inline-block;background:#26304a;color:#c7d2fe;border-radius:8px;
  padding:1px 8px;font-size:11px;font-weight:600;margin-left:6px;}
sup.cite {background:#26304a;color:#8fb4ff;border-radius:5px;padding:0 4px;
  font-size:9px;font-weight:700;margin:0 1px 0 2px;cursor:help;vertical-align:super;}
.bullet {position:relative;padding-left:18px;margin:7px 0;line-height:1.6;}
.bullet:before {content:'▸';position:absolute;left:2px;color:#4c8dff;}
.movechip {display:inline-block;border-radius:6px;padding:1px 8px;font-size:11px;
  font-weight:600;margin:4px 6px 0 0;}
.grouptitle {font-weight:700;color:#cfe0ff;font-size:0.92rem;margin-bottom:2px;}
.evtdate {display:inline-block;font-weight:700;color:#8fb4ff;font-size:0.86rem;
  letter-spacing:.02em;}
.muted {color:#9aa4b8;font-size:0.9rem;margin:2px 0 10px;line-height:1.55;}
.srcfoot {margin-top:12px;padding-top:8px;border-top:1px solid #26304a;
  font-size:11px;color:#7a8699;line-height:2;}
.srcfoot .cite {margin-right:4px;}
.cardtitle {font-weight:700;color:#e6edf7;font-size:0.98rem;margin-bottom:2px;}
.cardmeta {color:#9aa4b8;font-size:0.8rem;margin-bottom:6px;}
.kvlabel {color:#9aa4b8;font-size:0.8rem;text-transform:uppercase;
  letter-spacing:.04em;}
.quote {border-left:3px solid #3a4a6b;padding:4px 0 4px 10px;margin:6px 0;
  color:#c7d2fe;font-style:italic;line-height:1.55;}
.chip {display:inline-block;background:#26304a;color:#c7d2fe;border-radius:8px;
  padding:2px 10px;font-size:12px;margin:3px 6px 3px 0;}
.srclink {word-break:break-all;}
</style>"""
