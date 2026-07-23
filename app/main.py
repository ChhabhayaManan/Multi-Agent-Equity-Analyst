"""Home (index) page. `streamlit run app/main.py`.
Search an NSE ticker -> generate report (live progress) -> render -> PDF.
The Chatbot lives in app/pages/1_Chatbot.py."""
import sys
from datetime import datetime
from pathlib import Path

# Streamlit only puts this script's own dir on sys.path, not repo root,
# so absolute `app.*`/`tools.*`/`workflow.*` imports fail on deploy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from streamlit_searchbox import st_searchbox

from app.pdf_builder import build_pdf
from app.quote import get_quote
from app.report_store import load_report, namespace_of, save_report
from app.ui_helpers import (css_block, extract_citations, fmt_market_cap,
                            fmt_num, move_chip_html, parse_price_moves, prose_html,
                            section_note, split_bullets, split_labeled,
                            strip_move_sentence)
from tools.market_tools import search_ticker
from utils.tracing import init_tracing
from workflow.graph import stream_report

init_tracing()

SECTION_ORDER = ["fundamentals", "competitors", "events", "news", "docs"]
SECTION_TITLES = {
    "fundamentals": "Company & Fundamentals", "competitors": "Competitive Landscape",
    "events": "Event Timeline", "news": "News Analysis", "docs": "Financial Documents"}
PROGRESS_ROWS = ["fundamentals", "competitor", "news", "events", "docs", "synthesis"]
PROGRESS_LABELS = {
    "fundamentals": "Company & Fundamentals", "competitor": "Competitive Landscape",
    "news": "News Analysis", "events": "Event Timeline",
    "docs": "Financial Documents", "synthesis": "Synthesis"}
_ICONS = {"passed": "✅", "no_data": "📭", "failed_partial": "⚠️",
          "running": "🔄", "pending": "⏳"}

st.set_page_config(page_title="Stock Research Platform",
                   page_icon="📈", layout="wide")
st.markdown(css_block(), unsafe_allow_html=True)
st.session_state.setdefault("sessions", {})   # ticker -> ChatSession (chatbot page)

st.title("📈 Research an NSE Company")
st.caption("NSE-only equity research. Read-only. Not investment advice.")
st.divider()


def _search(query: str):
    if not query:
        return []
    try:
        matches = search_ticker(query)
    except Exception:
        return []
    return [(f"{m['ticker']} — {m['name']} ({m['exchange']})", (m["ticker"], m["name"]))
            for m in matches
            if (m.get("ticker") or "").upper().endswith(".NS")][:8]


picked = st_searchbox(_search, key="ticker_search",
                      placeholder="Type an NSE company or ticker…")

if not picked:
    st.stop()

ticker, company_name = picked
existing = load_report(ticker)


def _metric_tiles(symbol: str) -> None:
    """Live yfinance quote row. Silent no-op if the fetch fails."""
    q = get_quote(symbol)
    if not q:
        return
    c1, c2, c3, c4 = st.columns(4)
    delta = (f"{q['day_change_pct']:+.2f}%"
             if q["day_change_pct"] is not None else None)
    c1.metric("Price", fmt_num(q["price"], "₹"), delta)
    c2.metric("Market Cap", fmt_market_cap(q["market_cap"]))
    c3.metric("P/E (TTM)", fmt_num(q["pe"]))
    hi, lo = q["year_high"], q["year_low"]
    rng = (f"₹{lo:,.0f} – ₹{hi:,.0f}" if hi and lo else "—")
    c4.metric("52-week range", rng)


_DOC_ICONS = {
    "guidance": "🎯", "target": "🎯", "outlook": "🎯",
    "risk": "⚠️", "headwind": "⚠️",
    "strategy": "🧭", "strateg": "🧭", "priorit": "🧭",
    "refiner": "🏭", "expansion": "🏭", "capacity": "🏭", "manufactur": "🏭",
    "financ": "💰", "margin": "💰",
    "tone": "🗣️", "management": "🗣️",
}


def _doc_icon(label: str) -> str:
    low = label.lower()
    for key, icon in _DOC_ICONS.items():
        if key in low:
            return icon
    return "•"


def _sources_footer(cites: list) -> None:
    if not cites:
        return
    body = " ".join(f"<span class='cite'>{i}</span>{c}"
                    for i, c in enumerate(cites, 1))
    st.markdown(f"<div class='srcfoot'>{body}</div>", unsafe_allow_html=True)


def _render_bullets(text: str) -> None:
    """Generic digestible render: citations -> chips, prose -> bullets + footer."""
    clean, cites = extract_citations(text)
    for b in split_bullets(clean):
        st.markdown(f"<div class='bullet'>{prose_html(b, cites)}</div>",
                    unsafe_allow_html=True)
    _sources_footer(cites)


def _render_events(text: str) -> None:
    clean, cites = extract_citations(text)
    intro, groups = split_labeled(clean)
    if not groups:
        _render_bullets(text)
        return
    if intro:
        st.markdown(f"<div class='muted'>{prose_html(intro, cites)}</div>",
                    unsafe_allow_html=True)
    for date, body in groups:
        d1, d5 = parse_price_moves(body)
        summary = strip_move_sentence(body)
        with st.container(border=True):
            left, right = st.columns([1, 3.4])
            left.markdown(f"<span class='evtdate'>{date}</span>",
                          unsafe_allow_html=True)
            with right:
                if summary:
                    st.markdown(prose_html(summary, cites),
                                unsafe_allow_html=True)
                chips = move_chip_html("1D", d1) + move_chip_html("5D", d5)
                if chips:
                    st.markdown(chips, unsafe_allow_html=True)
    _sources_footer(cites)


def _render_docs(text: str) -> None:
    clean, cites = extract_citations(text)
    intro, groups = split_labeled(clean)
    if not groups:
        _render_bullets(text)
        return
    if intro:
        st.markdown(f"<div class='muted'>{prose_html(intro, cites)}</div>",
                    unsafe_allow_html=True)
    for label, body in groups:
        with st.container(border=True):
            st.markdown(f"<div class='grouptitle'>{_doc_icon(label)} {label}"
                        "</div>", unsafe_allow_html=True)
            for b in split_bullets(body):
                st.markdown(f"<div class='bullet'>{prose_html(b, cites)}</div>",
                            unsafe_allow_html=True)
    _sources_footer(cites)


_SECTION_RENDERERS = {"events": _render_events, "docs": _render_docs}


def _render_section(key: str, content: str) -> None:
    """Dispatch to a section-specific layout; fail soft to raw markdown."""
    if not content or not content.strip() or "unavailable" in content.lower():
        st.info("Data unavailable for this section.")
        return
    try:
        _SECTION_RENDERERS.get(key, _render_bullets)(content)
    except Exception:
        st.markdown(content)   # never let a parse quirk hide the content


def _render_report(stored: dict) -> None:
    report = stored["report"]
    st.markdown(f"## {stored['company_name']} "
                f"<span class='pill'>{stored['ticker']}</span>",
                unsafe_allow_html=True)
    st.caption(f"Generated {stored['generated_at']}")
    _metric_tiles(ticker)
    with st.container(border=True):
        st.markdown("#### Executive Summary")
        summary_clean, summary_cites = extract_citations(report.exec_summary)
        st.markdown(prose_html(summary_clean, summary_cites),
                    unsafe_allow_html=True)
        _sources_footer(summary_cites)
    tabs = st.tabs([SECTION_TITLES[k] for k in SECTION_ORDER])
    for tab, key in zip(tabs, SECTION_ORDER):
        with tab:
            if section_note(report.missing_sections, key):
                if key == "news":
                    st.info("No recent news found for this stock.")
                else:
                    st.info("Data unavailable for this section.")
            else:
                _render_section(key, report.sections.get(key, ""))
    if report.sources:
        with st.expander(f"Sources ({len(report.sources)})"):
            for s in report.sources:
                st.markdown(f"- {s}")
    act1, act2 = st.columns([1, 1])
    with act1:
        st.download_button(
            "⬇️ Download PDF", data=build_pdf(stored),
            file_name=f"{stored['ticker']}_research_{datetime.now():%Y%m%d}.pdf",
            mime="application/pdf", use_container_width=True)
    with act2:
        st.page_link("pages/1_Chatbot.py",
                     label="💬 Ask questions about this company",
                     use_container_width=True)


def _generate() -> None:
    with st.status("Generating report…", expanded=True) as status:
        rows = {name: st.empty() for name in PROGRESS_ROWS}
        for name in PROGRESS_ROWS:
            rows[name].markdown(f"⏳ {PROGRESS_LABELS[name]}")
        final = None
        for update in stream_report(ticker, company_name):
            for name in PROGRESS_ROWS:
                run = update["runs"].get(name)
                state = run["status"] if run else None
                if name == "synthesis" and update["report"] is not None:
                    state = "passed"
                rows[name].markdown(
                    f"{_ICONS.get(state, '⏳')} {PROGRESS_LABELS[name]}")
            if update["done"]:
                final = update
        if final and final["report"] is not None:
            status.update(label="Report ready", state="complete")
            generated_at = datetime.now().isoformat(timespec="seconds")
            save_report(ticker, company_name, final["report"], generated_at)
            st.session_state["_fresh_report"] = {
                "ticker": namespace_of(ticker), "company_name": company_name,
                "generated_at": generated_at, "report": final["report"]}
        else:
            status.update(label="No report produced", state="error")


button_label = "♻️ Regenerate" if existing else "🚀 Generate report"
if existing:
    _render_report(existing)
else:
    st.info("No saved report for this ticker yet.")

if st.button(button_label):
    _generate()
    fresh = st.session_state.pop("_fresh_report", None)
    if fresh:
        _render_report(fresh)
