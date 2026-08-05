"""Home (index) page. `streamlit run app/main.py`.
Search an NSE ticker -> generate report (live progress) -> render -> PDF.
The Chatbot lives in app/pages/1_Chatbot.py."""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from streamlit_searchbox import st_searchbox

from app.pdf_builder import build_pdf
from app.quote import get_quote
from app.report_store import load_report, namespace_of, save_report
from app.sections import (render_competitors, render_docs, render_events,
                          render_fundamentals, render_news, render_run_detail,
                          render_sources)
from app.ui_helpers import css_block, fmt_market_cap, fmt_num, status_icon
from tools.market_tools import search_ticker
from utils.tracing import init_tracing
from workflow.graph import stream_report

init_tracing()

AGENT_TABS = {"fundamentals": "Company & Fundamentals",
              "competitor": "Competitive Landscape",
              "events": "Event Timeline", "news": "News Analysis",
              "docs": "Financial Documents"}
PROGRESS_ROWS = ["fundamentals", "competitor", "news", "events", "docs", "synthesis"]
PROGRESS_LABELS = {**AGENT_TABS, "synthesis": "Synthesis"}
_ICONS = {"passed": "✅", "no_data": "📭", "failed_partial": "⚠️",
          "running": "🔄", "pending": "⏳"}

st.set_page_config(page_title="Stock Research Platform",
                   page_icon="📈", layout="wide")
st.markdown(css_block(), unsafe_allow_html=True)
st.session_state.setdefault("sessions", {})

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


def _render_report(stored: dict) -> None:
    report = stored["report"]
    specialists = stored.get("specialists", {})
    runs = stored.get("runs", {})
    st.markdown(f"## {stored['company_name']} "
                f"<span class='pill'>{stored['ticker']}</span>",
                unsafe_allow_html=True)
    st.caption(f"Generated {stored['generated_at']}")
    _metric_tiles(ticker)
    with st.container(border=True):
        st.markdown("#### Executive Summary")
        st.markdown(report.exec_summary)
    labels = [f"{status_icon((runs.get(name) or {}).get('status'))} {title}"
              for name, title in AGENT_TABS.items()] + ["📚 Sources"]
    tabs = st.tabs(labels)
    for tab, name in zip(tabs, AGENT_TABS):
        with tab:
            render_run_detail(runs.get(name))
            obj = specialists.get(name)
            if name == "fundamentals":
                render_fundamentals(obj, stored["generated_at"])
            elif name == "competitor":
                render_competitors(obj, specialists.get("fundamentals"),
                                   stored["ticker"], stored["company_name"])
            elif name == "events":
                render_events(obj)
            elif name == "news":
                render_news(obj)
            else:
                render_docs(obj)
    with tabs[-1]:
        render_sources(specialists)
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
            save_report(ticker, company_name, final["report"], generated_at,
                        final["specialists"], final["runs"])
            st.session_state["_fresh_report"] = {
                "ticker": namespace_of(ticker), "company_name": company_name,
                "generated_at": generated_at, "report": final["report"],
                "specialists": final["specialists"], "runs": final["runs"]}
        else:
            status.update(label="No report produced", state="error")


button_label = "♻️ Regenerate" if existing else "🚀 Generate report"
if existing and existing.get("legacy"):
    st.warning("This report was saved in an older format that did not keep the "
               "detailed agent data. Regenerate it to see the full breakdown.")
elif existing:
    _render_report(existing)
else:
    st.info("No saved report for this ticker yet.")

if st.button(button_label):
    _generate()
    fresh = st.session_state.pop("_fresh_report", None)
    if fresh:
        _render_report(fresh)
