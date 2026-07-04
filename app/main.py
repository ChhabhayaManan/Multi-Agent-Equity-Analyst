"""Home (index) page. `streamlit run app/main.py`.
Search an NSE ticker -> generate report (live progress) -> render -> PDF.
The Chatbot lives in app/pages/1_Chatbot.py."""
from datetime import datetime

import streamlit as st
from streamlit_searchbox import st_searchbox

from app.pdf_builder import build_pdf
from app.report_store import load_report, namespace_of, save_report
from app.ui_helpers import section_note
from tools.market_tools import search_ticker
from utils.tracing import init_tracing
from workflow.graph import stream_report

init_tracing()

SECTION_ORDER = ["fundamentals", "competitors", "events", "news", "docs"]
SECTION_TITLES = {
    "fundamentals": "Company & Fundamentals", "competitors": "Competitive Landscape",
    "events": "Event Timeline", "news": "News Analysis", "docs": "Financial Documents"}
PROGRESS_ROWS = ["fundamentals", "competitor", "news", "events", "docs", "synthesis"]
_ICONS = {"passed": "✅", "failed_partial": "⚠️", "running": "🔄", "pending": "⏳"}

st.set_page_config(page_title="Stock Research Platform",
                   page_icon="📈", layout="wide")
st.session_state.setdefault("sessions", {})   # ticker -> ChatSession (chatbot page)

st.title("📈 Research an NSE Company")
st.caption("NSE-only equity research. Read-only. Not investment advice.")


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


def _render_report(stored: dict) -> None:
    report = stored["report"]
    st.subheader(f"{stored['company_name']} ({stored['ticker']})")
    st.caption(f"Generated {stored['generated_at']}")
    st.markdown("### Executive Summary")
    st.markdown(report.exec_summary)
    for key in SECTION_ORDER:
        st.markdown(f"### {SECTION_TITLES[key]}")
        note = section_note(report.missing_sections, key)
        if note:
            st.markdown(note)
        st.markdown(report.sections.get(key, ""))
    if report.sources:
        st.markdown("### Sources")
        for s in report.sources:
            st.markdown(f"- {s}")
    st.download_button(
        "⬇️ Download PDF", data=build_pdf(stored),
        file_name=f"{stored['ticker']}_research_{datetime.now():%Y%m%d}.pdf",
        mime="application/pdf")
    st.page_link("pages/1_Chatbot.py", label="💬 Ask questions about this company")


def _generate() -> None:
    with st.status("Generating report…", expanded=True) as status:
        rows = {name: st.empty() for name in PROGRESS_ROWS}
        for name in PROGRESS_ROWS:
            rows[name].markdown(f"⏳ {name}")
        final = None
        for update in stream_report(ticker, company_name):
            for name in PROGRESS_ROWS:
                run = update["runs"].get(name)
                state = run["status"] if run else None
                if name == "synthesis" and update["report"] is not None:
                    state = "passed"
                rows[name].markdown(f"{_ICONS.get(state, '⏳')} {name}")
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
