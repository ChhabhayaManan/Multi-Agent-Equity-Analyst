"""Chatbot: pick a researched ticker -> chat (status + inline charts) + doc list."""
import streamlit as st
import streamlit.components.v1 as components

from app.report_store import list_reports
from app.ui_helpers import chart_iframe_html, css_block
from chatbot.chatbot_agent import ChatSession
from tools.pinecone_tools import list_documents, namespace_exists
from utils.tracing import init_tracing

init_tracing()

st.set_page_config(page_title="Research Chatbot", page_icon="💬", layout="wide")
st.markdown(css_block(), unsafe_allow_html=True)

st.title("💬 Research Chatbot")
st.caption("Answers grounded in this company's indexed documents only.")
st.divider()

AVATARS = {"user": "🧑", "assistant": "📈"}

# Candidate tickers: those with saved reports (each implies an indexed namespace).
reports = list_reports()
if not reports:
    st.info("No researched companies yet. Generate a report on the Home page first.")
    st.stop()

labels = {f"{r['company_name']} ({r['ticker']})": r for r in reports}
choice = st.selectbox("Company", list(labels.keys()))
meta = labels[choice]
ticker, company_name = meta["ticker"], meta["company_name"]

if not namespace_exists(ticker):
    st.warning("This company's data is not indexed yet. Regenerate its report on Home.")
    st.stop()

# One ChatSession per ticker, cached so history survives reruns / page switches.
sessions = st.session_state.setdefault("sessions", {})
if ticker not in sessions:
    try:
        sessions[ticker] = ChatSession(ticker, company_name)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
session = sessions[ticker]
history = st.session_state.setdefault(f"history_{ticker}", [])

chat_col, doc_col = st.columns([2.5, 1])

with doc_col:
    st.subheader("Indexed documents")
    docs = list_documents(ticker)
    if not docs:
        st.caption("No documents found.")
    for source_type in sorted(docs.keys()):
        items = docs[source_type]
        with st.container(border=True):
            st.markdown(f"**{source_type}** "
                        f"<span class='pill'>{len(items)}</span>",
                        unsafe_allow_html=True)
            for d in items:
                date = f" · {d['date']}" if d.get("date") else ""
                st.caption(f"{d['document_id']}{date}")

with chat_col:
    for turn in history:
        with st.chat_message(turn["role"], avatar=AVATARS.get(turn["role"])):
            st.markdown(turn["content"])
            for chart in turn.get("charts", []):
                html = chart_iframe_html(chart)
                if html:
                    components.html(html, height=420, scrolling=True)

# chat_input at top level (not inside a column) so it pins to the page bottom;
# new turns render into chat_col above it, keeping order user -> assistant.
prompt = st.chat_input(f"Ask about {company_name}…")
if prompt:
    with chat_col:
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            status = st.status("Thinking…", expanded=False)
            # `ask` emits status lines via the private _status hook; wire it per turn.
            session._status = lambda line: status.update(label=line)
            response = session.ask(prompt)
            status.update(label="Done", state="complete")
            st.markdown(response.answer)
            if response.sources_used:
                st.caption("Tools: " + ", ".join(response.sources_used))
            charts = []
            for chart in response.charts:
                html = chart_iframe_html(chart)
                if html:
                    components.html(html, height=420, scrolling=True)
                    charts.append(chart)
        history.append({"role": "assistant", "content": response.answer, "charts": charts})
