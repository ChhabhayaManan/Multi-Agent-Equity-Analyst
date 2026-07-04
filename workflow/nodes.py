"""LangGraph node functions. AGENT_RUNNERS maps agent name -> run callable
(uniform contract: run(ticker, company_name, retry_feedback) ->
(output, fetch_count)); tests monkeypatch entries here."""

from datetime import datetime, timezone

from agents.competitor_intelligence_agent import run as run_competitor
from agents.event_timeline_creator import run as run_events
from agents.financial_docs_analyzer import run as run_docs
from agents.fundamentals_agent import run as run_fundamentals
from agents.news_analysis_generator import run as run_news
from agents.synthesis_agent import run as run_synthesis
from tools.pinecone_tools import delete_namespace, namespace_exists, store_to_pinecone
from utils.helpers import get_logger
from workflow.state import AGENTS, GraphState, new_run
from workflow.validator import scan_advice, validate

logger = get_logger(__name__)

AGENT_RUNNERS = {
    "fundamentals": run_fundamentals,
    "competitor": run_competitor,
    "news": run_news,
    "events": run_events,
    "docs": run_docs,
}

MAX_ATTEMPTS = 2  # 1 initial + 2 validator retries


def prepare(state: GraphState) -> dict:
    """Freshness: stock data staleness policy is 'always regenerate' - wipe
    the ticker namespace so every agent re-fetches and re-indexes."""
    ticker = state["ticker"]
    fresh = False
    if namespace_exists(ticker):
        delete_namespace(ticker)
        fresh = True
        logger.info("Deleted stale namespace %s", ticker)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "namespace_fresh": fresh,
        "runs": {name: new_run() for name in AGENTS},
        "report": None,
        **{name: None for name in AGENTS},
    }


def make_run_node(name: str):
    def node(state: GraphState) -> dict:
        run = state["runs"][name]
        feedback = ""
        if run["failure_reasons"]:
            feedback = ("Previous attempt was rejected: "
                        + "; ".join(run["failure_reasons"]) + ". Fix these issues.")
        attempts = run["attempts"] + 1
        try:
            output, fetch_count = AGENT_RUNNERS[name](
                state["ticker"], state["company_name"], feedback)
            new = {**run, "status": "running", "attempts": attempts,
                   "fetch_count": fetch_count}
            return {name: output, "runs": {name: new}}
        except Exception as e:
            logger.exception("%s agent raised on attempt %d", name, attempts)
            new = {**run, "status": "running", "attempts": attempts,
                   "fetch_count": -1,  # unknown - allow retry
                   "failure_reasons": [f"agent error: {e}"]}
            return {name: None, "runs": {name: new}}

    return node


def make_validate_node(name: str):
    def node(state: GraphState) -> dict:
        run = state["runs"][name]
        result = validate(name, state.get(name), run)
        if result.passed:
            new = {**run, "status": "passed", "failure_reasons": []}
        elif result.empty_data or run["attempts"] >= MAX_ATTEMPTS:
            new = {**run, "status": "failed_partial",
                   "failure_reasons": result.reasons}
            logger.warning("%s failed_partial after %d attempts: %s",
                           name, run["attempts"], result.reasons)
        else:
            new = {**run, "status": "running", "failure_reasons": result.reasons}
        return {"runs": {name: new}}

    return node


def synthesis(state: GraphState) -> dict:
    report = None
    feedback = ""
    for attempt in range(2):  # one retry on advice-language failures OR exceptions
        try:
            candidate = run_synthesis(state, feedback)
        except Exception as e:
            if attempt == 0:
                logger.warning("synthesis raised on attempt 1: %s", e)
                continue
            logger.exception("synthesis raised on attempt 2; giving up")
            break
        problems = scan_advice(candidate)
        if not candidate.exec_summary.strip():
            problems.append("exec_summary is empty")
        if not problems:
            report = candidate
            break
        feedback = ("Previous attempt was rejected: " + "; ".join(problems)
                    + ". Fix these issues.")
        logger.warning("synthesis rejected: %s", problems)
    if report is not None:
        text = report.exec_summary + "\n\n" + "\n\n".join(
            report.sections.get(k, "") for k in
            ("fundamentals", "competitors", "events", "news", "docs"))
        try:
            store_to_pinecone(state["ticker"], [text], "report",
                              meta={"document_id": "final-report"})
        except Exception:
            logger.exception("report storage failed (non-fatal)")
    return {"report": report}
