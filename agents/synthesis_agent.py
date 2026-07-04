"""Report Synthesis: pure aggregation, one LLM call, no tools.
missing_sections and sources are computed in code and overwrite whatever
the LLM returned for those fields."""

import json

from templates.prompts.synthesis_agent import SYNTHESIS_PROMPT
from templates.schemas.outputs import ReportOutput
from utils.helpers import get_logger
from utils.llm import get_llm
from workflow.state import AGENTS

logger = get_logger(__name__)


def _collect_sources(state: dict) -> list[str]:
    sources: list[str] = []
    news = state.get("news")
    if news:
        sources += [it.source_url for it in news.items if it.source_url]
    docs = state.get("docs")
    if docs:
        sources += [g.source for g in docs.guidance] + [r.source for r in docs.risks]
    events = state.get("events")
    if events:
        sources += [e.filing_ref for e in events.events if e.filing_ref]
    seen, unique = set(), []
    for s in sources:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def run(state: dict, retry_feedback: str = "") -> ReportOutput:
    missing = [n for n in AGENTS
               if state["runs"].get(n, {}).get("status") == "failed_partial"]
    outputs = {
        n: (state[n].model_dump() if state.get(n) is not None
            else "MISSING - data unavailable")
        for n in AGENTS
    }
    llm = get_llm(ReportOutput)
    report = llm.invoke(SYNTHESIS_PROMPT.invoke({
        "ticker": state["ticker"], "company_name": state["company_name"],
        "missing": json.dumps(missing),
        "specialist_outputs": json.dumps(outputs, indent=2, default=str),
        "retry_feedback": retry_feedback}))
    return report.model_copy(update={
        "missing_sections": missing, "sources": _collect_sources(state)})
