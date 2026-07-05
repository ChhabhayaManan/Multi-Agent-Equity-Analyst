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

# AGENTS name -> report.sections key (only "competitor" differs from its section).
_SECTION_KEY = {"fundamentals": "fundamentals", "competitor": "competitors",
                "news": "news", "events": "events", "docs": "docs"}


def _is_empty(name: str, obj) -> bool:
    """True when a specialist genuinely produced no data (vs. failed a rule
    but still returned content). Only genuinely-empty sections get blanked."""
    if obj is None:
        return True
    if name == "competitor":
        return not obj.peers
    if name == "news":
        return not obj.items
    if name == "events":
        return not obj.events
    if name == "docs":
        return not obj.guidance and not obj.risks
    return False  # fundamentals: a present object is always usable


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
    # "missing" is driven by genuine data-emptiness, NOT validation status: a
    # section that failed a rule but still has content flows into the report in
    # full. Only truly-empty sections are blanked.
    missing = [n for n in AGENTS if _is_empty(n, state.get(n))]
    outputs = {
        n: ("MISSING - data unavailable" if _is_empty(n, state.get(n))
            else state[n].model_dump())
        for n in AGENTS
    }
    llm = get_llm(ReportOutput)
    report = llm.invoke(SYNTHESIS_PROMPT.invoke({
        "ticker": state["ticker"], "company_name": state["company_name"],
        "missing": json.dumps(missing),
        "specialist_outputs": json.dumps(outputs, indent=2, default=str),
        "retry_feedback": retry_feedback}))
    # Guarantee no section renders blank: any empty body becomes an explicit
    # absence note (never an empty string), whatever the LLM returned.
    sections = dict(report.sections)
    for name in AGENTS:
        key = _SECTION_KEY[name]
        if not sections.get(key, "").strip():
            sections[key] = (f"No {key} data available for "
                             f"{state['company_name']} ({state['ticker']}).")
    return report.model_copy(update={
        "sections": sections,
        "missing_sections": missing, "sources": _collect_sources(state)})
