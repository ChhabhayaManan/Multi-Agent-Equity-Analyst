"""Report Synthesis: pure aggregation, one LLM call, no tools.
missing_sections and sources are computed in code and overwrite whatever
the LLM returned for those fields. build_index_text serializes the whole
run for the chatbot's retrievable report document."""

import json

from templates.prompts.synthesis_agent import SYNTHESIS_PROMPT
from templates.schemas.outputs import ReportOutput
from utils.helpers import get_logger
from utils.llm import get_llm
from workflow.state import AGENTS

logger = get_logger(__name__)


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
    return False


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
    return report.model_copy(update={
        "missing_sections": missing, "sources": _collect_sources(state)})


def build_index_text(state: dict, report: ReportOutput) -> str:
    """Plain-text serialization of the whole report for the Pinecone
    'final-report' doc. Deterministic, no LLM: the chatbot retrieves this."""
    parts = [f"EXECUTIVE SUMMARY\n{report.exec_summary}"]

    f = state.get("fundamentals")
    if f:
        parts.append(
            "FUNDAMENTALS\n"
            f"{f.summary}\n"
            f"profile: {json.dumps(f.company_profile, default=str)}\n"
            f"valuation: {json.dumps(f.valuation, default=str)}\n"
            f"price: {json.dumps(f.price_snapshot, default=str)}\n"
            f"shareholding: {json.dumps(f.shareholding, default=str)}")

    c = state.get("competitor")
    if c:
        rows = "\n".join(
            f"- {p.name} ({p.ticker}): intensity {p.competition_intensity}, "
            f"target {p.target_standing}. {p.reason_for_inclusion} "
            f"metrics {json.dumps(p.metrics, default=str)}"
            for p in c.peers)
        parts.append(f"COMPETITORS (target standing {c.overall_standing})\n"
                     f"{c.comparison_summary}\n{rows}")

    n = state.get("news")
    if n:
        rows = "\n".join(
            f"- {it.date} {it.title} [{it.sentiment} {it.sentiment_score}] "
            f"{it.summary} Impact: {it.impact_on_stock} "
            f"Sector: {it.sector_impact} ({it.source_url})"
            for it in n.items)
        parts.append(f"NEWS (overall {n.overall_sentiment})\n{n.narrative}\n{rows}")

    e = state.get("events")
    if e:
        rows = "\n".join(
            f"- {ev.date} [{ev.type}/{ev.significance}] {ev.summary} "
            f"Meaning: {ev.what_it_meant} Effect: {ev.how_it_affected} "
            f"1D {ev.price_move_1d} 5D {ev.price_move_5d} "
            f"ref {ev.filing_ref or '-'}"
            for ev in e.events)
        highlights = "\n".join(f"* {h}" for h in e.highlights)
        parts.append(f"EVENTS\n{highlights}\n{rows}")

    d = state.get("docs")
    if d:
        guid = "\n".join(
            f"- {g.metric}: {g.value} for {g.period} ({g.source}) \"{g.quote}\""
            for g in d.guidance)
        risks = "\n".join(f"- {r.risk} ({r.source}) \"{r.quote}\"" for r in d.risks)
        strat = "\n".join(f"* {s}" for s in d.strategy_highlights)
        parts.append(
            f"DOCUMENTS (tone {d.management_tone}; {d.tone_trend})\n"
            f"{d.narrative}\nGUIDANCE\n{guid}\nRISKS\n{risks}\nSTRATEGY\n{strat}")

    return "\n\n".join(parts)
