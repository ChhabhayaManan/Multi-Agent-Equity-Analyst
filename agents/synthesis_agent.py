"""An Agent that synthesis the main report at the end."""

import json

from templates.prompts.synthesis_agent import SYNTHESIS_PROMPT
from templates.schemas.outputs import ReportOutput
from utils.helpers import get_logger, scrub_nan
from utils.llm import get_llm
from workflow.state import AGENTS

logger = get_logger(__name__)

MAX_PAYLOAD_CHARS = 18000
MAX_TEXT_CHARS = 400
CAPS = {"peers": 5, "news": 10, "events": 10,
        "guidance": 8, "risks": 8, "strategy": 6}
MISSING = "MISSING - data unavailable"
_SIGNIFICANCE = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# truncate over 400 chars, add ...
def _clip(text):
    if not isinstance(text, str) or len(text) <= MAX_TEXT_CHARS:
        return text
    return text[:MAX_TEXT_CHARS].rstrip() + "..."

#sorts news by absolute sentiment score, events by significance, and returns the top cap items
def _top_news(items, cap):
    return sorted(items, key=lambda it: -abs(it.sentiment_score))[:cap]

#sorts events by significance, and returns the top cap items
def _top_events(events, cap):
    return sorted(events, key=lambda e: _SIGNIFICANCE.get(e.significance, 3))[:cap]


def _project(name: str, obj, caps: dict) -> dict:
    """compressor for specific agent outputs."""

    if name == "fundamentals":
        return {"summary": _clip(obj.summary),
                "sector": obj.company_profile.get("sector"),
                "industry": obj.company_profile.get("industry"),
                "valuation": obj.valuation,
                "price": obj.price_snapshot,
                "shareholding": obj.shareholding}
    if name == "competitor":
        return {"standing": obj.overall_standing,
                "summary": _clip(obj.comparison_summary),
                "peers": [{"name": p.name, "intensity": p.competition_intensity,
                           "standing": p.target_standing}
                          for p in obj.peers[:caps["peers"]]]}
    if name == "news":
        return {"sentiment": obj.overall_sentiment,
                "narrative": _clip(obj.narrative),
                "items": [{"date": it.date, "title": _clip(it.title),
                           "sentiment": it.sentiment,
                           "impact": _clip(it.impact_on_stock)}
                          for it in _top_news(obj.items, caps["news"])]}
    if name == "events":
        return {"highlights": obj.highlights,
                "events": [{"date": e.date, "type": e.type,
                            "significance": e.significance,
                            "summary": _clip(e.summary),
                            "move_1d": e.price_move_1d}
                           for e in _top_events(obj.events, caps["events"])]}
    if name == "docs":
        return {"tone": obj.management_tone, "tone_trend": _clip(obj.tone_trend),
                "narrative": _clip(obj.narrative),
                "strategy": obj.strategy_highlights[:caps["strategy"]],
                "guidance": [{"metric": g.metric, "value": g.value,
                              "period": g.period}
                             for g in obj.guidance[:caps["guidance"]]],
                "risks": [{"risk": _clip(r.risk)}
                          for r in obj.risks[:caps["risks"]]]}
    return {}


def _payload(state: dict) -> str:
    """serialize the specialist outputs into a single JSON blob, truncating each section"""

    caps = dict(CAPS)
    while True:
        outputs = {n: (MISSING if _is_empty(n, state.get(n))
                       else _project(n, state[n], caps))
                   for n in AGENTS}
        payload = json.dumps(scrub_nan(outputs), default=str, allow_nan=False,
                             separators=(",", ":"))
        if len(payload) <= MAX_PAYLOAD_CHARS or min(caps.values()) <= 1:
            if len(payload) > MAX_PAYLOAD_CHARS:
                logger.warning("synthesis payload %d chars at minimum caps",
                               len(payload))
            return payload
        caps = {k: max(1, v // 2) for k, v in caps.items()}


def _is_empty(name: str, obj) -> bool:
    """True when a specialist genuinely produced no data."""

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
    """collect all source URLs from the specialist outputs, deduplicated"""

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
    """adds missing sections and sources to the final report, and returns the LLM output"""

    missing = [n for n in AGENTS if _is_empty(n, state.get(n))]
    llm = get_llm(ReportOutput)
    report = llm.invoke(SYNTHESIS_PROMPT.invoke({
        "ticker": state["ticker"], "company_name": state["company_name"],
        "missing": json.dumps(missing),
        "specialist_outputs": _payload(state),
        "retry_feedback": retry_feedback}))
    return report.model_copy(update={
        "missing_sections": missing, "sources": _collect_sources(state)})


def build_index_text(state: dict, report: ReportOutput) -> str:
    """builds a text blob for indexing in Pinecone, with all the specialist outputs and the final report summary"""
    
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
