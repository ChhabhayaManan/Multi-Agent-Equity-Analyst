"""Deterministic output validation. No LLM. Failure reasons are written to
feed the retry prompt, so keep them imperative and specific."""

import re
from datetime import datetime

from pydantic import BaseModel, Field

from utils.helpers import get_logger

logger = get_logger(__name__)

ADVICE_RE = re.compile(
    r"\b(buy|sell|hold|recommend|invest|accumulate|book profit)\b", re.IGNORECASE)

# Verbatim-external fields: quoted sources may legitimately say "invest".
_EXEMPT_FIELDS = {"quote", "title", "source", "source_url", "filing_ref"}


class ValidationResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    empty_data: bool = False


def _iter_strings(value, field_name=""):
    if isinstance(value, str):
        if field_name not in _EXEMPT_FIELDS:
            yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _iter_strings(v, str(k))
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_strings(v, field_name)


def scan_advice(output: BaseModel) -> list[str]:
    hits = set()
    for text in _iter_strings(output.model_dump()):
        for m in ADVICE_RE.finditer(text):
            hits.add(m.group().lower())
    return [f"advice word '{w}' is forbidden; rephrase without trading advice"
            for w in sorted(hits)]


def _rules_fundamentals(o) -> list[str]:
    # Quantity/length floors removed: a smaller-but-valid output is not a
    # rejection. Only the integrity check (price present) remains; types,
    # keys and enums are already enforced by the Pydantic schema.
    r = []
    if o.price_snapshot.get("price") is None:
        r.append("price_snapshot.price is missing")
    return r


def _rules_competitor(o) -> list[str]:
    # Peer-count and per-peer metric floors removed. Structure (>=3 peers,
    # metric keys, enums) is already enforced by the CompetitorOutput schema.
    return []


def _rules_news(o) -> list[str]:
    r = []
    if not o.items:
        r.append("no news items")
    if any(not it.source_url.strip() for it in o.items):
        r.append("an item is missing source_url")
    if o.items and not _mentions_any_title(o.narrative, o.items):
        r.append("narrative does not reference any article title")
    return r


def _mentions_any_title(narrative: str, items) -> bool:
    nl = narrative.lower()
    for it in items:
        title = it.title.lower()
        if title in nl:
            return True
        words = re.findall(r"[a-z]{6,}", title)
        if any(w in nl for w in words):
            return True
    return False


def _rules_events(o) -> list[str]:
    r = []
    if not o.events:
        r.append("no events")
    dates = []
    for e in o.events:
        try:
            dates.append(datetime.fromisoformat(e.date))
        except ValueError:
            r.append(f"unparseable event date '{e.date}' (need YYYY-MM-DD)")
    if dates and dates != sorted(dates):
        r.append("events are not in chronological order (oldest first)")
    return r


def _rules_docs(o) -> list[str]:
    r = []
    if not o.guidance and not o.risks:
        r.append("neither guidance nor risks extracted")
    return r  # quote length + tone enum enforced by the schema


_RULES = {
    "fundamentals": _rules_fundamentals,
    "competitor": _rules_competitor,
    "news": _rules_news,
    "events": _rules_events,
    "docs": _rules_docs,
}


def validate(agent_name: str, output, run) -> ValidationResult:
    if output is None:
        reasons = run.get("failure_reasons") or ["agent returned no output"]
        return ValidationResult(passed=False, reasons=list(reasons),
                                empty_data=run.get("fetch_count") == 0)
    reasons = scan_advice(output) + _RULES[agent_name](output)
    if reasons:
        empty = run.get("fetch_count") == 0
        if empty:
            logger.info("%s failed with empty source data; no retry", agent_name)
        return ValidationResult(passed=False, reasons=reasons, empty_data=empty)
    return ValidationResult(passed=True)
