"""Event Timeline agent: fetch 90d of announcements, attach tool-computed
price moves, LLM classifies + interprets. Price moves are re-applied from
the tool data after the LLM call, keyed by event date."""

import json

from templates.prompts.event_timeline_creator import EVENTS_PROMPT
from templates.schemas.outputs import EventOutput
from tools.fetch_tools import fetch_bse_announcements
from tools.market_tools import price_move_around
from tools.pinecone_tools import store_to_pinecone
from utils.helpers import get_logger
from utils.llm import get_llm

logger = get_logger(__name__)


def run(ticker: str, company_name: str, retry_feedback: str = ""):
    announcements = fetch_bse_announcements(ticker, days=90)
    fetch_count = len(announcements)

    moves_by_date: dict[str, dict] = {}
    for ann in announcements:
        if ann["date"] not in moves_by_date:
            moves_by_date[ann["date"]] = price_move_around(ticker, ann["date"])
        ann["price_moves"] = moves_by_date[ann["date"]]

    if announcements:
        try:
            docs = [f"[{a['date']}] {a['title']} (moves: {json.dumps(a['price_moves'])})"
                    for a in announcements]
            store_to_pinecone(ticker, docs, "events", meta={"document_id": "events-batch"})
        except Exception:
            logger.exception("events: pinecone store failed (non-fatal)")

    payload = (json.dumps(announcements, indent=2, default=str)
               if announcements else "NO ANNOUNCEMENTS FOUND")
    llm = get_llm(EventOutput)
    out = llm.invoke(EVENTS_PROMPT.invoke({
        "ticker": ticker, "company_name": company_name,
        "announcements": payload, "retry_feedback": retry_feedback}))

    # Belt and braces: price moves come from the tool, keyed by date.
    fixed = []
    for event in out.events:
        moves = moves_by_date.get(event.date, {"pct_1d": None, "pct_5d": None})
        fixed.append(event.model_copy(update={
            "price_move_1d": moves["pct_1d"], "price_move_5d": moves["pct_5d"]}))
    out = out.model_copy(update={"events": fixed})
    return out, fetch_count
