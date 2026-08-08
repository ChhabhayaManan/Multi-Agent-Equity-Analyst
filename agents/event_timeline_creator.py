"""Event Timeline Agent"""

import json

from templates.prompts.event_timeline_creator import EVENTS_PROMPT
from templates.schemas.outputs import EventOutput
from tools.fetch_tools import fetch_bse_announcements
from tools.market_tools import price_move_around
from tools.pinecone_tools import store_to_pinecone
from utils.helpers import get_logger, scrub_nan
from utils.llm import get_llm

logger = get_logger(__name__)


def run(ticker: str, company_name: str, retry_feedback: str = ""):
    announcements = fetch_bse_announcements(ticker, days=90) #fetch the announcements
    fetch_count = len(announcements)

    moves_by_date: dict[str, dict] = {}
    for ann in announcements:
        if ann["date"] not in moves_by_date:
            moves_by_date[ann["date"]] = scrub_nan(
                price_move_around(ticker, ann["date"]))
        ann["price_moves"] = moves_by_date[ann["date"]]

    if announcements:
        try:
            docs = [f"[{a['date']}] {a['title']} (moves: {json.dumps(a['price_moves'])})"
                    for a in announcements]
            store_to_pinecone(ticker, docs, "events", meta={"document_id": "events-batch"})
        except Exception:
            logger.exception("events: pinecone store failed (non-fatal)")

    payload = (json.dumps(scrub_nan(announcements), indent=2, default=str,
                          allow_nan=False)
               if announcements else "NO ANNOUNCEMENTS FOUND")
    llm = get_llm(EventOutput)
    out = llm.invoke(EVENTS_PROMPT.invoke({
        "ticker": ticker, "company_name": company_name,
        "announcements": payload, "retry_feedback": retry_feedback}))

    out = out.model_copy(update={"events": [
        event.model_copy(update={
            "price_move_1d": (m := moves_by_date.get(event.date, {})).get("pct_1d"),
            "price_move_5d": m.get("pct_5d"),
        })
        for event in out.events
    ]})
    return out, fetch_count
