"""Push current composed prompt templates to LangSmith Hub for versioning.

Code stays the source of truth; this is push-only (no runtime pull). Run
manually when prompts change. No-ops without a LangSmith key."""

from langchain_core.prompts import ChatPromptTemplate

from templates.prompts.chatbot import CHATBOT_SYSTEM
from templates.prompts.competitor_intelligence_agent import (
    COMPETITOR_STRUCT_PROMPT, COMPETITOR_SYSTEM)
from templates.prompts.event_timeline_creator import EVENTS_PROMPT
from templates.prompts.financial_docs_analyzer import DOCS_PROMPT
from templates.prompts.fundamentals_agent import FUNDAMENTALS_PROMPT
from templates.prompts.news_analysis_generator import NEWS_PROMPT
from templates.prompts.synthesis_agent import SYNTHESIS_PROMPT
from utils.helpers import get_logger
from utils.tracing import init_tracing

logger = get_logger(__name__)


def _wrap(system_text: str, human: str = "{input}") -> ChatPromptTemplate:
    """Wrap a plain system-string prompt so LangSmith can version it."""
    return ChatPromptTemplate.from_messages(
        [("system", system_text), ("human", human)])


PROMPTS = {
    "stock-research-fundamentals": FUNDAMENTALS_PROMPT,
    "stock-research-news": NEWS_PROMPT,
    "stock-research-events": EVENTS_PROMPT,
    "stock-research-docs": DOCS_PROMPT,
    "stock-research-synthesis": SYNTHESIS_PROMPT,
    "stock-research-competitor-struct": COMPETITOR_STRUCT_PROMPT,
    "stock-research-competitor-system": _wrap(
        COMPETITOR_SYSTEM, "Ticker: {ticker} ({company_name})"),
    "stock-research-chatbot": _wrap(CHATBOT_SYSTEM),
}


def _client():
    from langsmith import Client
    return Client()


def push_all() -> int:
    """Push every prompt to LangSmith Hub. Returns the count pushed (0 if
    tracing/LangSmith is unavailable)."""
    if not init_tracing():
        logger.warning("Tracing disabled (no LANGCHAIN_API_KEY); skipping push")
        return 0
    client = _client()
    n = 0
    for name, prompt in PROMPTS.items():
        client.push_prompt(name, object=prompt)
        logger.info("pushed prompt %s", name)
        n += 1
    logger.info("pushed %d prompts", n)
    return n


if __name__ == "__main__":
    count = push_all()
    print(f"pushed {count} prompts")
