"""News agent. Fetches 48h of newsdata.io articles, stores them for the
chatbot, and feeds them DIRECTLY to the LLM (not via a Pinecone query:
serverless is eventually consistent, an immediate query could miss)."""

import json

from templates.prompts.news_analysis_generator import NEWS_PROMPT
from templates.schemas.outputs import NewsOutput
from tools.fetch_tools import fetch_news_articles
from tools.pinecone_tools import store_to_pinecone
from utils.helpers import get_logger
from utils.llm import get_llm

logger = get_logger(__name__)


def _article_text(a: dict) -> str:
    return (f"[{a.get('pubDate', '')}] {a.get('title', '')} "
            f"({a.get('source_name', '')}, {a.get('link', '')})\n"
            f"{a.get('description') or ''}\n{a.get('content') or ''}")


def run(ticker: str, company_name: str, retry_feedback: str = ""):
    articles = fetch_news_articles(ticker, company_name)
    fetch_count = len(articles)
    if articles:
        try:
            store_to_pinecone(ticker, [_article_text(a) for a in articles], "news",
                              meta={"document_id": "news-batch"})
        except Exception:
            logger.exception("news: pinecone store failed (non-fatal)")

    payload = json.dumps(articles, indent=2, default=str) if articles else "NO ARTICLES FOUND"
    llm = get_llm(NewsOutput)
    out = llm.invoke(NEWS_PROMPT.invoke({
        "ticker": ticker, "company_name": company_name,
        "articles": payload, "retry_feedback": retry_feedback}))
    return out, fetch_count
