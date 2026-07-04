"""Financial Docs Analyzer: last 12 concall transcripts + latest annual
report -> Pinecone -> 5 fixed retrieval queries -> structured extraction."""

import json

from templates.prompts.financial_docs_analyzer import DOCS_PROMPT
from templates.schemas.outputs import DocsOutput
from tools.fetch_tools import (fetch_annual_report_url, fetch_concall_transcripts,
                               index_pdf_document)
from tools.pinecone_tools import query_pinecone, wait_for_vectors
from utils.helpers import get_logger
from utils.llm import get_llm

logger = get_logger(__name__)

MAX_TRANSCRIPTS = 12
QUERIES = (
    "revenue margin growth guidance targets",
    "risk factors headwinds concerns flagged by management",
    "competitive strategy priorities expansion plans",
    "management tone confidence outlook commentary",
    "future outlook next year expectations",
)


def run(ticker: str, company_name: str, retry_feedback: str = ""):
    indexed = 0
    for t in fetch_concall_transcripts(ticker)[:MAX_TRANSCRIPTS]:
        try:
            index_pdf_document(t["url"], ticker, "docs",
                               meta={"document_id": f"concall-{t.get('date', 'unknown')}"})
            indexed += 1
        except Exception:
            logger.exception("docs: failed to index transcript %s", t.get("url"))
    try:
        ar_url = fetch_annual_report_url(ticker)
        index_pdf_document(ar_url, ticker, "docs", meta={"document_id": "annual-report"})
        indexed += 1
    except Exception:
        logger.warning("docs: no annual report indexed for %s", ticker)

    contexts: dict[str, list] = {}
    if indexed:
        if not wait_for_vectors(ticker, timeout_s=90):
            logger.warning("docs: vectors not visible after store; queries may be empty")
        for q in QUERIES:
            try:
                contexts[q] = [hit["text"] for hit in query_pinecone(ticker, q, "docs", k=5)]
            except Exception:
                logger.exception("docs: query failed for %r", q)
                contexts[q] = []

    payload = json.dumps(contexts, indent=2) if contexts else "NO DOCUMENTS AVAILABLE"
    llm = get_llm(DocsOutput)
    out = llm.invoke(DOCS_PROMPT.invoke({
        "ticker": ticker, "company_name": company_name,
        "contexts": payload, "retry_feedback": retry_feedback}))
    return out, indexed
