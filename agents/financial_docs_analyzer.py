"""Financial Docs Analyzer: last 12 concall transcripts + latest annual
report -> Pinecone -> 5 fixed retrieval queries -> structured extraction."""

import json
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, as_completed

from templates.prompts.financial_docs_analyzer import DOCS_PROMPT
from templates.schemas.outputs import DocsOutput
from tools.fetch_tools import (fetch_annual_report_url, fetch_concall_transcripts,
                               index_pdf_document)
from tools.pinecone_tools import get_index, query_pinecone, wait_for_vectors
from utils.helpers import get_logger
from utils.llm import get_llm

logger = get_logger(__name__)

MAX_TRANSCRIPTS = 12
INDEX_WORKERS = 6  # concurrent PDF download+parse+embed+upsert jobs
QUERIES = (
    "revenue margin growth guidance targets",
    "risk factors headwinds concerns flagged by management",
    "competitive strategy priorities expansion plans",
    "management tone confidence outlook commentary",
    "future outlook next year expectations",
)


def run(ticker: str, company_name: str, retry_feedback: str = ""):
    # Build the list of (url, document_id) to index: concall transcripts + the
    # latest annual report. Each job is an independent PDF download+parse+embed
    # +upsert, so they run concurrently instead of one-at-a-time (the old
    # sequential loop was the dominant latency in this branch).
    jobs = [(t["url"], f"concall-{t.get('date', 'unknown')}")
            for t in fetch_concall_transcripts(ticker)[:MAX_TRANSCRIPTS]]
    try:
        jobs.append((fetch_annual_report_url(ticker), "annual-report"))
    except Exception:
        logger.warning("docs: no annual report found for %s", ticker)

    get_index()  # warm the shared Pinecone singleton before spawning threads
    indexed = 0
    with ThreadPoolExecutor(max_workers=INDEX_WORKERS) as pool:
        futures = {
            pool.submit(copy_context().run, index_pdf_document, url, ticker, "docs",
                        {"document_id": doc_id}): url
            for url, doc_id in jobs
        }
        for fut in as_completed(futures):
            try:
                fut.result()
                indexed += 1
            except Exception:
                logger.exception("docs: failed to index %s", futures[fut])

    contexts: dict[str, list] = {}
    if indexed:
        if not wait_for_vectors(ticker, timeout_s=90):
            logger.warning("docs: vectors not visible after store; queries may be empty")
        for q in QUERIES:
            try:
                contexts[q] = [hit["text"] for hit in query_pinecone(ticker, q, "docs", k=3)]
            except Exception:
                logger.exception("docs: query failed for %r", q)
                contexts[q] = []

    payload = json.dumps(contexts, indent=2) if contexts else "NO DOCUMENTS AVAILABLE"
    llm = get_llm(DocsOutput)
    out = llm.invoke(DOCS_PROMPT.invoke({
        "ticker": ticker, "company_name": company_name,
        "contexts": payload, "retry_feedback": retry_feedback}))
    return out, indexed
