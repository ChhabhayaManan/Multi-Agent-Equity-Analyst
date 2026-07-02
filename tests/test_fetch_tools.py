import time
import uuid
from datetime import datetime, timedelta

import pytest
import requests

from utils.helpers import load_config

TICKER = "WAAREEENER"
COMPANY_NAME = "Waaree Energies"

_cfg = load_config()
_HAS_NEWS_KEY = bool(_cfg.get("NEWSDATA_API_KEY"))
_HAS_PINECONE_KEY = bool(_cfg.get("PINECONE_API_KEY"))


def _screener_status():
    """screener.in may block datacenter IPs — probe once and skip scrape tests if blocked."""
    from tools import fetch_tools

    try:
        r = requests.get(
            f"{fetch_tools.SCREENER_BASE}/{TICKER}/consolidated/",
            headers=fetch_tools._HEADERS,
            timeout=30,
        )
        return r.status_code
    except requests.RequestException as e:
        return f"request failed: {e}"


_STATUS = _screener_status()
needs_screener = pytest.mark.skipif(
    _STATUS != 200, reason=f"screener.in blocked/unreachable (status: {_STATUS})"
)


def test_resolve_screener_slug():
    from tools.fetch_tools import resolve_screener_slug

    assert resolve_screener_slug("WAAREEENER.NS") == "WAAREEENER"
    assert resolve_screener_slug("waareeener.bo") == "WAAREEENER"
    assert resolve_screener_slug("WAAREEENER") == "WAAREEENER"


def test_parse_announcement_date():
    from tools.fetch_tools import _parse_announcement_date

    assert _parse_announcement_date("2d").date() == (datetime.now() - timedelta(days=2)).date()
    assert _parse_announcement_date("5h").date() == datetime.now().date()
    assert _parse_announcement_date("19 Mar 2025") == datetime(2025, 3, 19)
    parsed = _parse_announcement_date("28 Jun")
    assert parsed is not None and parsed.month == 6 and parsed.day == 28
    assert _parse_announcement_date("garbage") is None


@needs_screener
def test_fetch_screener_page():
    from tools.fetch_tools import fetch_screener_page

    html = fetch_screener_page(TICKER)
    assert "Waaree" in html
    assert 'id="documents"' in html


@needs_screener
def test_fetch_bse_announcements():
    from tools.fetch_tools import fetch_bse_announcements

    anns = fetch_bse_announcements(TICKER, days=90)
    assert isinstance(anns, list) and len(anns) > 0
    cutoff = datetime.now() - timedelta(days=91)
    for a in anns:
        assert set(a) == {"date", "title", "pdf_url"}
        assert datetime.strptime(a["date"], "%Y-%m-%d") >= cutoff
        assert a["title"]
        assert a["pdf_url"].startswith("http")


@needs_screener
def test_fetch_concall_transcripts():
    from tools.fetch_tools import fetch_concall_transcripts

    transcripts = fetch_concall_transcripts(TICKER)
    assert isinstance(transcripts, list) and len(transcripts) > 0
    for t in transcripts:
        assert set(t) == {"date", "url"}
        assert t["url"].startswith("http")
        assert t["date"]  # e.g. 'May 2026'


@needs_screener
def test_fetch_investor_presentations():
    from tools.fetch_tools import fetch_investor_presentations

    ppts = fetch_investor_presentations(TICKER)
    assert isinstance(ppts, list) and len(ppts) > 0
    for p in ppts:
        assert set(p) == {"date", "url"}
        assert p["url"].startswith("http")


@needs_screener
def test_fetch_annual_report_url():
    from tools.fetch_tools import fetch_annual_report_url

    url = fetch_annual_report_url(TICKER)
    assert url.startswith("http")


@needs_screener
def test_parse_pdf_to_text():
    from tools.fetch_tools import fetch_concall_transcripts, parse_pdf_to_text

    # Transcripts are small (tens of pages) vs annual reports (hundreds).
    url = fetch_concall_transcripts(TICKER)[0]["url"]
    text = parse_pdf_to_text(url)
    assert isinstance(text, str) and len(text) > 500
    assert "waaree" in text.lower()


@pytest.mark.skipif(not _HAS_NEWS_KEY, reason="NEWSDATA_API_KEY not set in .env")
def test_fetch_news_articles():
    from tools.fetch_tools import fetch_news_articles

    articles = fetch_news_articles(TICKER, COMPANY_NAME, hours=48)
    # A 48h window may legitimately be empty — assert shape, not length.
    assert isinstance(articles, list)
    for a in articles:
        assert {"article_id", "title", "link", "pubDate", "source_name"} <= set(a)
        blob = f"{a.get('title') or ''} {a.get('description') or ''} {a.get('content') or ''}"
        assert COMPANY_NAME.lower() in blob.lower()


@pytest.mark.skipif(
    not _HAS_PINECONE_KEY or _STATUS != 200,
    reason=f"needs PINECONE_API_KEY and screener access (status: {_STATUS})",
)
def test_index_pdf_document():
    from tools.fetch_tools import fetch_concall_transcripts, index_pdf_document
    from tools.pinecone_tools import get_index, query_pinecone

    ns = f"WAAREEENER-TEST-{uuid.uuid4().hex[:6]}"
    url = fetch_concall_transcripts(TICKER)[0]["url"]
    index = get_index()
    try:
        index_pdf_document(url, ns, "docs", meta={"document_id": "concall-test"})
        for _ in range(30):  # serverless: eventually consistent
            stats = index.describe_index_stats()
            if ns in (stats.namespaces or {}):
                break
            time.sleep(2)
        time.sleep(2)
        results = query_pinecone(ns, "revenue and margins", "docs", k=2)
        assert len(results) > 0
        assert results[0]["metadata"]["source_type"] == "docs"
        assert results[0]["metadata"]["document_id"] == "concall-test"
    finally:
        try:
            index.delete(delete_all=True, namespace=ns)
        except Exception:
            pass  # namespace may not exist if indexing failed before any upsert
