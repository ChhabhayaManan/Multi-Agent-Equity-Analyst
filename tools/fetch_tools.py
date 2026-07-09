import re
from datetime import datetime, timedelta
from typing import List, Optional

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

from utils.helpers import disk_cache, get_logger, load_config
from utils.tracing import traceable

logger = get_logger(__name__)

NEWSDATA_URL = "https://newsdata.io/api/1/latest"
SCREENER_BASE = "https://www.screener.in/company"
# screener.in serves 403 to default python-requests UA; a real browser UA is required.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
}
_ARTICLE_FIELDS = (
    "article_id",
    "title",
    "description",
    "content",
    "link",
    "pubDate",
    "source_name",
    "keywords",
)


@disk_cache(ttl_hours=6)
@traceable(name="fetch_news_articles")
def fetch_news_articles(ticker: str, company_name: str, hours: int = 48) -> List[dict]:
    """Fetch recent news for company_name from newsdata.io (/api/1/latest).

    KNOWN LIMITATION: `timeframe` is capped at 48 hours on paid plans and is
    NOT a 30-day window; true historical ranges need the paid Archive API.
    In practice the free plan rejects `timeframe` outright (422 'upgrade your
    plan'), so on that error we retry without it and take whatever recency
    /latest gives (free-plan results are already delayed ~12h). `size` is
    capped at 10 per request on the free plan.

    company_name is sent as a quoted exact phrase and each article is
    post-filtered to contain the name verbatim, to avoid pulling news of
    similarly-named companies (e.g. Waaree Energies vs Waaree Renewable Energy).
    """
    # Key goes in a header, not the URL, so it can't leak via logs/tracebacks.
    headers = {"X-ACCESS-KEY": load_config()["NEWSDATA_API_KEY"]}
    params = {
        "q": f'"{company_name}"',
        "country": "in",
        "language": "en",
        "timeframe": min(hours, 48),
        "size": 10,
    }
    articles: List[dict] = []
    # First page + at most 2 nextPage follows, to stay within the daily credit budget.
    for _ in range(3):
        resp = requests.get(NEWSDATA_URL, params=params, headers=headers, timeout=30)
        if resp.status_code == 422 and "timeframe" in params and "timeframe" in resp.text:
            del params["timeframe"]
            resp = requests.get(NEWSDATA_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("results") or []:
            blob = " ".join(
                str(item.get(f) or "") for f in ("title", "description", "content")
            ).lower()
            if company_name.lower() not in blob:
                continue
            articles.append({f: item.get(f) for f in _ARTICLE_FIELDS})
        next_page = data.get("nextPage")
        if not next_page:
            break
        params["page"] = next_page
    return articles


def resolve_screener_slug(ticker: str) -> str:
    return re.sub(r"\.(NS|BO)$", "", ticker.upper())


@disk_cache(ttl_hours=24)
@traceable(name="fetch_screener_page")
def fetch_screener_page(ticker: str) -> str:
    """Raw HTML of the screener.in company page (consolidated view, standalone fallback)."""
    slug = resolve_screener_slug(ticker)
    resp = requests.get(f"{SCREENER_BASE}/{slug}/consolidated/", headers=_HEADERS, timeout=30)
    if resp.status_code == 404:
        resp = requests.get(f"{SCREENER_BASE}/{slug}/", headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_announcement_date(raw: str) -> Optional[datetime]:
    """Screener announcement dates: '2d', '5h', '28 Jun', '19 Mar 2025'."""
    raw = raw.strip()
    m = re.match(r"^(\d+)d$", raw)
    if m:
        return datetime.now() - timedelta(days=int(m.group(1)))
    m = re.match(r"^(\d+)[hm]$", raw)
    if m:
        return datetime.now()
    try:
        return datetime.strptime(raw, "%d %b %Y")
    except ValueError:
        pass
    try:  # day + month only -> assume current year (last year if that lands in the future)
        dt = datetime.strptime(f"{raw} {datetime.now().year}", "%d %b %Y")
        if dt > datetime.now() + timedelta(days=1):  # e.g. '28 Dec' seen in Jan
            dt = dt.replace(year=dt.year - 1)
        return dt
    except ValueError:
        return None


def _documents_section(ticker: str) -> BeautifulSoup:
    soup = BeautifulSoup(fetch_screener_page(ticker), "lxml")
    section = soup.find(id="documents")
    if section is None:
        raise ValueError(f"No documents section on screener page for {ticker}")
    return section


def fetch_bse_announcements(ticker: str, days: int = 90) -> List[dict]:
    """BSE announcements from screener's Announcements panel: [{date, title, pdf_url}].

    Date label sits in a .ink-600.smaller div/span as '2d - summary' / '28 Jun' /
    '19 Mar 2025'; entries whose dates can't be parsed are skipped.
    """
    section = _documents_section(ticker)
    tab = section.find(id="company-announcements-tab") or section
    cutoff = datetime.now() - timedelta(days=days)
    out = []
    for a in tab.select("ul.list-links li a[href]"):
        date_el = a.find(class_="ink-600")
        if date_el is None:
            continue
        date_label = date_el.get_text(strip=True).split(" - ")[0]
        dt = _parse_announcement_date(date_label)
        if dt is None or dt < cutoff:
            continue
        title = a.find(string=True, recursive=False)
        out.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "title": (title or "").strip(),
                "pdf_url": a["href"],
            }
        )
    return out


def _concall_links(ticker: str, label: str) -> List[dict]:
    """Concalls panel rows: a date div ('May 2026') followed by <a class=concall-link>
    labeled Transcript/PPT/REC (missing items render as plain <div>, not links)."""
    section = _documents_section(ticker)
    concalls = section.find("div", class_="concalls")
    if concalls is None:
        return []
    out = []
    for li in concalls.select("ul.list-links li"):
        date_div = li.find("div", class_="nowrap")
        date = date_div.get_text(strip=True) if date_div else ""
        for a in li.find_all("a", class_="concall-link"):
            if a.get_text(strip=True) == label and a.get("href"):
                out.append({"date": date, "url": a["href"]})
    return out


def fetch_concall_transcripts(ticker: str) -> List[dict]:
    return _concall_links(ticker, "Transcript")


def fetch_investor_presentations(ticker: str) -> List[dict]:
    return _concall_links(ticker, "PPT")


def fetch_annual_report_url(ticker: str) -> str:
    """Most recent annual report PDF (first entry in the Annual reports panel)."""
    section = _documents_section(ticker)
    panel = section.find("div", class_="annual-reports")
    if panel is None:
        raise ValueError(f"No annual reports panel for {ticker}")
    link = panel.select_one("ul.list-links li a[href]")
    if link is None:
        raise ValueError(f"No annual report links for {ticker}")
    return link["href"]


@traceable(
    name="parse_pdf_to_text",
    process_outputs=lambda out: {"chars": len(out or "")},
)
def parse_pdf_to_text(url: str) -> str:
    """Download a PDF and extract text with PyMuPDF, pages joined by newlines.

    PyMuPDF (fitz) is ~20-30x faster than pdfplumber on large (200-400pg)
    annual reports, which dominated the docs branch latency. No OCR — assumes
    digitally-generated PDFs; layout/column order is not preserved, which is
    fine since the text is chunked and embedded for semantic search.
    """
    resp = requests.get(url, headers=_HEADERS, timeout=60)
    resp.raise_for_status()
    pages = []
    with fitz.open(stream=resp.content, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


@traceable(name="index_pdf_document")
def index_pdf_document(url: str, ticker: str, source_type: str, meta: Optional[dict] = None) -> None:
    """Parse a PDF and store its text into Pinecone namespace=ticker in one call."""
    from tools.pinecone_tools import store_to_pinecone

    text = parse_pdf_to_text(url)
    if not text.strip():
        logger.warning("No text extracted from %s; skipping indexing", url)
        return
    store_to_pinecone(ticker, [text], source_type, meta)


def fetch_shareholding(ticker: str) -> dict:
    """Latest-quarter shareholding pattern from screener.in's Shareholding section.

    Screener renders a quarterly table (section id='shareholding'): rows are
    'Promoters +', 'FIIs +', 'DIIs +', 'Public +'; the last column is the most
    recent quarter. Returns percentages as floats, None for anything missing.
    """
    empty = {"promoter": None, "fii": None, "dii": None, "public": None, "quarter": None}
    soup = BeautifulSoup(fetch_screener_page(ticker), "lxml")
    section = soup.find(id="shareholding")
    if section is None:
        return empty
    table = section.find("table")
    if table is None:
        return empty
    out = dict(empty)
    headers = [th.get_text(strip=True) for th in table.select("thead th")]
    if len(headers) > 1:
        out["quarter"] = headers[-1]
    row_map = (("promoter", "promoter"), ("fii", "fii"), ("dii", "dii"), ("public", "public"))
    for row in table.select("tbody tr"):
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True).lower()
        raw = cells[-1].get_text(strip=True).replace("%", "").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        for prefix, key in row_map:
            if label.startswith(prefix) and out[key] is None:
                out[key] = value
                break
    return out
