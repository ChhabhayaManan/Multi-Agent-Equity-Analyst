"""Fundamentals agent: tool-call + one LLM call for the summary.
All numeric fields are overwritten from tool data after the LLM call -
the LLM's numbers never ship."""

import json

from templates.prompts.fundamentals_agent import FUNDAMENTALS_PROMPT
from templates.schemas.outputs import FundamentalsOutput
from tools.fetch_tools import fetch_shareholding
from tools.market_tools import get_fundamentals, get_price_history, get_stock_info
from tools.pinecone_tools import store_to_pinecone
from utils.helpers import get_logger, scrub_nan
from utils.llm import get_llm

logger = get_logger(__name__)


def _close(closes, position: int):
    """Close as a float, or None when the bar is missing (yfinance NaN)."""
    return scrub_nan(float(closes.iloc[position]))


def _pct(closes, trading_days_back: int):
    if len(closes) <= trading_days_back:
        return None
    prev = _close(closes, -1 - trading_days_back)
    last = _close(closes, -1)
    if not prev or last is None:
        return None
    return round((last / prev - 1) * 100, 2)


def _snapshot(hist, market_cap) -> dict:
    if hist.empty or "Close" not in hist.columns:
        return {k: None for k in
                ("price", "high_52w", "low_52w", "ret_1m", "ret_6m", "ret_1y", "mktcap_cr")}
    closes = hist["Close"]
    last = _close(closes, -1)
    high, low = scrub_nan(float(closes.max())), scrub_nan(float(closes.min()))
    return {
        "price": round(last, 2) if last is not None else None,
        "high_52w": round(high, 2) if high is not None else None,
        "low_52w": round(low, 2) if low is not None else None,
        "ret_1m": _pct(closes, 21),
        "ret_6m": _pct(closes, 126),
        "ret_1y": _pct(closes, len(closes) - 1),
        "mktcap_cr": round(market_cap / 1e7, 2) if market_cap else None,
    }


def run(ticker: str, company_name: str, retry_feedback: str = ""):
    info = get_stock_info(ticker)
    fund = get_fundamentals(ticker)
    hist = get_price_history(ticker, period="1y")
    share = fetch_shareholding(ticker)

    company_profile = {k: info.get(k) for k in ("sector", "industry", "description")}
    valuation = scrub_nan({
        "pe": fund.get("pe_ratio"), "pb": fund.get("pb_ratio"),
        "roe": fund.get("roe"), "roce": None,  # yfinance has no ROCE
        "debt_equity": fund.get("debt_to_equity"),
        "dividend_yield": fund.get("dividend_yield"),
    })
    price_snapshot = _snapshot(hist, info.get("market_cap"))
    shareholding = scrub_nan(
        {k: share.get(k) for k in ("promoter", "fii", "dii", "public")})
    fetch_count = sum(v is not None for v in valuation.values())

    context = json.dumps({
        "company_profile": company_profile, "valuation": valuation,
        "price_snapshot": price_snapshot, "shareholding": shareholding,
        "shareholding_quarter": share.get("quarter"),
    }, indent=2, default=str, allow_nan=False)

    llm = get_llm(FundamentalsOutput)
    out = llm.invoke(FUNDAMENTALS_PROMPT.invoke({
        "ticker": ticker, "company_name": company_name,
        "context": context, "retry_feedback": retry_feedback}))

    # Belt and braces: numbers come from tools, whatever the LLM returned.
    out = out.model_copy(update={
        "company_profile": company_profile, "valuation": valuation,
        "price_snapshot": price_snapshot, "shareholding": shareholding})

    doc = (f"Fundamentals summary for {company_name} ({ticker}): {out.summary}\n"
           f"Valuation: {json.dumps(valuation)}\nPrice: {json.dumps(price_snapshot)}\n"
           f"Shareholding: {json.dumps(shareholding)}")
    try:
        store_to_pinecone(ticker, [doc], "fundamentals",
                          meta={"document_id": "fundamentals-summary"})
    except Exception:
        logger.exception("fundamentals: pinecone store failed (non-fatal)")
    return out, fetch_count
