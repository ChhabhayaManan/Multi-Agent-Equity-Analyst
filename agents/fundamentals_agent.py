"""Fundamentals agent: tool-call + one LLM call for the summary.
All numeric fields are overwritten from tool data after the LLM call -
the LLM's numbers never ship."""

import json

from templates.prompts.fundamentals_agent import FUNDAMENTALS_PROMPT
from templates.schemas.outputs import FundamentalsOutput
from tools.fetch_tools import fetch_shareholding
from tools.market_tools import get_fundamentals, get_price_history, get_stock_info
from tools.pinecone_tools import store_to_pinecone
from utils.helpers import get_logger
from utils.llm import get_llm

logger = get_logger(__name__)


def _pct(closes, trading_days_back: int):
    if len(closes) <= trading_days_back:
        return None
    prev = float(closes.iloc[-1 - trading_days_back])
    return round((float(closes.iloc[-1]) / prev - 1) * 100, 2) if prev else None


def _snapshot(hist, market_cap) -> dict:
    if hist.empty or "Close" not in hist.columns:
        return {k: None for k in
                ("price", "high_52w", "low_52w", "ret_1m", "ret_6m", "ret_1y", "mktcap_cr")}
    closes = hist["Close"]
    return {
        "price": round(float(closes.iloc[-1]), 2),
        "high_52w": round(float(closes.max()), 2),
        "low_52w": round(float(closes.min()), 2),
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
    valuation = {
        "pe": fund.get("pe_ratio"), "pb": fund.get("pb_ratio"),
        "roe": fund.get("roe"), "roce": None,  # yfinance has no ROCE
        "debt_equity": fund.get("debt_to_equity"),
        "dividend_yield": fund.get("dividend_yield"),
    }
    price_snapshot = _snapshot(hist, info.get("market_cap"))
    shareholding = {k: share.get(k) for k in ("promoter", "fii", "dii", "public")}
    fetch_count = sum(v is not None for v in valuation.values())

    context = json.dumps({
        "company_profile": company_profile, "valuation": valuation,
        "price_snapshot": price_snapshot, "shareholding": shareholding,
        "shareholding_quarter": share.get("quarter"),
    }, indent=2, default=str)

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
