"""Competitor Intelligence agent: langchain `create_agent` ReAct loop over
market tools with `response_format=CompetitorOutput` for structured output."""

import json

from langchain_core.tools import tool

from templates.prompts.competitor_intelligence_agent import COMPETITOR_SYSTEM
from templates.schemas.outputs import CompetitorOutput
from tools import market_tools
from tools.pinecone_tools import store_to_pinecone
from utils.helpers import get_logger
from utils.llm import get_chat_model

logger = get_logger(__name__)


@tool
def get_stock_info(ticker: str) -> str:
    """Sector, industry, market cap and description for a ticker (e.g. HDFCBANK.NS)."""
    return json.dumps(market_tools.get_stock_info(ticker), default=str)


@tool
def search_sector_peers(sector: str, mktcap_low: float, mktcap_high: float) -> str:
    """NSE tickers in `sector` with market cap (INR) between mktcap_low and mktcap_high."""
    return json.dumps(market_tools.search_sector_peers(sector, (mktcap_low, mktcap_high)))


@tool
def get_fundamentals(ticker: str) -> str:
    """P/E, P/B, ROE, debt/equity, revenue and dividend yield for a ticker."""
    return json.dumps(market_tools.get_fundamentals(ticker), default=str)


@tool
def get_price_returns(ticker: str) -> str:
    """1-month, 3-month and 6-month % price returns for a ticker."""
    df = market_tools.get_price_history(ticker, period="6mo")
    if df.empty or "Close" not in df.columns:
        return json.dumps({"ret_1m": None, "ret_3m": None, "ret_6m": None})
    closes = df["Close"]

    def pct(days):
        if len(closes) <= days:
            return None
        prev = float(closes.iloc[-1 - days])
        return round((float(closes.iloc[-1]) / prev - 1) * 100, 2) if prev else None

    return json.dumps({"ret_1m": pct(21), "ret_3m": pct(63),
                       "ret_6m": pct(len(closes) - 1)})


_TOOLS = [get_stock_info, search_sector_peers, get_fundamentals, get_price_returns]


def _build_agent(system_prompt: str):
    from langchain.agents import create_agent
    return create_agent(model=get_chat_model(), tools=_TOOLS,
                        system_prompt=system_prompt,
                        response_format=CompetitorOutput)


def run(ticker: str, company_name: str, retry_feedback: str = ""):
    system = COMPETITOR_SYSTEM.format(ticker=ticker, company_name=company_name)
    task = f"Analyze competitors for {ticker} ({company_name})."
    if retry_feedback:
        task += f"\n{retry_feedback}"
    agent = _build_agent(system)
    result = agent.invoke({"messages": [("user", task)]})
    out: CompetitorOutput = result["structured_response"]

    fetch_count = sum(
        1 for p in out.peers if any(v is not None for v in p.metrics.values()))
    doc = (f"Competitor comparison for {company_name} ({ticker}): "
           f"{out.comparison_summary}\nPeers: "
           + "; ".join(f"{p.name} ({p.ticker}) intensity={p.competition_intensity} "
                       f"standing={p.target_standing} metrics={json.dumps(p.metrics)}"
                       for p in out.peers))
    try:
        store_to_pinecone(ticker, [doc], "competitor",
                          meta={"document_id": "competitor-comparison"})
    except Exception:
        logger.exception("competitor: pinecone store failed (non-fatal)")
    return out, fetch_count
