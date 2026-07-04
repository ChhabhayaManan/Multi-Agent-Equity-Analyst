"""LangChain @tool wrappers for the research chatbot.

Every tool returns a JSON string. Tools never raise: failures come back as
{"error": "..."} so the agent can work around them. Outputs are truncated to
MAX_TOOL_CHARS before reaching the LLM.
"""

import asyncio
import json
import time
from pathlib import Path

from langchain_core.tools import tool

from tools import market_tools
from tools.fetch_tools import fetch_news_articles
from tools.pinecone_tools import query_pinecone
from tools.rerank_tools import cohere_rerank
from utils.helpers import get_cache, get_logger, load_config

logger = get_logger(__name__)

MAX_TOOL_CHARS = 2000
CHART_DIR = Path("data/charts")
SOURCE_TYPES = {"news", "docs", "events", "competitor", "report"}


def _to_json(payload) -> str:
    text = json.dumps(payload, default=str)
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + "...[truncated]"
    return text


def _safe(fn):
    try:
        return _to_json(fn())
    except Exception as exc:
        logger.warning("chatbot tool failed: %s", exc)
        return _to_json({"error": str(exc)})


def _save_chart(fig, name: str) -> str:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / f"{name}_{int(time.time())}.html"
    fig.write_html(str(path))
    return str(path)


def build_local_tools(ticker: str, company_name: str) -> list:
    """All local tools, closed over the session's primary ticker."""
    session_ticker = ticker

    @tool
    def search_research(query: str, source_type: str = "") -> str:
        """Search the indexed research documents for the session's primary
        stock. source_type: one of news|docs|events|competitor|report, or
        empty to search all sources. Returns the top 5 most relevant chunks."""

        def run():
            st = source_type if source_type in SOURCE_TYPES else None
            hits = query_pinecone(session_ticker, query, st, k=10)
            docs = [h["text"] for h in hits if h["text"]]
            if not docs:
                return {"results": [], "note": "no indexed chunks matched"}
            ranked = cohere_rerank(query, docs, top_n=5)
            by_text = {h["text"]: h["metadata"] for h in hits}
            return {"results": [
                {"text": r["document"],
                 "relevance": round(r["relevance_score"], 3),
                 "metadata": by_text.get(r["document"], {})}
                for r in ranked]}

        return _safe(run)

    @tool
    def get_live_price(ticker: str) -> str:
        """Live/last traded price for any NSE/BSE ticker (e.g. TCS.NS)."""
        return _safe(lambda: {"ticker": ticker,
                              "price": market_tools.get_live_price(ticker)})

    @tool
    def get_stock_info(ticker: str) -> str:
        """Sector, industry, market cap and business description for any
        NSE/BSE ticker."""
        return _safe(lambda: market_tools.get_stock_info(ticker))

    @tool
    def get_fundamentals(ticker: str) -> str:
        """P/E, P/B, ROE, debt/equity, revenue and dividend yield for any
        NSE/BSE ticker."""
        return _safe(lambda: market_tools.get_fundamentals(ticker))

    @tool
    def get_price_history(ticker: str, period: str = "1mo") -> str:
        """OHLC price summary over a period (1mo|3mo|6mo|1y|2y|5y) for any
        NSE/BSE ticker: first/last close, min, max, % change."""

        def run():
            df = market_tools.get_price_history(ticker, period)
            if df.empty or "Close" not in df.columns:
                return {"error": f"no price history for {ticker}"}
            closes = df["Close"]
            first, last = float(closes.iloc[0]), float(closes.iloc[-1])
            return {
                "ticker": ticker, "period": period, "rows": len(df),
                "first_close": round(first, 2), "last_close": round(last, 2),
                "min_close": round(float(closes.min()), 2),
                "max_close": round(float(closes.max()), 2),
                "pct_change": round((last / first - 1) * 100, 2) if first else None,
            }

        return _safe(run)

    @tool
    def price_move_around(ticker: str, date: str) -> str:
        """% price move around an event date (YYYY-MM-DD): 1-day and 5-day
        change vs the prior close."""
        return _safe(lambda: market_tools.price_move_around(ticker, date))

    @tool
    def resolve_ticker(query: str) -> str:
        """Resolve a company name to NSE/BSE tickers (e.g. 'Tata Consultancy'
        -> TCS.NS). Use before other tools when the user names a company
        without a ticker."""
        return _safe(lambda: {"matches": market_tools.search_ticker(query)[:5]})

    @tool
    def get_recent_news(ticker: str, company_name: str, hours: int = 48) -> str:
        """Recent news headlines for any listed company (newsdata.io). Use
        only when the user explicitly asks for news. company_name must be the
        full listed name."""

        def run():
            articles = fetch_news_articles(ticker, company_name, hours=hours)
            return {"articles": [
                {"title": a.get("title"), "description": a.get("description"),
                 "link": a.get("link"), "pubDate": a.get("pubDate"),
                 "source_name": a.get("source_name")}
                for a in articles[:8]]}

        return _safe(run)

    @tool
    def plot_price_chart(ticker: str, period: str = "6mo") -> str:
        """Render a candlestick price chart for one ticker and save it as an
        HTML file. Use when the user asks to see/plot/visualize a price."""

        def run():
            import plotly.graph_objects as go
            df = market_tools.get_price_history(ticker, period)
            if df.empty or "Close" not in df.columns:
                return {"error": f"no price history for {ticker}"}
            fig = go.Figure(go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"]))
            fig.update_layout(title=f"{ticker} — {period}",
                              xaxis_rangeslider_visible=False)
            path = _save_chart(fig, f"{ticker}_{period}")
            closes = df["Close"]
            return {"chart_path": path, "ticker": ticker, "period": period,
                    "last_close": round(float(closes.iloc[-1]), 2),
                    "min_close": round(float(closes.min()), 2),
                    "max_close": round(float(closes.max()), 2)}

        return _safe(run)

    @tool
    def plot_comparison_chart(tickers: str, period: str = "6mo") -> str:
        """Render a normalized (base-100) close-price comparison chart for
        2-5 comma-separated tickers and save it as an HTML file."""

        def run():
            import plotly.graph_objects as go
            symbols = [t.strip() for t in tickers.split(",") if t.strip()][:5]
            fig = go.Figure()
            plotted, last_values = [], {}
            for sym in symbols:
                df = market_tools.get_price_history(sym, period)
                if df.empty or "Close" not in df.columns:
                    continue
                closes = df["Close"]
                base = float(closes.iloc[0])
                if not base:
                    continue
                fig.add_trace(go.Scatter(
                    x=df.index, y=(closes / base * 100), mode="lines", name=sym))
                plotted.append(sym)
                last_values[sym] = round(float(closes.iloc[-1] / base * 100) - 100, 2)
            if not plotted:
                return {"error": "no price history for any requested ticker"}
            fig.update_layout(title=f"Normalized comparison — {period}",
                              yaxis_title="Indexed to 100")
            path = _save_chart(fig, "compare_" + "_".join(p.split(".")[0] for p in plotted))
            return {"chart_path": path, "tickers": plotted, "period": period,
                    "pct_change_over_period": last_values}

        return _safe(run)

    return [search_research, get_live_price, get_stock_info, get_fundamentals,
            get_price_history, price_move_around, resolve_ticker,
            get_recent_news, plot_price_chart, plot_comparison_chart]


# --- Alpha Vantage MCP ------------------------------------------------------

AV_ALLOWLIST = {
    # fundamentals
    "COMPANY_OVERVIEW", "INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW",
    "EARNINGS",
    # technicals
    "RSI", "SMA", "MACD",
    # alpha intelligence
    "NEWS_SENTIMENT", "TOP_GAINERS_LOSERS", "INSIDER_TRANSACTIONS",
    "EARNINGS_CALL_TRANSCRIPT", "ANALYTICS_FIXED_WINDOW",
}
AV_CACHE_TTL_S = 24 * 3600  # free tier: 25 requests/day — cache aggressively


def _cache_coroutine(tool):
    """Wrap an MCP tool's coroutine with the shared disk cache."""
    original = tool.coroutine
    cache = get_cache()

    async def cached(**kwargs):
        key = ("alphavantage", tool.name, tuple(sorted(kwargs.items())))
        if key in cache:
            return cache[key]
        result = await original(**kwargs)
        cache.set(key, result, expire=AV_CACHE_TTL_S)
        return result

    tool.coroutine = cached
    return tool


def filter_av_tools(tools: list) -> list:
    """Keep only allowlisted Alpha Vantage tools; disk-cache their calls."""
    return [_cache_coroutine(t) for t in tools if t.name.upper() in AV_ALLOWLIST]


def load_alphavantage_tools() -> list:
    """Alpha Vantage remote MCP tools, or [] if unavailable. Never raises."""
    key = load_config().get("ALPHAVANTAGE_API_KEY")
    if not key:
        logger.warning("ALPHAVANTAGE_API_KEY not set; running without AV tools")
        return []
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        client = MultiServerMCPClient({
            "alphavantage": {
                "url": f"https://mcp.alphavantage.co/mcp?apikey={key}",
                "transport": "streamable_http",
            }
        })
        tools = asyncio.run(client.get_tools())
        kept = filter_av_tools(tools)
        logger.info("Loaded %d Alpha Vantage MCP tools", len(kept))
        return kept
    except Exception as exc:
        logger.warning("Alpha Vantage MCP unavailable; continuing without: %s",
                       str(exc).replace(key, "***"))
        return []
