from typing import List, Tuple
import pandas as pd
import yfinance as yf
from utils.helpers import get_logger
from utils.tracing import traceable

logger = get_logger(__name__)


_EXCHANGE_NAMES = {"NSI": "NSE", "BSE": "BSE"}


def search_ticker(query: str) -> List[dict]:
    # yf.Search ranks by global relevance and frequently omits the NSE/BSE
    # listing entirely for common Indian company names (e.g. "Infosys",
    # "TCS", "HDFC Bank") in favor of NYSE ADRs or unrelated tickers.
    # yf.Lookup returns every matching symbol unranked, so the NSE/BSE
    # listing is reliably present.
    df = yf.Lookup(query).get_stock(count=25)
    results = [
        {
            "ticker": symbol,
            "name": row["shortName"],
            "exchange": _EXCHANGE_NAMES.get(row["exchange"], row["exchange"]),
        }
        for symbol, row in df.iterrows()
        if row.get("quoteType") == "equity" and symbol
    ]
    results.sort(key=lambda r: not r["ticker"].endswith(".NS"))
    return results


@traceable(name="get_stock_info")
def get_stock_info(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "description": info.get("longBusinessSummary"),
    }


@traceable(name="get_price_history")
def get_price_history(ticker: str, period: str = "1mo") -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period)


@traceable(name="get_fundamentals")
def get_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).info
    return {
        "pe_ratio": info.get("trailingPE"),
        "roe": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "revenue": info.get("totalRevenue"),
        "pb_ratio": info.get("priceToBook"),
        "dividend_yield": info.get("dividendYield"),
    }


def get_live_price(ticker: str) -> float:
    # fast_info key is camelCase "lastPrice" in yfinance 1.5.1
    return float(yf.Ticker(ticker).fast_info["lastPrice"])


@traceable(name="search_sector_peers")
def search_sector_peers(sector: str, mktcap_range: Tuple[float, float]) -> List[str]:
    """Return NSE tickers in `sector` whose market cap (INR) falls in mktcap_range.

    Uses yf.Sector(key, region="IN").top_companies, which does return Indian
    listings (mix of .NS and .BO symbols). LIMITATION: Yahoo only exposes the
    sector's largest companies by market weight (~a few dozen), so smaller NSE
    names are missing and results may vary over time. top_companies carries no
    market-cap column, so caps are fetched per candidate via fast_info.
    """
    key = sector.lower().replace(" ", "-")
    try:
        companies = yf.Sector(key, region="IN").top_companies
    except Exception as e:
        logger.warning("Sector lookup failed for %r: %s", sector, e)
        return []
    if companies is None or companies.empty:
        return []
    lo, hi = mktcap_range
    peers = []
    for symbol in companies.index:
        if not str(symbol).endswith(".NS"):
            continue
        try:
            cap = yf.Ticker(symbol).fast_info["marketCap"]
        except Exception:
            continue
        if cap is not None and lo <= cap <= hi:
            peers.append(str(symbol))
    return peers


def price_move_around(ticker: str, date: str) -> dict:
    """% price change around an event date, from daily closes.

    pct_1d: close on the event day (or nearest next trading day) vs the
    previous trading close. pct_5d: close 5 trading days after the event day
    vs that same previous close. None when history doesn't cover the date.
    """
    out = {"pct_1d": None, "pct_5d": None}
    df = get_price_history(ticker, period="1y")
    if df.empty or "Close" not in df.columns:
        return out
    closes = df["Close"]
    days = closes.index.tz_localize(None).normalize()
    target = pd.Timestamp(date)
    positions = [i for i, d in enumerate(days) if d >= target]
    if not positions or positions[0] == 0:  # date after history end, or no prior close
        return out
    pos = positions[0]
    prev_close = float(closes.iloc[pos - 1])
    if prev_close == 0:
        return out
    out["pct_1d"] = round((float(closes.iloc[pos]) / prev_close - 1) * 100, 2)
    if pos + 5 < len(closes):
        out["pct_5d"] = round((float(closes.iloc[pos + 5]) / prev_close - 1) * 100, 2)
    return out
