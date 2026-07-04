"""Real-network tests against WAAREEENER.NS — assert shapes/types, not values."""

import pandas as pd
from datetime import date, timedelta

from tools import market_tools

TICKER = "WAAREEENER.NS"


def test_search_ticker():
    results = market_tools.search_ticker("Waaree Energies")
    assert isinstance(results, list) and results
    first = results[0]
    assert set(first) == {"ticker", "name", "exchange"}
    assert first["ticker"].endswith(".NS")  # NSE results sorted first


def test_get_stock_info():
    info = market_tools.get_stock_info(TICKER)
    assert set(info) == {"sector", "industry", "market_cap", "description"}
    assert isinstance(info["market_cap"], int)
    assert isinstance(info["description"], str) and info["description"]


def test_get_price_history():
    df = market_tools.get_price_history(TICKER, period="1mo")
    assert isinstance(df, pd.DataFrame) and not df.empty
    assert "Close" in df.columns


def test_get_fundamentals():
    fundamentals = market_tools.get_fundamentals(TICKER)
    assert {"pe_ratio", "roe", "debt_to_equity", "revenue"} <= set(fundamentals)
    assert isinstance(fundamentals["revenue"], (int, float))


def test_get_live_price():
    price = market_tools.get_live_price(TICKER)
    assert isinstance(price, float) and price > 0


def test_search_sector_peers():
    # Yahoo's region="IN" sector data only covers top companies and shifts
    # over time — assert shape only; an empty list is a legitimate outcome.
    peers = market_tools.search_sector_peers("Technology", (1e10, 1e15))
    assert isinstance(peers, list)
    assert all(isinstance(p, str) and p.endswith(".NS") for p in peers)


def test_price_move_around_recent_date():
    event_date = (date.today() - timedelta(days=30)).isoformat()
    moves = market_tools.price_move_around(TICKER, event_date)
    assert set(moves) == {"pct_1d", "pct_5d"}
    assert isinstance(moves["pct_1d"], float)
    assert isinstance(moves["pct_5d"], float)
    assert -50 < moves["pct_1d"] < 50  # sanity: single-day move


def test_price_move_around_future_date_returns_none():
    moves = market_tools.price_move_around(TICKER, (date.today() + timedelta(days=30)).isoformat())
    assert moves == {"pct_1d": None, "pct_5d": None}


def test_get_fundamentals_extended_keys():
    fund = market_tools.get_fundamentals(TICKER)
    assert {"pe_ratio", "roe", "debt_to_equity", "revenue",
            "pb_ratio", "dividend_yield"} <= set(fund)
