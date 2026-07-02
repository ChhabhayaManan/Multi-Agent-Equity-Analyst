"""Real-network tests against WAAREEENER.NS — assert shapes/types, not values."""

import pandas as pd

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
    assert set(fundamentals) == {"pe_ratio", "roe", "debt_to_equity", "revenue"}
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
