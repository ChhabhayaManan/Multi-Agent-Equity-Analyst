"""Live quote for the report header metric tiles. Cached so Streamlit reruns
don't re-hit yfinance. Kept out of tools/market_tools.py because the st cache
is a frontend concern; fails soft (returns None -> tiles hidden)."""
from typing import Optional

import streamlit as st
import yfinance as yf


@st.cache_data(ttl=900, show_spinner=False)
def get_quote(symbol: str) -> Optional[dict]:
    try:
        info = yf.Ticker(symbol).info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price is None:
            return None
        prev = info.get("previousClose")
        change_pct = ((price - prev) / prev * 100) if prev else None
        return {
            "price": price,
            "day_change_pct": change_pct,
            "market_cap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "year_high": info.get("fiftyTwoWeekHigh"),
            "year_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return None
