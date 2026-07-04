import pandas as pd
import pytest

from templates.schemas.outputs import FundamentalsOutput


@pytest.fixture
def patched(monkeypatch):
    from agents import fundamentals_agent as mod

    hist = pd.DataFrame(
        {"Close": [100.0 + i for i in range(250)]},
        index=pd.date_range("2025-07-01", periods=250, freq="B", tz="Asia/Kolkata"))
    monkeypatch.setattr(mod, "get_stock_info", lambda t: {
        "sector": "Banks", "industry": "Private", "market_cap": 1_000_000_000_000,
        "description": "A bank."})
    monkeypatch.setattr(mod, "get_fundamentals", lambda t: {
        "pe_ratio": 19.2, "roe": 16.9, "debt_to_equity": 1.1,
        "revenue": 5e11, "pb_ratio": 2.8, "dividend_yield": 1.1})
    monkeypatch.setattr(mod, "get_price_history", lambda t, period="1y": hist)
    monkeypatch.setattr(mod, "fetch_shareholding", lambda t: {
        "promoter": 25.0, "fii": 30.0, "dii": 20.0, "public": 25.0,
        "quarter": "Mar 2026"})
    stored = {}
    monkeypatch.setattr(mod, "store_to_pinecone",
                        lambda ticker, docs, st, meta=None: stored.update(
                            {"ticker": ticker, "docs": docs, "source_type": st}))

    llm_output = FundamentalsOutput(
        company_profile={"sector": "WRONG"},          # must be overwritten
        valuation={"pe": 999.0},                      # must be overwritten
        price_snapshot={"price": 999.0},              # must be overwritten
        shareholding={"promoter": 999.0},             # must be overwritten
        summary="A large private bank trading near the top of its 52-week "
                "range after a steady one-year climb in the shares.")

    class FakeLLM:
        def invoke(self, prompt):
            return llm_output

    monkeypatch.setattr(mod, "get_llm", lambda schema=None: FakeLLM())
    return mod, stored


def test_run_overwrites_numbers_from_tools(patched):
    mod, stored = patched
    out, fetch_count = mod.run("HDFCBANK.NS", "HDFC Bank Ltd")
    assert isinstance(out, FundamentalsOutput)
    assert out.company_profile["sector"] == "Banks"       # tool value, not LLM's
    assert out.valuation["pe"] == 19.2
    assert out.valuation["roce"] is None
    assert out.price_snapshot["price"] == 349.0        # last close of the fake history
    assert out.shareholding["promoter"] == 25.0
    assert fetch_count == 5                               # pe, pb, roe, debt_equity, div_yield
    assert stored["source_type"] == "fundamentals"
    assert "52-week" in stored["docs"][0] or "bank" in stored["docs"][0].lower()


def test_price_snapshot_math(patched):
    mod, _ = patched
    out, _ = mod.run("HDFCBANK.NS", "HDFC Bank Ltd")
    snap = out.price_snapshot
    assert snap["high_52w"] == 349.0 and snap["low_52w"] == 100.0
    assert snap["ret_1y"] is not None and snap["ret_1y"] > 0
    assert snap["mktcap_cr"] == 100000.0                  # 1e12 INR / 1e7
