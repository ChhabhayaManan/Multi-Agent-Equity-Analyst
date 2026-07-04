import json

import pandas as pd
import pytest

import chatbot.chatbot_tools as ct


@pytest.fixture
def tools_by_name():
    tools = ct.build_local_tools("HDFCBANK.NS", "HDFC Bank Ltd")
    return {t.name: t for t in tools}


def test_all_expected_tools_present(tools_by_name):
    assert set(tools_by_name) == {
        "search_research", "get_live_price", "get_stock_info",
        "get_fundamentals", "get_price_history", "price_move_around",
        "resolve_ticker", "get_recent_news", "plot_price_chart",
        "plot_comparison_chart",
    }


def test_search_research_wires_rerank(monkeypatch, tools_by_name):
    monkeypatch.setattr(ct, "query_pinecone",
                        lambda ticker, query, source_type=None, k=5: [
                            {"text": f"chunk{i}", "score": 0.9,
                             "metadata": {"source_type": "news"}}
                            for i in range(10)])
    monkeypatch.setattr(ct, "cohere_rerank",
                        lambda query, docs, top_n=5: [
                            {"document": docs[0], "relevance_score": 0.99, "index": 0}])
    out = tools_by_name["search_research"].invoke(
        {"query": "margin guidance", "source_type": "news"})
    data = json.loads(out)
    assert data["results"][0]["text"] == "chunk0"


def test_search_research_all_sources_passes_none(monkeypatch, tools_by_name):
    captured = {}

    def fake_query(ticker, query, source_type=None, k=5):
        captured["source_type"] = source_type
        return []

    monkeypatch.setattr(ct, "query_pinecone", fake_query)
    tools_by_name["search_research"].invoke({"query": "anything"})
    assert captured["source_type"] is None


def test_tool_error_becomes_json_not_raise(monkeypatch, tools_by_name):
    monkeypatch.setattr(ct.market_tools, "get_live_price",
                        lambda ticker: (_ for _ in ()).throw(RuntimeError("boom")))
    out = tools_by_name["get_live_price"].invoke({"ticker": "TCS.NS"})
    assert "error" in json.loads(out)


def test_truncation(monkeypatch, tools_by_name):
    monkeypatch.setattr(ct.market_tools, "get_stock_info",
                        lambda ticker: {"description": "x" * 5000})
    out = tools_by_name["get_stock_info"].invoke({"ticker": "TCS.NS"})
    assert len(out) <= ct.MAX_TOOL_CHARS + len("...[truncated]")


def _fake_history(rows=30):
    idx = pd.date_range("2026-01-01", periods=rows, freq="B")
    base = {"Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 102.0}
    return pd.DataFrame([base] * rows, index=idx)


def test_plot_price_chart_writes_html(monkeypatch, tmp_path, tools_by_name):
    monkeypatch.setattr(ct, "CHART_DIR", tmp_path)
    monkeypatch.setattr(ct.market_tools, "get_price_history",
                        lambda ticker, period="1mo": _fake_history())
    out = json.loads(tools_by_name["plot_price_chart"].invoke(
        {"ticker": "HDFCBANK.NS", "period": "1mo"}))
    assert out["chart_path"].endswith(".html")
    assert (tmp_path / out["chart_path"].split("\\")[-1].split("/")[-1]).exists()
    assert out["last_close"] == 102.0


def test_plot_comparison_chart(monkeypatch, tmp_path, tools_by_name):
    monkeypatch.setattr(ct, "CHART_DIR", tmp_path)
    monkeypatch.setattr(ct.market_tools, "get_price_history",
                        lambda ticker, period="1mo": _fake_history())
    out = json.loads(tools_by_name["plot_comparison_chart"].invoke(
        {"tickers": "HDFCBANK.NS,ICICIBANK.NS", "period": "1mo"}))
    assert out["chart_path"].endswith(".html")
    assert out["tickers"] == ["HDFCBANK.NS", "ICICIBANK.NS"]


def test_get_recent_news(monkeypatch, tools_by_name):
    monkeypatch.setattr(ct, "fetch_news_articles",
                        lambda ticker, company, hours=48: [
                            {"title": "T", "description": "D", "link": "L",
                             "pubDate": "2026-07-01", "source_name": "ET"}])
    out = json.loads(tools_by_name["get_recent_news"].invoke(
        {"ticker": "TCS.NS", "company_name": "Tata Consultancy Services"}))
    assert out["articles"][0]["title"] == "T"


class _FakeMCPTool:
    def __init__(self, name):
        self.name = name
        self.calls = 0

        async def coro(**kwargs):
            self.calls += 1
            return f"{name}-result"

        self.coroutine = coro


def test_filter_av_tools_allowlist():
    tools = [_FakeMCPTool("COMPANY_OVERVIEW"), _FakeMCPTool("CRYPTO_INTRADAY"),
             _FakeMCPTool("news_sentiment")]
    kept = ct.filter_av_tools(tools)
    assert sorted(t.name.upper() for t in kept) == ["COMPANY_OVERVIEW", "NEWS_SENTIMENT"]


def test_filter_av_tools_caches_calls(monkeypatch):
    import asyncio

    store = {}

    class FakeCache(dict):
        def set(self, key, value, expire=None):
            self[key] = value

    monkeypatch.setattr(ct, "get_cache", lambda: FakeCache(store))
    t = _FakeMCPTool("EARNINGS")
    [wrapped] = ct.filter_av_tools([t])
    r1 = asyncio.run(wrapped.coroutine(symbol="TCS.BSE"))
    r2 = asyncio.run(wrapped.coroutine(symbol="TCS.BSE"))
    assert r1 == r2 == "EARNINGS-result"
    assert t.calls == 1  # second call served from cache


def test_load_alphavantage_tools_no_key(monkeypatch):
    monkeypatch.setattr(ct, "load_config", lambda: {"ALPHAVANTAGE_API_KEY": None})
    assert ct.load_alphavantage_tools() == []
