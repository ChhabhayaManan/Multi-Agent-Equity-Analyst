import pytest

from templates.schemas.outputs import NewsItem, NewsOutput

ARTICLES = [{
    "title": "Company wins large solar order", "link": "https://example.com/a",
    "description": "Order win", "content": "The company won a 500 MW order.",
    "pubDate": "2026-07-02 08:00:00", "source_name": "ET", "keywords": None,
}]

CANNED = NewsOutput(
    items=[NewsItem(title="Company wins large solar order", date="2026-07-02",
                    source_url="https://example.com/a", summary="Won 500 MW order.",
                    impact_on_stock="Order book strengthens.",
                    sector_impact="Healthy solar demand.", sentiment="POSITIVE",
                    sentiment_score=0.7)],
    narrative="Company wins large solar order dominated the tape.",
    overall_sentiment="POSITIVE")


@pytest.fixture
def patched(monkeypatch):
    from agents import news_analysis_generator as mod
    calls = {"stored": None, "prompt": None}
    monkeypatch.setattr(mod, "fetch_news_articles", lambda t, c: list(ARTICLES))
    monkeypatch.setattr(mod, "store_to_pinecone",
                        lambda ticker, docs, st, meta=None: calls.__setitem__("stored", (docs, st)))

    class FakeLLM:
        def invoke(self, prompt):
            calls["prompt"] = prompt.to_string()
            return CANNED

    monkeypatch.setattr(mod, "get_llm", lambda schema=None: FakeLLM())
    return mod, calls


def test_run_stores_and_analyzes(patched):
    mod, calls = patched
    out, fetch_count = mod.run("WAAREEENER.NS", "Waaree Energies Ltd")
    assert fetch_count == 1
    assert out.items[0].sentiment == "POSITIVE"
    docs, st = calls["stored"]
    assert st == "news" and "500 MW" in docs[0]
    assert "Waaree Energies Ltd" in calls["prompt"]      # articles fed directly to LLM
    assert "500 MW" in calls["prompt"]


def test_run_empty_articles_skips_store(patched, monkeypatch):
    mod, calls = patched
    monkeypatch.setattr(mod, "fetch_news_articles", lambda t, c: [])
    out, fetch_count = mod.run("WAAREEENER.NS", "Waaree Energies Ltd")
    assert fetch_count == 0
    assert calls["stored"] is None                        # nothing stored
