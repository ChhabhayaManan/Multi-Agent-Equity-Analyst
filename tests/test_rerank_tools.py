import pytest

from tools.rerank_tools import cohere_rerank
from utils.helpers import load_config

_HAS_KEY = bool(load_config().get("COHERE_API_KEY"))

QUERY = "solar panel manufacturer in India"
DOCS = [
    "How to make the perfect paneer butter masala at home in 30 minutes.",
    "Waaree Energies is India's largest solar PV module manufacturer, with "
    "12 GW of module manufacturing capacity across Gujarat.",
    "The history of cricket in Australia dates back to the 19th century.",
    "Top 10 budget travel destinations in Europe for backpackers.",
    "A beginner's guide to sourdough bread baking and starter maintenance.",
]
RELEVANT_INDEX = 1


@pytest.mark.skipif(not _HAS_KEY, reason="COHERE_API_KEY not set in .env")
def test_cohere_rerank_live():
    top_n = 3
    results = cohere_rerank(QUERY, DOCS, top_n=top_n)

    assert len(results) == top_n
    scores = [r["relevance_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["index"] == RELEVANT_INDEX
    for r in results:
        assert r["document"] == DOCS[r["index"]]


@pytest.mark.skipif(not _HAS_KEY, reason="COHERE_API_KEY not set in .env")
def test_cohere_rerank_top_n_capped():
    results = cohere_rerank(QUERY, DOCS[:2], top_n=10)
    assert len(results) == 2
