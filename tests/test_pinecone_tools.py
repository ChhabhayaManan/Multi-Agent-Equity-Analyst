import time
import types
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools import pinecone_tools
from utils.helpers import load_config

_HAS_KEY = bool(load_config().get("PINECONE_API_KEY"))

needs_key = pytest.mark.skipif(not _HAS_KEY, reason="PINECONE_API_KEY not set in .env")

TICKER = "WAAREEENER"
NS_TEST_TICKER = "WAAREEENER-NSTEST"
SOURCE_TYPE = "news"
DOCS = [
    "Waaree Energies reported strong quarterly results driven by solar module demand. "
    "The company expanded its manufacturing capacity in Gujarat to 12 GW.",
    "Waaree Energies won a large export order for solar PV modules from a US customer, "
    "strengthening its international order book.",
]


@pytest.fixture(scope="module")
def stored_docs():
    from tools.pinecone_tools import get_index, store_to_pinecone

    doc_id = f"test-{uuid.uuid4().hex[:8]}"
    store_to_pinecone(TICKER, DOCS, SOURCE_TYPE, meta={"document_id": doc_id})
    # Serverless indexes are eventually consistent — poll until vectors are visible.
    index = get_index()
    for _ in range(30):
        stats = index.describe_index_stats()
        ns = stats.namespaces or {}
        if TICKER in ns and ns[TICKER].vector_count > 0:
            break
        time.sleep(2)
    time.sleep(2)
    yield doc_id
    from tools.pinecone_tools import delete_namespace

    delete_namespace(TICKER)


def test_chunk_text():
    from tools.pinecone_tools import GeminiEmbedder

    embedder = GeminiEmbedder("fake-key")
    embedder.chunk_size = 500
    embedder.chunk_overlap = 50
    text = "x" * 1200
    chunks = embedder.chunk_text(text)
    assert all(len(c) <= 500 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1200  # overlap means total >= original
    assert embedder.chunk_text("short") == ["short"]
    assert embedder.chunk_text("   ") == []


def test_select_embedder_round_robins_api_keys(monkeypatch):
    router = pinecone_tools._GeminiRouter(["key-1", "key-2", "key-3"])
    monkeypatch.setattr(pinecone_tools, "_router", router)

    with ThreadPoolExecutor(max_workers=6) as pool:
        selected = list(pool.map(lambda _: pinecone_tools.select_embedder().api_key, range(6)))

    assert Counter(selected) == Counter({"key-1": 2, "key-2": 2, "key-3": 2})


def test_embed_texts_retries_on_quota_error(monkeypatch):
    class FakeClient:
        def __init__(self, key, fail=False):
            self.key = key
            self.fail = fail
            self.calls = 0

        def embed_documents(self, texts, **kwargs):
            self.calls += 1
            if self.fail:
                raise RuntimeError("429 rate limit exceeded")
            return [[float(len(self.key))] * pinecone_tools.EMBED_DIM for _ in texts]

        def embed_query(self, text, **kwargs):
            self.calls += 1
            if self.fail:
                raise RuntimeError("429 rate limit exceeded")
            return [float(len(self.key))] * pinecone_tools.EMBED_DIM

    clients = {
        "key-1": FakeClient("key-1", fail=True),
        "key-2": FakeClient("key-2"),
        "key-3": FakeClient("key-3"),
    }
    monkeypatch.setattr(pinecone_tools, "_router", pinecone_tools._GeminiRouter(list(clients)))
    monkeypatch.setattr(pinecone_tools, "_gemini_client", lambda api_key: clients[api_key])

    vectors = pinecone_tools.embed_texts(["hello", "world"], input_type="passage")

    assert len(vectors) == 2
    assert clients["key-1"].calls == 1
    assert clients["key-2"].calls == 1
    assert clients["key-3"].calls == 0


@needs_key
def test_embed_texts_shape(stored_docs):
    from tools.pinecone_tools import EMBED_DIM, embed_texts

    vecs = embed_texts(["hello world", "solar energy"], input_type="query")
    assert len(vecs) == 2
    assert all(len(v) == EMBED_DIM for v in vecs)


@needs_key
def test_check_index_true(stored_docs):
    from tools.pinecone_tools import check_index

    assert check_index(TICKER, SOURCE_TYPE) is True


@needs_key
def test_check_index_false_for_other_source_type(stored_docs):
    from tools.pinecone_tools import check_index

    assert check_index(TICKER, "docs") is False


@needs_key
def test_query_round_trip(stored_docs):
    from tools.pinecone_tools import query_pinecone

    results = query_pinecone(TICKER, "solar module manufacturing capacity", SOURCE_TYPE, k=3)
    assert len(results) > 0
    top = results[0]
    assert "Waaree" in top["text"]
    assert isinstance(top["score"], float)
    assert top["metadata"]["source_type"] == SOURCE_TYPE
    assert top["metadata"]["ticker"] == TICKER
    assert top["metadata"]["document_id"] == stored_docs
    assert "chunk_id" in top["metadata"]


@needs_key
def test_namespace_exists_and_delete():
    from tools.pinecone_tools import (
        delete_namespace,
        namespace_exists,
        store_to_pinecone,
        wait_for_vectors,
    )

    # Use a dedicated throwaway namespace so this test's delete doesn't wipe
    # the shared TICKER namespace that other live tests depend on.
    store_to_pinecone(NS_TEST_TICKER, DOCS[:1], SOURCE_TYPE, meta={"document_id": "ns-test"})
    assert wait_for_vectors(NS_TEST_TICKER, timeout_s=60) is True
    assert namespace_exists(NS_TEST_TICKER) is True
    delete_namespace(NS_TEST_TICKER)
    for _ in range(30):  # deletion is eventually consistent too
        if not namespace_exists(NS_TEST_TICKER):
            break
        time.sleep(2)
    assert namespace_exists(NS_TEST_TICKER) is False
    delete_namespace(NS_TEST_TICKER)  # idempotent: deleting a missing namespace must not raise


def test_namespace_exists_false_for_unknown(monkeypatch):
    from tools import pinecone_tools

    class FakeStats:
        namespaces = {}

    class FakeIndex:
        def describe_index_stats(self):
            return FakeStats()

    monkeypatch.setattr(pinecone_tools, "get_index", lambda: FakeIndex())
    assert pinecone_tools.namespace_exists("NOSUCH") is False


def test_query_pinecone_no_source_type_omits_filter(monkeypatch):
    """source_type=None must query without a metadata filter."""
    captured = {}

    class FakeIndex:
        def query(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(matches=[])

    monkeypatch.setattr(pinecone_tools, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(pinecone_tools, "embed_texts",
                        lambda texts, input_type="passage": [[0.1, 0.2]])

    result = pinecone_tools.query_pinecone("HDFCBANK.NS", "revenue growth", None, k=7)

    assert result == []
    assert captured["filter"] is None
    assert captured["top_k"] == 7


def test_query_pinecone_with_source_type_keeps_filter(monkeypatch):
    captured = {}

    class FakeIndex:
        def query(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(matches=[])

    monkeypatch.setattr(pinecone_tools, "get_index", lambda: FakeIndex())
    monkeypatch.setattr(pinecone_tools, "embed_texts",
                        lambda texts, input_type="passage": [[0.1, 0.2]])

    pinecone_tools.query_pinecone("HDFCBANK.NS", "revenue", "news")

    assert captured["filter"] == {"source_type": {"$eq": "news"}}


def _fake_vec(vid, source_type, document_id, date=None):
    md = {"source_type": source_type, "ticker": "HDFCBANK",
          "chunk_id": 0, "text": "x", "document_id": document_id}
    if date:
        md["date"] = date
    return types.SimpleNamespace(id=vid, metadata=md)


def test_list_documents_groups_and_dedupes(monkeypatch):
    fetched = {
        "HDFCBANK-news-1-0": _fake_vec("HDFCBANK-news-1-0", "news", "news-1", "2026-07-01"),
        "HDFCBANK-news-1-1": _fake_vec("HDFCBANK-news-1-1", "news", "news-1", "2026-07-01"),
        "HDFCBANK-docs-9-0": _fake_vec("HDFCBANK-docs-9-0", "docs", "docs-9", "2026-04-19"),
    }

    class FakeIndex:
        def list(self, namespace=None):
            yield list(fetched.keys())

        def fetch(self, ids, namespace=None):
            return types.SimpleNamespace(
                vectors={i: fetched[i] for i in ids if i in fetched})

    monkeypatch.setattr(pinecone_tools, "namespace_exists", lambda t: True)
    monkeypatch.setattr(pinecone_tools, "get_index", lambda: FakeIndex())

    out = pinecone_tools.list_documents("HDFCBANK")
    assert set(out.keys()) == {"news", "docs"}
    assert len(out["news"]) == 1                      # two chunks -> one doc
    assert out["news"][0]["document_id"] == "news-1"
    assert out["news"][0]["chunk_count"] == 2
    assert out["news"][0]["date"] == "2026-07-01"
    assert out["docs"][0]["document_id"] == "docs-9"


def test_list_documents_absent_namespace_returns_empty(monkeypatch):
    monkeypatch.setattr(pinecone_tools, "namespace_exists", lambda t: False)
    assert pinecone_tools.list_documents("NOPE") == {}
