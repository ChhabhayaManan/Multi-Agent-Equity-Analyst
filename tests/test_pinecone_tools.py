import time
import uuid

import pytest

from utils.helpers import load_config

_HAS_KEY = bool(load_config().get("PINECONE_API_KEY"))

needs_key = pytest.mark.skipif(not _HAS_KEY, reason="PINECONE_API_KEY not set in .env")

TICKER = "WAAREEENER"
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
    index.delete(delete_all=True, namespace=TICKER)


def test_chunk_text():
    from tools.pinecone_tools import _chunk_text

    text = "x" * 1200
    chunks = _chunk_text(text, size=500, overlap=50)
    assert all(len(c) <= 500 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1200  # overlap means total >= original
    assert _chunk_text("short") == ["short"]
    assert _chunk_text("   ") == []


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
