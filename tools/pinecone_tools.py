import time
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from utils.helpers import get_logger, load_config
logger = get_logger(__name__)

INDEX_NAME = "stock-research"
EMBED_MODEL = "llama-text-embed-v2"
EMBED_DIM = 1024

_pc: Optional[Pinecone] = None
_index = None


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=load_config()["PINECONE_API_KEY"])
    return _pc


def get_index():
    """Lazy singleton for the shared 'stock-research' serverless index (creates it if missing)."""
    global _index
    if _index is None:
        pc = _get_client()
        if not pc.has_index(INDEX_NAME):
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(INDEX_NAME).status["ready"]:
                time.sleep(1)
        _index = pc.Index(INDEX_NAME)
    return _index


def embed_texts(texts: List[str], input_type: str = "passage") -> List[List[float]]:
    """Embed via Pinecone hosted inference. input_type: 'passage' for docs, 'query' for queries."""
    pc = _get_client()
    vectors: List[List[float]] = []
    # llama-text-embed-v2 rejects >96 inputs per request (400 INVALID_ARGUMENT).
    for start in range(0, len(texts), 96):
        result = pc.inference.embed(
            model=EMBED_MODEL,
            inputs=texts[start : start + 96],
            parameters={"input_type": input_type, "truncate": "END"},
        )
        vectors.extend(item.values for item in result.data)
    return vectors


def _chunk_text(text: str, size: int = 500, overlap: int = 50) -> List[str]:
    if len(text) <= size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
        start += size - overlap
    return chunks


def check_index(ticker: str, source_type: str) -> bool:
    """True if namespace=ticker holds at least one vector tagged with source_type."""
    index = get_index()
    stats = index.describe_index_stats()
    ns = stats.namespaces or {}
    if ticker not in ns or ns[ticker].vector_count == 0:
        return False
    # Namespace exists; confirm at least one vector matches source_type via a filtered query.
    res = index.query(
        namespace=ticker,
        vector=[0.0] * (EMBED_DIM - 1) + [1.0],
        filter={"source_type": {"$eq": source_type}},
        top_k=1,
    )
    return len(res.matches) > 0


def store_to_pinecone(
    ticker: str, docs: List[str], source_type: str, meta: Optional[dict] = None
) -> None:
    """Chunk (~500 chars / 50 overlap), embed as passages, upsert into namespace=ticker.

    Each vector's metadata carries source_type, ticker, chunk_id, the chunk text
    itself (so queries can return it), plus any caller-supplied meta (date, document_id).
    """
    index = get_index()
    chunks: List[str] = []
    for doc in docs:
        chunks.extend(_chunk_text(doc))
    if not chunks:
        return
    embeddings = embed_texts(chunks, input_type="passage")
    doc_key = (meta or {}).get("document_id", f"{source_type}-{int(time.time())}")
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        metadata = {
            "source_type": source_type,
            "ticker": ticker,
            "chunk_id": i,
            "text": chunk,
        }
        if meta:
            metadata.update(meta)
        vectors.append({"id": f"{ticker}-{doc_key}-{i}", "values": emb, "metadata": metadata})
    for start in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[start : start + 100], namespace=ticker, show_progress=False)
    logger.info("Upserted %d vectors to %s/%s (%s)", len(vectors), INDEX_NAME, ticker, source_type)


def query_pinecone(
    ticker: str, query: str, source_type: Optional[str] = None, k: int = 5
) -> List[dict]:
    """Semantic search within namespace=ticker, optionally filtered by source_type.

    source_type=None searches across all source types.
    Returns [{text, score, metadata}] sorted by score desc.
    """
    index = get_index()
    query_vec = embed_texts([query], input_type="query")[0]
    res = index.query(
        namespace=ticker,
        vector=query_vec,
        filter={"source_type": {"$eq": source_type}} if source_type else None,
        top_k=k,
        include_metadata=True,
    )
    return [
        {
            "text": (m.metadata or {}).get("text", ""),
            "score": m.score,
            "metadata": dict(m.metadata or {}),
        }
        for m in res.matches
    ]


def namespace_exists(ticker: str) -> bool:
    """True if namespace=ticker holds at least one vector."""
    stats = get_index().describe_index_stats()
    ns = stats.namespaces or {}
    return ticker in ns and getattr(ns[ticker], "vector_count", 0) > 0


def _doc_id_from_vid(vid: str, ticker: str) -> str:
    """Fallback: parse '{ticker}-{doc_key}-{i}' -> doc_key when metadata lacks document_id."""
    core = vid[len(ticker) + 1:] if vid.startswith(ticker + "-") else vid
    return core.rsplit("-", 1)[0] if "-" in core else core


def list_documents(ticker: str) -> dict:
    """Distinct indexed documents in namespace=ticker, grouped by source_type.

    Returns {source_type: [{document_id, date, source_type, chunk_count}]}, one
    entry per distinct document_id (chunks collapsed). {} if namespace absent.
    """
    if not namespace_exists(ticker):
        return {}
    index = get_index()
    ids: List[str] = []
    for page in index.list(namespace=ticker):
        ids.extend(page)
    docs: dict = {}
    for start in range(0, len(ids), 100):
        resp = index.fetch(ids=ids[start:start + 100], namespace=ticker)
        vectors = getattr(resp, "vectors", {}) or {}
        for vec in vectors.values():
            md = getattr(vec, "metadata", None) or {}
            st = md.get("source_type", "unknown")
            doc_id = md.get("document_id") or _doc_id_from_vid(getattr(vec, "id", ""), ticker)
            bucket = docs.setdefault(st, {})
            entry = bucket.setdefault(
                doc_id,
                {"document_id": doc_id, "date": md.get("date"),
                 "source_type": st, "chunk_count": 0})
            entry["chunk_count"] += 1
            if entry["date"] is None and md.get("date"):
                entry["date"] = md["date"]
    return {st: list(by_id.values()) for st, by_id in docs.items()}


def delete_namespace(ticker: str) -> None:
    """Drop every vector in namespace=ticker. No-op if the namespace is absent."""
    try:
        get_index().delete(delete_all=True, namespace=ticker)
        logger.info("Deleted namespace %s from %s", ticker, INDEX_NAME)
    except Exception as e:  # pinecone raises NotFoundException for missing namespaces
        if "not found" in str(e).lower() or "404" in str(e):
            return
        raise


def wait_for_vectors(ticker: str, timeout_s: int = 20) -> bool:
    """Poll until namespace=ticker is visible (serverless is eventually
    consistent between upsert and query). True once visible, False on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if namespace_exists(ticker):
            return True
        time.sleep(2)
    return False
