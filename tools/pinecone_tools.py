import re
import os
import time
from functools import lru_cache
from typing import List, Optional
from threading import Lock
from pinecone import Pinecone, ServerlessSpec
from abc import ABC, abstractmethod
from utils.helpers import get_logger, load_config
from utils.tracing import traceable

logger = get_logger(__name__)

INDEX_NAME = "stock-research"
INDEX_NAME = "stock-research-gemini-3072"
EMBED_DIM = 3072
GEMINI_MODEL = "gemini-embedding-2-preview"
GEMINI_BATCH_SIZE = 100


def _split_keys(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[\n,;]+", raw) if part.strip()]


def _google_api_keys() -> List[str]:
    keys: List[str] = []
    cfg = load_config()
    keys.extend(_split_keys(cfg.get("GOOGLE_API_KEY")))
    keys.extend(_split_keys(os.getenv("GOOGLE_API_KEYS")))
    keys.extend(_split_keys(os.getenv("GEMINI_API_KEY")))
    for suffix in ("1", "2", "3"):
        keys.extend(_split_keys(os.getenv(f"GOOGLE_API_KEY_{suffix}")))
        keys.extend(_split_keys(os.getenv(f"GEMINI_API_KEY_{suffix}")))
    deduped: List[str] = []
    seen = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


@lru_cache(maxsize=None)
def _gemini_client(api_key: str):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_MODEL,
        api_key=api_key,
        output_dimensionality=EMBED_DIM,
    )


def namespace_of(ticker: str) -> str:
    """HDFCBANK.NS -> HDFCBANK. Strips exchange suffix, keeps alnum, uppercases.

    Pinecone namespaces are per-company (spec: one namespace per ticker, e.g.
    "HDFCBANK"), but callers hold the exchange-suffixed symbol (e.g.
    "HDFCBANK.NS") because that's what yfinance needs for market data. Every
    public function below normalizes through this so callers never have to."""
    base = re.sub(r"\.(NS|BO)$", "", ticker.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9]", "", base).upper()

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


class Embedder(ABC):
    chunk_size: int
    chunk_overlap: int
    
    def chunk_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        chunks = []
        start = 0
        while start < len(text):
            chunk = text[start : start + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    @abstractmethod
    def embed(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        pass

class GeminiEmbedder(Embedder):
    chunk_size = 24000
    chunk_overlap = 2000

    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def embed(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        client = _gemini_client(self.api_key)
        task_type = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
        if len(texts) == 1:
            return [client.embed_query(
                texts[0],
                task_type=task_type,
                output_dimensionality=EMBED_DIM,
            )]
        return client.embed_documents(
            texts,
            batch_size=GEMINI_BATCH_SIZE,
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        )

class _GeminiRouter:
    def __init__(self, api_keys: List[str]):
        if not api_keys:
            raise RuntimeError("No Google/Gemini embedding API keys configured.")
        self._api_keys = api_keys
        self._cursor = 0
        self._lock = Lock()

    def reset(self) -> None:
        with self._lock:
            self._cursor = 0

    def size(self) -> int:
        return len(self._api_keys)

    def next_key(self) -> str:
        with self._lock:
            key = self._api_keys[self._cursor]
            self._cursor = (self._cursor + 1) % len(self._api_keys)
            return key

    def select(self) -> GeminiEmbedder:
        return GeminiEmbedder(self.next_key())


_router: Optional[_GeminiRouter] = None


def _get_router() -> _GeminiRouter:
    global _router
    if _router is None:
        _router = _GeminiRouter(_google_api_keys())
    return _router


def reset_embed_floor():
    if _router is not None:
        _router.reset()


def get_current_embedder() -> Embedder:
    return _get_router().select()


def select_embedder() -> Embedder:
    return get_current_embedder()


def advance_embedder_tier():
    if _router is not None:
        _router.next_key()

def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("429" in text or "resource_exhausted" in text
            or "quota" in text or "rate limit" in text
            or "rate_limit" in text or "token limit" in text)


def _embed_with_round_robin(texts: List[str], input_type: str) -> List[List[float]]:
    last_error: Optional[Exception] = None
    router = _get_router()
    for _ in range(router.size()):
        embedder = router.select()
        try:
            return embedder.embed(texts, input_type)
        except Exception as exc:
            if not is_quota_error(exc):
                raise
            last_error = exc
            logger.warning("Gemini key %s quota exhausted; trying next", embedder.api_key)
    if last_error is not None:
        raise last_error
    return []


@traceable(
    name="embed_texts",
    process_inputs=lambda inp: {"n_texts": len(inp.get("texts") or []),
                                "input_type": inp.get("input_type")},
    process_outputs=lambda out: {"n_vectors": len(out or [])},
)
def embed_texts(texts: List[str], input_type: str = "passage") -> List[List[float]]:
    """Embed via Gemini key rotation. input_type: 'passage' for docs, 'query' for queries."""
    return _embed_with_round_robin(texts, input_type)


def check_index(ticker: str, source_type: str) -> bool:
    """True if namespace=ticker holds at least one vector tagged with source_type."""
    ticker = namespace_of(ticker)
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


@traceable(
    name="store_to_pinecone",
    process_inputs=lambda inp: {"ticker": inp.get("ticker"),
                                "n_docs": len(inp.get("docs") or []),
                                "source_type": inp.get("source_type")},
)
def store_to_pinecone(
    ticker: str, docs: List[str], source_type: str, meta: Optional[dict] = None
) -> None:
    """Chunk dynamically by model limit, embed, upsert into namespace=ticker."""
    ticker = namespace_of(ticker)
    index = get_index()
    doc_key = (meta or {}).get("document_id", f"{source_type}-{int(time.time())}")

    chunker = GeminiEmbedder(api_key="")
    chunks: List[str] = []
    for doc in docs:
        chunks.extend(chunker.chunk_text(doc))
    if not chunks:
        return

    embeddings = _embed_with_round_robin(chunks, input_type="passage")
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


@traceable(name="query_pinecone")
def query_pinecone(
    ticker: str, query: str, source_type: Optional[str] = None, k: int = 5
) -> List[dict]:
    """Semantic search within namespace=ticker, optionally filtered by source_type."""
    ticker = namespace_of(ticker)
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
    ticker = namespace_of(ticker)
    stats = get_index().describe_index_stats()
    ns = stats.namespaces or {}
    return ticker in ns and getattr(ns[ticker], "vector_count", 0) > 0


def _doc_id_from_vid(vid: str, ticker: str) -> str:
    """Fallback: parse '{ticker}-{doc_key}-{i}' -> doc_key when metadata lacks document_id."""
    core = vid[len(ticker) + 1:] if vid.startswith(ticker + "-") else vid
    return core.rsplit("-", 1)[0] if "-" in core else core


def list_documents(ticker: str) -> dict:
    """Distinct indexed documents in namespace=ticker, grouped by source_type."""
    ticker = namespace_of(ticker)
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
    ticker = namespace_of(ticker)
    try:
        get_index().delete(delete_all=True, namespace=ticker)
        logger.info("Deleted namespace %s from %s", ticker, INDEX_NAME)
    except Exception as e:  # pinecone raises NotFoundException for missing namespaces
        if "not found" in str(e).lower() or "404" in str(e):
            return
        raise


@traceable(name="wait_for_vectors")
def wait_for_vectors(ticker: str, timeout_s: int = 20) -> bool:
    """Poll until namespace=ticker is visible (serverless is eventually
    consistent between upsert and query). True once visible, False on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if namespace_exists(ticker):
            return True
        time.sleep(2)
    return False
