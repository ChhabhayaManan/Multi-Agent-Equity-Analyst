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

# ==== Embedding configuration (NVIDIA NIM) ====
INDEX_NAME = "stock-research"
EMBED_DIM = 4096  # NV-Embed-v1 outputs 4096-dim vectors
NVIDIA_MODEL = "nvidia/nv-embed-v1"

# Chunk size in characters (~6k-8k tokens assuming ~4 chars/token)
# Using 24000 chars ≈ 6000 tokens, safe for most LLMs.
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 500  # ~50 tokens overlap


def _split_keys(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [part.strip() for part in re.split(r"[\\n,;]+", raw) if part.strip()]


def _get_nvidia_api_key() -> str:
    """Fetch NVIDIA API key from environment or config."""
    cfg = load_config()
    key = (
        cfg.get("NVIDIA_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
    )
    # Basic validation: key should be a non-empty string.
    if not key or not isinstance(key, str):
        raise RuntimeError("NVIDIA API key not found. Set NVIDIA_API_KEY in .env or config.")
    return key.strip()


class Embedder(ABC):
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP

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


class NVIDIAEmbedder(Embedder):
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Lazy import to avoid hard dependency if not used
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

        self._client = NVIDIAEmbeddings(
            model=NVIDIA_MODEL,
            api_key=api_key,
            truncate="NONE",
        )

    def embed(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        """
        Embed a list of strings.
        For a single query (len(texts)==1 and input_type=='query') we use embed_query,
        otherwise we use embed_documents (batched internally by the NVIDIA client).
        """
        if len(texts) == 1 and input_type == "query":
            return [self._client.embed_query(texts[0])]
        # For passages or multiple texts, use embed_documents.
        # The NVIDIA endpoint handles batching; we can optionally chunk here if needed.
        return self._client.embed_documents(texts)


# ----- Singleton embedder (lazy, cached) -----
@lru_cache(maxsize=None)
def _get_embedder() -> NVIDIAEmbedder:
    return NVIDIAEmbedder(_get_nvidia_api_key())


def get_current_embedder() -> Embedder:
    return _get_embedder()


def select_embedder() -> Embedder:
    """Compatibility alias used elsewhere."""
    return get_current_embedder()


def reset_embed_floor() -> None:
    """Reset the cached embedder (no-op for singleton, but kept for compatibility)."""
    _get_embedder.cache_clear()


# ==== Pinecone utilities (unchanged) ====
def namespace_of(ticker: str) -> str:
    """HDFCBANK.NS -> HDFCBANK. Strips exchange suffix, keeps alphanum, uppercases."""
    base = re.sub(r"\\.(NS|BO)$", "", ticker.strip(), flags=re.IGNORECASE)
    return re.sub(r"[^A-Za-z0-9]", "", base).upper()


_pc: Optional[Pinecone] = None
_index = None


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=load_config()["PINECONE_API_KEY"])
    return _pc


def get_index():
    """Lazy singleton for the shared 'stock-research' serverless index (creates if missing)."""
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


# ==== Core embedding function ====
@traceable(
    name="embed_texts",
    process_inputs=lambda inp: {"n_texts": len(inp.get("texts") or []),
                                "input_type": inp.get("input_type")},
    process_outputs=lambda out: {"n_vectors": len(out or [])},
)
def embed_texts(texts: List[str], input_type: str = "passage") -> List[List[float]]:
    """
    Embed texts via the configured NVIDIA NIM model.
    input_type: 'passage' for documents, 'query' for queries.
    """
    embedder = get_current_embedder()
    return embedder.embed(texts, input_type=input_type)


# ==== Remaining Pinecone helpers (unchanged) ====
def check_index(ticker: str, source_type: str) -> bool:
    """True if namespace=ticker holds at least one vector tagged with source_type."""
    ticker = namespace_of(ticker)
    index = get_index()
    stats = index.describe_index_stats()
    ns = stats.namespaces or {}
    if ticker not in ns or ns[ticker].vector_count == 0:
        return False
    res = index.query(
        namespace=ticker,
        vector=[0.0] * (EMBED_DIM - 1) + [1.0],
        filter={"source_type": {"$eq": source_type}},
        top_k=1,
    )
    return len(res.matches) > 0


@traceable(
    name="store_to_pinecone",
    process_inputs=lambda inp: {
        "ticker": inp.get("ticker"),
        "n_docs": len(inp.get("docs") or []),
        "source_type": inp.get("source_type"),
    },
)
def store_to_pinecone(
    ticker: str, docs: List[str], source_type: str, meta: Optional[dict] = None
) -> None:
    """Chunk dynamically by model limit, embed, upsert into namespace=ticker."""
    ticker = namespace_of(ticker)
    index = get_index()
    doc_key = (meta or {}).get("document_id", f"{source_type}-{int(time.time())}")

    # Use a dummy key just for chunking (chunk_text does not need the API key)
    chunker = NVIDIAEmbedder(api_key="")
    chunks: List[str] = []
    for doc in docs:
        chunks.extend(chunker.chunk_text(doc))
    if not chunks:
        return

    embeddings = embed_texts(chunks, input_type="passage")
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
        vectors.append(
            {"id": f"{ticker}-{doc_key}-{i}", "values": emb, "metadata": metadata}
        )
    for start in range(0, len(vectors), 100):
        index.upsert(
            vectors=vectors[start : start + 100],
            namespace=ticker,
            show_progress=False,
        )
    logger.info(
        "Upserted %d vectors to %s/%s (%s)", len(vectors), INDEX_NAME, ticker, source_type
    )


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
    core = vid[len(ticker) + 1 :] if vid.startswith(ticker + "-") else vid
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
                 "source_type": st, "chunk_count": 0},
            )
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
    """Poll until namespace=ticker is visible (serverless is eventually consistent)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if namespace_exists(ticker):
            return True
        time.sleep(2)
    return False