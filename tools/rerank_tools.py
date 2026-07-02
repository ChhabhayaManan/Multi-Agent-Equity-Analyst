from typing import List, Optional
import cohere
from utils.helpers import get_logger, load_config

logger = get_logger(__name__)
_client: Optional[cohere.ClientV2] = None


def _get_client() -> cohere.ClientV2:
    global _client
    if _client is None:
        _client = cohere.ClientV2(api_key=load_config()["COHERE_API_KEY"])
    return _client


def cohere_rerank(query: str, docs: List[str], top_n: int = 5) -> List[dict]:
    """Rerank docs against query with Cohere rerank-v3.5.

    Returns [{document, relevance_score, index}] sorted by relevance_score desc,
    where index refers to the position in the original docs list.
    """
    if not docs:
        return []
    top_n = min(top_n, len(docs))
    response = _get_client().rerank(
        model="rerank-v3.5", query=query, documents=docs, top_n=top_n
    )
    return [
        {
            "document": docs[r.index],
            "relevance_score": r.relevance_score,
            "index": r.index,
        }
        for r in response.results
    ]
