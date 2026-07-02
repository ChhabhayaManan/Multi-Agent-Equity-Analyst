import math
import re
from typing import List

from guard.messages import JUDGE_ERROR_MESSAGE, NO_GROUNDED_INFO_MESSAGE
from guard.result import GuardrailResult
from tools.pinecone_tools import embed_texts
from utils.helpers import get_logger

logger = get_logger(__name__)

# A response sentence is grounded if its cosine similarity to at least one
# retrieved chunk clears this threshold (llama-text-embed-v2 embeddings).
GROUNDEDNESS_THRESHOLD = 0.6

# Advice language patterns. Deliberately require advice *context*, not bare
# keywords — "promoters hold 74%" and "FIIs bought shares" are factual and
# must survive; "you should buy this stock" must not.
_ACTION = r"(buy|sell|hold|invest|exit|accumulat\w*|book(?:ing)? profits?|enter)"
ADVICE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        rf"\b(recommend\w*|suggest\w*|advis\w*|advice)\b.{{0,60}}\b{_ACTION}",
        rf"\b(should|must|ought to|good time to|worth|better to)\b.{{0,40}}\b{_ACTION}\w*\b",
        rf"\b(buy|sell|hold)\s+(rating|call|recommendation|signal)\b",
        rf"^\s*{_ACTION}\b.{{0,40}}\b(stock|share|now|today)",
        rf"\b(strong|good|clear)\s+(buy|sell)\b",
    ]
]


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _is_advice(sentence: str) -> bool:
    return any(p.search(sentence) for p in ADVICE_PATTERNS)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _grounded_flags(sentences: List[str], chunks: List[str]) -> List[bool]:
    """One flag per sentence: True if it clears GROUNDEDNESS_THRESHOLD vs any chunk."""
    if not chunks:
        return [False] * len(sentences)
    sentence_vecs = embed_texts(sentences, input_type="query")
    chunk_vecs = embed_texts(chunks, input_type="passage")
    return [
        max(_cosine(sv, cv) for cv in chunk_vecs) >= GROUNDEDNESS_THRESHOLD
        for sv in sentence_vecs
    ]


class OutputGuardrail:
    def validate(self, llm_response: str, retrieved_chunks: List[str]) -> GuardrailResult:
        try:
            sentences = _split_sentences(llm_response)
            violations = []

            kept = [s for s in sentences if not _is_advice(s)]
            if len(kept) < len(sentences):
                violations.append("advice")

            if kept:
                flags = _grounded_flags(kept, retrieved_chunks)
                grounded = [s for s, ok in zip(kept, flags) if ok]
                if len(grounded) < len(kept):
                    violations.append("ungrounded")
                kept = grounded

            if not kept:
                return GuardrailResult(
                    False, NO_GROUNDED_INFO_MESSAGE, violations or ["ungrounded"],
                    "entire response stripped",
                )
            return GuardrailResult(not violations, " ".join(kept), violations, None)
        except Exception as exc:
            logger.warning("Output guardrail failed, failing closed: %s", exc)
            return GuardrailResult(False, JUDGE_ERROR_MESSAGE, ["guardrail_error"], str(exc))
