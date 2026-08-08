import math
import re
from typing import List

from guard.messages import JUDGE_ERROR_MESSAGE, NO_GROUNDED_INFO_MESSAGE
from guard.result import GuardrailResult
from tools.pinecone_tools import embed_texts
from utils.helpers import get_logger

logger = get_logger(__name__)

GROUNDEDNESS_THRESHOLD = 0.6

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")

REFUSAL_PHRASE = "I don't have that information"

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

# pattern check, if matches --> advice
def _is_advice(sentence: str) -> bool:
    return any(p.search(sentence) for p in ADVICE_PATTERNS)

# cosine similarity between two vectors
def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0

# response is considered a refusal if it contains the refusal phrase
def _is_refusal(sentence: str) -> bool:
    return REFUSAL_PHRASE.lower() in sentence.lower().replace("’", "'")


def _numeric_overlap(sentence: str, chunks_norm: List[str]) -> bool:
    """
    Check if any numeric value in the sentence matches a numeric value in the chunks.
    with rounding to the same number of decimal places as in the sentence.
    """
    for raw_num in _NUMBER_RE.findall(sentence):
        clean = raw_num.replace(",", "")
        try:
            sent_val = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        for chunk in chunks_norm:
            for chunk_raw_num in _NUMBER_RE.findall(chunk):
                try:
                    chunk_val = float(chunk_raw_num.replace(",", ""))
                except ValueError:
                    continue
                if round(chunk_val, decimals) == sent_val:
                    return True
    return False


def _grounded_flags(sentences: List[str], chunks: List[str]) -> List[bool]:
    """checking if each sentence is grounded in the retrieved chunks, using embeddings and cosine similarity.
    if grounded --> True
    else --> False --> will be removed from the final response"""
    chunks_norm = [c.replace(",", "") for c in chunks]
    flags = []
    pending = []  # indices still needing the embedding check
    for i, sentence in enumerate(sentences):
        if _is_refusal(sentence) or (
            chunks_norm and _numeric_overlap(sentence, chunks_norm)
        ):
            flags.append(True)
        else:
            flags.append(False)
            pending.append(i)
    if not chunks or not pending:
        return flags
    sentence_vecs = embed_texts([sentences[i] for i in pending], input_type="query")
    chunk_vecs = embed_texts(chunks, input_type="passage")
    for i, sv in zip(pending, sentence_vecs):
        flags[i] = max(_cosine(sv, cv) for cv in chunk_vecs) >= GROUNDEDNESS_THRESHOLD
    return flags


class OutputGuardrail:
    
    def validate(self, llm_response: str, retrieved_chunks: List[str]) -> GuardrailResult:
        """Validate the LLM response against advice and grounding rules."""
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
