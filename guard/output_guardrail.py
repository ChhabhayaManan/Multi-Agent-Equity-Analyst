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

# Embeddings can't match short numeric prose ("price is 3595.0 INR") to terse
# JSON tool output ('{"price": 3595.0}') — measured sim ~0.4-0.5, under the
# threshold. A sentence citing a number that appears verbatim in a grounding
# chunk is therefore also considered grounded (numeric-overlap path).
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")

# The system prompt mandates this exact refusal; it asserts no facts, so it
# must never be stripped as ungrounded.
REFUSAL_PHRASE = "i don't have that information"

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


def _is_refusal(sentence: str) -> bool:
    return REFUSAL_PHRASE in sentence.lower().replace("’", "'")


def _numeric_overlap(sentence: str, chunks_norm: List[str]) -> bool:
    """True if any number cited in the sentence matches a number in any
    grounding chunk, within the precision the sentence itself uses.

    Tool outputs carry raw float precision ("2859.199951171875") while the
    LLM reports a rounded value ("2859.20") — a plain substring/exact-string
    match would reject that as ungrounded, so chunk values are rounded to the
    sentence number's own decimal-place count before comparing.
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
    """One flag per sentence: True if the sentence is the mandated refusal,
    cites a number present in a chunk, or clears GROUNDEDNESS_THRESHOLD vs
    any chunk by embedding similarity."""
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
