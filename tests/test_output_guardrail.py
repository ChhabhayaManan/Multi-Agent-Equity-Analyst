from unittest.mock import patch

import pytest

from guard.messages import JUDGE_ERROR_MESSAGE, NO_GROUNDED_INFO_MESSAGE
from guard.output_guardrail import OutputGuardrail, _is_advice
from utils.helpers import load_config

_HAS_KEY = bool(load_config().get("PINECONE_API_KEY"))

CHUNKS = [
    "Waaree Energies is India's largest solar PV module manufacturer, with "
    "12 GW of module manufacturing capacity across Gujarat.",
    "Waaree Energies reported consolidated revenue of Rs 3,457 crore in Q2, "
    "up 28% year on year, driven by strong export orders.",
]

GROUNDED_SENTENCE = (
    "Waaree Energies reported consolidated revenue of Rs 3,457 crore in Q2, "
    "up 28% year on year."
)
UNGROUNDED_SENTENCE = (
    "The company also announced a merger with a large European automaker last week."
)
ADVICE_SENTENCE = "Based on this growth, you should buy the stock now."


@pytest.fixture(scope="module")
def guardrail():
    return OutputGuardrail()


# --- Advice regex unit tests (no API calls) ---


@pytest.mark.parametrize(
    "sentence",
    [
        "You should buy this stock now.",
        "I recommend selling your shares.",
        "It is a good time to invest in Waaree.",
        "Analysts gave it a strong buy.",
        "This stock has a buy rating from brokerages.",
        "Buy this stock today.",
        "It would be better to exit the position.",
    ],
)
def test_advice_sentences_detected(sentence):
    assert _is_advice(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "Promoters hold 74% of the company.",
        "FIIs bought shares worth Rs 200 crore in October.",
        "The company sells solar modules to US customers.",
        "Waaree invested Rs 600 crore in a new Gujarat facility.",
        "Revenue grew 28% year on year.",
    ],
)
def test_factual_sentences_not_flagged(sentence):
    assert not _is_advice(sentence)


# --- Full validate() flows (real Pinecone inference embeddings) ---


@pytest.mark.skipif(not _HAS_KEY, reason="PINECONE_API_KEY not set in .env")
def test_clean_grounded_response_passes(guardrail):
    result = guardrail.validate(GROUNDED_SENTENCE, CHUNKS)
    assert result.passed
    assert result.violations == []
    assert result.cleaned_text == GROUNDED_SENTENCE


@pytest.mark.skipif(not _HAS_KEY, reason="PINECONE_API_KEY not set in .env")
def test_advice_sentence_stripped(guardrail):
    result = guardrail.validate(f"{GROUNDED_SENTENCE} {ADVICE_SENTENCE}", CHUNKS)
    assert not result.passed
    assert "advice" in result.violations
    assert "should buy" not in result.cleaned_text
    assert "Rs 3,457 crore" in result.cleaned_text


@pytest.mark.skipif(not _HAS_KEY, reason="PINECONE_API_KEY not set in .env")
def test_ungrounded_sentence_stripped(guardrail):
    result = guardrail.validate(f"{GROUNDED_SENTENCE} {UNGROUNDED_SENTENCE}", CHUNKS)
    assert not result.passed
    assert "ungrounded" in result.violations
    assert "automaker" not in result.cleaned_text
    assert "Rs 3,457 crore" in result.cleaned_text


@pytest.mark.skipif(not _HAS_KEY, reason="PINECONE_API_KEY not set in .env")
def test_fully_stripped_response_falls_back(guardrail):
    result = guardrail.validate(ADVICE_SENTENCE, CHUNKS)
    assert not result.passed
    assert result.cleaned_text == NO_GROUNDED_INFO_MESSAGE


def test_no_chunks_means_nothing_grounded(guardrail):
    result = guardrail.validate(GROUNDED_SENTENCE, [])
    assert not result.passed
    assert result.cleaned_text == NO_GROUNDED_INFO_MESSAGE


# --- Numeric-overlap grounding + refusal exemption (no API calls: embeddings
# --- mocked orthogonal so only the new non-embedding paths can ground) ---


def _orthogonal_embeds(texts, input_type="passage"):
    vec = [1.0, 0.0] if input_type == "query" else [0.0, 1.0]
    return [vec for _ in texts]


def test_numeric_overlap_grounds_tool_data(guardrail):
    with patch("guard.output_guardrail.embed_texts", side_effect=_orthogonal_embeds):
        result = guardrail.validate(
            "The current price of Waaree Energies is 3595.0 INR [live: yfinance].",
            ['{"ticker": "WAAREEENER.NS", "price": 3595.0}'],
        )
    assert result.passed
    assert "3595.0" in result.cleaned_text


def test_numeric_overlap_normalizes_commas(guardrail):
    with patch("guard.output_guardrail.embed_texts", side_effect=_orthogonal_embeds):
        result = guardrail.validate(
            "Revenue was Rs 16,736 crore last quarter [live: yfinance].",
            ['{"revenue": 16736}'],
        )
    assert result.passed


def test_numeric_overlap_requires_full_number(guardrail):
    # 35 must NOT match inside 3595.0
    with patch("guard.output_guardrail.embed_texts", side_effect=_orthogonal_embeds):
        result = guardrail.validate(
            "The price rose 35 percent.",
            ['{"price": 3595.0}'],
        )
    assert not result.passed
    assert result.cleaned_text == NO_GROUNDED_INFO_MESSAGE


def test_numeric_overlap_matches_rounded_float(guardrail):
    # LLM rounds "2859.199951171875" to "2859.20" - must still ground.
    with patch("guard.output_guardrail.embed_texts", side_effect=_orthogonal_embeds):
        result = guardrail.validate(
            "The current price of Waaree Energies is 2859.20 INR [live: yfinance].",
            ['{"ticker": "WAAREEENER.NS", "price": 2859.199951171875}'],
        )
    assert result.passed
    assert "2859.20" in result.cleaned_text


def test_no_number_hallucination_still_stripped(guardrail):
    with patch("guard.output_guardrail.embed_texts", side_effect=_orthogonal_embeds):
        result = guardrail.validate(
            "The company announced a merger with a European automaker.",
            ['{"price": 3595.0}'],
        )
    assert not result.passed
    assert result.cleaned_text == NO_GROUNDED_INFO_MESSAGE


def test_refusal_phrase_passes(guardrail):
    with patch("guard.output_guardrail.embed_texts", side_effect=_orthogonal_embeds):
        result = guardrail.validate(
            "I don't have that information.",
            ['{"results": [], "note": "no indexed chunks matched"}'],
        )
    assert result.passed
    assert result.cleaned_text == "I don't have that information."


def test_refusal_phrase_passes_even_without_chunks(guardrail):
    result = guardrail.validate("I don't have that information.", [])
    assert result.passed


def test_embed_error_fails_closed(guardrail):
    # UNGROUNDED_SENTENCE has no numeric overlap with CHUNKS, so it must go
    # through the embedding path — which is broken here.
    with patch(
        "guard.output_guardrail.embed_texts", side_effect=RuntimeError("api down")
    ):
        result = guardrail.validate(UNGROUNDED_SENTENCE, CHUNKS)
    assert not result.passed
    assert result.violations == ["guardrail_error"]
    assert result.cleaned_text == JUDGE_ERROR_MESSAGE
