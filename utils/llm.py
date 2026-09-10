"""LLM factory. Ladder per invoke(): Groq gpt-oss-20b -> Gemini. No
module-level provider state: parallel branches must not downgrade each
other."""

import os
from functools import lru_cache
from typing import Optional, Type

from pydantic import BaseModel

from utils.helpers import get_logger, load_config

logger = get_logger(__name__)

GROQ_MODELS = ("openai/gpt-oss-20b",)
GEMINI_MODEL = "gemini-3.8-flash"
GROQ_TIMEOUT_S = 15


@lru_cache(maxsize=None)
def _groq(model: str = GROQ_MODELS[0]):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, temperature=0, timeout=GROQ_TIMEOUT_S,
                    reasoning_effort="low",
                    api_key=load_config()["GROQ_API_KEY"])


@lru_cache(maxsize=1)
def _gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


def _is_too_large(exc: Exception) -> bool:
    """True when the request is too large for the model to handle."""
    text = str(exc).lower()
    return ("request too large" in text or "reduce your message size" in text
            or "context_length_exceeded" in text
            or "context length" in text or "too many tokens" in text
            or getattr(exc, "status_code", None) == 413)


def _is_rate_limit(exc: Exception) -> bool: #function to check if the error is about the rate limit
    text = str(exc).lower()
    return ("429" in text or "rate limit" in text or "rate_limit" in text
            or "quota" in text or "over capacity" in text
            or "ratelimit" in type(exc).__name__.lower())


def _is_timeout(exc: Exception) -> bool:
    return ("timeout" in str(exc).lower()
            or "timeout" in type(exc).__name__.lower())


def _is_tool_use_failed(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("tool_use_failed" in text
            or ("tool use" in text and "failed" in text)
            or ("tool calling" in text and "error" in text)
            or ("tool use" in text and "error" in text))


def _is_provider_error(exc: Exception) -> bool:
    """Provider-side failure, worth another tier. Local errors are not."""
    return (getattr(exc, "status_code", None) is not None
            or "error code:" in str(exc).lower()
            or _is_timeout(exc))


class _BackoffLLM:
    """Walks the ladder forward; no tier can end the run before Gemini."""

    _MAX_TRIES_PER_TIER = 2

    def __init__(self, schema: Optional[Type[BaseModel]] = None):
        self._schema = schema

    def _bind(self, model):
        return model.with_structured_output(self._schema) if self._schema else model

    def _client(self, provider: str, model_name: str):
        return _groq(model_name) if provider == "groq" else _gemini()

    def invoke(self, input, **kwargs):
        tiers = [("groq", m) for m in GROQ_MODELS] + [("gemini", GEMINI_MODEL)]
        gemini_index = len(tiers) - 1

        index, tries, last_exception = 0, 0, None
        while index < len(tiers):
            provider, model_name = tiers[index]
            tries += 1
            try:
                return self._bind(self._client(provider, model_name)).invoke(
                    input, **kwargs)
            except Exception as e:
                last_exception = e
                if _is_too_large(e) and index < gemini_index:
                    # no smaller Groq bucket can hold it
                    logger.warning("%s: request too large, escalating to %s",
                                   model_name, GEMINI_MODEL)
                    index, tries = gemini_index, 0
                    continue
                if _is_tool_use_failed(e) and tries < self._MAX_TRIES_PER_TIER:
                    logger.warning("%s: tool-call parse failed, retrying once",
                                   model_name)
                    continue
                if not _is_provider_error(e):
                    raise
                logger.warning("%s failed (%s), falling through to next tier",
                               model_name, type(e).__name__)
                index, tries = index + 1, 0
        raise last_exception


def get_llm(structured_schema: Optional[Type[BaseModel]] = None):
    """LLM for prose/structured agent calls, with fallback built in."""
    if os.getenv("LLM_PROVIDER", "groq").lower() == "gemini":
        model = _gemini()
        return model.with_structured_output(structured_schema) if structured_schema else model
    return _BackoffLLM(structured_schema)


def get_chat_model():
    """Raw ChatGroq for create_agent, which needs bind_tools."""
    if os.getenv("LLM_PROVIDER", "groq").lower() == "gemini":
        return _gemini()
    return _groq()
