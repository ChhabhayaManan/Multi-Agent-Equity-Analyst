"""LLM factory for all agents.

Groq (llama-3.3-70b-versatile) primary; per-call backoff 10/15/20s (+/-2s
jitter) on rate limits, then a single Gemini (gemini-2.0-flash) fallback.

Concurrency: 5 graph branches call this in parallel. There is deliberately
NO module-level "current provider" state — a flag flipped by one branch's
429 would race with and silently downgrade the others. Base clients are
built once and treated as immutable; all retry/fallback state is local to
each invoke() call.
"""

import os
import random
import time
from functools import lru_cache
from typing import Optional, Type

from pydantic import BaseModel

from utils.helpers import get_logger, load_config

logger = get_logger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-3.5-flash"
_BACKOFF_S = (10, 15, 20)
_JITTER_S = 2


@lru_cache(maxsize=1)
def _groq():
    from langchain_groq import ChatGroq
    return ChatGroq(model=GROQ_MODEL, temperature=0,
                    api_key=load_config()["GROQ_API_KEY"])


@lru_cache(maxsize=1)
def _gemini():
    from langchain_google_genai import ChatGoogleGenerativeAI
    # No explicit google_api_key kwarg: let the client resolve the key itself
    # from the environment (GOOGLE_API_KEY or GEMINI_API_KEY). Passing
    # google_api_key=None here would override that dual lookup when only
    # GEMINI_API_KEY is set, since load_config() has no GEMINI_API_KEY entry.
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("429" in text or "rate limit" in text or "rate_limit" in text
            or "quota" in text or "over capacity" in text
            or "ratelimit" in type(exc).__name__.lower())


class _BackoffLLM:
    """Per-call Groq backoff + Gemini fallback. No shared mutable state."""

    def __init__(self, schema: Optional[Type[BaseModel]] = None):
        self._schema = schema

    def _bind(self, model):
        return model.with_structured_output(self._schema) if self._schema else model

    def invoke(self, input, **kwargs):
        primary = self._bind(_groq())
        for delay in _BACKOFF_S:
            try:
                return primary.invoke(input, **kwargs)
            except Exception as e:
                if not _is_rate_limit(e):
                    raise
                wait = delay + random.uniform(-_JITTER_S, _JITTER_S)
                logger.warning("Groq rate-limited; sleeping %.1fs", wait)
                time.sleep(wait)
        try:
            return primary.invoke(input, **kwargs)
        except Exception as e:
            if not _is_rate_limit(e):
                raise
            logger.warning("Groq still rate-limited; falling back to Gemini")
            return self._bind(_gemini()).invoke(input, **kwargs)


def get_llm(structured_schema: Optional[Type[BaseModel]] = None):
    """LLM for prose/structured agent calls, with backoff + fallback built in."""
    if os.getenv("LLM_PROVIDER", "groq").lower() == "gemini":
        model = _gemini()
        return model.with_structured_output(structured_schema) if structured_schema else model
    return _BackoffLLM(structured_schema)


def get_chat_model():
    """Raw ChatGroq for create_react_agent (needs bind_tools, which the
    backoff wrapper can't expose). A 429 mid-ReAct surfaces as an agent
    error and is absorbed by the workflow's validator-retry loop."""
    if os.getenv("LLM_PROVIDER", "groq").lower() == "gemini":
        return _gemini()
    return _groq()
