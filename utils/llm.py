"""LLM factory for all agents.

Fallback ladder per invoke(): Groq llama-3.3-70b -> Groq openai/gpt-oss-120b-> Gemini. Each Groq client carries a 15s request timeout; a rate-limit (429)
or timeout on one tier fails over immediately to the next (no backoff sleeps —
the 8b model is a separate rate bucket, and Gemini is the final safety net).

Concurrency: 5 graph branches call this in parallel. There is deliberately
NO module-level "current provider" state — a flag flipped by one branch's
429 would race with and silently downgrade the others. Base clients are
built once and treated as immutable; all retry/fallback state is local to
each invoke() call.
"""

import os
from functools import lru_cache
from typing import Optional, Type

from pydantic import BaseModel

from utils.helpers import get_logger, load_config

logger = get_logger(__name__)

# Tier order: 70b (best quality) first, 8b-instant (fast, separate bucket) next.
GROQ_MODELS = ("llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct")
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_TIMEOUT_S = 15


@lru_cache(maxsize=None)
def _groq(model: str = GROQ_MODELS[0]):
    from langchain_groq import ChatGroq
    return ChatGroq(model=model, temperature=0, timeout=GROQ_TIMEOUT_S,
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


def _is_timeout(exc: Exception) -> bool:
    return ("timeout" in str(exc).lower()
            or "timeout" in type(exc).__name__.lower())


class _BackoffLLM:
    """Per-call Groq ladder (70b -> 8b) + Gemini fallback. No shared state.
    Any exception on a tier fails over to the next; the last exception is
    raised only if Gemini also fails."""

    def __init__(self, schema: Optional[Type[BaseModel]] = None):
        self._schema = schema

    def _bind(self, model):
        return model.with_structured_output(self._schema) if self._schema else model

    def invoke(self, input, **kwargs):
        for model in GROQ_MODELS:
            try:
                return self._bind(_groq(model)).invoke(input, **kwargs)
            except Exception as e:
                reason = ("rate limit" if _is_rate_limit(e)
                          else "timeout" if _is_timeout(e) else "error")
                logger.warning("Groq %s failed (%s: %s); trying next tier",
                               model, reason, e)
        logger.warning("All Groq tiers failed; falling back to Gemini")
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
