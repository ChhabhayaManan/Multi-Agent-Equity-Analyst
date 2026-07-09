# utils/tracing.py
"""LangSmith tracing bootstrap. Opt-in via env; a no-op without a key so
local runs never hard-depend on LangSmith."""

import os

from dotenv import load_dotenv

from utils.helpers import get_logger

logger = get_logger(__name__)

_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
_DEFAULT_PROJECT = "stock-research"
_initialized = False


def init_tracing() -> bool:
    """Enable LangSmith tracing iff LANGCHAIN_API_KEY is set. Idempotent.
    Returns True when tracing is enabled, False when it is a no-op."""
    global _initialized
    load_dotenv()
    if not os.getenv("LANGCHAIN_API_KEY"):
        if not _initialized:
            logger.info("LangSmith tracing disabled (no LANGCHAIN_API_KEY)")
        _initialized = True
        return os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_ENDPOINT", _DEFAULT_ENDPOINT)
    os.environ.setdefault("LANGCHAIN_PROJECT", _DEFAULT_PROJECT)
    if not _initialized:
        logger.info("LangSmith tracing enabled (project=%s)",
                    os.environ["LANGCHAIN_PROJECT"])
    _initialized = True
    return True


def set_run_metadata(metadata: dict) -> None:
    """Attach metadata to the current LangSmith run, if any. No-op when
    tracing is off or langsmith is unavailable."""
    try:
        from langsmith.run_helpers import get_current_run_tree
    except Exception:
        return
    try:
        rt = get_current_run_tree()
        if rt is not None:
            rt.add_metadata(metadata)
    except Exception:
        pass


def traceable(*dargs, **dkwargs):
    """`langsmith.traceable` when importable, else a no-op decorator that
    supports both `@traceable` and `@traceable(...)` usage."""
    try:
        from langsmith import traceable as _lt
        return _lt(*dargs, **dkwargs)
    except Exception:
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        def deco(fn):
            return fn

        return deco
