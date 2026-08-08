"""Utils related to the tracing of langchain calls and tools, via langsmith."""

import os
from dotenv import load_dotenv
from utils.helpers import get_logger

logger = get_logger(__name__)

_DEFAULT_ENDPOINT = "https://aws.api.smith.langchain.com"
_DEFAULT_PROJECT = "stock-research"
_initialized = False


def init_tracing() -> bool:
    """Initialize LangSmith tracing, if the api key exists."""
    global _initialized
    load_dotenv()
    if not os.getenv("LANGSMITH_API_KEY"):
        if not _initialized:
            logger.info("LangSmith tracing disabled (no LANGSMITH_API_KEY)")
        _initialized = True
        return os.environ.get("LANGSMITH_TRACING") == "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_ENDPOINT", _DEFAULT_ENDPOINT)
    os.environ.setdefault("LANGSMITH_PROJECT", _DEFAULT_PROJECT)
    if not _initialized:
        logger.info("LangSmith tracing enabled (project=%s)",
                    os.environ["LANGSMITH_PROJECT"])
    _initialized = True
    return True


def set_run_metadata(metadata: dict) -> None:
    """Attach metadata to the current process, if Langsmith is not available this is a no-op."""
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
    """returns langsmith.traceable if available, else a no-op decorator."""
    try:
        from langsmith import traceable as _lt
        return _lt(*dargs, **dkwargs)
    except Exception:
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]

        def deco(fn):
            return fn

        return deco
