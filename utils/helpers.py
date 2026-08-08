import logging
import math
import numbers
import os
from functools import wraps
from pathlib import Path

from diskcache import Cache
from dotenv import load_dotenv

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_cache = Cache(str(_CACHE_DIR))


def load_config() -> dict:
    load_dotenv()
    return {
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "NEWSDATA_API_KEY": os.getenv("NEWSDATA_API_KEY"),
        "COHERE_API_KEY": os.getenv("COHERE_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "ALPHAVANTAGE_API_KEY": os.getenv("ALPHAVANTAGE_API_KEY")
        or os.getenv("ALPHA_VANTAGE_API"),
        "HUGGINGFACEHUB_API_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    }


def scrub_nan(value):
    """Recursively replace NaN/inf with None. to avoid JSON serialization errors."""
    if isinstance(value, dict):
        return {k: scrub_nan(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_nan(v) for v in value]
    if isinstance(value, numbers.Real) and not isinstance(value, (bool, int)):
        return None if math.isnan(value) or math.isinf(value) else value
    return value


def get_cache() -> Cache:
    """Shared diskcache instance, for callers needing manual get/set."""
    return _cache


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

# provides a decorator for caching function results to disk with a specified time-to-live (TTL) in hours.
def disk_cache(ttl_hours: float = 24):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__module__, func.__qualname__, args, tuple(sorted(kwargs.items())))
            if key in _cache:
                return _cache[key]
            result = func(*args, **kwargs)
            _cache.set(key, result, expire=ttl_hours * 3600)
            return result

        return wrapper

    return decorator
