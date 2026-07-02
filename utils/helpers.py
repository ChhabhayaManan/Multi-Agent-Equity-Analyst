import logging
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
    }


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


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
