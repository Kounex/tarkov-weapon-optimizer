"""
API caching logic.
"""

import hashlib
import json
import os
import time

from loguru import logger

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".cache")
CACHE_TTL = 3600  # 1 hour in seconds
CACHE_VERSION = 7  # Increment when data format changes


def get_cache_path(query, variables):
    """Generate a cache file path based on query hash."""
    key = hashlib.md5((query + json.dumps(variables or {}, sort_keys=True)).encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.json")


def load_cache(cache_path):
    """Load cached data if it exists, is not expired, and matches current version."""
    if not os.path.exists(cache_path):
        logger.debug(f"Cache miss: file does not exist at {cache_path}")
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        # Check version and TTL
        if cached.get("version") != CACHE_VERSION:
            logger.debug(f"Cache invalidated: version mismatch (cached={cached.get('version')}, current={CACHE_VERSION})")
            return None  # Version mismatch, invalidate cache
        age = time.time() - cached.get("timestamp", 0)
        if age < CACHE_TTL:
            logger.debug(f"Cache hit: loaded data from {cache_path} (age={age:.0f}s)")
            return cached.get("data")
        logger.debug(f"Cache expired: age={age:.0f}s > TTL={CACHE_TTL}s")
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Cache load error: {e}")
    return None


def save_cache(cache_path, data):
    """Save data to cache with timestamp and version."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": time.time(), "version": CACHE_VERSION, "data": data}, f)
    logger.debug(f"Cache saved to {cache_path}")
