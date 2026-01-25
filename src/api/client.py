"""
Tarkov.dev API client.
"""

import time

import requests
from loguru import logger

from .cache import get_cache_path, load_cache, save_cache
from .queries import GUNS_QUERY, MODS_QUERY

API_URL = "https://api.tarkov.dev/graphql"


def run_query(query, variables=None, max_retries=3):
    """Execute a GraphQL query against the Tarkov API with 1-hour cache and retry logic."""
    cache_path = get_cache_path(query, variables)

    # Try to load from cache
    cached_data = load_cache(cache_path)
    if cached_data is not None:
        return cached_data

    # Fetch from API with retry logic
    logger.info("Fetching data from Tarkov.dev API...")
    last_error = None

    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:
            resp = requests.post(
                API_URL,
                json={"query": query, "variables": variables or {}},
                timeout=90,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            elapsed = time.time() - start_time
            logger.debug(f"API request completed in {elapsed:.2f}s (status={resp.status_code})")

            data = resp.json()
            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                raise RuntimeError(data["errors"])

            result = data["data"]
            save_cache(cache_path, result)
            return result

        except requests.exceptions.ChunkedEncodingError as e:
            last_error = e
            logger.warning(f"API request failed (attempt {attempt}/{max_retries}): Response ended prematurely")
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.warning(f"API request timed out (attempt {attempt}/{max_retries})")
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.warning(f"API connection error (attempt {attempt}/{max_retries}): {e}")
        except requests.RequestException as e:
            last_error = e
            logger.warning(f"API request failed (attempt {attempt}/{max_retries}): {e}")

        if attempt < max_retries:
            wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
            logger.info(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)

    logger.error(f"API request failed after {max_retries} attempts: {last_error}")
    raise RuntimeError(f"Failed to fetch data from API after {max_retries} attempts: {last_error}")


def fetch_all_data():
    """Fetch all guns and mods from the API (cached for 1 hour)."""
    logger.info("Fetching guns...")
    guns_data = run_query(GUNS_QUERY)
    logger.info(f"Found {len(guns_data['itemsByType'])} guns")

    logger.info("Fetching mods...")
    mods_data = run_query(MODS_QUERY)
    logger.info(f"Found {len(mods_data['itemsByType'])} mods")

    return guns_data["itemsByType"], mods_data["itemsByType"]
