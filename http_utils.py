import random
import time
from typing import Optional

import requests


def request_with_retry(
    method: str,
    url: str,
    *,
    params=None,
    timeout: int = 20,
    max_retries: int = 5,
    backoff_base_seconds: float = 1.0,
    logger=None,
    headers: Optional[dict] = None,
):
    """
    HTTP request wrapper with retry/backoff for transient failures.
    Retries on:
    - network/request exceptions
    - HTTP 429
    - HTTP 5xx
    """
    for attempt in range(max_retries):
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                timeout=timeout,
                headers=headers,
            )
            status = response.status_code
            retryable = status == 429 or 500 <= status < 600
            if not retryable:
                response.raise_for_status()
                return response
            if logger:
                logger.warning(
                    "Retryable HTTP status %s for %s %s (attempt %d/%d).",
                    status,
                    method.upper(),
                    url,
                    attempt + 1,
                    max_retries,
                )
        except requests.RequestException as exc:
            if logger:
                logger.warning(
                    "HTTP request error for %s %s (attempt %d/%d): %s",
                    method.upper(),
                    url,
                    attempt + 1,
                    max_retries,
                    exc,
                )
            response = None

        if attempt == max_retries - 1:
            if response is not None:
                response.raise_for_status()
            raise requests.RequestException(
                f"Failed {method.upper()} {url} after {max_retries} attempts."
            )

        sleep_seconds = (backoff_base_seconds * (2 ** attempt)) + random.uniform(0.1, 0.7)
        time.sleep(sleep_seconds)


def jitter_sleep(min_seconds: float = 1.0, max_seconds: float = 5.0):
    time.sleep(random.uniform(min_seconds, max_seconds))
