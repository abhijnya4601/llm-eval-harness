import time
import functools
import logging
from typing import Callable, TypeVar, Any

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    fn: F,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> F:
    """Return a wrapped version of fn with exponential backoff retry.

    Retries on RateLimitError, requests.Timeout, and requests.ConnectionError.
    Non-retryable HTTP errors (4xx except 429) propagate immediately.
    """
    import requests
    # Import here to avoid circular; retry.py is imported by providers.

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        from harness.providers.groq_provider import RateLimitError

        last_exc: Exception | None = None
        delay = base_delay

        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except (RateLimitError, requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt == max_attempts:
                    break
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.1fs.",
                    attempt, max_attempts, exc, delay,
                )
                time.sleep(delay)
                delay *= backoff_factor
            except requests.HTTPError as exc:
                # 4xx client errors other than 429 are not transient — don't retry.
                raise

        raise last_exc  # type: ignore[misc]

    return wrapper  # type: ignore[return-value]
