import re
import time
import functools
import logging
from typing import Callable, TypeVar, Any

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RateLimitError(Exception):
    def __init__(self, message: str, body: str = ""):
        super().__init__(message)
        self.retry_after: float | None = _parse_retry_after(body)


def _parse_retry_after(body: str) -> float | None:
    # Matches "try again in 5.3s" (Groq) and "retry in 4s" (Gemini)
    match = re.search(r"retry in ([\d.]+)s", body, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 0.5  # pad slightly so the window actually resets
    return None


def with_retry(
    fn: F,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> F:
    import requests  # avoid circular import; retry.py is imported by providers

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc: Exception | None = None
        delay = base_delay

        for attempt in range(1, max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except (RateLimitError, requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt == max_attempts:
                    break
                actual_delay = getattr(exc, "retry_after", None) or delay
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.1fs.",
                    attempt, max_attempts, exc, actual_delay,
                )
                time.sleep(actual_delay)
                delay *= backoff_factor
            except requests.HTTPError:
                raise

        raise last_exc  # type: ignore[misc]

    return wrapper  # type: ignore[return-value]
