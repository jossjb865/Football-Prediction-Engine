"""
Thread-safe token bucket rate limiter for API requests.
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Thread-safe token-bucket rate limiter that strictly respects QPS limits."""

    def __init__(self, rate: float, capacity: int = 3):
        """
        Initialize rate limiter.

        Args:
            rate: Tokens refilled per second (QPS limit)
            capacity: Maximum burst capacity
        """
        self.rate = float(rate)
        self.capacity = int(capacity)
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()
        self._wait_times = deque(maxlen=100)
        logger.info(f"Rate limiter initialized: rate={rate} QPS, capacity={capacity}")

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket, blocking if necessary.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum wait time in seconds

        Returns:
            True if tokens acquired, False if timeout
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            with self.lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

                needed = tokens - self.tokens
                sleep_for = needed / self.rate

            if deadline is not None and time.monotonic() + sleep_for > deadline:
                logger.warning("Rate limiter timeout")
                return False

            time.sleep(min(sleep_for, 0.05))
            self._wait_times.append(sleep_for)

        return True

    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self.lock:
            return {
                "tokens_available": self.tokens,
                "rate": self.rate,
                "capacity": self.capacity,
                "avg_wait_last_100": (
                    sum(self._wait_times) / len(self._wait_times)
                    if self._wait_times else 0.0
                ),
            }
