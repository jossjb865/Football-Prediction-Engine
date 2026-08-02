"""
Filesystem-based cache manager with TTL support for API responses.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """Filesystem-based cache with TTL support for iSportsAPI responses."""

    def __init__(self, cache_dir: str, ttl_seconds: int = 86400):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory path for cache storage
            ttl_seconds: Time-to-live for cached items (default: 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        logger.info(f"Cache initialized at {self.cache_dir} with TTL={ttl_seconds}s")

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to filesystem path using SHA256 hash."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached item if exists and not expired.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found/expired
        """
        path = self._key_to_path(key)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            if time.time() - payload["timestamp"] > self.ttl_seconds:
                logger.debug(f"Cache expired for key: {key}")
                path.unlink(missing_ok=True)
                return None

            logger.debug(f"Cache hit: {key}")
            return payload["data"]

        except (json.JSONDecodeError, KeyError, OSError) as exc:
            logger.warning(f"Cache read failed for {key}: {exc}")
            path.unlink(missing_ok=True)
            return None

    def set(self, key: str, data: Any) -> None:
        """
        Store data in cache with current timestamp.

        Args:
            key: Cache key
            data: Data to cache (must be JSON serializable)
        """
        path = self._key_to_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"timestamp": time.time(), "data": data},
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.debug(f"Cache set: {key}")
        except OSError as exc:
            logger.error(f"Cache write failed for {key}: {exc}")

    def clear(self) -> None:
        """Remove all cached files."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink(missing_ok=True)
            count += 1
        logger.info(f"Cache cleared: {count} files removed")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "files": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "cache_dir": str(self.cache_dir)
        }
