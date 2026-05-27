"""
Disk-backed JSON cache for LLM responses and scoring results.
Provides deterministic caching with optional TTL expiration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """
    Simple file-backed key-value cache.

    Each entry is stored as a JSON file named by its key hash.
    Supports TTL-based expiration.
    """

    def __init__(self, cache_dir: str = "./data/cache", ttl_hours: float = 168.0) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl_seconds = ttl_hours * 3600
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """
        Retrieve a cached value.

        Args:
            key: Cache key (typically a hash string).

        Returns:
            Cached dict, or None if missing/expired.
        """
        path = self._key_path(key)
        if not path.exists():
            self._misses += 1
            return None

        try:
            with open(path, encoding="utf-8") as f:
                entry = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug(f"Cache read error for key {key[:8]}: {exc}")
            self._misses += 1
            return None

        # Check TTL
        stored_at = entry.get("_stored_at", 0)
        if self._ttl_seconds > 0 and (time.time() - stored_at) > self._ttl_seconds:
            path.unlink(missing_ok=True)
            self._misses += 1
            return None

        self._hits += 1
        return entry.get("value")

    def set(self, key: str, value: dict[str, Any]) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key.
            value: Dict to store.
        """
        path = self._key_path(key)
        entry = {"_stored_at": time.time(), "value": value}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
        except OSError as exc:
            logger.warning(f"Cache write error for key {key[:8]}: {exc}")

    def delete(self, key: str) -> None:
        """Remove a cache entry."""
        self._key_path(key).unlink(missing_ok=True)

    def clear(self) -> int:
        """Remove all cache entries. Returns count of removed files."""
        count = 0
        for p in self._dir.glob("*.json"):
            p.unlink()
            count += 1
        logger.info(f"Cleared {count} cache entries")
        return count

    def stats(self) -> dict[str, Any]:
        """Return cache hit/miss statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "entries_on_disk": len(list(self._dir.glob("*.json"))),
        }

    def _key_path(self, key: str) -> Path:
        """Map a cache key to a file path."""
        # Use first 2 chars as subdirectory for filesystem efficiency
        subdir = self._dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.json"
