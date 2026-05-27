"""
Deterministic hashing utilities for cache keys and deduplication.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_string(text: str) -> str:
    """Return SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_dict(obj: dict[str, Any]) -> str:
    """
    Return a deterministic SHA-256 hex digest of a dict.
    Keys are sorted to ensure consistency regardless of insertion order.
    """
    serialized = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hash_string(serialized)


def hash_prompt_input(prompt: str, document: str, model: str) -> str:
    """
    Generate a cache key for a specific (prompt, document, model) triple.
    Used for LLM response caching.
    """
    return hash_dict({"prompt": prompt, "document": document, "model": model})


def hash_score_input(prediction: Any, ground_truth: Any, metric: str) -> str:
    """
    Generate a cache key for a scoring operation.
    Used for semantic similarity caching.
    """
    return hash_dict(
        {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "metric": metric,
        }
    )


def short_hash(text: str, length: int = 8) -> str:
    """Return a short prefix of the SHA-256 hash for display purposes."""
    return hash_string(text)[:length]
