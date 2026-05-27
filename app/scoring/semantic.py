"""
Semantic similarity scoring with deterministic caching.
Uses RapidFuzz as the primary engine (no GPU required).
Optional: sentence-transformers for higher quality embeddings.
"""

from __future__ import annotations

from typing import Optional

from rapidfuzz import fuzz

from app.persistence.cache import ResponseCache
from app.utils.hashing import hash_score_input
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Try to import sentence-transformers for higher quality semantic scoring
_ST_AVAILABLE = False
_encoder = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    pass


def _load_encoder(model_name: str = "all-MiniLM-L6-v2") -> Optional[object]:
    """Lazy-load the sentence transformer encoder."""
    global _encoder
    if _encoder is None and _ST_AVAILABLE:
        try:
            _encoder = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer: {model_name}")
        except Exception as exc:
            logger.warning(f"Could not load sentence transformer: {exc}")
    return _encoder


class SemanticScorer:
    """
    Computes semantic similarity scores between text strings.

    Uses sentence-transformers if available, falls back to RapidFuzz token matching.
    All computations are cached for determinism and performance.
    """

    def __init__(
        self,
        cache: Optional[ResponseCache] = None,
        use_transformers: bool = False,
        transformer_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._cache = cache
        self._use_transformers = use_transformers and _ST_AVAILABLE
        if self._use_transformers:
            _load_encoder(transformer_model)

    def score(self, predicted: str, ground_truth: str) -> float:
        """
        Compute semantic similarity in [0, 1].

        Args:
            predicted: Predicted string.
            ground_truth: Ground truth string.

        Returns:
            Similarity score in [0, 1].
        """
        if not predicted or not ground_truth:
            return 0.0

        cache_key = hash_score_input(predicted, ground_truth, "semantic")

        if self._cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return float(cached.get("score", 0.0))

        if self._use_transformers and _encoder is not None:
            score = self._transformer_score(predicted, ground_truth)
        else:
            score = self._rapidfuzz_score(predicted, ground_truth)

        if self._cache:
            self._cache.set(cache_key, {"score": score})

        return score

    @staticmethod
    def _rapidfuzz_score(a: str, b: str) -> float:
        """RapidFuzz token sort ratio similarity."""
        return fuzz.token_sort_ratio(a.strip().lower(), b.strip().lower()) / 100.0

    @staticmethod
    def _transformer_score(a: str, b: str) -> float:
        """Cosine similarity from sentence-transformer embeddings."""
        import numpy as np
        try:
            embs = _encoder.encode([a, b], normalize_embeddings=True)
            cos_sim = float(np.dot(embs[0], embs[1]))
            # Clamp to [0, 1]
            return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))
        except Exception as exc:
            logger.warning(f"Transformer scoring failed: {exc}")
            return SemanticScorer._rapidfuzz_score(a, b)
