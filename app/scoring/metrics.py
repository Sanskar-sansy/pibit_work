"""
ExtractBench-compatible scoring metrics.

Supports:
- string_exact: exact string match
- string_semantic: embedding cosine similarity via RapidFuzz fallback
- integer_exact: exact integer match
- number_tolerance: match within tolerance
- array_llm: set-based alignment with semantic matching
"""

from __future__ import annotations

import math
from typing import Any, Optional

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def score_string_exact(predicted: Optional[str], ground_truth: Optional[str]) -> float:
    """Exact string match (case-insensitive, stripped)."""
    if predicted is None or ground_truth is None:
        return 0.0
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0


def score_string_semantic(
    predicted: Optional[str],
    ground_truth: Optional[str],
    threshold: float = 0.5,
) -> float:
    """
    Semantic string similarity using RapidFuzz token_sort_ratio.
    Falls back gracefully without requiring sentence-transformers.

    Score is 1.0 if similarity >= threshold, else proportional.
    """
    if predicted is None or ground_truth is None:
        return 0.0
    if not predicted.strip() or not ground_truth.strip():
        return 0.0

    # Normalize ratio to [0,1]
    ratio = fuzz.token_sort_ratio(predicted.strip().lower(), ground_truth.strip().lower()) / 100.0
    return ratio


def score_integer_exact(predicted: Optional[int], ground_truth: Optional[int]) -> float:
    """Exact integer match."""
    if predicted is None or ground_truth is None:
        return 0.0
    try:
        return 1.0 if int(predicted) == int(ground_truth) else 0.0
    except (TypeError, ValueError):
        return 0.0


def score_number_tolerance(
    predicted: Optional[float],
    ground_truth: Optional[float],
    tolerance: float = 0.01,
) -> float:
    """
    Numeric match within relative tolerance.

    Score is 1.0 if |pred - gt| <= tolerance * max(|gt|, 1).
    Degrades linearly beyond tolerance up to 2x tolerance.
    """
    if predicted is None or ground_truth is None:
        return 0.0
    try:
        pred = float(predicted)
        gt = float(ground_truth)
    except (TypeError, ValueError):
        return 0.0

    if math.isnan(pred) or math.isnan(gt):
        return 0.0

    scale = max(abs(gt), 1.0)
    diff = abs(pred - gt)
    relative_diff = diff / scale

    if relative_diff <= tolerance:
        return 1.0
    if relative_diff >= tolerance * 2:
        return 0.0
    # Linear decay between tolerance and 2*tolerance
    return 1.0 - (relative_diff - tolerance) / tolerance


def score_array_simple(
    predicted: Optional[list],
    ground_truth: Optional[list],
    item_score_fn=None,
) -> tuple[float, float, float]:
    """
    Array scoring using greedy maximum bipartite matching.

    Returns:
        Tuple of (precision, recall, f1).
    """
    if predicted is None:
        predicted = []
    if ground_truth is None:
        ground_truth = []

    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not ground_truth:
        return 0.0, 0.0, 0.0

    if item_score_fn is None:
        item_score_fn = score_string_semantic

    # Build similarity matrix
    matrix = [
        [item_score_fn(str(p), str(g)) for g in ground_truth]
        for p in predicted
    ]

    # Greedy matching: pick highest similarity pairs
    matched_pred = set()
    matched_gt = set()
    match_scores: list[float] = []

    # Collect all (score, pred_idx, gt_idx) triples
    triples = [
        (matrix[pi][gi], pi, gi)
        for pi in range(len(predicted))
        for gi in range(len(ground_truth))
    ]
    triples.sort(key=lambda x: -x[0])

    for score, pi, gi in triples:
        if pi not in matched_pred and gi not in matched_gt:
            match_scores.append(score)
            matched_pred.add(pi)
            matched_gt.add(gi)

    # Precision: of predicted items, how many matched well?
    precision = sum(match_scores) / len(predicted) if predicted else 0.0
    # Recall: of ground truth items, how many were covered?
    recall = sum(match_scores) / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


# ---------------------------------------------------------------------------
# Dispatch by field type
# ---------------------------------------------------------------------------

def score_field(
    field_type: str,
    predicted: Any,
    ground_truth: Any,
    tolerance: float = 0.01,
) -> float:
    """
    Dispatch scoring to the appropriate metric based on field type.

    Args:
        field_type: One of the ExtractBench field types.
        predicted: Predicted value.
        ground_truth: Ground truth value.
        tolerance: Tolerance for number_tolerance type.

    Returns:
        Score in [0, 1].
    """
    if field_type == "string_exact":
        return score_string_exact(predicted, ground_truth)

    if field_type == "string_semantic":
        return score_string_semantic(predicted, ground_truth)

    if field_type == "integer_exact":
        return score_integer_exact(predicted, ground_truth)

    if field_type == "number_tolerance":
        return score_number_tolerance(predicted, ground_truth, tolerance=tolerance)

    if field_type == "array_llm":
        _, _, f1 = score_array_simple(predicted, ground_truth)
        return f1

    # Default fallback
    logger.warning(f"Unknown field type '{field_type}', using exact string match")
    return score_string_exact(str(predicted) if predicted else None,
                              str(ground_truth) if ground_truth else None)


def compute_precision_recall_f1(
    field_scores: dict[str, float],
    fields_with_ground_truth: set[str],
    fields_predicted: set[str],
) -> tuple[float, float, float]:
    """
    Compute macro-averaged precision, recall, and F1 from per-field scores.

    Args:
        field_scores: Dict of field_name -> score.
        fields_with_ground_truth: Fields that exist in ground truth.
        fields_predicted: Fields that the model output.

    Returns:
        Tuple of (precision, recall, f1).
    """
    if not field_scores:
        return 0.0, 0.0, 0.0

    # Precision: average score over predicted fields
    if fields_predicted:
        precision = sum(field_scores.get(f, 0.0) for f in fields_predicted) / len(fields_predicted)
    else:
        precision = 0.0

    # Recall: average score over ground truth fields
    if fields_with_ground_truth:
        recall = sum(field_scores.get(f, 0.0) for f in fields_with_ground_truth) / len(
            fields_with_ground_truth
        )
    else:
        recall = 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1
