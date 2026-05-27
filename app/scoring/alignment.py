"""
Array alignment utilities for ExtractBench-compatible evaluation.
Implements greedy maximum bipartite matching for array fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from rapidfuzz import fuzz

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class AlignmentResult:
    """Result of aligning two arrays."""

    matched_pairs: list[tuple[int, int, float]]  # (pred_idx, gt_idx, score)
    unmatched_pred: list[int]
    unmatched_gt: list[int]
    precision: float
    recall: float
    f1: float


def string_similarity(a: str, b: str) -> float:
    """Token sort ratio similarity between two strings, normalized to [0,1]."""
    return fuzz.token_sort_ratio(a.strip().lower(), b.strip().lower()) / 100.0


def align_arrays(
    predicted: list[Any],
    ground_truth: list[Any],
    similarity_fn: Optional[Callable[[Any, Any], float]] = None,
    threshold: float = 0.5,
) -> AlignmentResult:
    """
    Align two arrays using greedy maximum bipartite matching.

    This implements the ExtractBench repeated-array alignment policy:
    each ground truth item can be matched to at most one predicted item,
    and each predicted item to at most one ground truth item.

    Args:
        predicted: Predicted array items.
        ground_truth: Ground truth array items.
        similarity_fn: Function(a, b) -> float in [0,1]. Defaults to string similarity.
        threshold: Minimum similarity to count as a match.

    Returns:
        AlignmentResult with matched pairs and P/R/F1 metrics.
    """
    if similarity_fn is None:
        similarity_fn = lambda a, b: string_similarity(str(a), str(b))

    if not predicted and not ground_truth:
        return AlignmentResult(
            matched_pairs=[], unmatched_pred=[], unmatched_gt=[],
            precision=1.0, recall=1.0, f1=1.0
        )
    if not predicted:
        return AlignmentResult(
            matched_pairs=[], unmatched_pred=[],
            unmatched_gt=list(range(len(ground_truth))),
            precision=0.0, recall=0.0, f1=0.0
        )
    if not ground_truth:
        return AlignmentResult(
            matched_pairs=[], unmatched_pred=list(range(len(predicted))),
            unmatched_gt=[],
            precision=0.0, recall=0.0, f1=0.0
        )

    # Build similarity matrix
    n_pred = len(predicted)
    n_gt = len(ground_truth)
    matrix: list[list[float]] = [
        [similarity_fn(predicted[pi], ground_truth[gi]) for gi in range(n_gt)]
        for pi in range(n_pred)
    ]

    # Greedy: sort all (score, pi, gi) descending, assign greedily
    triples = sorted(
        [(matrix[pi][gi], pi, gi) for pi in range(n_pred) for gi in range(n_gt)],
        key=lambda x: -x[0],
    )

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matched_pairs: list[tuple[int, int, float]] = []

    for score, pi, gi in triples:
        if score < threshold:
            break  # remaining scores are all below threshold
        if pi not in matched_pred and gi not in matched_gt:
            matched_pairs.append((pi, gi, score))
            matched_pred.add(pi)
            matched_gt.add(gi)

    unmatched_pred = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]

    match_score_sum = sum(s for _, _, s in matched_pairs)
    precision = match_score_sum / n_pred
    recall = match_score_sum / n_gt
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return AlignmentResult(
        matched_pairs=matched_pairs,
        unmatched_pred=unmatched_pred,
        unmatched_gt=unmatched_gt,
        precision=precision,
        recall=recall,
        f1=f1,
    )
