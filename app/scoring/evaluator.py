"""
Batch evaluator: applies the scoring engine to extraction results
and aggregates metrics across a dataset split.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from app.datasets.schemas import DatasetSplit, ExtractionResult, FieldSpec, ScoredResult
from app.extraction.parser import normalize_extraction
from app.scoring.metrics import (
    compute_precision_recall_f1,
    score_field,
)
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    Scores extraction results against ground truth.

    Implements ExtractBench-compatible evaluation:
    - Per-field scoring
    - Aggregate precision / recall / F1
    - Subtree breakdown
    - Parse failure penalty
    """

    def __init__(self, fields: list[FieldSpec]) -> None:
        self._fields = fields
        self._field_map = {f.name: f for f in fields}

    def score_result(
        self,
        result: ExtractionResult,
        ground_truth: dict[str, Any],
    ) -> ScoredResult:
        """
        Score a single ExtractionResult against its ground truth.

        Args:
            result: Output from the extractor.
            ground_truth: Expected values dict.

        Returns:
            ScoredResult with per-field and aggregate metrics.
        """
        if not result.parse_success or result.parsed is None:
            return ScoredResult(
                sample_id=result.sample_id,
                field_scores={f.name: 0.0 for f in self._fields},
                precision=0.0,
                recall=0.0,
                f1=0.0,
                aggregate_score=0.0,
                parse_success=False,
                details={"error": "parse_failure"},
            )

        # Normalize extracted values
        normalized = normalize_extraction(result.parsed, self._fields)

        field_scores: dict[str, float] = {}
        for field in self._fields:
            name = field.name
            predicted = normalized.get(name)
            expected = ground_truth.get(name)

            # Both null -> full score (correctly identified absence)
            if predicted is None and expected is None:
                field_scores[name] = 1.0
                continue

            # Predicted null when expected non-null -> 0
            if predicted is None and expected is not None:
                field_scores[name] = 0.0
                continue

            # Predicted non-null when expected is null -> penalize slightly
            if predicted is not None and expected is None:
                field_scores[name] = 0.0
                continue

            tolerance = field.tolerance or 0.01
            field_scores[name] = score_field(
                field_type=field.type,
                predicted=predicted,
                ground_truth=expected,
                tolerance=tolerance,
            )

        # Determine which fields have ground truth vs were predicted
        fields_with_gt = {
            name for name, val in ground_truth.items()
            if val is not None and name in self._field_map
        }
        fields_predicted = {
            name for name, val in normalized.items()
            if val is not None
        }

        precision, recall, f1 = compute_precision_recall_f1(
            field_scores, fields_with_gt, fields_predicted
        )

        # Aggregate: mean of all field scores
        aggregate = sum(field_scores.values()) / len(field_scores) if field_scores else 0.0

        return ScoredResult(
            sample_id=result.sample_id,
            field_scores=field_scores,
            precision=precision,
            recall=recall,
            f1=f1,
            aggregate_score=aggregate,
            parse_success=True,
            details={
                "normalized": normalized,
                "ground_truth": ground_truth,
            },
        )

    def evaluate_batch(
        self,
        results: list[ExtractionResult],
        samples_by_id: dict[str, Any],
    ) -> list[ScoredResult]:
        """
        Score a batch of extraction results.

        Args:
            results: Extraction results from the extractor.
            samples_by_id: Dict mapping sample_id -> DatasetSample.

        Returns:
            List of ScoredResult in same order.
        """
        scored = []
        for result in results:
            sample = samples_by_id.get(result.sample_id)
            if sample is None:
                logger.warning(f"Sample {result.sample_id} not found in sample map")
                continue
            scored.append(self.score_result(result, sample.ground_truth))
        return scored

    def aggregate_scores(self, scored_results: list[ScoredResult]) -> dict[str, Any]:
        """
        Compute aggregate statistics over a list of scored results.

        Args:
            scored_results: Scored results from evaluate_batch.

        Returns:
            Dict with mean_f1, mean_precision, mean_recall, mean_aggregate,
            parse_success_rate, and per_field breakdown.
        """
        if not scored_results:
            return {
                "mean_f1": 0.0,
                "mean_precision": 0.0,
                "mean_recall": 0.0,
                "mean_aggregate": 0.0,
                "parse_success_rate": 0.0,
                "n_samples": 0,
                "per_field": {},
            }

        n = len(scored_results)
        parse_successes = sum(1 for r in scored_results if r.parse_success)

        mean_f1 = sum(r.f1 for r in scored_results) / n
        mean_precision = sum(r.precision for r in scored_results) / n
        mean_recall = sum(r.recall for r in scored_results) / n
        mean_aggregate = sum(r.aggregate_score for r in scored_results) / n

        # Per-field breakdown
        per_field: dict[str, float] = {}
        for field in self._fields:
            name = field.name
            scores = [
                r.field_scores.get(name, 0.0)
                for r in scored_results
                if r.parse_success
            ]
            per_field[name] = sum(scores) / len(scores) if scores else 0.0

        return {
            "mean_f1": round(mean_f1, 4),
            "mean_precision": round(mean_precision, 4),
            "mean_recall": round(mean_recall, 4),
            "mean_aggregate": round(mean_aggregate, 4),
            "parse_success_rate": round(parse_successes / n, 4),
            "n_samples": n,
            "per_field": {k: round(v, 4) for k, v in per_field.items()},
        }

    def to_dataframe(self, scored_results: list[ScoredResult]) -> pd.DataFrame:
        """Convert scored results to a pandas DataFrame for analysis."""
        rows = []
        for r in scored_results:
            row: dict[str, Any] = {
                "sample_id": r.sample_id,
                "f1": r.f1,
                "precision": r.precision,
                "recall": r.recall,
                "aggregate": r.aggregate_score,
                "parse_success": r.parse_success,
            }
            for fname, fscore in r.field_scores.items():
                row[f"field_{fname}"] = fscore
            rows.append(row)
        return pd.DataFrame(rows)
