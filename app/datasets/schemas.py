"""
Pydantic schemas for dataset samples, fields, and extraction targets.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class FieldSpec(BaseModel):
    """Specification for a single extraction field."""

    name: str
    type: str = "string_exact"  # string_exact | string_semantic | integer_exact | number_tolerance | array_llm
    required: bool = True
    tolerance: Optional[float] = None  # for number_tolerance
    description: Optional[str] = None


class DatasetSample(BaseModel):
    """A single sample in an extraction dataset."""

    id: str
    input_text: str
    ground_truth: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None


class DatasetSplit(BaseModel):
    """A named split of a dataset (train, validation, test)."""

    name: str  # train | validation | test
    samples: list[DatasetSample]
    fields: list[FieldSpec]

    @property
    def size(self) -> int:
        return len(self.samples)


class ExtractionTarget(BaseModel):
    """What the extractor should produce for one sample."""

    sample_id: str
    fields: list[FieldSpec]
    input_text: str


class ExtractionResult(BaseModel):
    """Raw output from the extractor for one sample."""

    sample_id: str
    raw_output: str
    parsed: Optional[dict[str, Any]] = None
    parse_success: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    model: str = "unknown"
    prompt_hash: str = ""


class ScoredResult(BaseModel):
    """Scored extraction result for a single sample."""

    sample_id: str
    field_scores: dict[str, float]  # field_name -> score [0,1]
    precision: float
    recall: float
    f1: float
    aggregate_score: float
    parse_success: bool = True
    details: dict[str, Any] = Field(default_factory=dict)
