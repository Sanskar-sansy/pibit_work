"""
Validates extraction outputs against field specs.
Identifies missing required fields, type mismatches, and empty results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.datasets.schemas import FieldSpec
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """Result of validating one extraction against its field specs."""

    is_valid: bool
    missing_required: list[str] = field(default_factory=list)
    type_mismatches: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.missing_required:
            parts.append(f"Missing required: {self.missing_required}")
        if self.type_mismatches:
            parts.append(f"Type mismatches: {self.type_mismatches}")
        if self.warnings:
            parts.append(f"Warnings: {self.warnings}")
        return "; ".join(parts) if parts else "OK"


def validate_extraction(
    extracted: Optional[dict[str, Any]],
    fields: list[FieldSpec],
) -> ValidationReport:
    """
    Validate an extraction result against expected field specifications.

    Args:
        extracted: The normalized extraction dict.
        fields: Expected field specs.

    Returns:
        ValidationReport with details about any issues found.
    """
    if extracted is None:
        missing = [f.name for f in fields if f.required]
        return ValidationReport(
            is_valid=False,
            missing_required=missing,
            warnings=["Extraction returned None (parse failure)"],
        )

    missing_required: list[str] = []
    type_mismatches: list[str] = []
    warnings: list[str] = []

    for field_spec in fields:
        name = field_spec.name
        value = extracted.get(name)

        if value is None:
            if field_spec.required:
                missing_required.append(name)
            continue

        ftype = field_spec.type
        if ftype in ("string_exact", "string_semantic") and not isinstance(value, str):
            type_mismatches.append(f"{name}: expected str, got {type(value).__name__}")

        elif ftype == "integer_exact" and not isinstance(value, int):
            type_mismatches.append(f"{name}: expected int, got {type(value).__name__}")

        elif ftype == "number_tolerance" and not isinstance(value, (int, float)):
            type_mismatches.append(
                f"{name}: expected number, got {type(value).__name__}"
            )

        elif ftype == "array_llm" and not isinstance(value, list):
            type_mismatches.append(f"{name}: expected list, got {type(value).__name__}")

        # Warn about empty strings
        if isinstance(value, str) and not value.strip():
            warnings.append(f"{name}: value is empty string")

        # Warn about empty lists
        if isinstance(value, list) and len(value) == 0:
            warnings.append(f"{name}: list is empty")

    is_valid = len(missing_required) == 0 and len(type_mismatches) == 0

    return ValidationReport(
        is_valid=is_valid,
        missing_required=missing_required,
        type_mismatches=type_mismatches,
        warnings=warnings,
    )
