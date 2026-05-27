"""
Output parsing utilities for extraction results.
Handles type coercion, null normalization, and field validation.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from app.datasets.schemas import FieldSpec
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

_NULL_STRINGS = {"null", "none", "n/a", "na", "not found", "not available", "unknown", ""}


def normalize_value(value: Any, field: FieldSpec) -> Optional[Any]:
    """
    Normalize an extracted value to the expected type for a field.

    Args:
        value: Raw extracted value.
        field: FieldSpec describing the expected type.

    Returns:
        Coerced value or None if unavailable.
    """
    if value is None:
        return None

    if isinstance(value, str) and value.strip().lower() in _NULL_STRINGS:
        return None

    ftype = field.type

    if ftype in ("string_exact", "string_semantic"):
        return str(value).strip()

    if ftype == "integer_exact":
        return _to_int(value)

    if ftype == "number_tolerance":
        return _to_float(value)

    if ftype == "array_llm":
        return _to_list(value)

    return value


def normalize_extraction(
    raw: Optional[dict[str, Any]],
    fields: list[FieldSpec],
) -> dict[str, Any]:
    """
    Normalize all fields in an extraction result.

    Args:
        raw: Parsed dict from LLM output.
        fields: Expected field specs.

    Returns:
        Normalized dict with all specified fields (None if missing).
    """
    if raw is None:
        return {f.name: None for f in fields}

    normalized: dict[str, Any] = {}
    for field in fields:
        raw_value = raw.get(field.name)
        normalized[field.name] = normalize_value(raw_value, field)

    return normalized


def _to_int(value: Any) -> Optional[int]:
    """Convert a value to int, stripping non-numeric characters."""
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        digits = re.sub(r"[^\d-]", "", value)
        if digits:
            try:
                return int(digits)
            except ValueError:
                pass
    return None


def _to_float(value: Any) -> Optional[float]:
    """Convert a value to float, handling currency symbols and commas."""
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value.replace(",", ""))
        try:
            return float(cleaned)
        except ValueError:
            pass
    return None


def _to_list(value: Any) -> Optional[list]:
    """Convert a value to a list."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None]
    if isinstance(value, str):
        # Try comma-separated
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts if parts else None
    return None
