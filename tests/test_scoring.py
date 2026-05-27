"""Tests for scoring metrics."""
import pytest
from app.scoring.metrics import (
    score_string_exact, score_string_semantic,
    score_integer_exact, score_number_tolerance,
    score_array_simple, score_field,
)


def test_string_exact_match():
    assert score_string_exact("hello", "hello") == 1.0
    assert score_string_exact("Hello", "hello") == 1.0
    assert score_string_exact("hello", "world") == 0.0
    assert score_string_exact(None, "hello") == 0.0
    assert score_string_exact("hello", None) == 0.0


def test_integer_exact():
    assert score_integer_exact(5, 5) == 1.0
    assert score_integer_exact(5, 6) == 0.0
    assert score_integer_exact(None, 5) == 0.0


def test_number_tolerance():
    assert score_number_tolerance(100.0, 100.0, tolerance=0.01) == 1.0
    assert score_number_tolerance(100.5, 100.0, tolerance=0.01) == 1.0  # within 1%
    assert score_number_tolerance(110.0, 100.0, tolerance=0.01) == 0.0  # 10% off
    assert score_number_tolerance(None, 100.0) == 0.0


def test_array_simple_exact():
    pred = ["a", "b", "c"]
    gt = ["a", "b", "c"]
    p, r, f1 = score_array_simple(pred, gt)
    assert f1 > 0.9


def test_array_simple_empty():
    p, r, f1 = score_array_simple([], [])
    assert p == 1.0 and r == 1.0 and f1 == 1.0

    p, r, f1 = score_array_simple([], ["a"])
    assert f1 == 0.0


def test_score_field_dispatch():
    assert score_field("string_exact", "foo", "foo") == 1.0
    assert score_field("integer_exact", 3, 3) == 1.0
    assert score_field("number_tolerance", 99.5, 100.0, tolerance=0.01) == 1.0
