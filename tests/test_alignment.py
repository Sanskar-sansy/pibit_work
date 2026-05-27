"""Tests for array alignment."""
import pytest
from app.scoring.alignment import align_arrays


def test_perfect_alignment():
    result = align_arrays(["apple", "banana"], ["apple", "banana"])
    assert result.f1 > 0.9
    assert len(result.matched_pairs) == 2


def test_empty_arrays():
    result = align_arrays([], [])
    assert result.f1 == 1.0


def test_no_overlap():
    result = align_arrays(["cat"], ["dog"])
    assert result.f1 < 0.5


def test_partial_match():
    result = align_arrays(["apple", "grape"], ["apple", "banana"])
    assert 0.3 < result.f1 < 0.9


def test_one_sided_empty_pred():
    result = align_arrays([], ["a", "b"])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_one_sided_empty_gt():
    result = align_arrays(["a"], [])
    assert result.f1 == 0.0
