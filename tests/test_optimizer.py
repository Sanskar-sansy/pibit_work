"""Tests for optimizer components."""
import pytest
from app.optimizer.acceptance import StrictAcceptance, SimulatedAnnealingAcceptance
from app.optimizer.trajectory import OptimizationTrajectory
from app.optimizer.mutation import list_strategies, get_strategy
from app.extraction.parser import normalize_value
from app.datasets.schemas import FieldSpec


def test_strict_acceptance_accepts_improvement():
    policy = StrictAcceptance(min_improvement=0.01)
    assert policy.should_accept(0.8, 0.75, 1, {}) is True


def test_strict_acceptance_rejects_regression():
    policy = StrictAcceptance(min_improvement=0.01)
    assert policy.should_accept(0.7, 0.75, 1, {}) is False


def test_strict_acceptance_rejects_tiny_improvement():
    policy = StrictAcceptance(min_improvement=0.01)
    assert policy.should_accept(0.7501, 0.75, 1, {}) is False


def test_sa_accepts_improvement():
    policy = SimulatedAnnealingAcceptance(initial_temperature=1.0)
    # Improvements always accepted
    for _ in range(10):
        assert policy.should_accept(0.9, 0.8, 1, {}) is True


def test_trajectory_records():
    traj = OptimizationTrajectory()
    scores = {"mean_f1": 0.7, "mean_precision": 0.7, "mean_recall": 0.7,
              "mean_aggregate": 0.7, "per_field": {}}
    traj.record(0, "abc123", None, scores, accepted=True)
    traj.record(1, "def456", "instruction_rewrite", {**scores, "mean_f1": 0.8}, accepted=True)
    assert traj.best_score == pytest.approx(0.8)
    assert len(traj.entries) == 2


def test_trajectory_serialization():
    traj = OptimizationTrajectory()
    scores = {"mean_f1": 0.6, "mean_precision": 0.6, "mean_recall": 0.6,
              "mean_aggregate": 0.6, "per_field": {}}
    traj.record(0, "abc", "seed", scores, accepted=True)
    data = traj.to_dict()
    restored = OptimizationTrajectory.from_dict(data)
    assert restored.best_score == pytest.approx(0.6)


def test_all_strategies_registered():
    strategies = list_strategies()
    expected = [
        "instruction_rewrite", "output_format_tighten", "verbosity_reduce",
        "hallucination_suppress", "schema_aware_refine", "field_constraint_add",
        "chain_of_thought_toggle", "few_shot_insert",
    ]
    for s in expected:
        assert s in strategies, f"Strategy '{s}' not registered"


def test_normalize_value_string():
    field = FieldSpec(name="title", type="string_exact")
    assert normalize_value("  Hello  ", field) == "Hello"
    assert normalize_value("null", field) is None
    assert normalize_value(None, field) is None


def test_normalize_value_number():
    field = FieldSpec(name="price", type="number_tolerance")
    assert normalize_value("$1,234.56", field) == pytest.approx(1234.56)
    assert normalize_value(99.9, field) == pytest.approx(99.9)


def test_normalize_value_integer():
    field = FieldSpec(name="years", type="integer_exact")
    assert normalize_value("5", field) == 5
    assert normalize_value(7, field) == 7
