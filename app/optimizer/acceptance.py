"""
Acceptance criteria for prompt mutation evaluation.

Determines whether a mutated prompt should be accepted based on
its score relative to the current best.
"""

from __future__ import annotations

import math
import random
from typing import Any

from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AcceptancePolicy:
    """Abstract base for acceptance policies."""

    def should_accept(
        self,
        candidate_score: float,
        current_score: float,
        iteration: int,
        context: dict[str, Any],
    ) -> bool:
        raise NotImplementedError


class StrictAcceptance(AcceptancePolicy):
    """
    Accept only if the candidate score strictly improves over current.
    Uses a configurable minimum improvement threshold.
    """

    def __init__(self, min_improvement: float = 0.001) -> None:
        self._min_improvement = min_improvement

    def should_accept(self, candidate_score, current_score, iteration, context):
        improvement = candidate_score - current_score
        accepted = improvement >= self._min_improvement
        logger.debug(
            f"[Strict] Δ={improvement:+.4f} (min={self._min_improvement:.4f}) "
            f"-> {'ACCEPT' if accepted else 'REJECT'}"
        )
        return accepted


class SimulatedAnnealingAcceptance(AcceptancePolicy):
    """
    Accept improvements always; accept regressions with probability
    based on simulated annealing temperature schedule.

    Useful for population-based search to escape local optima.
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.85,
        seed: int = 42,
    ) -> None:
        self._temp = initial_temperature
        self._cooling_rate = cooling_rate
        self._rng = random.Random(seed)

    def should_accept(self, candidate_score, current_score, iteration, context):
        delta = candidate_score - current_score

        if delta >= 0:
            # Always accept improvements
            self._cool()
            logger.debug(f"[SA] Δ={delta:+.4f} -> ACCEPT (improvement)")
            return True

        # Accept regression with probability exp(delta / T)
        current_temp = self._temp * (self._cooling_rate ** iteration)
        current_temp = max(current_temp, 1e-6)  # floor to prevent division by zero
        probability = math.exp(delta / current_temp)
        accepted = self._rng.random() < probability

        logger.debug(
            f"[SA] Δ={delta:+.4f}, T={current_temp:.4f}, "
            f"p={probability:.4f} -> {'ACCEPT' if accepted else 'REJECT'}"
        )
        self._cool()
        return accepted

    def _cool(self) -> None:
        self._temp *= self._cooling_rate


def build_acceptance_policy(
    policy_name: str,
    optimizer_config: dict[str, Any],
    seed: int = 42,
) -> AcceptancePolicy:
    """
    Factory to build an acceptance policy from config.

    Args:
        policy_name: 'strict' or 'simulated_annealing'.
        optimizer_config: Optimizer section of the config dict.
        seed: Random seed for stochastic policies.

    Returns:
        AcceptancePolicy instance.
    """
    min_improvement = optimizer_config.get("min_improvement", 0.001)

    if policy_name == "strict":
        return StrictAcceptance(min_improvement=min_improvement)

    if policy_name == "simulated_annealing":
        return SimulatedAnnealingAcceptance(
            initial_temperature=optimizer_config.get("temperature", 1.0),
            cooling_rate=optimizer_config.get("cooling_rate", 0.85),
            seed=seed,
        )

    logger.warning(f"Unknown acceptance policy '{policy_name}', using strict")
    return StrictAcceptance(min_improvement=min_improvement)
