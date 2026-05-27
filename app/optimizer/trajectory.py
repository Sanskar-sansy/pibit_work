"""
Optimization trajectory tracker.
Records the history of all prompt versions, scores, and decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TrajectoryEntry:
    """Record of one optimizer step."""

    iteration: int
    prompt_hash: str
    mutation_strategy: Optional[str]
    score: float
    f1: float
    precision: float
    recall: float
    parse_success_rate: float
    accepted: bool
    is_best: bool
    per_field_scores: dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class OptimizationTrajectory:
    """
    Maintains the full trajectory of the optimization run.
    Used for visualization, diff tracking, and report generation.
    """

    def __init__(self) -> None:
        self._entries: list[TrajectoryEntry] = []
        self._best_score: float = 0.0
        self._best_prompt_hash: Optional[str] = None

    def record(
        self,
        iteration: int,
        prompt_hash: str,
        mutation_strategy: Optional[str],
        scores: dict[str, Any],
        accepted: bool,
    ) -> TrajectoryEntry:
        """
        Record one iteration in the trajectory.

        Args:
            iteration: Iteration number.
            prompt_hash: Hash of the prompt used.
            mutation_strategy: Name of the mutation strategy applied.
            scores: Score dict from the evaluator.
            accepted: Whether this prompt was accepted.

        Returns:
            The created TrajectoryEntry.
        """
        score = scores.get("mean_f1", 0.0)
        is_best = score > self._best_score

        if is_best:
            self._best_score = score
            self._best_prompt_hash = prompt_hash

        entry = TrajectoryEntry(
            iteration=iteration,
            prompt_hash=prompt_hash,
            mutation_strategy=mutation_strategy,
            score=score,
            f1=scores.get("mean_f1", 0.0),
            precision=scores.get("mean_precision", 0.0),
            recall=scores.get("mean_recall", 0.0),
            parse_success_rate=scores.get("parse_success_rate", 0.0),
            accepted=accepted,
            is_best=is_best,
            per_field_scores=scores.get("per_field", {}),
        )
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> list[TrajectoryEntry]:
        return self._entries

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_prompt_hash(self) -> Optional[str]:
        return self._best_prompt_hash

    @property
    def accepted_entries(self) -> list[TrajectoryEntry]:
        return [e for e in self._entries if e.accepted]

    @property
    def rejected_entries(self) -> list[TrajectoryEntry]:
        return [e for e in self._entries if not e.accepted]

    def score_curve(self) -> list[tuple[int, float]]:
        """Return (iteration, score) pairs for plotting."""
        return [(e.iteration, e.score) for e in self._entries]

    def best_score_curve(self) -> list[tuple[int, float]]:
        """Return running best score at each iteration."""
        curve = []
        running_best = 0.0
        for e in self._entries:
            if e.score > running_best:
                running_best = e.score
            curve.append((e.iteration, running_best))
        return curve

    def to_dict(self) -> dict[str, Any]:
        """Serialize trajectory to a plain dict (for checkpointing)."""
        return {
            "entries": [
                {
                    "iteration": e.iteration,
                    "prompt_hash": e.prompt_hash,
                    "mutation_strategy": e.mutation_strategy,
                    "score": e.score,
                    "f1": e.f1,
                    "precision": e.precision,
                    "recall": e.recall,
                    "parse_success_rate": e.parse_success_rate,
                    "accepted": e.accepted,
                    "is_best": e.is_best,
                    "per_field_scores": e.per_field_scores,
                    "timestamp": e.timestamp,
                }
                for e in self._entries
            ],
            "best_score": self._best_score,
            "best_prompt_hash": self._best_prompt_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptimizationTrajectory":
        """Deserialize trajectory from a dict (for checkpoint resume)."""
        traj = cls()
        traj._best_score = data.get("best_score", 0.0)
        traj._best_prompt_hash = data.get("best_prompt_hash")
        for raw in data.get("entries", []):
            entry = TrajectoryEntry(
                iteration=raw["iteration"],
                prompt_hash=raw["prompt_hash"],
                mutation_strategy=raw.get("mutation_strategy"),
                score=raw["score"],
                f1=raw["f1"],
                precision=raw["precision"],
                recall=raw["recall"],
                parse_success_rate=raw.get("parse_success_rate", 1.0),
                accepted=raw["accepted"],
                is_best=raw["is_best"],
                per_field_scores=raw.get("per_field_scores", {}),
                timestamp=raw.get("timestamp", ""),
            )
            traj._entries.append(entry)
        return traj
