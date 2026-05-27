"""
Checkpoint manager for optimizer resumability.
Wraps DatabaseManager checkpoint operations with higher-level logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

from app.persistence.database import DatabaseManager
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizerState:
    """Serializable state of the optimizer at a checkpoint."""

    experiment_id: int
    iteration: int
    best_prompt_hash: str
    best_score: float
    current_beam: list[str]  # list of prompt hashes in beam
    budget_used: int
    trajectory: list[dict[str, Any]]  # list of iteration records


class CheckpointManager:
    """
    Saves and loads optimizer state for experiment resumability.
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def save(self, state: OptimizerState) -> None:
        """
        Persist optimizer state to database.

        Args:
            state: Current optimizer state.
        """
        self._db.save_checkpoint(
            experiment_id=state.experiment_id,
            iteration=state.iteration,
            best_prompt_hash=state.best_prompt_hash,
            best_score=state.best_score,
            state=asdict(state),
        )
        logger.debug(
            f"[Checkpoint] Saved state at iter {state.iteration}, "
            f"best_score={state.best_score:.4f}"
        )

    def load(self, experiment_id: int) -> Optional[OptimizerState]:
        """
        Load the latest checkpoint for an experiment.

        Args:
            experiment_id: Database experiment ID.

        Returns:
            OptimizerState if checkpoint found, else None.
        """
        raw = self._db.load_latest_checkpoint(experiment_id)
        if raw is None:
            logger.info(f"No checkpoint found for experiment #{experiment_id}")
            return None

        state = OptimizerState(**raw)
        logger.info(
            f"[Checkpoint] Resumed experiment #{experiment_id} "
            f"from iteration {state.iteration}, best_score={state.best_score:.4f}"
        )
        return state

    def find_latest_experiment_id(
        self, experiment_name: str
    ) -> Optional[int]:
        """
        Find the most recent experiment ID matching a name.

        Args:
            experiment_name: Experiment name to search for.

        Returns:
            Experiment ID or None.
        """
        experiments = self._db.list_experiments()
        for exp in experiments:  # already ordered by started_at desc
            if exp.name == experiment_name:
                return exp.id
        return None
