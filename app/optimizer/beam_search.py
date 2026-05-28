"""
Beam search and greedy prompt optimizer.

Greedy: maintains single best prompt, accepts only strict improvements.
Beam: maintains top-k candidates at each iteration.
"""

from __future__ import annotations

import random
from typing import Any, Optional

from app.datasets.schemas import DatasetSample, DatasetSplit, FieldSpec
from app.extraction.extractor import Extractor
from app.llm.ollama_client import OllamaClient
from app.optimizer.acceptance import AcceptancePolicy
from app.optimizer.mutation import MutationStrategy, get_strategy
from app.optimizer.trajectory import OptimizationTrajectory, TrajectoryEntry
from app.persistence.checkpoints import CheckpointManager, OptimizerState
from app.persistence.database import DatabaseManager
from app.scoring.evaluator import Evaluator
from app.utils.hashing import hash_string, short_hash
from app.utils.logging_utils import get_logger
# from app.reporting.plots import OptimizationPlotter
from app.reporting.plots import plot_score_curve


logger = get_logger(__name__)


class BeamSearchOptimizer:
    """
    Beam search prompt optimizer.

    At each iteration:
    1. For each prompt in the beam, generate M mutations.
    2. Evaluate all candidates.
    3. Keep the top-k as the new beam.

    With beam_width=1, this degenerates to greedy search.
    """

    def __init__(
        self,
        client: OllamaClient,
        extractor: Extractor,
        evaluator: Evaluator,
        mutator_model_config: dict[str, Any],
        optimizer_config: dict[str, Any],
        mutation_params: dict[str, Any],
        acceptance_policy: AcceptancePolicy,
        db: DatabaseManager,
        checkpoint_manager: CheckpointManager,
        fields: list[FieldSpec],
        experiment_id: int,
        seed: int = 42,
    ) -> None:
        self._client = client
        self._extractor = extractor
        self._evaluator = evaluator
        self._mutator_cfg = mutator_model_config
        self._opt_cfg = optimizer_config
        self._mutation_params = mutation_params
        self._acceptance = acceptance_policy
        self._db = db
        self._cp = checkpoint_manager
        self._fields = fields
        self._experiment_id = experiment_id
        self._rng = random.Random(seed)

        strategy_name = optimizer_config.get("strategy", "beam")
        self._beam_width: int = (
            optimizer_config.get("beam_width", 3)
            if strategy_name == "beam"
            else 1
        )
        self._max_iterations: int = optimizer_config.get("max_iterations", 8)
        self._budget: int = optimizer_config.get("budget", 80)
        self._mutations_per_iter: int = optimizer_config.get("mutations_per_iter", 5)
        self._strategy_names: list[str] = optimizer_config.get("mutation_strategies", [])

        self._trajectory = OptimizationTrajectory()
        self._budget_used: int = 0

    def optimize(
        self,
        seed_prompt: str,
        val_split: DatasetSplit,
        train_split: Optional[DatasetSplit] = None,
        resume_state: Optional[OptimizerState] = None,
    ) -> tuple[str, OptimizationTrajectory]:
        """
        Run beam search optimization.

        Args:
            seed_prompt: Starting prompt text.
            val_split: Validation split for evaluation.
            train_split: Optional training split for few-shot context.
            resume_state: Optional checkpoint state to resume from.

        Returns:
            Tuple of (best_prompt_text, trajectory).
        """
        samples = val_split.samples
        score_history: list[float] = []
        samples_by_id = {s.id: s for s in samples}
        train_samples = train_split.samples if train_split else []

        # Initialize or resume beam
        if resume_state:
            logger.info("Resuming beam search from checkpoint")
            self._trajectory = OptimizationTrajectory.from_dict(
                {
                    "entries": [vars(e) for e in []],
                    "best_score": resume_state.best_score,
                    "best_prompt_hash": resume_state.best_prompt_hash,
                }
            )
            self._budget_used = resume_state.budget_used
            # Reconstruct beam from stored hashes
            beam: list[tuple[str, float]] = []
            for h in resume_state.current_beam:
                text = self._db.get_prompt_by_hash(h)
                if text:
                    beam.append((text, 0.0))
            if not beam:
                beam = [(seed_prompt, resume_state.best_score)]
            start_iter = resume_state.iteration + 1
        else:
            logger.info(f"Starting beam search (width={self._beam_width}) from seed prompt")
            seed_score = self._evaluate_prompt(seed_prompt, samples, samples_by_id)
            score_history.append(seed_score)
            beam = [(seed_prompt, seed_score)]
            self._record(0, seed_prompt, None, seed_score, accepted=True)
            start_iter = 1

        for iteration in range(start_iter, self._max_iterations + 1):
            if self._budget_used >= self._budget:
                logger.info(f"Budget exhausted at iteration {iteration}")
                break

            logger.info(
                f"[Beam] Iteration {iteration}/{self._max_iterations} | "
                f"Beam size={len(beam)} | Budget {self._budget_used}/{self._budget}"
            )

            all_candidates: list[tuple[str, float, str]] = []  # (prompt, score, strategy)

            # Expand each beam member
            for parent_prompt, parent_score in beam:
                per_field = self._get_per_field(parent_prompt, samples, samples_by_id)
                context = {
                    "per_field_scores": per_field,
                    "train_samples": train_samples,
                    "num_few_shot": self._mutation_params.get(
                        "few_shot_insert", {}
                    ).get("num_examples", 2),
                    "failure_examples": "",
                }

                strategies = self._select_strategies(self._mutations_per_iter)
                for strategy_name in strategies:
                    if self._budget_used >= self._budget:
                        break
                    if not strategy_name:
                        continue

                    strategy = get_strategy(strategy_name)
                    mutated = strategy.mutate(
                        parent_prompt, self._client, self._mutator_cfg, self._fields, context
                    )
                    self._budget_used += 1

                    if not mutated or mutated == parent_prompt:
                        continue

                    score = self._evaluate_prompt(mutated, samples, samples_by_id)
                    all_candidates.append((mutated, score, strategy_name))

                    accepted = self._acceptance.should_accept(
                        score, parent_score, iteration, {}
                    )
                    self._record(iteration, mutated, strategy_name, score, accepted=accepted)
                    self._db.upsert_prompt(
                        hash_string(mutated), mutated,
                        parent_hash=short_hash(parent_prompt),
                        mutation_strategy=strategy_name,
                    )

            if not all_candidates:
                logger.info(f"[Beam] No valid mutations at iteration {iteration}")
                continue

            # Update beam: keep top-k by score, plus current beam members
            all_combined = [(p, s, st) for p, s, st in all_candidates] + [
                (p, s, "carry") for p, s in beam
            ]
            all_combined.sort(key=lambda x: x[1], reverse=True)
            beam = [(p, s) for p, s, _ in all_combined[:self._beam_width]]
            best_iteration_score = beam[0][1]
            score_history.append(best_iteration_score)

            logger.info(
                f"[Beam] Top scores after expansion: "
                f"{[round(s, 4) for _, s in beam]}"
            )

            # Checkpoint
        self._save_checkpoint(iteration, beam)

        trajectory_entries = []

        for entry in self._trajectory.entries:

            trajectory_entries.append({
                "iteration": entry.iteration,
                "score": entry.score,
                "accepted": entry.accepted,
            })

        plot_path = plot_score_curve(
            trajectory_entries,
            "reports/score_curve.png",
            title="Beam Search Optimization"
        )

        logger.info(f"Saved optimization plot to: {plot_path}")

        best_prompt, best_score = beam[0] if beam else (seed_prompt, 0.0)

        logger.info(f"[Beam] Done. Best F1={best_score:.4f}")

        return best_prompt, self._trajectory

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_prompt(
        self,
        prompt: str,
        samples: list[DatasetSample],
        samples_by_id: dict,
    ) -> float:
        """Evaluate prompt on a random subset of validation samples."""
        batch_size = self._opt_cfg.get("batch_size", 5)
        subset_n = min(len(samples), batch_size)
        subset = self._rng.sample(samples, subset_n)
        results = self._extractor.extract_batch(prompt, subset, self._fields)
        scored = self._evaluator.evaluate_batch(results, samples_by_id)
        agg = self._evaluator.aggregate_scores(scored)
        return agg.get("mean_f1", 0.0)

    def _get_per_field(
        self, prompt: str, samples: list, samples_by_id: dict
    ) -> dict[str, float]:
        subset_n = min(len(samples), 5)
        subset = self._rng.sample(samples, subset_n)
        results = self._extractor.extract_batch(prompt, subset, self._fields)
        scored = self._evaluator.evaluate_batch(results, samples_by_id)
        agg = self._evaluator.aggregate_scores(scored)
        return agg.get("per_field", {})

    def _select_strategies(self, n: int) -> list[str]:
        """Select n strategy names, cycling through registered strategies."""
        if not self._strategy_names:
            return []
        return self._rng.choices(self._strategy_names, k=n)

    def _record(
        self,
        iteration: int,
        prompt: str,
        strategy: Optional[str],
        score: float,
        accepted: bool,
    ) -> None:
        """Record iteration in trajectory and database."""
        scores = {
            "mean_f1": score,
            "mean_precision": score,
            "mean_recall": score,
            "mean_aggregate": score,
            "parse_success_rate": 1.0,
            "per_field": {},
        }
        entry = self._trajectory.record(
            iteration=iteration,
            prompt_hash=short_hash(prompt),
            mutation_strategy=strategy,
            scores=scores,
            accepted=accepted,
        )
        self._db.log_iteration(
            experiment_id=self._experiment_id,
            iteration_number=iteration,
            prompt_hash=short_hash(prompt),
            mutation_strategy=strategy,
            scores=scores,
            accepted=accepted,
            is_best=entry.is_best,
        )

    def _save_checkpoint(self, iteration: int, beam: list[tuple[str, float]]) -> None:
        best_prompt, best_score = beam[0]
        state = OptimizerState(
            experiment_id=self._experiment_id,
            iteration=iteration,
            best_prompt_hash=hash_string(best_prompt),
            best_score=best_score,
            current_beam=[hash_string(p) for p, _ in beam],
            budget_used=self._budget_used,
            trajectory=[vars(e) for e in self._trajectory.entries],
        )
        self._cp.save(state)
