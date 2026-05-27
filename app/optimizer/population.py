"""
Population-based prompt optimizer.
Maintains a diverse population of candidate prompts,
applies mutations, and evolves toward higher-scoring variants.
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

logger = get_logger(__name__)


class PopulationOptimizer:
    """
    Evolutionary population-based prompt optimizer.

    Maintains a pool of candidate prompts, applies mutations,
    and replaces the weakest members with successful mutations.
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

        self._population_size: int = optimizer_config.get("population_size", 5)
        self._elite_size: int = optimizer_config.get("elite_size", 2)
        self._max_iterations: int = optimizer_config.get("max_iterations", 6)
        self._budget: int = optimizer_config.get("budget", 120)
        self._mutations_per_iter: int = optimizer_config.get("mutations_per_iter", 3)
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
        Run population-based optimization.

        Args:
            seed_prompt: Starting prompt.
            val_split: Validation split for evaluation.
            train_split: Optional training split for few-shot examples.
            resume_state: Optional state to resume from.

        Returns:
            Tuple of (best_prompt_text, trajectory).
        """
        samples = val_split.samples
        samples_by_id = {s.id: s for s in samples}

        # Initialize population
        if resume_state:
            logger.info("Resuming population optimizer from checkpoint")
            self._trajectory = OptimizationTrajectory.from_dict(
                {"entries": resume_state.trajectory, "best_score": resume_state.best_score,
                 "best_prompt_hash": resume_state.best_prompt_hash}
            )
            self._budget_used = resume_state.budget_used
            population: list[tuple[str, float]] = [
                (h, 0.0) for h in resume_state.current_beam
            ]
            start_iter = resume_state.iteration + 1
        else:
            logger.info("Starting population optimizer from seed prompt")
            seed_score = self._evaluate_prompt(seed_prompt, samples, samples_by_id, 0)
            population = [(seed_prompt, seed_score)]
            self._record_iteration(0, seed_prompt, None, seed_score, samples_by_id, accepted=True)
            start_iter = 1

        train_samples = train_split.samples if train_split else []

        for iteration in range(start_iter, self._max_iterations + 1):
            if self._budget_used >= self._budget:
                logger.info(f"Budget exhausted at iteration {iteration}")
                break

            logger.info(
                f"[Population] Iteration {iteration}/{self._max_iterations} | "
                f"Pop size={len(population)} | Budget used={self._budget_used}/{self._budget}"
            )

            new_candidates: list[tuple[str, float]] = []

            # Generate mutations from each population member
            for parent_prompt, parent_score in population:
                per_field_scores = self._get_per_field_scores(parent_prompt, samples, samples_by_id)
                context = {
                    "per_field_scores": per_field_scores,
                    "train_samples": train_samples,
                    "num_few_shot": self._mutation_params.get("few_shot_insert", {}).get("num_examples", 2),
                    "failure_examples": "",
                }

                for strategy_name in self._rng.choices(self._strategy_names, k=self._mutations_per_iter):
                    if self._budget_used >= self._budget:
                        break
                    strategy = get_strategy(strategy_name)
                    mutated = strategy.mutate(
                        parent_prompt, self._client, self._mutator_cfg, self._fields, context
                    )
                    self._budget_used += 1

                    if mutated and mutated != parent_prompt:
                        score = self._evaluate_prompt(mutated, samples, samples_by_id, iteration)
                        new_candidates.append((mutated, score))
                        accepted = self._acceptance.should_accept(score, parent_score, iteration, {})
                        self._record_iteration(
                            iteration, mutated, strategy_name, score, samples_by_id, accepted
                        )

            # Merge population with new candidates and select top members
            all_candidates = population + new_candidates
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            population = all_candidates[:self._population_size]

            # Save checkpoint after each iteration
            state = OptimizerState(
                experiment_id=self._experiment_id,
                iteration=iteration,
                best_prompt_hash=hash_string(population[0][0]),
                best_score=population[0][1],
                current_beam=[hash_string(p) for p, _ in population],
                budget_used=self._budget_used,
                trajectory=[vars(e) for e in self._trajectory.entries],
            )
            self._cp.save(state)

        best_prompt = population[0][0] if population else seed_prompt
        logger.info(
            f"[Population] Optimization complete. Best score: {population[0][1]:.4f}"
        )
        return best_prompt, self._trajectory

    def _evaluate_prompt(
        self,
        prompt: str,
        samples: list[DatasetSample],
        samples_by_id: dict,
        iteration: int,
    ) -> float:
        """Evaluate a prompt on the validation set and return mean F1."""
        subset_n = min(len(samples), self._opt_cfg.get("batch_size", 5))
        subset = self._rng.sample(samples, subset_n)
        results = self._extractor.extract_batch(prompt, subset, self._fields)
        scored = self._evaluator.evaluate_batch(results, samples_by_id)
        agg = self._evaluator.aggregate_scores(scored)
        return agg.get("mean_f1", 0.0)

    def _get_per_field_scores(self, prompt, samples, samples_by_id) -> dict:
        subset_n = min(len(samples), 5)
        subset = self._rng.sample(samples, subset_n)
        results = self._extractor.extract_batch(prompt, subset, self._fields)
        scored = self._evaluator.evaluate_batch(results, samples_by_id)
        agg = self._evaluator.aggregate_scores(scored)
        return agg.get("per_field", {})

    def _record_iteration(self, iteration, prompt, strategy, score, samples_by_id, accepted):
        """Record iteration in trajectory and database."""
        scores = {"mean_f1": score, "mean_precision": score, "mean_recall": score,
                  "mean_aggregate": score, "parse_success_rate": 1.0, "per_field": {}}
        entry = self._trajectory.record(
            iteration=iteration,
            prompt_hash=short_hash(prompt),
            mutation_strategy=strategy,
            scores=scores,
            accepted=accepted,
        )
        self._db.upsert_prompt(hash_string(prompt), prompt, mutation_strategy=strategy)
        self._db.log_iteration(
            experiment_id=self._experiment_id,
            iteration_number=iteration,
            prompt_hash=short_hash(prompt),
            mutation_strategy=strategy,
            scores=scores,
            accepted=accepted,
            is_best=entry.is_best,
        )
