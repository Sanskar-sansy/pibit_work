"""
Deterministic train/validation/test splitting for datasets.
"""

from __future__ import annotations

import random
from typing import Optional

from app.datasets.schemas import DatasetSample, DatasetSplit, FieldSpec
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DatasetSplitter:
    """
    Splits a list of DatasetSamples into train, validation, and test sets.
    Splitting is deterministic given a fixed seed.
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def split(
        self,
        samples: list[DatasetSample],
        fields: list[FieldSpec],
        validation_ratio: float = 0.2,
        test_ratio: float = 0.1,
    ) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
        """
        Split samples into train, validation, and test sets.

        Args:
            samples: All loaded dataset samples.
            fields: Field specs for the dataset.
            validation_ratio: Fraction to allocate to validation.
            test_ratio: Fraction to allocate to test.

        Returns:
            Tuple of (train_split, val_split, test_split).
        """
        if len(samples) == 0:
            raise ValueError("Cannot split an empty dataset")

        # Reproducible shuffle
        rng = random.Random(self._seed)
        shuffled = samples.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * validation_ratio))
        n_train = n - n_val - n_test

        if n_train <= 0:
            logger.warning(
                f"Dataset too small ({n} samples) for requested split ratios. "
                "Using 60/20/20 split instead."
            )
            n_test = max(1, n // 5)
            n_val = max(1, n // 5)
            n_train = n - n_val - n_test

        train_samples = shuffled[:n_train]
        val_samples = shuffled[n_train : n_train + n_val]
        test_samples = shuffled[n_train + n_val :]

        logger.info(
            f"Dataset split: train={len(train_samples)}, "
            f"val={len(val_samples)}, test={len(test_samples)}"
        )

        return (
            DatasetSplit(name="train", samples=train_samples, fields=fields),
            DatasetSplit(name="validation", samples=val_samples, fields=fields),
            DatasetSplit(name="test", samples=test_samples, fields=fields),
        )

    def subsample(
        self,
        split: DatasetSplit,
        n: int,
        seed: Optional[int] = None,
    ) -> DatasetSplit:
        """
        Return a sub-sampled version of a split for fast evaluation.

        Args:
            split: The source split to subsample from.
            n: Number of samples to include.
            seed: Random seed (defaults to instance seed).

        Returns:
            New DatasetSplit with n samples.
        """
        rng = random.Random(seed or self._seed)
        samples = split.samples.copy()
        rng.shuffle(samples)
        selected = samples[:n]
        return DatasetSplit(name=split.name, samples=selected, fields=split.fields)
