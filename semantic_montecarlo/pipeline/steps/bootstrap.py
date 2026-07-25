"""The bootstrap stage: aggregate numeric answers into a value distribution.

Confidence-weighted resampling. Each :class:`NumericAnswer`'s ``confidence``
becomes a sampling probability; answers are drawn with replacement and their
observed frequencies form the returned :class:`Distribution` PMF.

This is the final pipeline stage, taking ``search``'s output directly. There is
no separate scoring step — confidence already lives on each answer.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Sequence

from semantic_montecarlo.observability import get_logger
from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.schemas.search import NumericAnswer

_logger = get_logger(__name__)


def bootstrap(
    samples: Sequence[NumericAnswer],
    *,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> Distribution:
    """
    Build a confidence-weighted empirical distribution of numeric answers.

    Confidence scores are normalized into sampling probabilities. Numeric
    answers are then sampled with replacement, and their observed frequencies
    form the returned probability distribution.

    Args:
        samples: Numeric answers exposing ``value`` and ``confidence`` fields.
        n_resamples: Number of bootstrap resamples. Must be positive.
        seed: Optional seed for reproducible resampling.

    Returns:
        Each distinct numeric answer and its normalized sampling frequency.

    Raises:
        ValueError: If the configuration or any observation is invalid.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    values, weights = _validated_observations(samples)
    total_weight = math.fsum(weights)
    if total_weight == 0.0:
        raise ValueError("at least one sample must have positive confidence")

    probabilities = tuple(weight / total_weight for weight in weights)
    rng = random.Random(seed)
    estimate_counts = Counter(
        rng.choices(
            values,
            weights=probabilities,
            k=n_resamples,
        )
    )

    data = [
        (estimate, count / n_resamples)
        for estimate, count in sorted(estimate_counts.items())
    ]
    _logger.debug(
        "bootstrap: %d samples -> %d distinct values", len(samples), len(data)
    )
    return Distribution(data=data)


def _validated_observations(
    samples: Sequence[NumericAnswer],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not samples:
        raise ValueError("samples must not be empty")

    values: list[float] = []
    weights: list[float] = []
    for index, sample in enumerate(samples):
        value = sample.value
        confidence = sample.confidence

        if not math.isfinite(value):
            raise ValueError(f"sample {index} has a non-finite value")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"sample {index} confidence must be a finite number between 0 and 1"
            )

        values.append(value)
        weights.append(confidence)

    return tuple(values), tuple(weights)
