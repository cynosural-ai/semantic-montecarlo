"""
The bootstrap stage: aggregate numeric answers into a value distribution.

Missing answers are resampled as a categorical outcome according to their
observed frequency. The remaining probability mass is distributed among
numeric answers according to their confidence.

This is the final pipeline stage, taking ``search``'s output directly. There is
no separate scoring step.
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
    Build an empirical answerability and numeric value distribution.

    ``None`` is sampled according to its observed frequency. Conditional on
    receiving a numeric answer, confidence scores determine the sampling
    probabilities. Numeric frequencies are normalized over numeric draws only.

    Args:
        samples: Numeric answers exposing ``value`` and ``confidence`` fields.
        n_resamples: Number of bootstrap resamples. Must be positive.
        seed: Optional seed for reproducible resampling.

    Returns:
        The conditional numeric distribution and sampled no-answer probability.

    Raises:
        ValueError: If the configuration or any observation is invalid.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    values, weights, n_missing = _validated_observations(samples)
    if not values:
        return Distribution(data=[], no_answer_probability=1.0)

    total_weight = math.fsum(weights)
    if total_weight == 0.0:
        raise ValueError("at least one numeric sample must have positive confidence")

    no_answer_probability = n_missing / len(samples)
    numeric_probability = 1.0 - no_answer_probability
    outcomes: tuple[float | None, ...] = (None, *values)
    probabilities = (
        no_answer_probability,
        *(numeric_probability * weight / total_weight for weight in weights),
    )
    rng = random.Random(seed)
    estimate_counts = Counter(
        rng.choices(
            outcomes,
            weights=probabilities,
            k=n_resamples,
        )
    )

    no_answer_count = estimate_counts.pop(None, 0)
    numeric_count = n_resamples - no_answer_count
    numeric_counts = (
        (estimate, count)
        for estimate, count in estimate_counts.items()
        if estimate is not None
    )
    data = (
        [
            (estimate, count / numeric_count)
            for estimate, count in sorted(numeric_counts)
        ]
        if numeric_count
        else []
    )
    _logger.debug(
        "bootstrap: %d samples -> %d distinct values, %.3f no-answer probability",
        len(samples),
        len(data),
        no_answer_count / n_resamples,
    )
    return Distribution(
        data=data,
        no_answer_probability=no_answer_count / n_resamples,
    )


def _validated_observations(
    samples: Sequence[NumericAnswer],
) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    if not samples:
        raise ValueError("samples must not be empty")

    values: list[float] = []
    weights: list[float] = []
    n_missing = 0
    for index, sample in enumerate(samples):
        value = sample.value
        confidence = sample.confidence

        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"sample {index} confidence must be a finite number between 0 and 1"
            )
        if value is None:
            n_missing += 1
            continue
        if not math.isfinite(value):
            raise ValueError(f"sample {index} has a non-finite value")

        values.append(value)
        weights.append(confidence)

    return tuple(values), tuple(weights), n_missing
