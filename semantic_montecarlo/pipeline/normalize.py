"""
The normalize stage: collapse bootstrap samples into a :class:`Distribution`.

SCAFFOLD STUB. Converts the raw sample list (the bootstrap's output) into the
probability mass function that :class:`Distribution` enforces: one
``(value, probability)`` pair per distinct value, probabilities summing to 1.

Kept separate from the bootstrap so the bootstrap stays a pure sampling function
and this owns only the shape conversion — but the exact mechanism (count-and-
divide, kernel density, etc.) is TBD alongside the bootstrap decision.
"""

from __future__ import annotations

from semantic_montecarlo.schemas.models import Distribution


def normalize(samples: list[float]) -> Distribution:
    """
    Convert bootstrap samples into a normalized :class:`Distribution`.

    Args:
        samples: Raw bootstrap samples.

    Returns:
        A :class:`Distribution` PMF over the sampled values.

    Raises:
        NotImplementedError: pending the bootstrap design decision.
    """
    raise NotImplementedError

