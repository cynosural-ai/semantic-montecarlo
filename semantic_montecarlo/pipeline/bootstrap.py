"""
The bootstrap stage: resample scored answers into a value distribution.

SCAFFOLD STUB. The bootstrap is a load-bearing statistical design decision
(resampling vs. precision-weighting, the IID assumption, correlated-source
discounts) owned on the ``feat/weighted-bootstrap`` branch. This stub exists
only so the pipeline's final stage has a named, typed seam.

The intended contract is ``(scored answers, sample count) -> raw samples``,
which :func:`normalize` then collapses into a :class:`Distribution`.
"""

from __future__ import annotations

import numpy as np

from semantic_montecarlo.schemas.scored import ScoredAnswer

# Default sample count placeholder; the real value is owned by the bootstrap.
DEFAULT_N_SAMPLES = 1000


def bootstrap(
    scored: list[ScoredAnswer],
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """
    Resample scored answer values into a list of bootstrap samples.

    Args:
        scored: Scored answers (confidence-weighted).
        n_samples: Number of bootstrap samples to draw.
        rng: Optional NumPy generator for reproducibility.

    Returns:
        A list of float sample values.

    Raises:
        NotImplementedError: This stage is owned on ``feat/weighted-bootstrap``.
    """
    raise NotImplementedError

