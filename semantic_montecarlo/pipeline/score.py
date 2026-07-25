"""
The scoring stage: assign each answer a confidence weight.

SCAFFOLD STUB. The contract is ``list[SearchAnswer] -> list[ScoredAnswer]`` in
order. The real confidence logic — domain allowlists, recency, primary-vs-
secondary sourcing, source-correlation discounts — replaces this function's body
without changing its signature. Confidence is a per-answer weight in ``[0, 1]``
that scales resampling; it is not a probability and need not sum to 1.
"""

from __future__ import annotations

from semantic_montecarlo.schemas.scored import ScoredAnswer
from semantic_montecarlo.schemas.search import SearchAnswer


def score(answers: list[SearchAnswer]) -> list[ScoredAnswer]:
    """
    Assign a confidence weight to each answer.

    Args:
        answers: Search answers to score.

    Returns:
        One :class:`ScoredAnswer` per input, in order.

    Raises:
        NotImplementedError: This stage's body is owned separately.
    """
    raise NotImplementedError

