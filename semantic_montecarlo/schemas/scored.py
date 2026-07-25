"""
A search answer paired with a confidence weight.

Produced by the scoring stage and consumed by the bootstrap. The confidence is
a per-answer weight in ``[0, 1]`` that scales how often the answer's value is
resampled; it is not a probability and need not sum to 1 across answers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from semantic_montecarlo.schemas.search import SearchAnswer


class ScoredAnswer(BaseModel):
    """
    A search answer with its confidence weight.

    Attributes:
        answer: The underlying search answer.
        confidence: Weight in ``[0, 1]`` used by the bootstrap.
    """

    answer: SearchAnswer
    confidence: float = Field(ge=0.0, le=1.0)
