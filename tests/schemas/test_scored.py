"""Tests for the scored-answer schema contract."""

from __future__ import annotations

import pytest

from semantic_montecarlo.schemas.scored import ScoredAnswer
from semantic_montecarlo.schemas.search import SearchAnswer


def _answer() -> SearchAnswer:
    return SearchAnswer(reasoning="r", number=42.0)


def test_scored_holds_answer_and_confidence() -> None:
    s = ScoredAnswer(answer=_answer(), confidence=0.8)
    assert s.answer.number == 42.0
    assert s.confidence == 0.8


@pytest.mark.parametrize("bad", [-0.01, 1.01, -1.0, 2.0])
def test_confidence_must_be_in_unit_interval(bad: float) -> None:
    with pytest.raises(ValueError):
        ScoredAnswer(answer=_answer(), confidence=bad)


@pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
def test_confidence_accepts_endpoints(ok: float) -> None:
    # Bounds are inclusive: 0.0 and 1.0 are valid weights.
    assert ScoredAnswer(answer=_answer(), confidence=ok).confidence == ok
