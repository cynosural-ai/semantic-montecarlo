"""Tests for the search output schema contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from semantic_montecarlo.schemas.search import NumericAnswer


def test_answer_holds_value_and_confidence() -> None:
    a = NumericAnswer(
        reasoning="r",
        value=67.0,
        confidence=0.8,
        sources=["https://example.com"],
    )
    assert a.value == 67.0
    assert a.confidence == 0.8
    assert a.sources == ["https://example.com"]

    with pytest.raises(ValidationError):
        NumericAnswer(
            reasoning="r",
            value=float("nan"),
            confidence=0.8,
            sources=[],
        )


def test_answer_allows_no_numeric_value() -> None:
    answer = NumericAnswer(
        reasoning="The research did not contain enough information.",
        value=None,
        confidence=0.0,
        sources=[],
    )

    assert answer.value is None
