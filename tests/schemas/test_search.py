"""Tests for the search output schema contracts."""

from __future__ import annotations

from semantic_montecarlo.schemas.search import SearchAnswer, Source


def test_answer_defaults_number_to_none() -> None:
    # An answer that found no number preserves None rather than 0.0.
    a = SearchAnswer(reasoning="r")
    assert a.number is None
    assert a.sources == []


def test_answer_with_value_and_sources() -> None:
    a = SearchAnswer(
        reasoning="r",
        number=67.0,
        sources=[Source(url="https://example.com", title="Example")],
    )
    assert a.number == 67.0
    assert a.sources[0].url == "https://example.com"


def test_source_title_optional() -> None:
    s = Source(url="https://example.com")
    assert s.title is None
