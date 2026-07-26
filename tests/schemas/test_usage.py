"""Tests for the Usage dataclass, especially aggregation."""

from __future__ import annotations

from semantic_montecarlo.schemas.usage import Usage


def test_default_is_all_zeros() -> None:
    u = Usage()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.total_tokens == 0
    assert u.reasoning_tokens == 0
    assert u.cached_tokens == 0


def test_add_sums_each_field() -> None:
    a = Usage(prompt_tokens=10, completion_tokens=5, reasoning_tokens=2)
    b = Usage(prompt_tokens=3, completion_tokens=7, cached_tokens=1)
    total = a + b
    assert total.prompt_tokens == 13
    assert total.completion_tokens == 12
    assert total.total_tokens == 0  # both were zero
    assert total.reasoning_tokens == 2
    assert total.cached_tokens == 1


def test_zero_is_identity() -> None:
    u = Usage(prompt_tokens=10, completion_tokens=5)
    assert (u + Usage()) == u
    assert (Usage() + u) == u


def test_sum_over_iterable() -> None:
    usages = [
        Usage(prompt_tokens=1, completion_tokens=1),
        Usage(prompt_tokens=2, completion_tokens=2),
        Usage(prompt_tokens=3, completion_tokens=3),
    ]
    total = sum(usages, start=Usage())
    assert total.prompt_tokens == 6
    assert total.completion_tokens == 6


def test_is_frozen() -> None:
    import dataclasses

    assert Usage.__dataclass_params__.frozen is True
