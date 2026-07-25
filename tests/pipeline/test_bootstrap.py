"""Tests for the bootstrap stage.

Pins the contract of :func:`bootstrap`: confidence-weighted resampling into a
valid :class:`Distribution`, reproducibility, and the documented error cases.
"""

from __future__ import annotations

import pytest

from semantic_montecarlo.pipeline.bootstrap import bootstrap
from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.schemas.search import NumericAnswer


def _answer(value: float, confidence: float) -> NumericAnswer:
    return NumericAnswer(
        reasoning="r", value=value, confidence=confidence, sources=[]
    )


def test_returns_distribution() -> None:
    dist = bootstrap(
        [_answer(10.0, 1.0)], n_resamples=100, seed=0
    )
    assert isinstance(dist, Distribution)


def test_single_answer_concentrates_all_mass() -> None:
    dist = bootstrap(
        [_answer(42.0, 1.0)], n_resamples=100, seed=0
    )
    assert dist.data == [(42.0, 1.0)]


def test_two_equal_weights_split_fifty_fifty() -> None:
    dist = bootstrap(
        [_answer(10.0, 1.0), _answer(20.0, 1.0)],
        n_resamples=10000,
        seed=0,
    )
    as_dict = dict(dist.data)
    assert as_dict[10.0] == pytest.approx(0.5, abs=0.02)
    assert as_dict[20.0] == pytest.approx(0.5, abs=0.02)


def test_higher_confidence_drawn_more_often() -> None:
    # 10.0 has 9x the weight of 20.0; with many samples it dominates.
    dist = bootstrap(
        [_answer(10.0, 0.9), _answer(20.0, 0.1)],
        n_resamples=10000,
        seed=0,
    )
    as_dict = dict(dist.data)
    assert as_dict[10.0] > as_dict[20.0]
    assert 0.85 < as_dict[10.0] < 0.95


def test_probabilities_sum_to_one() -> None:
    # Distribution's validator already enforces this; confirms bootstrap passes.
    dist = bootstrap(
        [_answer(1.0, 0.5), _answer(2.0, 0.5), _answer(3.0, 0.5)],
        n_resamples=1000,
        seed=1,
    )
    assert sum(p for _, p in dist.data) == pytest.approx(1.0)


def test_reproducible_with_seed() -> None:
    samples = [_answer(10.0, 1.0), _answer(20.0, 1.0)]
    a = bootstrap(samples, n_resamples=500, seed=42)
    b = bootstrap(samples, n_resamples=500, seed=42)
    assert a.data == b.data


def test_empty_samples_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap([], n_resamples=100)


def test_zero_resamples_raise() -> None:
    with pytest.raises(ValueError, match="positive"):
        bootstrap([_answer(1.0, 1.0)], n_resamples=0)


def test_all_zero_confidence_raises() -> None:
    with pytest.raises(ValueError, match="positive confidence"):
        bootstrap(
            [_answer(1.0, 0.0), _answer(2.0, 0.0)], n_resamples=100
        )


def test_confidence_above_one_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        bootstrap([_answer(1.0, 1.5)], n_resamples=100)


def test_non_finite_value_raises() -> None:
    bad = NumericAnswer(reasoning="r", value=float("nan"), confidence=1.0, sources=[])
    with pytest.raises(ValueError, match="non-finite value"):
        bootstrap([bad], n_resamples=100)
