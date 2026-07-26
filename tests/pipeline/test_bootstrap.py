"""
Tests for the bootstrap stage.

Pins the contract of :func:`bootstrap`: confidence-weighted resampling into a
valid :class:`Distribution`, reproducibility, and the documented error cases.
"""

from __future__ import annotations

import pytest

from semantic_montecarlo.pipeline import bootstrap
from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.schemas.search import NumericAnswer


def _answer(value: float | None, confidence: float) -> NumericAnswer:
    return NumericAnswer(reasoning="r", value=value, confidence=confidence, sources=[])


def test_returns_distribution() -> None:
    empirical, _ = bootstrap([_answer(10.0, 1.0)], n_resamples=100, seed=0)
    assert isinstance(empirical, Distribution)


def test_single_answer_concentrates_all_mass() -> None:
    empirical, _ = bootstrap([_answer(42.0, 1.0)], n_resamples=100, seed=0)
    assert empirical.data == [(42.0, 1.0)]
    assert empirical.no_answer_probability == 0.0


def test_missing_answers_are_sampled_separately() -> None:
    # No-answer probability is the *confidence-weighted* mass of None answers,
    # not their count. Two None answers with total confidence 1.0 vs. two
    # numeric answers with total confidence 1.0 -> no_answer_probability ~0.5.
    empirical, _ = bootstrap(
        [
            _answer(None, 0.5),
            _answer(None, 0.5),
            _answer(10.0, 0.9),
            _answer(20.0, 0.1),
        ],
        n_resamples=10_000,
        seed=0,
    )

    assert empirical.no_answer_probability == pytest.approx(0.5, abs=0.02)
    # Numeric probabilities are renormalized over numeric mass only.
    assert dict(empirical.data)[10.0] == pytest.approx(0.9, abs=0.02)
    assert dict(empirical.data)[20.0] == pytest.approx(0.1, abs=0.02)
    assert sum(probability for _, probability in empirical.data) == pytest.approx(1.0)


def test_all_missing_answers_return_no_numeric_distribution() -> None:
    # Every answer is None with positive confidence -> empty numeric
    # distribution, all mass on no-answer.
    empirical, _ = bootstrap(
        [_answer(None, 0.5), _answer(None, 0.5)],
        n_resamples=100,
        seed=0,
    )

    assert empirical.data == []
    assert empirical.no_answer_probability == 1.0


def test_two_equal_weights_split_fifty_fifty() -> None:
    empirical, _ = bootstrap(
        [_answer(10.0, 1.0), _answer(20.0, 1.0)],
        n_resamples=10000,
        seed=0,
    )
    as_dict = dict(empirical.data)
    assert as_dict[10.0] == pytest.approx(0.5, abs=0.02)
    assert as_dict[20.0] == pytest.approx(0.5, abs=0.02)


def test_higher_confidence_drawn_more_often() -> None:
    # 10.0 has 9x the weight of 20.0; with many samples it dominates.
    empirical, _ = bootstrap(
        [_answer(10.0, 0.9), _answer(20.0, 0.1)],
        n_resamples=10000,
        seed=0,
    )
    as_dict = dict(empirical.data)
    assert as_dict[10.0] > as_dict[20.0]
    assert 0.85 < as_dict[10.0] < 0.95


def test_probabilities_sum_to_one() -> None:
    # Distribution's validator already enforces this; confirms bootstrap passes.
    empirical, _ = bootstrap(
        [_answer(1.0, 0.5), _answer(2.0, 0.5), _answer(3.0, 0.5)],
        n_resamples=1000,
        seed=1,
    )
    assert sum(p for _, p in empirical.data) == pytest.approx(1.0)


def test_reproducible_with_seed() -> None:
    samples = [_answer(None, 0.0), _answer(10.0, 1.0), _answer(20.0, 1.0)]
    a = bootstrap(samples, n_resamples=500, seed=42)
    b = bootstrap(samples, n_resamples=500, seed=42)
    assert a == b
