"""
Tests for the bootstrap stage.

Pins the contract of :func:`bootstrap}: it returns two distributions — the
empirical (confidence-weighted) distribution and the bootstrap resample-mean
distribution. Both must be valid PMFs and reproducible; the resample-mean
additionally excludes None answers from each resample's mean.
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
    # No-answer probability comes from observed frequency. Confidence only
    # distributes the remaining mass among numeric answers.
    empirical, _ = bootstrap(
        [
            _answer(None, 0.0),
            _answer(None, 0.0),
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
    empirical, bootstrap_mean = bootstrap(
        [_answer(None, 0.0), _answer(None, 0.0)],
        n_resamples=100,
        seed=0,
    )

    assert empirical.data == []
    assert empirical.no_answer_probability == 1.0
    assert bootstrap_mean.data == []
    assert bootstrap_mean.no_answer_probability == 1.0


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


def test_zero_numeric_confidences_use_equal_probabilities() -> None:
    empirical, _ = bootstrap(
        [_answer(10.0, 0.0), _answer(20.0, 0.0)],
        n_resamples=100,
        seed=0,
    )

    assert dict(empirical.data) == {10.0: 0.5, 20.0: 0.5}


def test_reproducible_with_seed() -> None:
    samples = [_answer(None, 0.0), _answer(10.0, 1.0), _answer(20.0, 1.0)]
    a = bootstrap(samples, n_resamples=500, seed=42)
    b = bootstrap(samples, n_resamples=500, seed=42)
    assert a == b


def test_empty_samples_raise() -> None:
    with pytest.raises(ValueError, match="samples must not be empty"):
        bootstrap([])


def test_non_positive_resamples_raise() -> None:
    with pytest.raises(ValueError, match="n_resamples must be positive"):
        bootstrap([_answer(10.0, 1.0)], n_resamples=0)


# --- Bootstrap resample-mean distribution (the second returned element) ---
# Every test above discards it via `empirical, _ = ...`; these pin its behavior,
# since it is the distribution the benchmark scores against (RunResult.bootstrap_mean).


def test_bootstrap_mean_single_value_concentrates_mass() -> None:
    # Every resample draws only 42.0 -> every resample mean is 42.0.
    _, bootstrap_mean = bootstrap([_answer(42.0, 1.0)], n_resamples=100, seed=0)
    assert bootstrap_mean.data == [(42.0, 1.0)]
    assert bootstrap_mean.no_answer_probability == 0.0


def test_bootstrap_mean_centers_on_midpoint_for_equal_weights() -> None:
    # Two values, equal confidence: resample means cluster around 15 (midpoint).
    _, bootstrap_mean = bootstrap(
        [_answer(10.0, 1.0), _answer(20.0, 1.0)],
        n_resamples=10_000,
        seed=0,
    )
    values = [v for v, _ in bootstrap_mean.data]
    probs = [p for _, p in bootstrap_mean.data]
    mean_of_means = sum(v * p for v, p in zip(values, probs, strict=True))
    assert mean_of_means == pytest.approx(15.0, abs=0.5)


def test_bootstrap_mean_shifts_toward_higher_confidence_value() -> None:
    # 10.0 has 9x the weight of 20.0 -> mean of means is pulled toward 10.
    _, bootstrap_mean = bootstrap(
        [_answer(10.0, 0.9), _answer(20.0, 0.1)],
        n_resamples=10_000,
        seed=0,
    )
    values = [v for v, _ in bootstrap_mean.data]
    probs = [p for _, p in bootstrap_mean.data]
    mean_of_means = sum(v * p for v, p in zip(values, probs, strict=True))
    # Unweighted across all answers: 11.0; weighted toward 10 -> below 11.
    assert mean_of_means < 11.0
    assert 10.5 < mean_of_means < 11.0


def test_bootstrap_mean_is_reproducible_with_seed() -> None:
    samples = [_answer(10.0, 0.5), _answer(20.0, 0.5)]
    _, a = bootstrap(samples, n_resamples=500, seed=42)
    _, b = bootstrap(samples, n_resamples=500, seed=42)
    assert a.data == b.data


def test_bootstrap_mean_excludes_all_none_resamples() -> None:
    # When a resample draws only None answers, it cannot produce a numeric mean
    # and is excluded from the mean distribution (counted as no-answer instead).
    _, bootstrap_mean = bootstrap(
        [
            _answer(None, 0.0),
            _answer(None, 0.0),
            _answer(10.0, 0.1),
            _answer(20.0, 0.1),
        ],
        n_resamples=5_000,
        seed=0,
    )
    # Some resamples drew only Nones -> positive no-answer probability.
    assert bootstrap_mean.no_answer_probability == pytest.approx(0.0625, abs=0.02)
    # Conditional numeric means are still valid: probabilities sum to 1.
    assert sum(p for _, p in bootstrap_mean.data) == pytest.approx(1.0)


def test_bootstrap_mean_combines_float_artifacts() -> None:
    _, bootstrap_mean = bootstrap(
        [_answer(0.1, 1.0), _answer(0.2, 1.0), _answer(0.3, 1.0)],
        n_resamples=10_000,
        seed=0,
    )

    close_to_point_two = [
        probability
        for value, probability in bootstrap_mean.data
        if value == pytest.approx(0.2)
    ]

    assert len(close_to_point_two) == 1
