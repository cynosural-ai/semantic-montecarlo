"""Tests for the normalized variance complement."""

from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.stats import norm_var_comp


def test_no_numeric_answer_has_zero_confidence() -> None:
    distribution = Distribution(data=[], no_answer_probability=1.0)

    assert norm_var_comp(distribution) == 0.0


def test_answerability_discounts_numeric_concentration() -> None:
    distribution = Distribution(
        data=[(42.0, 1.0)],
        no_answer_probability=0.25,
    )

    assert norm_var_comp(distribution) == 0.75
