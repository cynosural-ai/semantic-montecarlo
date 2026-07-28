"""Tests for static distribution visualization."""

from pathlib import Path

from semantic_montecarlo.plotting import plot_result, save_result_plot
from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.schemas.run_result import RunResult
from semantic_montecarlo.schemas.usage import Usage


def _result(
    *,
    empirical: Distribution,
    bootstrap_mean: Distribution,
) -> RunResult:
    return RunResult(
        question="What was NVIDIA's total revenue for fiscal 2026?",
        unit="USD billions",
        paraphrases=[],
        answers=[],
        distribution=empirical,
        bootstrap_mean=bootstrap_mean,
        elapsed_seconds=1.0,
        model="test-model",
        paraphrase_usage=Usage(),
        search_usage=Usage(),
    )


def test_saves_weighted_kde_plot(tmp_path: Path) -> None:
    result = _result(
        empirical=Distribution(data=[(215.8, 0.2), (215.9, 0.8)]),
        bootstrap_mean=Distribution(data=[(215.8, 0.1), (215.9, 0.7), (216.0, 0.2)]),
    )
    path = tmp_path / "distribution.png"

    save_result_plot(result, path)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_point_mass_uses_single_distribution_axes() -> None:
    result = _result(
        empirical=Distribution(data=[(42.0, 1.0)]),
        bootstrap_mean=Distribution(data=[(42.0, 1.0)]),
    )

    figure = plot_result(result)

    assert len(figure.axes) == 1


def test_no_answer_adds_indicator_and_handles_empty_numeric_data() -> None:
    no_answer = Distribution(data=[], no_answer_probability=1.0)

    figure = plot_result(_result(empirical=no_answer, bootstrap_mean=no_answer))

    assert len(figure.axes) == 2
    assert figure.axes[0].texts[0].get_text() == "No answer  100.0%"
    assert figure.axes[1].texts[0].get_text() == "No numeric estimate available"
