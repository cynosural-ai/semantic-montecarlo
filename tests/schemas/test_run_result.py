"""Tests for the RunResult data shape."""

from __future__ import annotations

from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.schemas.run_result import RunResult
from semantic_montecarlo.schemas.search import NumericAnswer
from semantic_montecarlo.schemas.usage import Usage


def _answer(value: float) -> NumericAnswer:
    return NumericAnswer(
        reasoning="r", value=value, confidence=0.5, sources=["https://example.com"]
    )


def _result(**overrides: object) -> RunResult:
    defaults: dict[str, object] = {
        "question": "How many?",
        "unit": "people",
        "paraphrases": ["How many?"],
        "answers": [_answer(67.0)],
        "distribution": Distribution(data=[(67.0, 1.0)]),
        "elapsed_seconds": 1.5,
        "model": "openrouter/free",
        "paraphrase_usage": Usage(prompt_tokens=10, total_tokens=15),
        "search_usage": Usage(prompt_tokens=40, total_tokens=60),
    }
    defaults.update(overrides)
    return RunResult(**defaults)  # type: ignore[arg-type]


def test_holds_all_fields() -> None:
    r = _result()
    assert r.question == "How many?"
    assert r.unit == "people"
    assert r.paraphrases == ["How many?"]
    assert r.answers[0].value == 67.0
    assert r.distribution.data == [(67.0, 1.0)]
    assert r.elapsed_seconds == 1.5
    assert r.model == "openrouter/free"
    assert r.paraphrase_usage.total_tokens == 15
    assert r.search_usage.total_tokens == 60


def test_per_stage_usage_sums_to_run_total() -> None:
    r = _result()
    assert (r.paraphrase_usage + r.search_usage).total_tokens == 75


def test_unit_may_be_none() -> None:
    assert _result(unit=None).unit is None


def test_is_frozen() -> None:
    # RunResult is declared frozen=True; mutation must raise at runtime.
    import dataclasses

    assert dataclasses.is_dataclass(RunResult)
    # The __dataclass_params__ attribute carries the frozen flag set at class
    # creation — this is the declaration we depend on for immutability.
    assert RunResult.__dataclass_params__.frozen is True
