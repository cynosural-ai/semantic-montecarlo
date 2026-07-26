"""Tests for the pipeline ``run`` composition and RunResult packaging.

Mocks ``LLMClient`` so the full ``paraphrase -> search -> bootstrap`` chain runs
end-to-end with no network. Asserts the ``RunResult`` contract: type, elapsed
timing, model capture, and a valid distribution.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.pipeline import run
from semantic_montecarlo.schemas.paraphrase import ParaphraseOutput
from semantic_montecarlo.schemas.run_result import RunResult
from semantic_montecarlo.schemas.search import NumericEstimate


def _stub_client(model: str = "openrouter/free") -> MagicMock:
    """Return a mock client matching the search step's two-call contract.

    ``search`` calls ``complete(...)`` for research (returns ``(research,
    sources)``) and ``complete_structured(..., NumericEstimate)`` for parsing.
    Paraphrase goes through ``complete_structured(..., ParaphraseOutput)``.
    """
    client = MagicMock(spec=LLMClient)
    client.default_model = model

    # complete() returns (research_text, sources_list).
    client.complete.return_value = ("some research text", ["https://example.com"])

    def fake_structured(prompt, schema, **kwargs):  # noqa: ANN001
        if schema is ParaphraseOutput:
            return ParaphraseOutput(paraphrases=["alt phrasing?"])
        if schema is NumericEstimate:
            return NumericEstimate(reasoning="r", value=67.0, confidence=1.0)
        raise AssertionError(f"unexpected schema {schema}")

    client.complete_structured.side_effect = fake_structured
    return client


def test_run_returns_run_result() -> None:
    result = run("How many?", client=_stub_client(), n_resamples=50)
    assert isinstance(result, RunResult)


def test_run_records_model() -> None:
    client = _stub_client(model="anthropic/claude-3.5-sonnet")
    result = run("How many?", client=client, n_resamples=50)
    assert result.model == "anthropic/claude-3.5-sonnet"


def test_run_elapsed_is_non_negative() -> None:
    result = run("How many?", client=_stub_client(), n_resamples=50)
    assert result.elapsed_seconds >= 0.0


def test_run_carries_paraphrases_and_answers() -> None:
    result = run("How many?", client=_stub_client(), n_paraphrases=1, n_resamples=50)
    # Original prepended + the one generated paraphrase.
    assert result.paraphrases[0] == "How many?"
    assert len(result.paraphrases) == 2
    assert len(result.answers) == 2
    assert result.answers[0].value == 67.0


def test_run_distribution_is_valid_pmf() -> None:
    result = run("How many?", client=_stub_client(), n_resamples=100)
    total = sum(p for _, p in result.distribution.data)
    assert abs(total - 1.0) < 1e-9


def test_run_carries_sources_on_answers() -> None:
    result = run("How many?", client=_stub_client(), n_paraphrases=1, n_resamples=50)
    assert result.answers[0].sources == ["https://example.com"]
