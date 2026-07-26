"""
Tests for the paraphrase stage.

Mocks at the ``LLMClient`` boundary: the stage calls
``complete_structured(prompt, ParaphraseOutput, temperature=...)`` and we assert
what it passes through and how it post-processes the result.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semantic_montecarlo.llm import LLMClient, StructuredCompletion
from semantic_montecarlo.pipeline import paraphrase
from semantic_montecarlo.schemas.paraphrase import ParaphraseOutput
from semantic_montecarlo.schemas.usage import Usage


def _client_returning(paraphrases: list[str]) -> MagicMock:
    """Return a mock ``LLMClient`` yielding canned paraphrases.

    The mock's ``complete_structured`` returns the given paraphrases wrapped in
    a :class:`StructuredCompletion` around :class:`ParaphraseOutput`.
    """
    client = MagicMock(spec=LLMClient)
    client.complete_structured.return_value = StructuredCompletion(
        data=ParaphraseOutput(paraphrases=paraphrases), usage=Usage()
    )
    return client


def test_prepends_original_first() -> None:
    client = _client_returning(["What is the population of France?"])
    result = paraphrase("How many people live in France?", client=client)
    assert result[0] == "How many people live in France?"


def test_returns_n_plus_one_before_dedup() -> None:
    client = _client_returning(["A?", "B?", "C?"])
    result = paraphrase("Q?", n=3, client=client)
    # original + 3 paraphrases.
    assert result == ["Q?", "A?", "B?", "C?"]


def test_passes_temperature_through() -> None:
    client = _client_returning(["A?"])
    paraphrase("Q?", client=client, temperature=0.9)
    kwargs = client.complete_structured.call_args.kwargs
    assert kwargs["temperature"] == 0.9


def test_default_temperature_is_0_7() -> None:
    client = _client_returning(["A?"])
    paraphrase("Q?", client=client)
    assert client.complete_structured.call_args.kwargs["temperature"] == 0.7


def test_dedup_removes_duplicate_of_original() -> None:
    # The model echoes the original with different casing/whitespace; the
    # stage must drop it rather than return it twice.
    client = _client_returning(["  HOW MANY people live in France?  "])
    result = paraphrase("How many people live in France?", client=client)
    assert result == ["How many people live in France?"]


def test_dedup_removes_duplicates_among_paraphrases() -> None:
    client = _client_returning(
        ["What is the population of France?", "what is the POPULATION of france?"]
    )
    result = paraphrase("How many people live in France?", client=client)
    # Only the first of the two duplicates survives.
    assert result == [
        "How many people live in France?",
        "What is the population of France?",
    ]


def test_preserves_first_seen_order() -> None:
    client = _client_returning(["B?", "A?", "B?", "A?", "C?"])
    result = paraphrase("Q?", client=client)
    assert result == ["Q?", "B?", "A?", "C?"]


def test_prompt_includes_question_and_n() -> None:
    client = _client_returning(["A?"])
    paraphrase("How many people live in France?", n=7, client=client)
    prompt = client.complete_structured.call_args.args[0]
    assert "How many people live in France?" in prompt
    assert "7" in prompt


def test_uses_paraphrase_output_schema() -> None:
    client = _client_returning(["A?"])
    paraphrase("Q?", client=client)
    schema = client.complete_structured.call_args.args[1]
    assert schema is ParaphraseOutput


def test_empty_model_output_yields_just_original() -> None:
    client = _client_returning([])
    result = paraphrase("Q?", client=client)
    assert result == ["Q?"]


@pytest.mark.parametrize("n", [1, 3, 8])
def test_various_n(n: int) -> None:
    client = _client_returning([f"P{i}?" for i in range(n)])
    result = paraphrase("Q?", n=n, client=client)
    assert len(result) == n + 1
