"""Hierarchy checks for the LLM error types."""

import pytest

from semantic_montecarlo.llm.errors import LLMError, StructuredOutputError


def test_llm_error_is_runtime_error() -> None:
    assert issubclass(LLMError, RuntimeError)


def test_structured_output_error_is_llm_error() -> None:
    # Callers catching LLMError must also catch parse failures.
    assert issubclass(StructuredOutputError, LLMError)


def test_structured_output_error_preserves_cause() -> None:
    original = ValueError("bad json")
    with pytest.raises(StructuredOutputError) as exc_info:
        raise StructuredOutputError("wrap") from original
    assert exc_info.value.__cause__ is original
