"""
Typed exceptions for the LLM layer.

Errors are split by failure mode so callers can decide what to recover from.
``LLMError`` covers transport/provider problems (missing keys, HTTP failures);
``StructuredOutputError`` covers the model failing to produce a valid response
that conforms to a requested schema. Both subclass :class:`RuntimeError` to
match the rest of the codebase's error convention.
"""

from __future__ import annotations


class LLMError(RuntimeError):
    """
    Base error for the LLM layer.

    Raised for provider/transport-level failures: missing API keys, an unknown
    model, or any failure that is not a structured-output parse problem.
    """


class StructuredOutputError(LLMError):
    """
    The model returned a response that could not be parsed into the schema.

    Wraps the underlying validation error so callers can catch the parse failure
    distinctly from transport failures while still catching :class:`LLMError`.
    """
