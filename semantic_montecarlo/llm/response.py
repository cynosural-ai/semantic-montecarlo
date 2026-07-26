"""Return types for the LLM client methods.

These wrap the model's output together with token :class:`~semantic_montecarlo.schemas.usage.Usage`,
so callers get the response and its cost in one value rather than via a side
channel. They are client return types (specific to the LLM call boundary), so
they live in ``llm/`` rather than ``schemas/`` — unlike :class:`Usage` itself,
which is shared vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from semantic_montecarlo.schemas.usage import Usage

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Completion:
    """Return value of :meth:`LLMClient.complete`.

    Attributes:
        text: The model's free-form text response.
        sources: Citation URLs scraped from OpenRouter ``url_citation``
            annotations. Empty when the call did not use web search.
        usage: Token usage for the call (placeholder zeros in Stage 1).
    """

    text: str
    sources: list[str]
    usage: Usage


@dataclass(frozen=True)
class StructuredCompletion(Generic[T]):
    """Return value of :meth:`LLMClient.complete_structured`.

    Attributes:
        data: The validated pydantic instance.
        usage: Token usage for the call (placeholder zeros in Stage 1).
    """

    data: T
    usage: Usage
