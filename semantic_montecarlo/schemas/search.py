"""
Output of a single web search.

A search agent turns one phrasing of the question into a structured answer: the
model's reasoning, the numeric value it found (if any), and the sources it
cited. The numeric value is ``None`` when the model found no number — that case
is preserved through the pipeline rather than silently dropped.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Source(BaseModel):
    """
    A citation returned by a search.

    Attributes:
        url: Source URL.
        title: Page or document title, if known.
    """

    url: str
    title: str | None = None


class SearchAnswer(BaseModel):
    """
    Structured answer to one phrasing of the question.

    Attributes:
        reasoning: The model's reasoning about which number to report and why.
        number: The numeric value found, or ``None`` if none was found.
        sources: Citations backing the answer.
    """

    reasoning: str
    number: float | None = None
    sources: list[Source] = Field(default_factory=list)
