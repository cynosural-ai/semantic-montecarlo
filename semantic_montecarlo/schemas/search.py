"""Schemas produced by the search stage."""

from pydantic import BaseModel


class NumericAnswer(BaseModel):
    """Numeric answer and supporting context for one paraphrase."""

    reasoning: str
    value: float
    confidence: float
    sources: list[str]
