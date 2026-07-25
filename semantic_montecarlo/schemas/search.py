"""Schemas produced by the search stage."""

from pydantic import BaseModel, Field


class NumericAnswer(BaseModel):
    """Numeric answer and supporting context for one paraphrase."""

    reasoning: str
    value: float
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]
