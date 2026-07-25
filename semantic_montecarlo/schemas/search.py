"""Schemas produced by the search stage."""

from pydantic import BaseModel, Field


class NumericEstimate(BaseModel):
    """Numeric fields generated from researched evidence."""

    reasoning: str
    value: float | None = Field(
        allow_inf_nan=False,
        description=(
            "Source-grounded numeric answer, or null when the research cannot "
            "support a defensible estimate."
        ),
    )
    confidence: float = Field(ge=0.0, le=1.0)


class NumericAnswer(NumericEstimate):
    """Numeric estimate and verified sources for one paraphrase."""

    sources: list[str]
