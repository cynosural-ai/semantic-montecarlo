"""Output schema for the paraphrase stage."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ParaphraseOutput(BaseModel):
    """
    Structured output of the paraphrase model call.

    The model emits only the generated paraphrases; the stage prepends the
    verbatim original question afterwards (see :func:`paraphrase`), so this
    schema intentionally does not include the original.

    Attributes:
        paraphrases: Generated phrasings, never empty.
    """

    paraphrases: list[str] = Field(
        description="Diverse phrasings of the question, each a complete sentence."
    )
