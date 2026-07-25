"""Research paraphrases and convert the results into numeric answers."""

from __future__ import annotations

from semantic_montecarlo.llm import LLMClient, StructuredOutputError
from semantic_montecarlo.prompts import load
from semantic_montecarlo.schemas.search import NumericAnswer, NumericEstimate

_SEARCH_PROMPT = load("search")


def search(
    paraphrases: list[str],
    *,
    client: LLMClient,
    unit: str | None = None,
) -> list[NumericAnswer]:
    """
    Research and parse each paraphrase into a numeric answer.

    Input order is preserved.
    """
    answers: list[NumericAnswer] = []
    unit_requirement = _unit_requirement(unit)

    for paraphrase in paraphrases:
        research, sources = client.complete(
            _SEARCH_PROMPT.render(
                "research",
                paraphrase=paraphrase,
                unit_requirement=unit_requirement,
            ),
            web_search="auto",
        )

        try:
            estimate = client.complete_structured(
                _SEARCH_PROMPT.render(
                    "parse",
                    paraphrase=paraphrase,
                    research=research,
                    unit_requirement=unit_requirement,
                ),
                NumericEstimate,
            )
        except StructuredOutputError:
            continue
        answers.append(NumericAnswer(**estimate.model_dump(), sources=sources))

    if not answers:
        raise StructuredOutputError("Search produced no valid numeric answers.")

    return answers


def _unit_requirement(unit: str | None) -> str:
    if unit is None:
        return "Use one explicit, consistent unit for the final numeric answer."
    return (
        f"Return the final numeric answer in exactly {unit}. "
        "Convert values reported in other scales or units before returning the result."
    )
