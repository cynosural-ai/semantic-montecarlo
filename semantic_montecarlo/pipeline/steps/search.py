"""Research paraphrases and convert the results into numeric answers."""

from __future__ import annotations

from semantic_montecarlo.llm import LLMClient, StructuredOutputError
from semantic_montecarlo.observability.logging.setup import get_logger
from semantic_montecarlo.prompts import load
from semantic_montecarlo.schemas.search import NumericAnswer, NumericEstimate
from semantic_montecarlo.schemas.usage import Usage

_SEARCH_PROMPT = load("search")

_logger = get_logger(__name__)

def search(
    paraphrases: list[str],
    *,
    client: LLMClient,
    unit: str | None = None,
) -> tuple[list[NumericAnswer], Usage]:
    """
    Research and parse each paraphrase into a numeric answer.

    Input order is preserved.

    Returns:
        A ``(answers, usage)`` tuple. ``usage`` is the sum of token usage across
        every research + parse call in the stage.
    """
    answers: list[NumericAnswer] = []
    usage = Usage()
    unit_requirement = _unit_requirement(unit)

    for paraphrase in paraphrases:
        _logger.debug("searching: %s", paraphrase)
        research = client.complete(
            _SEARCH_PROMPT.render(
                "research",
                paraphrase=paraphrase,
                unit_requirement=unit_requirement,
            ),
            web_search="auto",
        )
        usage = usage + research.usage

        try:
            parsed = client.complete_structured(
                _SEARCH_PROMPT.render(
                    "parse",
                    paraphrase=paraphrase,
                    research=research.text,
                    unit_requirement=unit_requirement,
                ),
                NumericEstimate,
            )
            _logger.debug("search results: %s", parsed.data)
        except StructuredOutputError:
            continue
        usage = usage + parsed.usage
        answers.append(
            NumericAnswer(**parsed.data.model_dump(), sources=research.sources)
        )

    if not answers:
        raise StructuredOutputError("Search produced no valid numeric answers.")

    return answers, usage


def _unit_requirement(unit: str | None) -> str:
    if unit is None:
        return "Use one explicit, consistent unit for the final numeric answer."
    return (
        f"Return the final numeric answer in exactly {unit}. "
        "Convert values reported in other scales or units before returning the result."
    )
