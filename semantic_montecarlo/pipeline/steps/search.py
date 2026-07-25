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
) -> list[NumericAnswer]:
    """
    Research and parse each paraphrase into a numeric answer.

    Input order is preserved.
    """
    answers: list[NumericAnswer] = []

    for paraphrase in paraphrases:
        research, sources = client.complete(
            _SEARCH_PROMPT.render(
                "research",
                paraphrase=paraphrase,
            ),
            web_search="auto",
        )

        try:
            estimate = client.complete_structured(
                _SEARCH_PROMPT.render(
                    "parse",
                    paraphrase=paraphrase,
                    research=research,
                ),
                NumericEstimate,
            )
        except StructuredOutputError:
            continue
        answers.append(NumericAnswer(**estimate.model_dump(), sources=sources))

    if not answers:
        raise StructuredOutputError("Search produced no valid numeric answers.")

    return answers
