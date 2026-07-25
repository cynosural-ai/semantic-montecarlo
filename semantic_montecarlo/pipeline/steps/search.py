from __future__ import annotations

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.prompts import load
from semantic_montecarlo.schemas.search import NumericAnswer

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
        research = client.complete(
            _SEARCH_PROMPT.render(
                "research",
                paraphrase=paraphrase,
            ),
            web_search="auto",
        )

        answer = client.complete_structured(
            _SEARCH_PROMPT.render(
                "parse",
                paraphrase=paraphrase,
                research=research,
            ),
            NumericAnswer,
        )
        answers.append(answer)

    return answers