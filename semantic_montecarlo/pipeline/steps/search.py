"""
The search stage: turn phrasings into numeric answers.

SCAFFOLD STUB. The contract below (signature + return type) is the stable seam
the pipeline depends on. The implementation — structured search via the LLM
client, the OpenRouter web-search plugin (``extra_body={"plugins": [{"id":
"web"}]}``), retries, and concurrency — is owned separately and lives in this
function's body when ready. See ADR 0001 for the structured-output decision.

The returned :class:`NumericAnswer` carries both the numeric ``value`` and the
``confidence`` weight that ``weighted_bootstrap`` consumes, so there is no
separate scoring step in the pipeline.
"""

from __future__ import annotations

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.schemas.search import NumericAnswer


def search(
    paraphrases: list[str],
    *,
    client: LLMClient,
) -> list[NumericAnswer]:
    """
    Answer each paraphrase through OpenRouter web search.

    Returns one structured numeric answer per input paraphrase, preserving
    input order.
    """
    answers: list[NumericAnswer] = []

    for paraphrase in paraphrases:
        prompt = f"""
Research the following question using web search:

{paraphrase}

Return:
- reasoning: a concise explanation of how the value was determined
- value: the single best numeric answer
- confidence: a number from 0.0 to 1.0
- sources: URLs for the sources used

Use current, reliable sources. Do not include units or symbols in `value`.
""".strip()

        answer = client.complete_structured(
            prompt,
            NumericAnswer,
            web_search="auto",
        )
        answers.append(answer)

    return answers
