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
    Answer each phrasing via a structured LLM call.

    Args:
        paraphrases: Query phrasings to search.
        client: LLM client used for the structured generation calls.

    Returns:
        One :class:`NumericAnswer` per input phrasing, in order. Each answer
        carries its own ``confidence`` weight for the bootstrap.

    Raises:
        NotImplementedError: This stage's body is owned separately.
    """
    raise NotImplementedError
