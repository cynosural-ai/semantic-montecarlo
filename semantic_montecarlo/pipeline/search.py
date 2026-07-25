"""
The search stage: turn phrasings into structured answers.

SCAFFOLD STUB. The contract below (signature + return type) is the stable seam
the pipeline depends on. The implementation — structured search via the LLM
client, the OpenRouter web-search plugin (``extra_body={"plugins": [{"id":
"web"}]}``), retries, and concurrency — is owned separately and lives in this
function's body when ready. See ADR 0001 for the structured-output decision.
"""

from __future__ import annotations

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.schemas.search import SearchAnswer


def search(
    paraphrases: list[str],
    *,
    client: LLMClient,
) -> list[SearchAnswer]:
    """
    Answer each phrasing via a structured LLM call.

    Args:
        paraphrases: Query phrasings to search.
        client: LLM client used for the structured generation calls.

    Returns:
        One :class:`SearchAnswer` per input phrasing, in order. Answers with no
        numeric value carry ``number=None`` and are preserved (the scoring and
        bootstrap stages decide how to treat them).

    Raises:
        NotImplementedError: This stage's body is owned separately.
    """
    raise NotImplementedError

