"""
The paraphrase stage: turn one question into several search-ready phrasings.

This is a pipeline stage, not an agent: it is a deterministic transformation
that happens to use an LLM. Given the user's question, it generates ``n``
paraphrasings via the LLM and returns the original prepended, yielding ``n + 1``
phrasings total.

Two post-processing rules are applied in the stage, never delegated to the
prompt:

- The original question is prepended verbatim. The model never gets a chance to
  "improve" it — its job is to add diversity, not to rewrite the user's words.
- Outputs are deduplicated case-insensitively after whitespace normalization.
  This is cheap and deterministic; instructing the model about dedup is
  unreliable.
"""

from __future__ import annotations

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.observability import get_logger
from semantic_montecarlo.prompts import load
from semantic_montecarlo.schemas.paraphrase import ParaphraseOutput

_logger = get_logger(__name__)

DEFAULT_TEMPERATURE = 0.7


def paraphrase(
    question: str,
    *,
    n: int = 5,
    client: LLMClient,
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[str]:
    """
    Generate phrasings of ``question`` for downstream search.

    Args:
        question: The user's question, returned verbatim as the first element.
        n: Number of paraphrases to ask the model for. The returned list has
            ``n + 1`` items before dedup (original first).
        client: LLM client used for the structured generation call.
        temperature: Sampling temperature for the generation.

    Returns:
        A list starting with ``question`` verbatim, followed by up to ``n``
        deduplicated paraphrases. Duplicates of the original or of each other
        are removed, so the result may be shorter than ``n + 1``.
    """
    template = load("paraphraser")
    prompt = template.render("user", question=question, n=n)

    output = client.complete_structured(
        prompt,
        ParaphraseOutput,
        temperature=temperature,
    )
    phrasings = _dedup(question, output.data.paraphrases)
    _logger.debug("paraphrase: requested %d, got %d after dedup", n, len(phrasings) - 1)
    return phrasings


def _dedup(original: str, paraphrases: list[str]) -> list[str]:
    """
    Prepend ``original`` and drop case/whitespace-normalized duplicates.

    Args:
        original: The verbatim question; always first in the result.
        paraphrases: Model-generated phrasings.

    Returns:
        ``[original, ...unique paraphrases]`` with normalized duplicates
        removed, preserving first-seen order.
    """
    seen: set[str] = {_normalize(original)}
    result: list[str] = [original]
    for p in paraphrases:
        key = _normalize(p)
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def _normalize(s: str) -> str:
    """Normalize for duplicate comparison: lowercase, collapse whitespace."""
    return " ".join(s.lower().split())
