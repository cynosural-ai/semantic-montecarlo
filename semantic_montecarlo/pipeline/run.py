"""
The pipeline entry point: question in, :class:`Distribution` out.

Composes three stages:

    paraphrase -> search -> bootstrap

Confidence lives on :class:`NumericAnswer` (produced by ``search``), so there is
no separate scoring step. Aggregation is owned by :func:`bootstrap`, which
produces the final :class:`Distribution`.

This is the ``Callable[[str], Distribution]`` that the benchmark consumes
(see :mod:`semantic_montecarlo.scripts.benchmark`).
"""

from __future__ import annotations

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.pipeline.bootstrap import bootstrap
from semantic_montecarlo.pipeline.paraphrase import paraphrase
from semantic_montecarlo.pipeline.search import search
from semantic_montecarlo.schemas.models import Distribution


def run(
    question: str,
    *,
    client: LLMClient,
    n_paraphrases: int = 5,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> Distribution:
    """
    Estimate the value distribution for a numeric ``question``.

    Args:
        question: The user's numeric question.
        client: LLM client shared across the paraphrase and search stages.
        n_paraphrases: Number of paraphrases to generate (plus the original).
        n_resamples: Bootstrap resample count passed to ``bootstrap``.
        seed: Optional seed for reproducible bootstrapping.

    Returns:
        A :class:`Distribution` over the numeric values the searches found.
    """
    paraphrases = paraphrase(question, n=n_paraphrases, client=client)
    answers = search(paraphrases, client=client)
    return bootstrap(answers, n_resamples=n_resamples, seed=seed)
