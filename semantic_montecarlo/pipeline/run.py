"""
The pipeline entry point: question in, :class:`Distribution` out.

Composes three stages:

    paraphrase -> search -> bootstrap

Confidence lives on numeric :class:`NumericAnswer` outcomes produced by
``search``. Missing outcomes contribute to answerability, while aggregation is
owned by :func:`bootstrap`.

This is the ``Callable[[str], Distribution]`` that the CLI consumes
(see :mod:`semantic_montecarlo.cli`).
"""

from __future__ import annotations

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.observability import get_logger
from semantic_montecarlo.pipeline.steps.bootstrap import bootstrap
from semantic_montecarlo.pipeline.steps.paraphrase import paraphrase
from semantic_montecarlo.pipeline.steps.search import search
from semantic_montecarlo.schemas.models import Distribution

_logger = get_logger(__name__)


def run(
    question: str,
    *,
    client: LLMClient,
    unit: str | None = None,
    n_paraphrases: int = 5,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> Distribution:
    """
    Estimate the value distribution for a numeric ``question``.

    Args:
        question: The user's numeric question.
        client: LLM client shared across the paraphrase and search stages.
        unit: Unit required for every numeric answer.
        n_paraphrases: Number of paraphrases to generate (plus the original).
        n_resamples: Bootstrap resample count passed to ``bootstrap``.
        seed: Optional seed for reproducible bootstrapping.

    Returns:
        The conditional numeric distribution and no-answer probability.
    """
    _logger.info("Pipeline run starting: %r", question)
    paraphrases = paraphrase(question, n=n_paraphrases, client=client)
    answers = search(paraphrases, client=client, unit=unit)
    distribution = bootstrap(answers, n_resamples=n_resamples, seed=seed)
    _logger.info(
        "Pipeline run finished: %d paraphrases -> %d answers -> "
        "distribution over %d values (no-answer probability %.3f)",
        len(paraphrases),
        len(answers),
        len(distribution.data),
        distribution.no_answer_probability,
    )
    return distribution
