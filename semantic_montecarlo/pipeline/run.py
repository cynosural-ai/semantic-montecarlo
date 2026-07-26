"""
The pipeline entry point: question in, :class:`RunResult` out.

Composes three stages:

    paraphrase -> search -> bootstrap

Confidence lives on numeric :class:`NumericAnswer` outcomes produced by
``search``. Missing outcomes contribute to answerability, while aggregation is
owned by :func:`bootstrap`, which produces the final :class:`Distribution`.
``run`` packages the stages' outputs plus run-level metadata (elapsed time,
model) into a :class:`RunResult`.

Consumed by the CLI (see :mod:`semantic_montecarlo.cli`) and by
:mod:`scripts.benchmark`.
"""

from __future__ import annotations

import time

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.observability import get_logger
from semantic_montecarlo.pipeline.steps.bootstrap import bootstrap
from semantic_montecarlo.pipeline.steps.paraphrase import paraphrase
from semantic_montecarlo.pipeline.steps.search import search
from semantic_montecarlo.schemas.run_result import RunResult

_logger = get_logger(__name__)


def run(
    question: str,
    *,
    client: LLMClient,
    unit: str | None = None,
    n_paraphrases: int = 5,
    n_resamples: int = 10_000,
    seed: int | None = None,
) -> RunResult:
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
        A :class:`RunResult` bundling the paraphrases, answers, the conditional
        numeric distribution with no-answer probability, elapsed time, and the
        configured model.
    """
    _logger.info("Pipeline run starting: %r", question)
    start = time.perf_counter()
    paraphrases = paraphrase(question, n=n_paraphrases, client=client)
    answers = search(paraphrases, client=client, unit=unit)
    distribution = bootstrap(answers, n_resamples=n_resamples, seed=seed)
    elapsed = time.perf_counter() - start
    _logger.info(
        "Pipeline run finished: %d paraphrases -> %d answers -> "
        "distribution over %d values, no-answer probability = %.3f "
        "(time-elapsed = %.2fs)",
        len(paraphrases),
        len(answers),
        len(distribution.data),
        distribution.no_answer_probability,
        elapsed,
    )
    return RunResult(
        question=question,
        unit=unit,
        paraphrases=paraphrases,
        answers=answers,
        distribution=distribution,
        elapsed_seconds=elapsed,
        model=client.default_model,
    )
