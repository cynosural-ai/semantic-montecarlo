"""
The pipeline entry point: question in, :class:`Distribution` out.

Composes the four stages:

    paraphrase -> search -> score -> bootstrap -> normalize

This is the ``Callable[[str], Distribution]`` that the benchmark consumes
(see :mod:`semantic_montecarlo.scripts.benchmark`).
"""

from __future__ import annotations

import numpy as np

from semantic_montecarlo.llm import LLMClient
from semantic_montecarlo.pipeline.bootstrap import (
    DEFAULT_N_SAMPLES,
    bootstrap,
)
from semantic_montecarlo.pipeline.normalize import normalize
from semantic_montecarlo.pipeline.paraphrase import paraphrase
from semantic_montecarlo.pipeline.score import score
from semantic_montecarlo.pipeline.search import search
from semantic_montecarlo.schemas.models import Distribution


def run(
    question: str,
    *,
    client: LLMClient,
    n_paraphrases: int = 5,
    n_samples: int = DEFAULT_N_SAMPLES,
    rng: np.random.Generator | None = None,
) -> Distribution:
    """
    Estimate the value distribution for a numeric ``question``.

    Args:
        question: The user's numeric question.
        client: LLM client shared across the paraphrase and search stages.
        n_paraphrases: Number of paraphrases to generate (plus the original).
        n_samples: Bootstrap sample count.
        rng: Optional NumPy generator for reproducible bootstrapping.

    Returns:
        A :class:`Distribution` over the numeric values the searches found.
    """
    paraphrases = paraphrase(question, n=n_paraphrases, client=client)
    answers = search(paraphrases, client=client)
    scored = score(answers)
    samples = bootstrap(scored, n_samples=n_samples, rng=rng)
    return normalize(samples)
