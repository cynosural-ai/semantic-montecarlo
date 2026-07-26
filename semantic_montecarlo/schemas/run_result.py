"""
The result of one pipeline run.

Constructed by :func:`semantic_montecarlo.pipeline.run`, consumed by the CLI
(and by the benchmark). It bundles everything the run produced — the
intermediate paraphrases and answers, not just the final distribution — so that
a consumer can persist provenance (``answers`` carries reasoning, sources, and
confidence) alongside the headline result.

This is a plain data shape; it lives in ``schemas/`` because it is the shared
vocabulary between the pipeline and its consumers, not pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_montecarlo.schemas.models import Distribution
from semantic_montecarlo.schemas.search import NumericAnswer


@dataclass(frozen=True)
class RunResult:
    """
    Everything one pipeline run produced, plus run-level metadata.

    Attributes:
        question: The question the run was asked (after CLI resolution).
        unit: Required unit for the numeric answers, if any.
        paraphrases: Phrasings actually used by the search stage (original
            first). Useful for reproducing or debugging retrieval diversity.
        answers: Numeric answers with reasoning, sources, and confidence —
            the provenance behind the distribution.
        distribution: The final confidence-weighted value distribution.
        elapsed_seconds: Wall-clock time of the pipeline stages, in seconds.
        model: The model the client was configured with (``default_model``).
            Per-call ``model=`` overrides are not tracked here.
    """

    question: str
    unit: str | None
    paraphrases: list[str]
    answers: list[NumericAnswer]
    distribution: Distribution
    elapsed_seconds: float
    model: str
