"""
Shared pydantic schemas (the vocabulary between agents).

Agent input/output schemas — e.g. ``ParaphraseOutput`` for the paraphraser and
``NumericAnswer``/``Source`` for the searcher — live here so agents and the
aggregation layer share one definition of the data they exchange.
"""

from semantic_montecarlo.schemas.search import NumericAnswer

__all__ = ["NumericAnswer"]
