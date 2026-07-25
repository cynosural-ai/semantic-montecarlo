"""
LLM transport layer: provider routing, the cached client, and errors.

Public surface:

* :class:`LLMClient` — cached ``ChatOpenAI`` wrapper with structured output.
* :func:`resolve_provider`, :class:`ProviderConfig` — provider routing.
* :class:`LLMError`, :class:`StructuredOutputError` — typed exceptions.
"""

from semantic_montecarlo.llm.client import LLMClient
from semantic_montecarlo.llm.config import (
    DEFAULT_MODEL,
    OPENROUTER_BASE_URL,
    ProviderConfig,
    resolve_provider,
)
from semantic_montecarlo.llm.errors import LLMError, StructuredOutputError

__all__ = [
    "DEFAULT_MODEL",
    "LLMClient",
    "LLMError",
    "OPENROUTER_BASE_URL",
    "ProviderConfig",
    "StructuredOutputError",
    "resolve_provider",
]
