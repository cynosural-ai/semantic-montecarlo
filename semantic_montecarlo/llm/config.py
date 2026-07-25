"""
Provider configuration for the OpenRouter backend.

OpenRouter speaks the OpenAI-compatible chat API, so it is handled by the same
:class:`~langchain_openai.ChatOpenAI` client used elsewhere. This module
resolves the base URL, API key, and optional attribution headers into a plain
:class:`ProviderConfig` dataclass. Any model id is passed through to OpenRouter
unchanged (e.g. ``openrouter/free``, ``anthropic/claude-3.5-sonnet``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"

# Env var names — underscore-style so they are shell-friendly (see .env.template).
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_REFERER_ENV = "OPENROUTER_HTTP_REFERER"
OPENROUTER_TITLE_ENV = "OPENROUTER_APP_TITLE"


@dataclass(frozen=True)
class ProviderConfig:
    """
    Resolved provider details used to build a :class:`ChatOpenAI`.

    Attributes:
        base_url: OpenAI-compatible chat completions endpoint.
        api_key: Resolved API key (caller raises if missing).
        default_headers: Per-provider default HTTP headers (e.g. OpenRouter
            attribution); may be empty.
    """

    base_url: str
    api_key: str
    default_headers: dict[str, str] = field(default_factory=dict)


def _openrouter_headers() -> dict[str, str]:
    """Build OpenRouter attribution headers from env, dropping unset ones."""
    candidates = {
        "HTTP-Referer": os.getenv(OPENROUTER_REFERER_ENV),
        "X-Title": os.getenv(OPENROUTER_TITLE_ENV),
    }
    return {k: v for k, v in candidates.items() if v}


def resolve_provider(
    model_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> ProviderConfig:
    """
    Resolve OpenRouter provider settings for ``model_id``.

    Args:
        model_id: The OpenRouter model id (e.g. ``openrouter/free``).
        api_key: Optional explicit key; falls back to ``OPENROUTER_API_KEY``.
        base_url: Optional endpoint; overrides the OpenRouter default.

    Returns:
        A :class:`ProviderConfig` with the resolved key and headers.

    Raises:
        LLMError: If no API key can be resolved.
    """
    # Local import keeps the import cycle clean for callers who only need config.
    # Avoid importing load_dotenv side effects at module import time.
    from dotenv import load_dotenv  # noqa: PLC0415

    from semantic_montecarlo.llm.errors import LLMError

    load_dotenv()

    resolved_key = api_key or os.getenv(OPENROUTER_KEY_ENV)
    if not resolved_key:
        raise LLMError(f"Missing {OPENROUTER_KEY_ENV} for model {model_id!r}.")

    return ProviderConfig(
        base_url=base_url or OPENROUTER_BASE_URL,
        api_key=resolved_key,
        default_headers=_openrouter_headers(),
    )
