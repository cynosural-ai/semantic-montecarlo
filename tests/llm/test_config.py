"""
Tests for provider configuration in :mod:`semantic_montecarlo.llm.config`.

Key resolution reads env vars; tests monkeypatch ``os.environ`` so no real keys
or network are involved.
"""

from __future__ import annotations

import pytest

from semantic_montecarlo.llm import config
from semantic_montecarlo.llm.config import (
    OPENROUTER_BASE_URL,
    ProviderConfig,
    resolve_provider,
)
from semantic_montecarlo.llm.errors import LLMError


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all provider env vars so each test controls its inputs.

    Also stubs ``load_dotenv`` so the real ``.env`` (which carries a live key
    in dev) doesn't repopulate the env mid-test via ``resolve_provider``.
    """
    for var in (
        config.OPENROUTER_KEY_ENV,
        config.OPENROUTER_REFERER_ENV,
        config.OPENROUTER_TITLE_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    # resolve_provider imports load_dotenv lazily inside the function.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


def test_resolves_openrouter_endpoint(_clean_env: None) -> None:
    cfg = resolve_provider("openrouter/free", api_key="k")
    assert isinstance(cfg, ProviderConfig)
    assert cfg.base_url == OPENROUTER_BASE_URL
    assert cfg.api_key == "k"


def test_passes_through_arbitrary_model_id(_clean_env: None) -> None:
    # Any model id is forwarded to OpenRouter unchanged.
    cfg = resolve_provider("anthropic/claude-3.5-sonnet", api_key="k")
    assert cfg.base_url == OPENROUTER_BASE_URL


def test_key_falls_back_to_env(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv(config.OPENROUTER_KEY_ENV, "env-key")
    cfg = resolve_provider("openrouter/free")
    assert cfg.api_key == "env-key"


def test_missing_key_raises(_clean_env: None) -> None:
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        resolve_provider("openrouter/free")


def test_headers_from_env(monkeypatch: pytest.MonkeyPatch, _clean_env: None) -> None:
    monkeypatch.setenv(config.OPENROUTER_REFERER_ENV, "https://app.test")
    monkeypatch.setenv(config.OPENROUTER_TITLE_ENV, "My App")
    cfg = resolve_provider("openrouter/free", api_key="k")
    assert cfg.default_headers == {
        "HTTP-Referer": "https://app.test",
        "X-Title": "My App",
    }


def test_headers_omit_unset(_clean_env: None) -> None:
    cfg = resolve_provider("openrouter/free", api_key="k")
    assert cfg.default_headers == {}


def test_explicit_base_url_overrides_default(_clean_env: None) -> None:
    cfg = resolve_provider("openrouter/free", api_key="k", base_url="https://custom/v1")
    assert cfg.base_url == "https://custom/v1"
