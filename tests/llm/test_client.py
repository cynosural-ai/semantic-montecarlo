"""
Tests for :class:`LLMClient`.

Following the project convention: mock at the ``ChatOpenAI`` boundary (no
network). We inject a ``MagicMock`` as the cached client so we can assert what
the client passes through and that the cache holds across calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from semantic_montecarlo.llm.client import LLMClient
from semantic_montecarlo.llm.errors import StructuredOutputError


class _Answer(BaseModel):
    """Trivial schema for structured-output tests."""

    value: int


def _make_client_with_mock(
    model: str = "openrouter/free",
) -> tuple[LLMClient, MagicMock]:
    """
    Build an :class:`LLMClient` whose cached ``ChatOpenAI`` is a mock.

    Bypasses ``__init__`` to avoid key resolution, then seeds the per-model
    client cache with a ``MagicMock`` so calls return canned values.
    """
    client = LLMClient.__new__(LLMClient)
    client.default_model = model
    client.default_temperature = 0.0
    client.default_max_tokens = 4096
    client.max_retries = 2
    client._api_key = "test-key"
    client._base_url = None
    client._clients = {}
    mock_chat = MagicMock(name="ChatOpenAI")
    client._clients[(model, None)] = mock_chat
    return client, mock_chat


def test_complete_returns_response_content() -> None:
    client, mock_chat = _make_client_with_mock()
    mock_chat.invoke.return_value = MagicMock(content="hello world")

    result = client.complete("a prompt")

    assert result == "hello world"
    mock_chat.invoke.assert_called_once_with("a prompt")


def test_complete_uses_default_model() -> None:
    client, mock_chat = _make_client_with_mock(model="openrouter/free")
    mock_chat.invoke.return_value = MagicMock(content="ok")

    # No explicit model -> uses the pre-seeded default model client.
    client.complete("hi")
    assert mock_chat.invoke.call_count == 1


def test_complete_binds_per_call_options() -> None:
    client, mock_chat = _make_client_with_mock()
    bound = MagicMock(name="bound")
    mock_chat.bind.return_value = bound
    bound.invoke.return_value = MagicMock(content="ok")

    client.complete(
        "hi",
        temperature=0.7,
        max_tokens=128,
        extra_body={"plugins": [{"id": "web"}]},
    )

    mock_chat.bind.assert_called_once_with(
        temperature=0.7,
        max_tokens=128,
        extra_body={"plugins": [{"id": "web"}]},
    )
    bound.invoke.assert_called_once_with("hi")
    # The base client must not have been invoked directly when binding occurs.
    mock_chat.invoke.assert_not_called()


def test_complete_does_not_bind_when_no_options() -> None:
    client, mock_chat = _make_client_with_mock()
    mock_chat.invoke.return_value = MagicMock(content="ok")

    client.complete("hi")

    mock_chat.bind.assert_not_called()


def test_complete_structured_uses_with_structured_output() -> None:
    client, mock_chat = _make_client_with_mock()
    structured = MagicMock(name="structured")
    mock_chat.with_structured_output.return_value = structured
    expected = _Answer(value=42)
    structured.invoke.return_value = expected

    result = client.complete_structured("prompt", _Answer)

    mock_chat.with_structured_output.assert_called_once_with(_Answer)
    assert result is expected


def test_complete_structured_binds_per_call_options() -> None:
    client, mock_chat = _make_client_with_mock()
    structured = MagicMock(name="structured")
    bound = MagicMock(name="bound")
    mock_chat.with_structured_output.return_value = structured
    structured.bind.return_value = bound
    bound.invoke.return_value = _Answer(value=1)

    client.complete_structured("p", _Answer, temperature=0.9)

    structured.bind.assert_called_once_with(temperature=0.9)


def test_complete_structured_wraps_validation_error() -> None:
    client, mock_chat = _make_client_with_mock()
    structured = MagicMock(name="structured")
    mock_chat.with_structured_output.return_value = structured

    # Simulate the provider/validator rejecting the response.
    from pydantic import ValidationError

    structured.invoke.side_effect = ValidationError.from_exception_data("x", [])

    with pytest.raises(StructuredOutputError):
        client.complete_structured("p", _Answer)


def test_client_cache_holds_across_calls() -> None:
    # The same model id must reuse the cached client, not build a new one.
    client, mock_chat = _make_client_with_mock(model="openrouter/free")
    mock_chat.invoke.return_value = MagicMock(content="ok")

    client.complete("one")
    client.complete("two")

    # No new entry was added to the cache.
    assert client._clients.keys() == {("openrouter/free", None)}
    assert mock_chat.invoke.call_count == 2


def test_client_cache_separates_models(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two different model ids get two different cached clients.
    monkeypatch.setenv("OPENROUTER_API_KEY", "k1")

    client = LLMClient(default_model="openrouter/free")
    # Force construction of clients for two distinct OpenRouter models.
    c1 = client._get_client("openrouter/free")
    c2 = client._get_client("anthropic/claude-3.5-sonnet")

    assert c1 is not c2
    assert client._clients.keys() == {
        ("openrouter/free", None),
        ("anthropic/claude-3.5-sonnet", None),
    }
