"""
Tests for :class:`LLMClient`.

Following the project convention, mocks sit at the ``ChatOpenAI`` boundary so
no test performs network I/O.
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
    client cache with a mock so calls return canned values.
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


def test_complete_returns_content_and_citations() -> None:
    client, mock_chat = _make_client_with_mock()
    mock_chat.invoke.return_value = MagicMock(
        content=[
            {
                "type": "text",
                "text": "hello world",
                "annotations": [{"type": "url_citation", "url": "https://example.com"}],
            }
        ]
    )

    result = client.complete("a prompt")

    assert result == ("hello world", ["https://example.com"])
    mock_chat.invoke.assert_called_once_with("a prompt")


def test_complete_uses_default_model() -> None:
    client, mock_chat = _make_client_with_mock(model="openrouter/free")
    mock_chat.invoke.return_value = MagicMock(content="ok")

    client.complete("hi")

    assert mock_chat.invoke.call_count == 1


def test_complete_passes_per_call_options() -> None:
    client, mock_chat = _make_client_with_mock()
    bound = MagicMock(name="bound")
    mock_chat.bind.return_value = bound
    bound.invoke.return_value = MagicMock(content="ok")

    client.complete(
        "hi",
        temperature=0.7,
        max_tokens=128,
        web_search="auto",
    )

    mock_chat.bind.assert_called_once_with(
        tools=[
            {
                "type": "openrouter:web_search",
                "parameters": {"engine": "auto"},
            }
        ],
        temperature=0.7,
        max_tokens=128,
    )
    bound.invoke.assert_called_once_with("hi")


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

    # Per-call options are bound before structured output; with none passed,
    # with_structured_output is called on the base client directly.
    mock_chat.with_structured_output.assert_called_once_with(
        _Answer, method="json_schema"
    )
    assert result is expected


def test_complete_structured_binds_per_call_options() -> None:
    client, mock_chat = _make_client_with_mock()
    bound = MagicMock(name="bound")
    structured = MagicMock(name="structured")
    mock_chat.bind.return_value = bound
    bound.with_structured_output.return_value = structured
    structured.invoke.return_value = _Answer(value=1)

    client.complete_structured("p", _Answer, temperature=0.9)

    # Binding happens on the base client; structured output is built on the
    # bound runnable.
    mock_chat.bind.assert_called_once_with(temperature=0.9)
    bound.with_structured_output.assert_called_once_with(_Answer, method="json_schema")


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
    client, mock_chat = _make_client_with_mock(model="openrouter/free")
    mock_chat.invoke.return_value = MagicMock(content="ok")

    client.complete("one")
    client.complete("two")

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
