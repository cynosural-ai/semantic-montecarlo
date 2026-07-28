"""
Tests for :class:`LLMClient`.

Mocks sit at the ``openai.OpenAI`` boundary so no test performs network I/O. We
seed the client cache with a ``MagicMock`` whose ``chat.completions.create``
returns canned ``ChatCompletion``-shaped objects, then assert what the client
passes through and how it extracts text/sources/usage.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from openai import OpenAI
from pydantic import BaseModel

from semantic_montecarlo.llm.client import LLMClient
from semantic_montecarlo.llm.errors import LLMError, StructuredOutputError


class _Answer(BaseModel):
    """Trivial schema for structured-output tests."""

    value: int


def _make_client_with_mock(
    model: str = "openrouter/free",
) -> tuple[LLMClient, MagicMock]:
    """
    Build an :class:`LLMClient` whose cached ``OpenAI`` is a mock.

    Bypasses ``__init__`` to skip key resolution, then seeds the per-model
    client cache with a ``MagicMock`` so ``chat.completions.create`` is stubbed.
    """
    client = LLMClient.__new__(LLMClient)
    client.default_model = model
    client.default_temperature = 0.0
    client.default_max_tokens = 4096
    client.max_retries = 2
    client._api_key = "test-key"
    client._base_url = None
    client._clients = {}
    mock_openai = MagicMock(name="OpenAI", spec=OpenAI)
    client._clients[(model, None)] = mock_openai
    return client, mock_openai


def _chat_completion(
    content: str = "",
    annotations: list[object] | None = None,
    usage: object | None = None,
) -> MagicMock:
    """Build a ``ChatCompletion``-shaped mock with one choice."""
    message = MagicMock()
    message.content = content
    message.annotations = annotations or []
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _usage(
    prompt: int = 0,
    completion: int = 0,
    total: int = 0,
    reasoning: int = 0,
    cached: int = 0,
) -> MagicMock:
    """Build a ``CompletionUsage``-shaped mock."""
    completion_details = MagicMock(reasoning_tokens=reasoning)
    prompt_details = MagicMock(cached_tokens=cached)
    return MagicMock(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        completion_tokens_details=completion_details,
        prompt_tokens_details=prompt_details,
    )


def _url_citation(url: str) -> MagicMock:
    """Build an ``Annotation``-shaped mock for one ``url_citation``."""
    citation = MagicMock(url=url)
    return MagicMock(type="url_citation", url_citation=citation)


def test_complete_returns_text_sources_usage() -> None:
    client, mock_openai = _make_client_with_mock()
    mock_openai.chat.completions.create.return_value = _chat_completion(
        content="hello world",
        annotations=[_url_citation("https://example.com")],
        usage=_usage(prompt=5, completion=7, total=12, reasoning=3, cached=1),
    )

    result = client.complete("a prompt")

    assert result.text == "hello world"
    assert result.sources == ["https://example.com"]
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 12
    assert result.usage.reasoning_tokens == 3
    assert result.usage.cached_tokens == 1


def test_complete_uses_default_model() -> None:
    client, mock_openai = _make_client_with_mock(model="openrouter/free")
    mock_openai.chat.completions.create.return_value = _chat_completion()

    client.complete("hi")

    assert mock_openai.chat.completions.create.call_count == 1


def test_complete_retries_malformed_provider_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, mock_openai = _make_client_with_mock()
    malformed = json.JSONDecodeError("Expecting value", "invalid", 0)
    mock_openai.chat.completions.create.side_effect = [
        malformed,
        _chat_completion(content="recovered"),
    ]
    monkeypatch.setattr("semantic_montecarlo.llm.client.time.sleep", lambda _: None)

    assert client.complete("hi").text == "recovered"
    assert mock_openai.chat.completions.create.call_count == 2


def test_complete_wraps_repeated_malformed_provider_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, mock_openai = _make_client_with_mock()
    malformed = json.JSONDecodeError("Expecting value", "invalid", 0)
    mock_openai.chat.completions.create.side_effect = malformed
    monkeypatch.setattr("semantic_montecarlo.llm.client.time.sleep", lambda _: None)

    with pytest.raises(LLMError, match="after 3 attempts") as exc_info:
        client.complete("hi")

    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)
    assert mock_openai.chat.completions.create.call_count == 3


def test_complete_passes_per_call_options() -> None:
    client, mock_openai = _make_client_with_mock()
    mock_openai.chat.completions.create.return_value = _chat_completion()

    client.complete(
        "hi",
        temperature=0.7,
        max_tokens=128,
        web_search="auto",
    )

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_completion_tokens"] == 128
    assert kwargs["tools"] == [
        {
            "type": "openrouter:web_search",
            "parameters": {
                "engine": "auto",
                "max_results": 10,
                "max_total_results": 15,
            },
        }
    ]
    assert kwargs["extra_body"] == {"max_tool_calls": 3}
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_complete_strips_duplicate_sources() -> None:
    client, mock_openai = _make_client_with_mock()
    mock_openai.chat.completions.create.return_value = _chat_completion(
        annotations=[
            _url_citation("https://a.test"),
            _url_citation("https://b.test"),
            _url_citation("https://a.test"),
        ],
    )

    result = client.complete("hi")

    assert result.sources == ["https://a.test", "https://b.test"]


def test_complete_structured_validates_json_and_returns_data() -> None:
    client, mock_openai = _make_client_with_mock()
    mock_openai.chat.completions.create.return_value = _chat_completion(
        content='{"value": 42}',
        usage=_usage(prompt=10, completion=5, total=15),
    )

    result = client.complete_structured("prompt", _Answer)

    assert isinstance(result.data, _Answer)
    assert result.data.value == 42
    assert result.usage.total_tokens == 15
    # response_format must be the json_schema shape.
    rf = mock_openai.chat.completions.create.call_args.kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "_Answer"
    assert "value" in rf["json_schema"]["schema"]["properties"]


def test_complete_structured_binds_per_call_options() -> None:
    client, mock_openai = _make_client_with_mock()
    mock_openai.chat.completions.create.return_value = _chat_completion(
        content='{"value": 1}'
    )

    client.complete_structured("p", _Answer, temperature=0.9)

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0.9
    assert "response_format" in kwargs


def test_complete_structured_wraps_validation_error() -> None:
    client, mock_openai = _make_client_with_mock()
    # Schema mismatch: value must be int, content gives a string.
    mock_openai.chat.completions.create.return_value = _chat_completion(
        content='{"value": "not an int"}'
    )

    with pytest.raises(StructuredOutputError):
        client.complete_structured("p", _Answer)


def test_complete_structured_wraps_malformed_json() -> None:
    client, mock_openai = _make_client_with_mock()
    mock_openai.chat.completions.create.return_value = _chat_completion(
        content="not json at all"
    )

    with pytest.raises(StructuredOutputError):
        client.complete_structured("p", _Answer)


def test_usage_zero_when_usage_absent() -> None:
    client, mock_openai = _make_client_with_mock()
    mock_openai.chat.completions.create.return_value = _chat_completion(
        content="ok", usage=None
    )

    result = client.complete("hi")

    assert result.usage.prompt_tokens == 0
    assert result.usage.completion_tokens == 0


def test_client_cache_holds_across_calls() -> None:
    client, mock_openai = _make_client_with_mock(model="openrouter/free")
    mock_openai.chat.completions.create.return_value = _chat_completion()

    client.complete("one")
    client.complete("two")

    assert client._clients.keys() == {("openrouter/free", None)}
    assert mock_openai.chat.completions.create.call_count == 2


def test_client_cache_separates_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k1")

    client = LLMClient(default_model="openrouter/free")
    c1 = client._get_client("openrouter/free")
    c2 = client._get_client("anthropic/claude-3.5-sonnet")

    assert c1 is not c2
    assert client._clients.keys() == {
        ("openrouter/free", None),
        ("anthropic/claude-3.5-sonnet", None),
    }
