"""
The LLM client: a thin, cached wrapper around ``ChatOpenAI``.

This module is the *transport* layer — "give me a model response, optionally
parsed into this schema." It deliberately knows nothing about paraphrasing or
web search; those concerns live in :mod:`semantic_montecarlo.agents`.

Two methods form the public surface:

* :meth:`LLMClient.complete` — a plain string completion.
* :meth:`LLMClient.complete_structured` — a completion parsed into a pydantic
  schema via the model's native ``with_structured_output``. This replaces the
  old brace-matching JSON extraction: it is more reliable and cheaper because
  the schema is enforced by the provider (function/JSON-schema calling).

Provider routing and key resolution are delegated to
:func:`~semantic_montecarlo.llm.config.resolve_provider`. ``ChatOpenAI`` clients
are cached per ``(model_id, base_url)`` so repeated calls reuse the underlying
HTTP connection pool rather than rebuilding it on every invocation.

OpenRouter's web-search plugin is enabled per-call via ``extra_body`` — it lives
on the request, not the client, because the searcher wants it and the
paraphraser does not.
"""

from __future__ import annotations

from typing import TypeVar, cast

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from semantic_montecarlo.llm.config import DEFAULT_MODEL, resolve_provider
from semantic_montecarlo.llm.errors import StructuredOutputError

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Cached OpenAI-compatible chat client with structured-output support.

    A single client holds routing/key defaults but builds (and caches) one
    ``ChatOpenAI`` per ``(model_id, base_url)``. Per-call options (temperature,
    max tokens, the OpenRouter web-search plugin via ``extra_body``) are passed
    on each method so they never leak across unrelated calls.

    Attributes:
        default_model: Model id used when a method omits ``model``.
        default_temperature: Sampling temperature when omitted.
        default_max_tokens: Completion token cap when omitted.
        max_retries: Passed through to ``ChatOpenAI`` for transport retries.
    """

    def __init__(
        self,
        default_model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_temperature: float = 0.0,
        default_max_tokens: int = 4096,
        max_retries: int = 2,
    ) -> None:
        """
        Initialize the client with routing and per-call defaults.

        No network or key validation happens here; keys are resolved lazily by
        :func:`resolve_provider` when a client is first built for a model.
        """
        self.default_model = default_model
        self._api_key = api_key
        self._base_url = base_url
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        # Cache ChatOpenAI per (model_id, base_url) to reuse connection pools.
        self._clients: dict[tuple[str, str | None], ChatOpenAI] = {}

    def _get_client(self, model_id: str) -> ChatOpenAI:
        """
        Return a cached ``ChatOpenAI`` for ``model_id``, building it if new.

        Routing and key resolution happen in :func:`resolve_provider`; this
        method only constructs and memoizes the langchain client.
        """
        cache_key = (model_id, self._base_url)
        cached = self._clients.get(cache_key)
        if cached is not None:
            return cached

        provider = resolve_provider(
            model_id, api_key=self._api_key, base_url=self._base_url
        )
        client = ChatOpenAI(
            model=model_id,
            base_url=provider.base_url,
            api_key=provider.api_key,
            temperature=self.default_temperature,
            max_tokens=self.default_max_tokens,
            max_retries=self.max_retries,
            default_headers=provider.default_headers or None,
        )
        self._clients[cache_key] = client
        return client

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> str:
        """
        Run a plain completion and return the response text.

        Args:
            prompt: The user prompt to send.
            model: Model id; falls back to :attr:`default_model`.
            temperature: Sampling temperature; falls back to
                :attr:`default_temperature`.
            max_tokens: Completion token cap; falls back to
                :attr:`default_max_tokens`.
            extra_body: Extra request-body fields passed through to the provider.
                The OpenRouter web-search plugin is enabled with
                ``{"plugins": [{"id": "web"}]}``.

        Returns:
            The model's response content as a string.
        """
        client = self._get_client(model or self.default_model)
        bound = client
        # Bind per-call options so the cached base client stays untouched.
        bind_kwargs: dict[str, object] = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        if extra_body is not None:
            bind_kwargs["extra_body"] = extra_body
        if bind_kwargs:
            bound = client.bind(**bind_kwargs)
        response = bound.invoke(prompt)
        return str(response.content)

    def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> T:
        """
        Run a completion parsed into ``schema`` via native structured output.

        Uses ``ChatOpenAI.with_structured_output(schema)`` so the provider
        enforces the schema (JSON-schema / function calling) rather than
        parsing JSON out of free text. Parse or validation failures are wrapped
        in :class:`StructuredOutputError`.

        Args:
            prompt: The user prompt to send.
            schema: Pydantic model class to parse the response into.
            model: Model id; falls back to :attr:`default_model`.
            temperature: Sampling temperature; falls back to
                :attr:`default_temperature`.
            max_tokens: Completion token cap; falls back to
                :attr:`default_max_tokens`.
            extra_body: Extra request-body fields (see :meth:`complete`).

        Returns:
            An instance of ``schema``.

        Raises:
            StructuredOutputError: If the response cannot be parsed/validated.
        """
        client = self._get_client(model or self.default_model)
        structured = client.with_structured_output(schema)
        bind_kwargs: dict[str, object] = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens
        if extra_body is not None:
            bind_kwargs["extra_body"] = extra_body
        if bind_kwargs:
            structured = structured.bind(**bind_kwargs)
        try:
            result = structured.invoke(prompt)
        except ValidationError as exc:
            raise StructuredOutputError(
                f"Model response did not validate against {schema.__name__}."
            ) from exc
        # ``with_structured_output`` is typed as ``dict | BaseModel`` because
        # langchain also allows dict schemas; our signature constrains ``T`` to
        # BaseModel, so the dict branch cannot occur — cast, don't guard.
        return cast(T, result)
