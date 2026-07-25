"""OpenRouter-compatible completion client."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypeVar, cast

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from semantic_montecarlo.llm.config import DEFAULT_MODEL, resolve_provider
from semantic_montecarlo.llm.errors import StructuredOutputError

T = TypeVar("T", bound=BaseModel)

WebProvider: TypeAlias = Literal[
    "auto",
    "native",
    "exa",
    "parallel",
    "perplexity",
    "firecrawl",
]


class LLMClient:
    """Cached OpenAI-compatible client with optional OpenRouter web search."""

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
        """Configure completion defaults and lazy client caches."""
        self.default_model = default_model
        self._api_key = api_key
        self._base_url = base_url
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.max_retries = max_retries
        self._clients: dict[tuple[str, str | None], ChatOpenAI] = {}

    def _get_client(self, model_id: str) -> ChatOpenAI:
        cache_key = (model_id, self._base_url)
        cached = self._clients.get(cache_key)
        if cached is not None:
            return cached

        provider = resolve_provider(
            model_id,
            api_key=self._api_key,
            base_url=self._base_url,
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

    @staticmethod
    def _request_options(
        extra_body: dict[str, object] | None,
        web_search: WebProvider | None,
    ) -> dict[str, object]:
        body = dict(extra_body or {})
        tools = list(cast(list[object], body.pop("tools", [])))
        if web_search is not None:
            tools.append(
                {
                    "type": "openrouter:web_search",
                    "parameters": {"engine": web_search},
                }
            )

        options: dict[str, object] = {}
        if tools:
            options["tools"] = tools
        if body:
            options["extra_body"] = body
        return options

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        web_search: WebProvider | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, list[str]]:
        """Run a plain completion and return its text and verified citation URLs."""
        client = self._get_client(model or self.default_model)

        bind_kwargs = self._request_options(extra_body, web_search)
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens

        bound = client.bind(**bind_kwargs) if bind_kwargs else client
        response = bound.invoke(prompt)
        if isinstance(response.content, str):
            return response.content, []

        text: list[str] = []
        sources: list[str] = []
        for block in response.content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            if isinstance(block_text := block.get("text"), str):
                text.append(block_text)
            annotations = block.get("annotations")
            if isinstance(annotations, list):
                sources.extend(
                    annotation["url"]
                    for annotation in annotations
                    if isinstance(annotation, dict)
                    and annotation.get("type") == "url_citation"
                    and isinstance(annotation.get("url"), str)
                )

        return "".join(text), list(dict.fromkeys(sources))

    def complete_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        web_search: WebProvider | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> T:
        """Run a completion parsed into a Pydantic schema."""
        client = self._get_client(model or self.default_model)

        bind_kwargs: dict[str, object] = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_tokens"] = max_tokens

        bind_kwargs.update(self._request_options(extra_body, web_search))

        bound = client.bind(**bind_kwargs) if bind_kwargs else client
        structured = bound.with_structured_output(
            schema,
            method="json_schema",
        )

        try:
            result = structured.invoke(prompt)
        except ValidationError as exc:
            raise StructuredOutputError(
                f"Model response did not validate against {schema.__name__}."
            ) from exc

        return cast(T, result)
