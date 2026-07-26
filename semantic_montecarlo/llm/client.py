"""OpenRouter-compatible completion client (bare ``openai`` SDK)."""

from __future__ import annotations

import json
from typing import Any, Literal, TypeAlias, TypeVar, cast

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from semantic_montecarlo.llm.config import DEFAULT_MODEL, resolve_provider
from semantic_montecarlo.llm.errors import StructuredOutputError
from semantic_montecarlo.llm.response import Completion, StructuredCompletion
from semantic_montecarlo.schemas.usage import Usage

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
        self._clients: dict[tuple[str, str | None], OpenAI] = {}

    def _get_client(self, model_id: str) -> OpenAI:
        cache_key = (model_id, self._base_url)
        cached = self._clients.get(cache_key)
        if cached is not None:
            return cached

        provider = resolve_provider(
            model_id,
            api_key=self._api_key,
            base_url=self._base_url,
        )
        client = OpenAI(
            api_key=provider.api_key,
            base_url=provider.base_url,
            default_headers=provider.default_headers or None,
            max_retries=self.max_retries,
        )
        self._clients[cache_key] = client
        return client

    @staticmethod
    def _request_options(
        extra_body: dict[str, object] | None,
        web_search: WebProvider | None,
    ) -> dict[str, object]:
        """Build per-request ``tools``/``extra_body`` from web_search + body."""
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

    @staticmethod
    def _usage_from(response_usage: object) -> Usage:
        """Extract a :class:`Usage` from a raw ``ChatCompletion.usage``.

        All attribute reads are guarded: ``usage``, the ``_details`` subobjects,
        and the OpenRouter-only ``cost_details`` (untyped in the SDK) are all
        optional. Returns an all-zero :class:`Usage` when nothing is present.
        """
        if response_usage is None:
            return Usage()

        def _get(obj: object, name: str) -> int:
            value = getattr(obj, name, None)
            return int(value) if value is not None else 0

        prompt = _get(response_usage, "prompt_tokens")
        completion = _get(response_usage, "completion_tokens")
        total = _get(response_usage, "total_tokens")

        completion_details = getattr(response_usage, "completion_tokens_details", None)
        reasoning = (
            _get(completion_details, "reasoning_tokens") if completion_details else 0
        )

        prompt_details = getattr(response_usage, "prompt_tokens_details", None)
        cached = _get(prompt_details, "cached_tokens") if prompt_details else 0

        return Usage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=reasoning,
            cached_tokens=cached,
        )

    @staticmethod
    def _sources_from(message: object) -> list[str]:
        """Scrape ``url_citation`` URLs off ``message.annotations``.

        OpenRouter returns citations as typed ``Annotation`` objects with
        ``type == "url_citation"`` and a nested ``url_citation.url``. Order is
        preserved; duplicates are dropped.
        """
        annotations = getattr(message, "annotations", None) or []
        sources: list[str] = []
        for annotation in annotations:
            if getattr(annotation, "type", None) != "url_citation":
                continue
            citation = getattr(annotation, "url_citation", None)
            url = getattr(citation, "url", None) if citation is not None else None
            if isinstance(url, str):
                sources.append(url)
        return list(dict.fromkeys(sources))

    def _common_create_kwargs(
        self,
        prompt: str,
        model: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, object]:
        """Core kwargs shared by both completion methods."""
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": (
                self.default_temperature if temperature is None else temperature
            ),
            # SDK deprecates ``max_tokens`` in favour of ``max_completion_tokens``.
            "max_completion_tokens": (
                self.default_max_tokens if max_tokens is None else max_tokens
            ),
        }

    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        web_search: WebProvider | None = None,
        extra_body: dict[str, object] | None = None,
    ) -> Completion:
        """Run a plain completion and return its text and verified citation URLs."""
        client = self._get_client(model or self.default_model)
        resolved_model = model or self.default_model
        # Build kwargs and splat: the dict shape is dynamic (tools/extra_body are
        # optional), so we pass it as Any — the SDK validates at runtime.
        kwargs: Any = self._common_create_kwargs(
            prompt, resolved_model, temperature, max_tokens
        )
        kwargs.update(self._request_options(extra_body, web_search))
        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        text = message.content or ""
        return Completion(
            text=text,
            sources=self._sources_from(message),
            usage=self._usage_from(response.usage),
        )

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
    ) -> StructuredCompletion[T]:
        """Run a completion parsed into a Pydantic schema via ``json_schema``."""
        client = self._get_client(model or self.default_model)
        resolved_model = model or self.default_model
        # Same dynamic-shape splat as complete(); see the note there.
        kwargs: Any = self._common_create_kwargs(
            prompt, resolved_model, temperature, max_tokens
        )
        kwargs.update(self._request_options(extra_body, web_search))
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
                "strict": False,
            },
        }

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        try:
            data = schema.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise StructuredOutputError(
                f"Model response did not validate against {schema.__name__}."
            ) from exc

        return StructuredCompletion(data=data, usage=self._usage_from(response.usage))
