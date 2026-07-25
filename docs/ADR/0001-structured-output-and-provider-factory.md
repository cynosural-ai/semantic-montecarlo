# Native structured output, and provider routing as a factory

*Accepted 25-07-2026.*

## The question

How does the LLM layer get a model to return structured data, and how does it decide which provider (base URL, API key, headers) to talk to?

## Options considered

- **A. Prompt for JSON and parse it out of the response.** Tell the model to emit a fenced JSON block, then regex-extract and brace-match it back to valid JSON. This is what the layer was doing when copied in from another project: ~90 lines of fence detection, balanced-brace scanning, and a two-step "ask once, then ask again to parse" fallback. It works, mostly. It fails on nested braces inside strings, on truncated output, on models that ignore the fence, and the two-step path doubles latency and tokens to "be concise."
- **B. Native structured output.** Hand the pydantic schema to `ChatOpenAI.with_structured_output(schema)` and let the provider enforce it via JSON-schema / function calling. No parsing code on our side.
- **C. Some middle ground** — JSON mode (provider guarantees valid JSON, but not our schema) plus a local validation pass.

## Decision

Went with B. The schema is enforced by the provider, not by us, so the model literally cannot return a response that doesn't conform. `complete_structured` is now four lines: invoke, catch `ValidationError`, wrap in `StructuredOutputError`, return. The old `extract_json`, the brace matcher, and `two_step_parsing` are gone.

We keep one domain exception, `StructuredOutputError` (subclassing `LLMError`), for a concrete reason: the sampling pipeline will have a retry loop, and catching one named exception there is cleaner than catching a grab-bag of pydantic and stdlib types. The exception is two lines; the complexity was all in the handling, which we cut.

On provider routing: OpenRouter is the only provider, so routing is a straight-line factory function, `resolve_provider(model_id) -> ProviderConfig`, rather than branching inside the client or a provider class hierarchy. Two OpenAI-compatible endpoints did not justify a `Provider` ABC. Any model id passes through to OpenRouter unchanged.

Two smaller calls fall out of this:

- `ChatOpenAI` clients are cached per `(model_id, base_url)` so repeated calls reuse the connection pool, and per-call options (temperature, max tokens, the web-search plugin via `extra_body`) are bound on each call so the cached base client never carries state from one call into the next.
- `cast(T, result)` rather than an `isinstance` guard: `with_structured_output` is typed `dict | BaseModel` only because langchain also allows dict schemas, and our signature constrains `T` to `BaseModel`, so the dict branch cannot occur. The cast is honest where a runtime guard would be dead code pretending to be a branch.

## Consequences

Structured output depends on the provider supporting JSON-schema or function calling. OpenRouter does for the models we care about; a model that doesn't will fail at `with_structured_output` time, which is a clear error rather than a silent fall-back to prose parsing. If we ever need such a model, that's a separate ADR — we won't quietly bring `extract_json` back.

The OpenRouter web-search plugin (needed for the search agent) composes with structured output on the same request — that should be spiked before the searcher is built, since it determines whether `complete_structured` is enough or the client needs a separate "raw response with citations" path.
