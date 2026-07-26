"""Token and cost usage for one LLM call, or aggregated across a run.

Stage 1 (this commit) returns all-zero placeholders from the client; Stage 2
populates the fields for real from the OpenRouter ``response.usage`` payload.
The shape is fixed now so call sites and :class:`RunResult` can be built around
it before the real values arrive.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Usage:
    """Token counts for one LLM call.

    Attributes:
        prompt_tokens: Input tokens charged for the call.
        completion_tokens: Output tokens generated.
        total_tokens: ``prompt_tokens + completion_tokens``.
        reasoning_tokens: Output tokens spent on reasoning/thinking
            (``completion_tokens_details.reasoning_tokens`` from OpenRouter).
            Zero for models that do not report reasoning.
        cached_tokens: Prompt tokens served from cache
            (``prompt_tokens_details.cached_tokens``). Zero when nothing was
            cached.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        """Component-wise sum, so usage can be accumulated across calls/stages.

        ``Usage()`` is the identity, so ``sum(iterable, start=Usage())`` works.
        """
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )
