"""The single chokepoint for all LLM calls — now provider-agnostic.

Delegates ``complete``/``stream`` to the configured provider (Anthropic, OpenAI, Gemini,
Ollama, or any OpenAI-compatible endpoint) and implements structured output (``parse``)
generically via JSON-schema prompting so it behaves identically across providers. Callers
check ``available`` and fall back to deterministic behaviour when no provider is usable.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from lstm_forecast.ai.providers import LLMProvider, build_provider
from lstm_forecast.config import AISettings, get_settings

T = TypeVar("T", bound=BaseModel)


class AIUnavailableError(RuntimeError):
    """Raised when an AI call is attempted but no provider is available."""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response (tolerates ``` fences/prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
    if candidate is None:
        raise ValueError("No JSON object found in model response.")
    return json.loads(candidate)


class AIClient:
    """Thin wrapper around an :class:`LLMProvider` with graceful no-key handling."""

    def __init__(self, settings: AISettings | None = None) -> None:
        self.settings = settings or get_settings().ai
        self.provider: LLMProvider = build_provider(self.settings)

    @property
    def available(self) -> bool:
        """True when the configured provider can actually be called."""
        return self.settings.enabled and self.provider.available

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def _require(self) -> LLMProvider:
        if not self.available:
            raise AIUnavailableError(
                f"AI provider '{self.settings.provider}' is unavailable. Configure a key "
                "(or run Ollama locally) and install the matching extra."
            )
        return self.provider

    def complete(self, *, system: str, messages: list[dict[str, str]],
                 max_tokens: int | None = None) -> str:
        """Non-streaming completion."""
        return self._require().complete(
            system=system, messages=messages, max_tokens=max_tokens or self.settings.max_tokens
        )

    def stream(self, *, system: str, messages: list[dict[str, str]],
               max_tokens: int | None = None) -> Iterator[str]:
        """Stream text deltas (true streaming where the provider supports it)."""
        yield from self._require().stream(
            system=system, messages=messages, max_tokens=max_tokens or self.settings.max_tokens
        )

    def parse(self, *, system: str, user: str, schema: type[T],
              max_tokens: int | None = None, retries: int = 1) -> T:
        """Provider-agnostic structured output validated against a Pydantic ``schema``.

        Prompts the model to emit JSON matching the schema, extracts and validates it, and
        retries once with the validation error fed back if the first attempt is malformed.
        """
        provider = self._require()
        schema_json = json.dumps(schema.model_json_schema())
        sys_prompt = (
            f"{system}\n\nReturn ONLY a single JSON object that conforms to this JSON Schema. "
            f"No prose, no markdown fences.\nJSON Schema:\n{schema_json}"
        )
        messages = [{"role": "user", "content": user}]
        last_err: Exception | None = None
        for _ in range(retries + 1):
            text = provider.complete(
                system=sys_prompt, messages=messages, max_tokens=max_tokens or self.settings.max_tokens
            )
            try:
                return schema.model_validate(_extract_json(text))
            except (ValueError, ValidationError) as exc:
                last_err = exc
                messages = [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That was invalid ({exc}). Return only valid JSON."},
                ]
        raise AIUnavailableError(f"Model did not return valid structured output: {last_err}")
