"""Provider-agnostic LLM backends: Anthropic, OpenAI, Google (Gemini), Ollama, or any
OpenAI-compatible endpoint.

Each provider implements ``complete`` and ``stream`` over a normalised message format
(``[{"role": "user"|"assistant", "content": str}]`` plus a separate ``system`` string).
Structured output (``parse``) is handled generically in :mod:`lstm_forecast.ai.client`
via JSON-schema prompting, so it works identically across every provider.

All imports are lazy and optional — a provider is simply "unavailable" if its SDK isn't
installed or its key isn't set, and the AI layer falls back gracefully.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from lstm_forecast.config import AISettings

# Default local endpoint for Ollama's OpenAI-compatible API.
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class LLMProvider(ABC):
    """Base class for an LLM backend."""

    name: str = "base"

    def __init__(self, settings: AISettings) -> None:
        self.settings = settings
        self._client: Any = None

    @property
    @abstractmethod
    def available(self) -> bool:
        """True when this provider can actually be called (SDK present + creds set)."""

    @abstractmethod
    def complete(self, *, system: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        """Return a single completion string."""

    def stream(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int
    ) -> Iterator[str]:
        """Stream text deltas. Default: yield the full completion once."""
        yield self.complete(system=system, messages=messages, max_tokens=max_tokens)


class AnthropicProvider(LLMProvider):
    """Claude via the official Anthropic SDK (adaptive thinking + effort)."""

    name = "anthropic"

    def _ensure(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self.settings.api_key, timeout=self.settings.request_timeout
            )
        return self._client

    @property
    def available(self) -> bool:
        if not self.settings.api_key.strip():
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def complete(self, *, system: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        resp = self._ensure().messages.create(
            model=self.settings.resolved_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.settings.effort},
            thinking={"type": "adaptive"},
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    def stream(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int
    ) -> Iterator[str]:
        with self._ensure().messages.stream(
            model=self.settings.resolved_model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.settings.effort},
        ) as stream:
            yield from stream.text_stream


class OpenAIProvider(LLMProvider):
    """OpenAI, or any OpenAI-compatible endpoint (OpenRouter, Together, Groq, vLLM, Ollama).

    ``base_url`` selects the endpoint; ``keyless`` allows local servers (Ollama) with no key.
    """

    name = "openai"

    def __init__(self, settings: AISettings, *, base_url: str | None = None,
                 keyless: bool = False) -> None:
        super().__init__(settings)
        self._base_url = base_url or (settings.base_url or None)
        self._keyless = keyless

    def _ensure(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                api_key=self.settings.api_key or ("ollama" if self._keyless else None),
                base_url=self._base_url,
                timeout=self.settings.request_timeout,
            )
        return self._client

    @property
    def available(self) -> bool:
        if not self._keyless and not self.settings.api_key.strip():
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _msgs(self, system: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, *messages]

    def complete(self, *, system: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        resp = self._ensure().chat.completions.create(
            model=self.settings.resolved_model,
            messages=self._msgs(system, messages),
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int
    ) -> Iterator[str]:
        stream = self._ensure().chat.completions.create(
            model=self.settings.resolved_model,
            messages=self._msgs(system, messages),
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class GoogleProvider(LLMProvider):
    """Gemini via the google-generativeai SDK."""

    name = "google"

    @property
    def available(self) -> bool:
        if not self.settings.api_key.strip():
            return False
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            return False
        return True

    def _model(self, system: str) -> Any:
        import google.generativeai as genai

        genai.configure(api_key=self.settings.api_key)
        return genai.GenerativeModel(self.settings.resolved_model, system_instruction=system)

    @staticmethod
    def _to_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        # Gemini uses role "model" for assistant turns.
        role_map = {"assistant": "model", "user": "user"}
        return [
            {"role": role_map.get(m["role"], "user"), "parts": [m["content"]]}
            for m in messages
        ]

    def complete(self, *, system: str, messages: list[dict[str, str]], max_tokens: int) -> str:
        model = self._model(system)
        resp = model.generate_content(
            self._to_contents(messages),
            generation_config={"max_output_tokens": max_tokens},
        )
        return resp.text or ""

    def stream(
        self, *, system: str, messages: list[dict[str, str]], max_tokens: int
    ) -> Iterator[str]:
        model = self._model(system)
        for chunk in model.generate_content(
            self._to_contents(messages),
            generation_config={"max_output_tokens": max_tokens},
            stream=True,
        ):
            if getattr(chunk, "text", None):
                yield chunk.text


def build_provider(settings: AISettings) -> LLMProvider:
    """Instantiate the configured provider (defaults to Anthropic)."""
    provider = settings.provider.lower()
    if provider == "anthropic":
        return AnthropicProvider(settings)
    if provider in ("openai", "openai_compatible"):
        return OpenAIProvider(settings)
    if provider == "google":
        return GoogleProvider(settings)
    if provider == "ollama":
        return OpenAIProvider(
            settings, base_url=settings.base_url or OLLAMA_DEFAULT_BASE_URL, keyless=True
        )
    # Unknown provider → treat as disabled rather than crashing.
    return AnthropicProvider(settings)
