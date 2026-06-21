"""Centralised configuration via pydantic-settings.

Settings are read from environment variables (and an optional ``.env`` file) using the
``LSTM_FORECAST_`` prefix. Nested settings use a ``__`` delimiter, e.g.
``LSTM_FORECAST_AI__MODEL=claude-opus-4-8``.

Nothing here requires any secret to be present — the AI sub-config simply has an empty
API key by default, and the rest of the system runs fully without it.
"""

from __future__ import annotations

import functools
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Provider → its default model id, used when LSTM_FORECAST_AI__MODEL is left unset.
PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o",
    "google": "gemini-1.5-pro",
    "ollama": "llama3.1",
    "openai_compatible": "gpt-4o",
}


class AISettings(BaseSettings):
    """Configuration for the (provider-agnostic) AI layer.

    Defaults to Anthropic/Claude, but any provider can be selected via
    ``LSTM_FORECAST_AI__PROVIDER``: ``anthropic``, ``openai``, ``google`` (Gemini),
    ``ollama`` (local, no key needed), or ``openai_compatible`` (any OpenAI-style endpoint
    such as OpenRouter, Together, Groq, vLLM — set ``base_url``). Empty/unavailable config
    disables AI features and the rest of the system is unaffected.
    """

    model_config = SettingsConfigDict(
        env_prefix="LSTM_FORECAST_AI__", extra="ignore", populate_by_name=True
    )

    provider: str = Field(default="anthropic", description="anthropic|openai|google|ollama|openai_compatible")
    api_key: str = Field(
        default="",
        # Accept the prefixed name first, then common provider env vars for convenience.
        validation_alias=AliasChoices(
            "LSTM_FORECAST_AI__API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        ),
        description="API key for the selected provider. Not required for the 'ollama' provider.",
    )
    model: str = Field(default="", description="Model id. Empty → provider's default model.")
    base_url: str = Field(
        default="",
        description="Custom endpoint for 'openai_compatible'/'ollama' providers.",
    )
    effort: str = Field(default="high", description="Reasoning effort (Anthropic): low|medium|high|max.")
    max_tokens: int = Field(default=4096, ge=1)
    request_timeout: float = Field(default=60.0, gt=0)

    @property
    def resolved_model(self) -> str:
        """The model id to use, falling back to the provider's default."""
        return self.model.strip() or PROVIDER_DEFAULT_MODELS.get(self.provider, "")

    @property
    def enabled(self) -> bool:
        """True when the provider is usable (a key is set, or it's a keyless local provider)."""
        if self.provider == "ollama":
            return True  # local server, no key required
        return bool(self.api_key.strip())


class APISettings(BaseSettings):
    """Configuration for the FastAPI service."""

    model_config = SettingsConfigDict(env_prefix="LSTM_FORECAST_API__", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


class Settings(BaseSettings):
    """Top-level application settings."""

    model_config = SettingsConfigDict(
        env_prefix="LSTM_FORECAST_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cache_dir: Path = Field(default=Path(".cache"))
    device: str = Field(default="auto", description="torch device: auto|cpu|cuda|mps")
    seed: int = Field(default=20)

    ai: AISettings = Field(default_factory=AISettings)
    api: APISettings = Field(default_factory=APISettings)

    def ensure_cache_dir(self) -> Path:
        """Create the cache directory if necessary and return it."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Cached so repeated calls are cheap and consistent within a process. Tests that need
    to vary the environment can call ``get_settings.cache_clear()``.
    """
    return Settings()
