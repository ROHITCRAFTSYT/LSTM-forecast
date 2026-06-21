"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SeriesInput(BaseModel):
    """Either a ticker to download, or explicit values (+ optional ISO dates)."""

    ticker: str | None = None
    values: list[float] | None = None
    dates: list[str] | None = None
    allow_synthetic: bool = Field(
        default=True,
        description="If a provider download fails, fall back to synthetic data.",
    )

    @model_validator(mode="after")
    def _check_one_source(self) -> SeriesInput:
        if not self.ticker and not self.values:
            raise ValueError("Provide either 'ticker' or 'values'.")
        if self.values is not None and self.dates is not None and len(self.values) != len(self.dates):
            raise ValueError("'values' and 'dates' must be the same length.")
        return self


class ForecastRequest(BaseModel):
    series: SeriesInput
    horizon: int = Field(default=21, ge=1, le=365)
    test_length: int = Field(default=42, ge=2, le=1000)
    lags: int = Field(default=21, ge=2, le=512)
    hidden_size: int = Field(default=64, ge=4, le=512)
    epochs: int = Field(default=60, ge=1, le=1000)
    ensemble: int = Field(default=1, ge=1, le=10, description="Number of seeded models to average.")
    alpha: float = Field(default=0.1, gt=0, lt=1)
    seasonal_period: int | None = Field(default=5, ge=2)
    use_features: bool = Field(default=False, description="Add finance features (multivariate).")
    use_rag: bool = Field(default=False, description="Enable retrieval-augmented features.")
    include_insights: bool = Field(default=False, description="Attach an AI insight narrative.")


class BacktestRequest(ForecastRequest):
    backtest_windows: int = Field(default=10, ge=2, le=100)


class IntervalPoint(BaseModel):
    date: str
    point: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    name: str
    horizon: int
    alpha: float
    forecast: list[IntervalPoint]
    metrics: dict[str, dict[str, float]]
    interval: dict[str, float]
    significance: dict[str, object] = Field(default_factory=dict)
    best_model: str | None = None
    insights: str | None = None


class TransferRequest(BaseModel):
    source: SeriesInput
    target: SeriesInput
    horizon: int = Field(default=21, ge=1, le=365)
    test_length: int = Field(default=42, ge=2, le=1000)
    lags: int = Field(default=21, ge=2, le=512)
    epochs: int = Field(default=60, ge=1, le=1000)


class ChatRequest(BaseModel):
    question: str
    forecast: ForecastRequest | None = Field(
        default=None,
        description="Optional: run this forecast first so the assistant can ground answers in it.",
    )


class ChatResponse(BaseModel):
    answer: str
    ai_enabled: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    ai_enabled: bool
    ai_provider: str
    device: str
