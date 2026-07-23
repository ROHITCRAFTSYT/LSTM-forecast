"""lstm-forecast: production-grade LSTM time-series forecasting with RAG and Claude AI.

A custom PyTorch forecasting core with attention and probabilistic (quantile + conformal)
heads, a leakage-safe reversible transform pipeline, optional retrieval-augmented
forecasting, honest baseline benchmarking, and a provider-agnostic LLM layer for insights,
a chat assistant, and cross-validated tuning.

Typical usage::

    from lstm_forecast import Forecaster, Pipeline
    from lstm_forecast.transforms import default_finance_transformer
    from lstm_forecast.data import load_prices

    series = load_prices("AAPL")
    f = Forecaster(y=series["close"], current_dates=series.index, future_dates=21, test_length=42)
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    pipe = Pipeline(transformer=transformer, reverter=reverter)
    f = pipe.fit_predict(f, lags=21)
    print(f.results.metrics_frame())
"""

from __future__ import annotations

from lstm_forecast.config import Settings, get_settings
from lstm_forecast.forecasting.forecaster import Forecaster, ForecastResult
from lstm_forecast.pipelines import Pipeline

__all__ = [
    "ForecastResult",
    "Forecaster",
    "Pipeline",
    "Settings",
    "__version__",
    "get_settings",
]


def _resolve_version() -> str:
    """Single-source the version from installed package metadata (pyproject is the source).

    Falls back to a sentinel when running from a source tree that isn't installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("lstm-forecast")
    except PackageNotFoundError:  # pragma: no cover - only when not installed
        return "0.0.0+unknown"


__version__ = _resolve_version()
