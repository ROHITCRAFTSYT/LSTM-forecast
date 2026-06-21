"""Service layer: turn API requests into library calls (no business logic in routes)."""

from __future__ import annotations

import hashlib
import json
import threading

import pandas as pd

from lstm_forecast import __version__
from lstm_forecast.api.schemas import (
    ForecastRequest,
    ForecastResponse,
    IntervalPoint,
    SeriesInput,
)
from lstm_forecast.data import add_finance_features, load_prices
from lstm_forecast.forecasting.forecaster import Forecaster, ForecastResult
from lstm_forecast.transforms import default_finance_transformer


def _series_to_frame(series: SeriesInput) -> pd.DataFrame:
    """Resolve a :class:`SeriesInput` to a tidy frame with at least a 'close' column."""
    if series.ticker:
        return load_prices(series.ticker, allow_synthetic_fallback=series.allow_synthetic)
    index = (
        pd.to_datetime(series.dates)
        if series.dates
        else pd.RangeIndex(len(series.values or []))
    )
    return pd.DataFrame({"close": series.values}, index=index)


def build_forecaster(req: ForecastRequest) -> Forecaster:
    """Construct a configured :class:`Forecaster` from a forecast request."""
    df = _series_to_frame(req.series)
    if req.use_features:
        feat = add_finance_features(df, fourier_periods=(5.0,))
        exog = feat.drop(columns=["close"])
        target, dates = feat["close"], feat.index
    else:
        target, dates, exog = df["close"], df.index, None
    return Forecaster(
        y=target,
        current_dates=dates,
        future_dates=req.horizon,
        test_length=req.test_length,
        exog=exog,
        name="lstm",
    )


def run_forecast(
    req: ForecastRequest,
    *,
    run_backtest: bool = False,
    backtest_windows: int = 10,
) -> tuple[Forecaster, ForecastResult]:
    """Build, fit and forecast; optionally with dynamic (backtested) intervals."""
    f = build_forecaster(req)
    transformer, reverter = default_finance_transformer(seasonal_period=req.seasonal_period)
    f.attach_transformer(transformer, reverter)

    if req.use_rag:
        # Build the analog index from the transformed training portion (leakage-safe).
        import numpy as np

        from lstm_forecast.rag import build_analog_retriever

        split = f.y.size - f.test_length
        transformer.fit(f.y[:split], np.arange(split))
        ref = transformer.transform(f.y[:split], np.arange(split))
        f.attach_retriever(build_analog_retriever(ref, window_len=req.lags))

    from lstm_forecast.forecasting.forecaster import ModelSpec

    spec = ModelSpec(
        lags=req.lags, hidden_size=req.hidden_size, epochs=req.epochs, ensemble=req.ensemble
    )
    result = f.fit_predict(
        spec,
        alpha=req.alpha,
        run_backtest=run_backtest,
        backtest_windows=backtest_windows,
    )
    return f, result


def to_response(req: ForecastRequest, result: ForecastResult, *, insights: str | None) -> ForecastResponse:
    points = [
        IntervalPoint(date=str(d)[:10], point=float(p), lower=float(lo), upper=float(hi))
        for d, p, lo, hi in zip(
            result.future_dates, result.point, result.lower, result.upper, strict=False
        )
    ]
    best = result.metrics_frame().index[0] if result.metrics else None
    return ForecastResponse(
        name="lstm",
        horizon=req.horizon,
        alpha=result.alpha,
        forecast=points,
        metrics=result.metrics,
        interval=result.interval,
        significance=result.significance,
        best_model=str(best) if best is not None else None,
        insights=insights,
    )


# --------------------------------------------------------------------------- #
# Trained-model cache                                                          #
# --------------------------------------------------------------------------- #
# An in-memory cache of fitted Forecasters keyed by a stable signature of the
# request fields that affect training. Reusing a fitted model lets repeated
# requests skip (re)training and go straight to ``forecast_future``. This is a
# simple per-process cache; a multi-process deployment would persist fitted
# models to ``settings.cache_dir`` (via ``Forecaster.save``/``load``) or an
# external store instead.
_MODEL_CACHE: dict[str, tuple[Forecaster, ForecastResult]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def request_cache_key(req: ForecastRequest) -> str:
    """Compute a stable cache key from the training-relevant request fields.

    Fields that do not change the fitted model (e.g. ``include_insights``) are
    excluded so cosmetic differences still hit the cache.
    """
    payload = {
        "series": req.series.model_dump(),
        "horizon": req.horizon,
        "test_length": req.test_length,
        "lags": req.lags,
        "hidden_size": req.hidden_size,
        "epochs": req.epochs,
        "ensemble": req.ensemble,
        "alpha": req.alpha,
        "seasonal_period": req.seasonal_period,
        "use_features": req.use_features,
        "use_rag": req.use_rag,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_forecast_cached(req: ForecastRequest) -> tuple[Forecaster, ForecastResult]:
    """Forecast, reusing a cached fitted model when the request signature matches.

    On a cache hit the stored :class:`Forecaster` is reused via
    :meth:`Forecaster.forecast_future` (no retraining); the cached
    :class:`ForecastResult` (with its baseline metrics) is returned unchanged so
    the response is identical to the first call. On a miss the model is trained
    via :func:`run_forecast` and cached.
    """
    key = request_cache_key(req)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
    if cached is not None:
        f, result = cached
        # Re-run only the (cheap) future projection to confirm the model is usable.
        f.forecast_future(alpha=req.alpha)
        return f, result

    f, result = run_forecast(req)
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = (f, result)
    return f, result


def clear_model_cache() -> None:
    """Empty the in-memory model cache (mainly for tests)."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def version() -> str:
    return __version__
