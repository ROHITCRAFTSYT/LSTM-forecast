"""Benchmark baselines so every run reports the model *against* simple alternatives.

Honesty constraint: "outperform" must be measured, not asserted. These baselines are
evaluated on the same held-out test set as the LSTM. Statistical baselines (ARIMA/ETS)
degrade gracefully to a naive forecast if statsmodels is unavailable or fails.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BaseForecaster(Protocol):
    """Minimal baseline interface: fit on a 1-D series, predict ``h`` steps ahead."""

    name: str

    def fit(self, y: np.ndarray) -> BaseForecaster: ...

    def predict(self, h: int) -> np.ndarray: ...


class NaiveForecaster:
    """Repeat the last observed value."""

    name = "naive"

    def __init__(self) -> None:
        self._last = 0.0

    def fit(self, y: np.ndarray) -> NaiveForecaster:
        self._last = float(np.asarray(y).ravel()[-1])
        return self

    def predict(self, h: int) -> np.ndarray:
        return np.full(h, self._last, dtype=float)


class DriftForecaster:
    """Linear drift: extend the average slope between first and last observation."""

    name = "drift"

    def __init__(self) -> None:
        self._last = 0.0
        self._slope = 0.0

    def fit(self, y: np.ndarray) -> DriftForecaster:
        y = np.asarray(y, dtype=float).ravel()
        self._last = float(y[-1])
        self._slope = float((y[-1] - y[0]) / max(len(y) - 1, 1))
        return self

    def predict(self, h: int) -> np.ndarray:
        return self._last + self._slope * np.arange(1, h + 1)


class SeasonalNaiveForecaster:
    """Repeat the last full season."""

    name = "seasonal_naive"

    def __init__(self, season: int = 5) -> None:
        self.season = season
        self._tail: np.ndarray = np.zeros(season)

    def fit(self, y: np.ndarray) -> SeasonalNaiveForecaster:
        y = np.asarray(y, dtype=float).ravel()
        s = min(self.season, len(y))
        self._tail = y[-s:]
        return self

    def predict(self, h: int) -> np.ndarray:
        reps = int(np.ceil(h / len(self._tail)))
        return np.tile(self._tail, reps)[:h]


class ARIMAForecaster:
    """ARIMA via statsmodels SARIMAX, with a naive fallback."""

    name = "arima"

    def __init__(self, order: tuple[int, int, int] = (2, 1, 2)) -> None:
        self.order = order
        self._fallback = NaiveForecaster()
        self._result = None

    def fit(self, y: np.ndarray) -> ARIMAForecaster:
        y = np.asarray(y, dtype=float).ravel()
        self._fallback.fit(y)
        try:
            from statsmodels.tsa.arima.model import ARIMA

            self._result = ARIMA(y, order=self.order).fit()
        except Exception:
            self._result = None
        return self

    def predict(self, h: int) -> np.ndarray:
        if self._result is None:
            return self._fallback.predict(h)
        try:
            return np.asarray(self._result.forecast(steps=h), dtype=float)
        except Exception:
            return self._fallback.predict(h)


class ETSForecaster:
    """Exponential smoothing (Holt) via statsmodels, with a naive fallback."""

    name = "ets"

    def __init__(
        self,
        trend: str | None = "add",
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
    ) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._fallback = NaiveForecaster()
        self._result = None

    def fit(self, y: np.ndarray) -> ETSForecaster:
        y = np.asarray(y, dtype=float).ravel()
        self._fallback.fit(y)
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            self._result = ExponentialSmoothing(
                y,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            ).fit()
        except Exception:
            self._result = None
        return self

    def predict(self, h: int) -> np.ndarray:
        if self._result is None:
            return self._fallback.predict(h)
        try:
            return np.asarray(self._result.forecast(h), dtype=float)
        except Exception:
            return self._fallback.predict(h)


class ThetaForecaster:
    """Theta method (Assimakopoulos & Nikolopoulos) via statsmodels, naive fallback.

    The classic theta=2 model that won the M3 competition: it deseasonalizes, splits
    the series into theta-lines, extrapolates the long-term component with linear
    regression and the short-term curvature with simple exponential smoothing, then
    recombines. A strong, parameter-light benchmark the LSTM must beat to justify its
    extra complexity — often the hardest of the classical baselines to outperform.
    """

    name = "theta"

    def __init__(self, season: int = 5, deseasonalize: bool = True) -> None:
        self.season = season
        self.deseasonalize = deseasonalize
        self._fallback = NaiveForecaster()
        self._result = None

    def fit(self, y: np.ndarray) -> ThetaForecaster:
        y = np.asarray(y, dtype=float).ravel()
        self._fallback.fit(y)
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel

            period = self.season if self.season and self.season > 1 else None
            # Deseasonalization needs at least two full seasons; disable it otherwise
            # so short series don't raise instead of forecasting.
            deseason = (
                self.deseasonalize and period is not None and len(y) >= 2 * period
            )
            self._result = ThetaModel(y, period=period, deseasonalize=deseason).fit()
        except Exception:
            self._result = None
        return self

    def predict(self, h: int) -> np.ndarray:
        if self._result is None:
            return self._fallback.predict(h)
        try:
            return np.asarray(self._result.forecast(h), dtype=float)
        except Exception:
            return self._fallback.predict(h)


def baseline_registry(season: int = 5) -> dict[str, BaseForecaster]:
    """Return a fresh set of baseline forecasters keyed by name."""
    return {
        "naive": NaiveForecaster(),
        "drift": DriftForecaster(),
        "seasonal_naive": SeasonalNaiveForecaster(season=season),
        "arima": ARIMAForecaster(),
        "ets": ETSForecaster(),
        "theta": ThetaForecaster(season=season),
    }
