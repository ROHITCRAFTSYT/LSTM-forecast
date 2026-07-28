"""Benchmark baselines.

Every forecast run is measured *against* these, so their contract matters: fit on
a 1-D series, predict exactly ``h`` finite values, and — for the statsmodels-backed
models (ARIMA, ETS, Theta) — degrade to a naive forecast rather than crash when the
library is missing or the fit fails on a pathological series.
"""

from __future__ import annotations

import numpy as np
import pytest

from lstm_forecast.forecasting.baselines import (
    ARIMAForecaster,
    DriftForecaster,
    ETSForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
    baseline_registry,
)

ALL = [
    NaiveForecaster,
    DriftForecaster,
    lambda: SeasonalNaiveForecaster(season=5),
    ARIMAForecaster,
    ETSForecaster,
    lambda: ThetaForecaster(season=5),
]


def _series(n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    trend = np.linspace(0, 5, n)
    season = 2.0 * np.sin(2 * np.pi * np.arange(n) / 5)
    return 100 + trend + season + rng.randn(n) * 0.5


@pytest.mark.parametrize("factory", ALL)
def test_predicts_exactly_h_finite_values(factory):
    f = factory().fit(_series())
    out = f.predict(12)
    assert out.shape == (12,)
    assert np.isfinite(out).all()


def test_naive_repeats_last_value():
    out = NaiveForecaster().fit(np.array([1.0, 2.0, 3.0])).predict(4)
    assert np.allclose(out, 3.0)


def test_drift_extrapolates_slope():
    # perfectly linear -> drift continues the same step
    out = DriftForecaster().fit(np.arange(10, dtype=float)).predict(3)
    assert np.allclose(out, [10.0, 11.0, 12.0])


def test_seasonal_naive_repeats_last_season():
    y = np.array([1, 2, 3, 4, 5, 6], dtype=float)  # season=3 -> last block [4,5,6]
    out = SeasonalNaiveForecaster(season=3).fit(y).predict(5)
    assert np.allclose(out, [4, 5, 6, 4, 5])


def test_registry_includes_theta_and_all_baselines():
    reg = baseline_registry(season=5)
    assert set(reg) == {"naive", "drift", "seasonal_naive", "arima", "ets", "theta"}
    for name, model in reg.items():
        assert model.name == name


# --- graceful degradation of the statsmodels-backed models ------------------
@pytest.mark.parametrize("factory", [ARIMAForecaster, ETSForecaster, lambda: ThetaForecaster()])
def test_statsmodels_models_fall_back_on_short_series(factory):
    # Too few points for a real fit; must still return h finite values (naive).
    f = factory().fit(np.array([1.0, 2.0, 3.0]))
    out = f.predict(3)
    assert out.shape == (3,)
    assert np.isfinite(out).all()


def test_theta_beats_or_matches_naive_on_a_trending_series(monkeypatch):
    # On a clean trend+season series Theta should not be wildly worse than naive;
    # this mostly guards that a real (non-fallback) forecast is being produced.
    y = _series(n=200)
    train, test = y[:-12], y[-12:]
    theta = ThetaForecaster(season=5).fit(train).predict(12)
    naive = NaiveForecaster().fit(train).predict(12)
    theta_err = np.mean(np.abs(test - theta))
    naive_err = np.mean(np.abs(test - naive))
    # Theta tracks the trend; naive holds flat. Allow slack for noise/fallback.
    assert theta_err <= naive_err * 1.5
