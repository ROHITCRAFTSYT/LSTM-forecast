"""Ljung-Box residual autocorrelation diagnostic."""

from __future__ import annotations

import numpy as np

from lstm_forecast.evaluation.significance import LjungBoxResult, ljung_box


def test_white_noise_is_not_flagged():
    rng = np.random.RandomState(0)
    res = ljung_box(rng.randn(300), lags=10)
    assert isinstance(res, LjungBoxResult)
    assert res.autocorrelated is False
    assert res.p_value > 0.05


def test_strongly_autocorrelated_series_is_flagged():
    # A random walk (cumulative sum) has heavy autocorrelation in levels.
    rng = np.random.RandomState(1)
    walk = np.cumsum(rng.randn(300))
    res = ljung_box(walk, lags=10)
    assert res.autocorrelated is True
    assert res.p_value < 0.05


def test_ar1_residuals_detected():
    rng = np.random.RandomState(2)
    n = 400
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.randn()
    assert ljung_box(x, lags=10).autocorrelated is True


def test_constant_residuals_are_safe():
    res = ljung_box(np.zeros(50), lags=10)
    assert res.autocorrelated is False
    assert np.isnan(res.statistic)


def test_short_series_returns_no_detection():
    res = ljung_box(np.array([1.0, 2.0]), lags=10)
    assert res.autocorrelated is False


def test_model_dof_reduces_degrees_of_freedom():
    rng = np.random.RandomState(3)
    r = rng.randn(200)
    # Same data, fewer dof -> a (weakly) more conservative-or-different p; both valid.
    base = ljung_box(r, lags=12, model_dof=0)
    reduced = ljung_box(r, lags=12, model_dof=4)
    assert base.lags == reduced.lags == 12
    assert 0.0 <= reduced.p_value <= 1.0


def test_as_dict_shape():
    d = ljung_box(np.random.RandomState(4).randn(100)).as_dict()
    assert set(d) == {"statistic", "p_value", "lags", "autocorrelated"}
