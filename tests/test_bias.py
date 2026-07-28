"""Forecast bias — the signed mean error that magnitude metrics hide."""

from __future__ import annotations

import numpy as np

from lstm_forecast.evaluation.metrics import bias, mae, point_metrics


def test_unbiased_forecast_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert bias(y, y) == 0.0


def test_over_forecast_is_positive():
    y = np.array([1.0, 2.0, 3.0])
    assert bias(y, y + 2.0) == 2.0


def test_under_forecast_is_negative():
    y = np.array([1.0, 2.0, 3.0])
    assert bias(y, y - 1.5) == -1.5


def test_bias_can_be_small_while_mae_is_large():
    # symmetric errors cancel in bias but not in mae
    y = np.array([0.0, 0.0])
    pred = np.array([5.0, -5.0])
    assert bias(y, pred) == 0.0
    assert mae(y, pred) == 5.0


def test_point_metrics_includes_bias():
    out = point_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert out["bias"] == 0.5
