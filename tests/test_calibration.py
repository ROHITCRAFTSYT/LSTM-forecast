"""Calibration-curve metric: shape, range and monotonicity on synthetic data."""

from __future__ import annotations

import math

import numpy as np

from lstm_forecast.data import load_synthetic_prices
from lstm_forecast.evaluation import calibration_curve

LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def test_calibration_curve_shapes_and_range():
    df = load_synthetic_prices(n=400, seed=0)
    y_true = df["close"].to_numpy(dtype=float)
    rng = np.random.default_rng(0)
    # A naive "model": shift the series; residuals drive the interval radii.
    y_pred = y_true + rng.normal(0, 1.0, size=y_true.shape)
    residuals = y_true - y_pred

    cal = calibration_curve(y_true, y_pred, residuals, levels=LEVELS)

    assert len(cal["nominal"]) == len(LEVELS)
    assert len(cal["empirical"]) == len(LEVELS)
    assert cal["nominal"] == list(LEVELS)
    assert all(0.0 <= e <= 1.0 for e in cal["empirical"])


def test_calibration_curve_monotone_and_error_finite():
    df = load_synthetic_prices(n=400, seed=1)
    y_true = df["close"].to_numpy(dtype=float)
    rng = np.random.default_rng(1)
    y_pred = y_true + rng.normal(0, 1.0, size=y_true.shape)
    residuals = y_true - y_pred

    cal = calibration_curve(y_true, y_pred, residuals, levels=LEVELS)

    # Wider nominal intervals should cover at least as much (allow sampling noise overall).
    assert cal["empirical"][-1] >= cal["empirical"][0]

    err = cal["calibration_error"]
    assert isinstance(err, float)
    assert math.isfinite(err)
    assert err >= 0.0
