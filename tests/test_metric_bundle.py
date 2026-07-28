"""Regression guard on the assembled metric bundles.

point_metrics and interval_metrics feed the benchmark table, the API response
and the AI insights, so their key sets are a public contract. This pins every
key (including the finance additions dir_acc/bias and the proper interval score)
and sanity-checks the values, so an accidental removal or rename is caught.
"""

from __future__ import annotations

import numpy as np

from lstm_forecast.evaluation.metrics import interval_metrics, point_metrics


def test_point_metrics_full_key_set_with_train():
    y_true = np.array([10.0, 11.0, 12.0, 11.5])
    y_pred = np.array([10.2, 10.8, 12.1, 11.6])
    y_train = np.array([8.0, 8.5, 9.0, 9.5, 10.0])
    out = point_metrics(y_true, y_pred, y_train=y_train, season=1)
    assert set(out) == {"rmse", "mae", "mape", "smape", "r2", "dir_acc", "bias", "mase"}
    assert out["rmse"] >= out["mae"] >= 0.0
    assert 0.0 <= out["dir_acc"] <= 1.0
    assert -1e9 < out["bias"] < 1e9


def test_point_metrics_without_train_omits_mase_keeps_the_rest():
    out = point_metrics(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
    assert "mase" not in out
    assert {"rmse", "mae", "mape", "smape", "r2", "dir_acc", "bias"} <= set(out)


def test_interval_metrics_full_key_set():
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 1.0, 2.0])
    hi = np.array([2.0, 3.0, 4.0])
    out = interval_metrics(y, lo, hi, nominal=0.9)
    assert set(out) == {"coverage", "nominal", "coverage_gap", "mean_width", "interval_score"}
    assert 0.0 <= out["coverage"] <= 1.0
    assert out["mean_width"] >= 0.0
    assert out["interval_score"] >= 0.0
    assert out["coverage_gap"] == out["coverage"] - out["nominal"]
