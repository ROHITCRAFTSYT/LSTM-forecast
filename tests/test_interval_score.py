"""Winkler interval score — a proper scoring rule for prediction intervals."""

from __future__ import annotations

import numpy as np

from lstm_forecast.evaluation.metrics import interval_metrics, interval_score


def test_inside_interval_equals_mean_width():
    y = np.array([5.0, 5.0])
    lo = np.array([0.0, 0.0])
    hi = np.array([10.0, 10.0])
    # all inside -> score is just the width
    assert interval_score(y, lo, hi, alpha=0.1) == 10.0


def test_miss_is_penalized_proportional_to_distance():
    y = np.array([20.0])       # 10 above the upper bound
    lo = np.array([0.0])
    hi = np.array([10.0])
    # width 10 + (2/alpha)*(y-hi) = 10 + 20*10 = 210 at alpha=0.1
    assert interval_score(y, lo, hi, alpha=0.1) == 210.0


def test_farther_miss_scores_worse():
    lo, hi = np.array([0.0]), np.array([10.0])
    near = interval_score(np.array([12.0]), lo, hi, alpha=0.1)
    far = interval_score(np.array([30.0]), lo, hi, alpha=0.1)
    assert far > near


def test_narrower_interval_wins_when_both_cover():
    y = np.array([5.0, 5.0, 5.0])
    wide = interval_score(y, np.array([0.0] * 3), np.array([10.0] * 3), alpha=0.1)
    narrow = interval_score(y, np.array([4.0] * 3), np.array([6.0] * 3), alpha=0.1)
    assert narrow < wide  # sharper interval scores better when it still covers


def test_interval_metrics_includes_interval_score():
    y = np.array([1.0, 2.0, 3.0])
    lo = np.array([0.0, 1.0, 2.0])
    hi = np.array([2.0, 3.0, 4.0])
    m = interval_metrics(y, lo, hi, nominal=0.9)
    assert "interval_score" in m
    assert m["interval_score"] >= 0.0
