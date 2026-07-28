"""Directional accuracy — the sign-of-move metric that matters for finance."""

from __future__ import annotations

import numpy as np

from lstm_forecast.evaluation.metrics import directional_accuracy, point_metrics


def test_perfect_direction_scores_one():
    y_true = np.array([1.0, 2.0, 3.0, 2.5])
    y_pred = np.array([1.1, 2.2, 3.3, 2.4])  # same up/up/up/down pattern
    assert directional_accuracy(y_true, y_pred) == 1.0


def test_opposite_direction_scores_zero():
    y_true = np.array([1.0, 2.0, 3.0])   # up, up
    y_pred = np.array([3.0, 2.0, 1.0])   # down, down
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_anchor_includes_the_first_step():
    # last known value 1.0; actual first move is up, predicted first move is down.
    da = directional_accuracy(np.array([2.0, 3.0]), np.array([0.5, 1.5]), last=1.0)
    # transitions: [1->2 up vs 1->0.5 down] wrong, [2->3 up vs 0.5->1.5 up] right -> 0.5
    assert da == 0.5


def test_too_short_without_anchor_is_nan():
    assert np.isnan(directional_accuracy(np.array([5.0]), np.array([5.0])))


def test_flat_matches_only_flat():
    # actual flat then up; pred flat then up -> both transitions agree
    da = directional_accuracy(np.array([1.0, 1.0, 2.0]), np.array([9.0, 9.0, 10.0]))
    assert da == 1.0


def test_point_metrics_includes_dir_acc():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    out = point_metrics(y_true, y_pred, y_train=np.array([0.0, 0.5, 1.0]))
    assert "dir_acc" in out
    assert 0.0 <= out["dir_acc"] <= 1.0
