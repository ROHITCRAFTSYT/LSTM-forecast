"""Point, probabilistic and interval forecast metrics."""

from __future__ import annotations

import numpy as np


def _align(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%), ignoring zero-valued targets."""
    y_true, y_pred = _align(y_true, y_pred)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE (%)."""
    y_true, y_pred = _align(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int = 1) -> float:
    """Mean absolute scaled error, scaled by the in-sample seasonal naive error."""
    y_true, y_pred = _align(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if y_train.size <= season:
        return float("nan")
    scale = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def pinball(y_true: np.ndarray, q_pred: np.ndarray, quantile: float) -> float:
    """Pinball loss for a single quantile forecast."""
    y_true, q_pred = _align(y_true, q_pred)
    errors = y_true - q_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1) * errors)))


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations within ``[lower, upper]``."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    return float(np.mean(upper - lower))


def point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    season: int = 1,
) -> dict[str, float]:
    """Bundle the common point metrics into a dict."""
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, season=season)
    return out


def interval_metrics(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    nominal: float = 0.9,
) -> dict[str, float]:
    """Coverage, mean width and a coverage-vs-nominal gap."""
    cov = coverage(y_true, lower, upper)
    return {
        "coverage": cov,
        "nominal": nominal,
        "coverage_gap": cov - nominal,
        "mean_width": mean_interval_width(lower, upper),
    }


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> dict[str, list[float] | float]:
    """Empirical coverage of symmetric conformal-style intervals across nominal levels.

    For each nominal level ``p`` a symmetric interval ``[y_pred - r, y_pred + r]`` is built,
    where the radius ``r`` is the finite-sample-corrected ``p (1 + 1/n)`` empirical quantile
    of the absolute calibration ``residuals`` (the same construction as
    :func:`lstm_forecast.forecasting.conformal.conformal_quantile`). The empirical coverage
    is the fraction of ``y_true`` falling within that interval. A well-calibrated model has
    empirical coverage close to the nominal level (the diagonal ``y = x``).

    The computation is purely causal: the radius depends only on the supplied calibration
    residuals, never on ``y_true``.

    Parameters
    ----------
    y_true : np.ndarray
        Observed target values.
    y_pred : np.ndarray
        Point predictions aligned with ``y_true``.
    residuals : np.ndarray
        Calibration residuals (e.g. ``test_actual - test_pred``) used to size intervals.
        Absolute values are taken internally.
    levels : tuple of float, optional
        Nominal coverage levels in ``(0, 1)`` at which to evaluate calibration.

    Returns
    -------
    dict
        Mapping with keys ``'nominal'`` (list of the requested levels), ``'empirical'``
        (list of empirical coverages, one per level) and ``'calibration_error'`` (float,
        the mean absolute gap ``|empirical - nominal|`` across levels).
    """
    y_true, y_pred = _align(y_true, y_pred)
    res = np.abs(np.asarray(residuals, dtype=float).ravel())
    n = res.size
    if n == 0:
        raise ValueError("Need at least one calibration residual.")

    nominal: list[float] = []
    empirical: list[float] = []
    for p in levels:
        level = min(1.0, float(p) * (1 + 1 / n))
        radius = float(np.quantile(res, level, method="higher"))
        cov = float(np.mean(np.abs(y_true - y_pred) <= radius))
        nominal.append(float(p))
        empirical.append(cov)

    error = float(np.mean(np.abs(np.asarray(empirical) - np.asarray(nominal))))
    return {"nominal": nominal, "empirical": empirical, "calibration_error": error}
