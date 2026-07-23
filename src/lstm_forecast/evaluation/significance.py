"""Statistical significance of forecast-accuracy differences (Diebold-Mariano).

A lower RMSE than a baseline is only meaningful if the difference is statistically
significant. The Diebold-Mariano test compares two forecasts' loss differentials; the
Harvey-Leybourne-Newbold small-sample correction makes it usable on short test windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DMResult:
    """Diebold-Mariano test outcome."""

    statistic: float
    p_value: float
    better: str  # "a", "b", or "tie"
    significant: bool  # p < 0.05

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "better": self.better,
            "significant": self.significant,
        }


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    *,
    horizon: int = 1,
    power: float = 2.0,
    alpha: float = 0.05,
) -> DMResult:
    """Two-sided Diebold-Mariano test on two error series (HLN-corrected).

    Parameters
    ----------
    errors_a, errors_b:
        Forecast errors (actual - prediction) for models A and B over the same points.
    horizon:
        Forecast horizon ``h`` (controls the autocovariance lags used).
    power:
        Loss power (2 = squared error, 1 = absolute error).
    alpha:
        Significance level for the ``significant`` flag.

    Returns
    -------
    DMResult with the statistic, two-sided p-value, which model is better, and whether the
    difference is significant. ``better`` is "a" if A has lower loss, else "b" (or "tie").
    """
    ea = np.asarray(errors_a, dtype=float).ravel()
    eb = np.asarray(errors_b, dtype=float).ravel()
    if ea.shape != eb.shape:
        raise ValueError(f"error series must match: {ea.shape} vs {eb.shape}")
    n = ea.size
    if n < 4:
        return DMResult(
            statistic=float("nan"), p_value=float("nan"), better="tie", significant=False
        )

    # Loss differential d_t = L(e_a) - L(e_b); positive mean → A worse than B.
    d = np.abs(ea) ** power - np.abs(eb) ** power
    d_mean = d.mean()

    # Newey-West-style variance using autocovariances up to lag h-1.
    def autocov(k: int) -> float:
        return float(np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n) if k else float(d.var())

    gamma0 = autocov(0)
    var_d = gamma0 + 2 * sum(autocov(k) for k in range(1, horizon))
    if var_d <= 0:
        return DMResult(
            statistic=float("nan"), p_value=float("nan"), better="tie", significant=False
        )

    dm = d_mean / np.sqrt(var_d / n)
    # Harvey, Leybourne & Newbold (1997) small-sample correction.
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    dm_corrected = dm * hln
    p_value = 2 * (1 - stats.t.cdf(abs(dm_corrected), df=n - 1))

    better = "tie"
    if p_value < alpha:
        # d_mean > 0 ⇒ A's loss is higher ⇒ B is the better model.
        better = "b" if d_mean > 0 else "a"
    return DMResult(
        statistic=float(dm_corrected),
        p_value=float(p_value),
        better=better,
        significant=bool(p_value < alpha),
    )
