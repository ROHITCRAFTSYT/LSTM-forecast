"""Generate the images embedded in the README, from real library output.

Run:  python scripts/make_readme_assets.py
Outputs (committed to the repo):
  assets/forecast_example.png   — forecast with conformal intervals + test forecast
  assets/benchmark_example.png  — model vs baselines test-set RMSE

Deterministic (synthetic data + fixed seed) so the images regenerate identically and the
build needs no network or API keys.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.transforms import default_finance_transformer

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)

PRIMARY = "#1f4e79"
ACCENT = "#c0392b"
TEST = "#e67e22"


def _structured_series(n: int = 500, seed: int = 7) -> pd.Series:
    """A demo series WITH learnable structure (trend + seasonality + autocorrelation).

    Unlike a pure random walk (where naive is near-optimal), this has signal the LSTM can
    actually capture — the right setting to *demonstrate* the model, not to claim it beats
    naive on unpredictable markets.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 100 + 0.05 * t
    seasonal = 6 * np.sin(2 * np.pi * t / 20) + 3 * np.sin(2 * np.pi * t / 5)
    # AR(1) noise so recent context is genuinely informative.
    eps = rng.normal(0, 0.8, n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.6 * ar[i - 1] + eps[i]
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    return pd.Series(trend + seasonal + ar, index=dates, name="close")


def main() -> None:
    series = _structured_series()

    f = Forecaster(
        y=series, current_dates=series.index, future_dates=21, test_length=42
    )
    transformer, reverter = default_finance_transformer(seasonal_period=5)
    result = Pipeline(transformer=transformer, reverter=reverter).fit_predict(
        f, lags=21, hidden_size=48, epochs=120, alpha=0.1
    )

    # ---- Forecast plot -------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 5))
    hist_dates = result.history_dates[-150:]
    hist_vals = result.history_values[-150:]
    ax.plot(hist_dates, hist_vals, color=PRIMARY, lw=1.6, label="history")
    ax.plot(result.future_dates, result.point, color=ACCENT, lw=2.0, label="forecast")
    ax.fill_between(
        result.future_dates, result.lower, result.upper, color=ACCENT, alpha=0.18,
        label="90% conformal interval",
    )
    if result.test_dates is not None and result.test_pred is not None:
        ax.plot(result.test_dates, result.test_pred, "--", color=TEST, lw=1.6,
                label="test forecast")
    ax.set_title("LSTM forecast with conformal intervals (structured demo series)",
                 fontsize=13, weight="bold")
    ax.set_ylabel("price")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(ASSETS / "forecast_example.png", dpi=130)
    plt.close(fig)

    # ---- Benchmark bar chart -------------------------------------------------------------
    frame = result.metrics_frame()
    rmse = frame["rmse"].sort_values(ascending=False)
    colors = [ACCENT if name == "lstm" else "#95a5a6" for name in rmse.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(rmse.index, rmse.to_numpy(), color=colors)
    ax.set_title("Test-set RMSE: model vs baselines (lower is better)", fontsize=12,
                 weight="bold")
    ax.set_xlabel("RMSE")
    for i, v in enumerate(rmse.to_numpy()):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "benchmark_example.png", dpi=130)
    plt.close(fig)

    print("Wrote:")
    print("  assets/forecast_example.png")
    print("  assets/benchmark_example.png")
    print("\nForecast benchmark (RMSE-sorted):")
    print(frame.round(4).to_string())


if __name__ == "__main__":
    main()
