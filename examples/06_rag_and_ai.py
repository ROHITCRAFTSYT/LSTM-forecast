"""Retrieval-augmented forecasting + LLM AI + cross-validated tuning.

Demonstrates the full "LLM proposes, data decides" loop:
1. build an analog-window RAG index from training data and condition the LSTM on it,
2. ask Claude for a candidate hyperparameter grid (template grid if no API key),
3. cross-validate that grid and adopt the winner,
4. forecast, benchmark (incl. a Diebold-Mariano significance test), and explain it.

    python examples/06_rag_and_ai.py AAPL
"""

from __future__ import annotations

import sys

import numpy as np

from lstm_forecast import Forecaster
from lstm_forecast.ai import generate_insights, suggest_tuning
from lstm_forecast.data import load_prices
from lstm_forecast.forecasting.forecaster import ModelSpec
from lstm_forecast.forecasting.tuning import specs_from_suggestion
from lstm_forecast.rag import build_analog_retriever
from lstm_forecast.transforms import default_finance_transformer


def main(ticker: str = "AAPL") -> None:
    df = load_prices(ticker, allow_synthetic_fallback=True)
    f = Forecaster(y=df["close"], current_dates=df.index, future_dates=21, test_length=42)

    transformer, reverter = default_finance_transformer(seasonal_period=5)
    f.attach_transformer(transformer, reverter)

    # RAG: build the analog index from the transformed *training* portion only (leakage-safe).
    split = f.y.size - f.test_length
    transformer.fit(f.y[:split], np.arange(split))
    ref = transformer.transform(f.y[:split], np.arange(split))
    f.attach_retriever(build_analog_retriever(ref, window_len=21, k=8))

    # LLM proposes a grid → cross-validation picks the winner.
    suggestion = suggest_tuning(f.y)
    print("Claude/template suggested transforms:", suggestion.recommended_transforms)
    specs = specs_from_suggestion(suggestion, base=ModelSpec(epochs=60, ensemble=2))
    report = f.tune(specs, k=2)
    best = report["best_spec"]
    print(
        f"Best CV config: lags={best.lags} hidden={best.hidden_size} "
        f"(CV RMSE={report['best_cv_rmse']:.4f})"
    )

    result = f.fit_predict(best)

    print(f"\n=== {ticker} — benchmark (RMSE-sorted) ===")
    print(result.metrics_frame().round(4).to_string())
    dm = result.significance.get("vs_naive", {})
    if dm:
        print(f"\nDiebold-Mariano vs naive: winner={dm['winner']}, p={dm['p_value']:.3f}")

    print("\n=== AI insights ===")
    print(generate_insights(result, label=ticker))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "AAPL")
