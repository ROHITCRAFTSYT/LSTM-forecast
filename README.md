# lstm-forecast

**Production-grade LSTM time-series forecasting for finance — enhanced with retrieval-augmented forecasting (RAG), Claude-powered insights, and rigorous probabilistic intervals.**

[![CI](https://github.com/ROHITCRAFTSYT/lstm-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/ROHITCRAFTSYT/lstm-forecast/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This project reimagines the classic ["LSTM Model for Time Series, with Code"](https://towardsdatascience.com/) scalecast walkthrough as a **custom PyTorch system** you can actually ship: a clean library, a REST API, and a dashboard — all Dockerized, tested, and documented.

It keeps the article's five capabilities and upgrades each one:

| Capability | This project |
| --- | --- |
| Univariate forecasting | PyTorch LSTM **+ attention**, reversible transforms |
| Multivariate forecasting | exogenous features + engineered finance indicators |
| Probabilistic forecasting | **split-conformal** intervals with coverage guarantees |
| Dynamic probabilistic | **rolling backtest residual matrix** → horizon-aware intervals |
| Transfer learning | `transfer_predict` — reuse a fitted model on new/related series |

…and adds three things the article never had:

- 🔎 **Retrieval-augmented forecasting (RAG)** — index historical *analog* windows and condition the model on "what happened after shapes like the recent past".
- 🤖 **Claude AI layer** — natural-language forecast insights, a RAG **chat assistant** grounded in your docs + run results, and **LLM-assisted hyperparameter tuning** (Claude proposes, cross-validation decides).
- 📊 **Honest benchmarking** — every run is scored against Naive / Drift / Seasonal-Naive / ARIMA / ETS baselines on a held-out test set, so "it beats the baselines" is *measured*, not asserted.

> ⚠️ **Not financial advice.** This is a research/engineering framework. Forecasts are uncertain; markets are not guaranteed to be predictable. See the [model card](docs/model_card.md).

---

## Architecture

```
data ingest → feature engineering → reversible transform pipeline
     → [RAG analog retrieval] ⟶ feature fusion ⟶ PyTorch LSTM(+attention, quantile/conformal head)
     → backtest / conformal intervals → benchmark vs baselines
     → Claude: insights · chat (RAG over docs+results) · tuning suggestions
Surfaces:  Python API   ·   FastAPI service   ·   Streamlit dashboard      (all Dockerized)
```

A clean **core library** (`src/lstm_forecast`) is consumed by the API and dashboard — no business logic lives in the serving layers.

---

## Install

```bash
# Core library (PyTorch model + transforms + conformal + baselines)
pip install -e .

# Add what you need:
pip install -e ".[data]"       # yfinance / stooq data loaders
pip install -e ".[rag]"        # FAISS vector index (RAG; NumPy fallback works without it)
pip install -e ".[ai]"         # Claude insights / chat / tuning
pip install -e ".[api]"        # FastAPI service
pip install -e ".[dashboard]"  # Streamlit dashboard
pip install -e ".[all]"        # everything + dev tooling
```

The library runs **fully offline** with no API keys: data falls back to a synthetic generator, RAG uses a NumPy k-NN fallback if FAISS is absent, and the AI layer returns deterministic template summaries when `ANTHROPIC_API_KEY` is unset.

---

## Quickstart (Python)

```python
from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.data import load_prices, add_finance_features
from lstm_forecast.transforms import default_finance_transformer

df = load_prices("AAPL", allow_synthetic_fallback=True)          # falls back to synthetic offline
feat = add_finance_features(df, fourier_periods=(5.0,))           # RSI/MACD/Bollinger/Fourier...
exog = feat.drop(columns=["close"])

f = Forecaster(
    y=feat["close"], current_dates=feat.index,
    future_dates=21, test_length=42, exog=exog, name="lstm",
)
transformer, reverter = default_finance_transformer(seasonal_period=5)
pipe = Pipeline(transformer=transformer, reverter=reverter)

result = pipe.fit_predict(f, lags=21, hidden_size=64, epochs=60, alpha=0.1)

print(result.metrics_frame())     # model vs baselines, RMSE-sorted
f.plot(ci=True)                    # history + forecast + 90% interval + test forecast
```

### Retrieval-augmented forecasting

```python
import numpy as np
from lstm_forecast.rag import build_analog_retriever

ref = transformer.fit_transform(f.y, np.arange(f.y.size))
f.attach_retriever(build_analog_retriever(ref, window_len=21, k=8))
result = f.fit_predict(lags=21)    # now conditioned on retrieved analog regimes
```

### Dynamic (backtested) intervals

```python
result = f.fit_predict(lags=21, run_backtest=True, backtest_windows=10)
# result.lower/upper now widen with the horizon, sized from the backtest residual matrix
```

### Transfer learning

```python
target = Forecaster(y=other_series, future_dates=21, test_length=42)
target.transfer_predict(transfer_from=f)   # no retraining
```

### AI insights & tuning (needs `ANTHROPIC_API_KEY`)

```python
from lstm_forecast.ai import generate_insights, suggest_tuning
print(generate_insights(result, label="AAPL"))      # NL narrative (template if no key)
print(suggest_tuning(f.y).candidates)               # structured candidate grid to CV
```

---

## CLI

```bash
lstm-forecast forecast AAPL --features --insights --plot --allow-synthetic
lstm-forecast serve --port 8000          # launch the API
```

---

## REST API

```bash
uvicorn lstm_forecast.api.main:app --port 8000
# open http://localhost:8000/docs  for interactive Swagger
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness + whether AI is enabled + device |
| `POST /forecast` | forecast + conformal intervals + baseline metrics |
| `POST /backtest` | forecast + dynamic (backtested) intervals |
| `POST /insights` | run a forecast, return the AI narrative |
| `POST /chat` | RAG chat grounded in docs + an optional forecast run |
| `POST /transfer` | train on a source series, forecast a target series |

```bash
curl -s localhost:8000/forecast -H 'content-type: application/json' -d '{
  "series": {"ticker": "AAPL", "allow_synthetic": true},
  "horizon": 21, "test_length": 42, "epochs": 40
}' | jq '.best_model, .metrics'
```

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Pick a ticker, run a forecast, inspect the point path with intervals and the test-set
benchmark table, and chat with the Claude-powered assistant about the run.

---

## Docker

```bash
docker compose up --build
# API       → http://localhost:8000/docs
# Dashboard → http://localhost:8501
```

Pass `ANTHROPIC_API_KEY` via your shell or a `.env` file (see `.env.example`).

---

## How it works (the ML edge)

1. **Attention over LSTM states** weights informative lags (toggle for ablation).
2. **Probabilistic heads** — quantile (pinball) outputs, plus **conformal calibration** for marginal coverage guarantees.
3. **Dynamic intervals** from a **rolling-origin backtest residual matrix** — widths grow with the horizon.
4. **RAG analog conditioning** — z-normalised window k-NN gives the model a non-parametric memory of recurring regimes.
5. **Leakage-safe reversible transforms** — detrend / deseason / robust-scale fit on *train only*, auto-reverted (including at future positions).
6. **LLM-assisted tuning** — Claude proposes a structured candidate grid; cross-validation picks the winner.

See [`docs/architecture.md`](docs/architecture.md) for details and [`docs/model_card.md`](docs/model_card.md) for scope, limitations and the no-advice disclaimer.

---

## Development

```bash
pip install -e ".[all]"
pre-commit install
ruff check . && mypy src && pytest      # lint, type-check, test (Claude calls are mocked)
```

Tests run offline with no API key (synthetic data, NumPy RAG fallback, mocked Claude).

## License

MIT — see [LICENSE](LICENSE).
