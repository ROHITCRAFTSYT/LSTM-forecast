# lstm-forecast

**Production-grade LSTM time-series forecasting for finance — enhanced with retrieval-augmented forecasting (RAG), Claude-powered insights, and rigorous probabilistic intervals.**

[![CI](https://github.com/ROHITCRAFTSYT/lstm-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/ROHITCRAFTSYT/lstm-forecast/actions/workflows/ci.yml)
[![Docs](https://github.com/ROHITCRAFTSYT/lstm-forecast/actions/workflows/docs.yml/badge.svg)](https://github.com/ROHITCRAFTSYT/lstm-forecast/actions/workflows/docs.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)
![Tests](https://img.shields.io/badge/tests-66%20passing-brightgreen)
![Lint](https://img.shields.io/badge/lint-ruff-261230)
![Types](https://img.shields.io/badge/types-mypy-2a6db2)
![License](https://img.shields.io/badge/license-MIT-green)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

This project reimagines the classic ["LSTM Model for Time Series, with Code"](https://towardsdatascience.com/) scalecast walkthrough as a **custom PyTorch system** you can actually ship: a clean library, a REST API, and a dashboard — all Dockerized, tested, and documented.

<p align="center">
  <img src="assets/forecast_example.png" alt="LSTM forecast with conformal prediction intervals and the test-set forecast tracking the actual series" width="92%">
</p>
<p align="center"><em>Forecast with a 90% conformal interval; the dashed line is the held-out test forecast tracking reality. Regenerate with <code>python scripts/make_readme_assets.py</code>.</em></p>

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
- 🤖 **Provider-agnostic AI layer** — natural-language forecast insights, a RAG **chat assistant** grounded in your docs + run results, and **LLM-assisted hyperparameter tuning** (the LLM proposes, cross-validation decides). Use **any model**: Claude (default), OpenAI, Gemini, local **Ollama**, or any OpenAI-compatible endpoint.
- 📊 **Honest benchmarking** — every run is scored against Naive / Drift / Seasonal-Naive / ARIMA / ETS baselines on a held-out test set, with a **Diebold–Mariano significance test** vs naive, so "it beats the baselines" is *statistically measured*, not asserted.
- 🎛️ **Cross-validated tuning + ensembling** — the AI-proposed grid is evaluated by walk-forward CV (`Forecaster.tune`), and forecasts can average a multi-seed **ensemble** for robustness.

> ⚠️ **Not financial advice.** This is a research/engineering framework. Forecasts are uncertain; markets are not guaranteed to be predictable. See the [model card](docs/model_card.md).

---

## Contents

- [Architecture](#architecture)
- [Results — measured, not asserted](#results--measured-not-asserted)
- [Install](#install)
- [Quickstart (Python)](#quickstart-python)
- [CLI](#cli) · [REST API](#rest-api) · [Dashboard](#dashboard) · [Docker](#docker)
- [How it works (the ML edge)](#how-it-works-the-ml-edge)
- [Configuration](#configuration) · [Project structure](#project-structure)
- [Development](#development) · [Contributing](#contributing) · [License](#license)

---

## Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="Architecture: data → features → transforms → RAG retrieval → LSTM+attention → intervals/benchmark/Claude → library/API/dashboard surfaces" width="96%">
</p>

A clean **core library** (`src/lstm_forecast`) is consumed by the API and dashboard — no business logic lives in the serving layers. See [docs/architecture.md](docs/architecture.md) for the design decisions (delta-mode targets, leakage-safe positional inversion, the two-model evaluation flow, RAG conditioning).

---

## Results — measured, not asserted

On a demo series **with genuine structure** (trend + seasonality + autocorrelation), the LSTM clearly beats every baseline. The same harness scores the model on *your* data, so you always see whether it actually wins:

<p align="center">
  <img src="assets/benchmark_example.png" alt="Test-set RMSE bar chart: the LSTM beats naive, drift, seasonal-naive, ARIMA and ETS" width="70%">
</p>

| model | RMSE | MAE | MASE | R² |
| --- | ---: | ---: | ---: | ---: |
| **lstm** | **1.02** | **0.89** | **0.33** | **0.95** |
| ets / drift | 5.38 | 4.47 | 1.64 | −0.35 |
| naive | 5.81 | 4.70 | 1.72 | −0.58 |
| seasonal_naive | 7.33 | 6.07 | 2.23 | −1.51 |
| arima | 7.41 | 6.12 | 2.25 | −1.56 |

> **The honest caveat:** on a *near-random-walk* series (e.g. raw daily stock prices) naive is close to optimal and a tie is a good result — markets are hard. This framework is built to *measure* that per-series rather than overclaim. See the [model card](docs/model_card.md).

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

### Use any LLM provider

The AI layer is provider-agnostic — pick one via environment variables (no code change):

```bash
# Claude (default)
export LSTM_FORECAST_AI__PROVIDER=anthropic   ANTHROPIC_API_KEY=sk-ant-...
# OpenAI / OpenAI-compatible (OpenRouter, Together, Groq, vLLM)
export LSTM_FORECAST_AI__PROVIDER=openai       OPENAI_API_KEY=sk-...
# Google Gemini
export LSTM_FORECAST_AI__PROVIDER=google       GOOGLE_API_KEY=...
# Local Ollama (no key)
export LSTM_FORECAST_AI__PROVIDER=ollama       LSTM_FORECAST_AI__MODEL=llama3.1
```

Install the matching extra: `pip install -e ".[ai]"` (Claude), `".[ai-openai]"` (OpenAI/Ollama/compatible), `".[ai-google]"` (Gemini), or `".[ai-all]"`.

### Cross-validated tuning, ensembling & persistence

```python
from lstm_forecast import Forecaster
from lstm_forecast.forecasting.forecaster import ModelSpec
from lstm_forecast.forecasting.tuning import specs_from_suggestion

specs = specs_from_suggestion(suggest_tuning(f.y))      # LLM proposes a grid
report = f.tune(specs, k=3)                              # walk-forward CV picks the winner
result = f.fit_predict(ModelSpec(lags=21, ensemble=3))  # average a 3-model ensemble

f.save("artifacts/aapl.pt")                             # persist the fitted ensemble
later = Forecaster.load("artifacts/aapl.pt")
later.forecast_future()                                 # forecast again — no retraining
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
| `POST /jobs/forecast` | submit a forecast to run in the background → returns a `job_id` |
| `GET /jobs/{id}` | poll a background job's status and (when done) its forecast result |

```bash
curl -s localhost:8000/forecast -H 'content-type: application/json' -d '{
  "series": {"ticker": "AAPL", "allow_synthetic": true},
  "horizon": 21, "test_length": 42, "epochs": 40
}' | jq '.best_model, .metrics'
```

Fitted models are kept in an in-process cache keyed by the training-relevant request
fields, so repeating an identical request skips retraining and reuses the saved model.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Pick a ticker and an AI provider, run a forecast, inspect the point path with intervals,
the test-set benchmark table, and a **calibration (reliability) plot** of empirical vs
nominal interval coverage, and chat with the AI assistant about the run.

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
6. **LLM-assisted tuning, CV-decided** — Claude proposes a structured candidate grid; **walk-forward cross-validation** (`Forecaster.tune`) picks the winner.
7. **Ensembling** — average `ensemble=N` differently-seeded models to cut initialisation variance.
8. **Statistical rigor** — a **Diebold–Mariano test** reports whether the model's accuracy edge over naive is significant, not just numerically lower.
9. **Calibration check** — a `calibration_curve` metric measures empirical vs nominal interval coverage (and a mean calibration error), so interval *honesty* is quantified, not assumed.

See [`docs/architecture.md`](docs/architecture.md) for details and [`docs/model_card.md`](docs/model_card.md) for scope, limitations and the no-advice disclaimer.

---

## Configuration

All settings are optional and read from the environment (prefix `LSTM_FORECAST_`, nested with `__`) or a `.env` file (see [`.env.example`](.env.example)). **Nothing here is required** — the forecasting core and RAG retrieval run with zero configuration.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LSTM_FORECAST_AI__PROVIDER` | `anthropic` | LLM provider: `anthropic \| openai \| google \| ollama \| openai_compatible`. |
| `LSTM_FORECAST_AI__API_KEY` | _(empty)_ | Provider API key. `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` are also auto-detected. Empty → graceful fallbacks (and not required for `ollama`). |
| `LSTM_FORECAST_AI__MODEL` | _(provider default)_ | Model id, e.g. `claude-opus-4-8`, `gpt-4o`, `gemini-1.5-pro`, `llama3.1`. |
| `LSTM_FORECAST_AI__BASE_URL` | _(empty)_ | Endpoint for `openai_compatible` / `ollama` (e.g. `http://localhost:11434/v1`). |
| `LSTM_FORECAST_AI__EFFORT` | `high` | Reasoning effort (Anthropic): `low \| medium \| high \| max`. |
| `LSTM_FORECAST_DEVICE` | `auto` | torch device: `auto \| cpu \| cuda \| mps`. |
| `LSTM_FORECAST_SEED` | `20` | Global RNG seed (Python / NumPy / torch). |
| `LSTM_FORECAST_CACHE_DIR` | `.cache` | Where data downloads & artifacts are cached. |
| `LSTM_FORECAST_API__PORT` | `8000` | API port. |
| `LSTM_FORECAST_API__CORS_ORIGINS` | `*` | Comma-separated CORS origins. |

---

## Project structure

```
src/lstm_forecast/
├── data/          # loaders (yfinance/CSV/synthetic) + causal finance features
├── transforms/    # reversible, leakage-safe transforms (Detrend/Deseason/Scale/…)
├── models/        # windowing, LSTM(+attention), point/quantile heads, Trainer
├── forecasting/   # Forecaster (public API), baselines, conformal, backtest
├── rag/           # window embedder, vector AnalogStore, AnalogRetriever
├── ai/            # Claude client, insights, structured tuner, doc index, chat assistant
├── evaluation/    # point / interval / probabilistic metrics
├── api/           # FastAPI service (schemas, service layer, routers)
├── pipelines.py   # Transform → Forecast → Revert composer
├── config.py      # pydantic-settings configuration
└── cli.py         # `lstm-forecast` command-line interface
dashboard/         # Streamlit app          docker/      # API + dashboard images
examples/          # 6 runnable demos        docs/        # mkdocs site + model card
tests/             # 66 tests (offline)      scripts/     # README asset generation
```

---

## Development

```bash
pip install -e ".[all]"
pre-commit install
ruff check . && mypy src && pytest      # lint, type-check, test (Claude calls are mocked)
```

Tests run offline with no API key (synthetic data, NumPy RAG fallback, mocked Claude).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, the
quality gate (ruff + mypy + pytest), and conventions (causal/leakage-safe transforms,
framework-agnostic core, AI always optional, honest benchmarking). Good first issues:
new transforms, baselines, data loaders, or additional finance features.

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Built as a production-grade reimagining of the scalecast "LSTM for Time Series" walkthrough — custom PyTorch core, retrieval-augmented forecasting, and a Claude AI layer. Forecasts are uncertain and not financial advice.</sub>
