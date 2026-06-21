# Quickstart

## Install

```bash
pip install -e ".[all]"     # or pick extras: data, rag, ai, api, dashboard
```

Everything runs offline with no API keys (synthetic data fallback, NumPy RAG fallback,
template AI summaries).

## Forecast in Python

```python
from lstm_forecast import Forecaster, Pipeline
from lstm_forecast.data import load_prices, add_finance_features
from lstm_forecast.transforms import default_finance_transformer

df = load_prices("AAPL", allow_synthetic_fallback=True)
feat = add_finance_features(df, fourier_periods=(5.0,))

f = Forecaster(
    y=feat["close"], current_dates=feat.index,
    future_dates=21, test_length=42, exog=feat.drop(columns=["close"]),
)
transformer, reverter = default_finance_transformer(seasonal_period=5)
result = Pipeline(transformer=transformer, reverter=reverter).fit_predict(
    f, lags=21, epochs=60, alpha=0.1
)

print(result.metrics_frame())   # model vs baselines, RMSE-sorted
f.plot(ci=True)
```

## The five capabilities

Each has a runnable example under `examples/`:

```bash
python examples/01_univariate.py AAPL
python examples/02_multivariate.py AAPL
python examples/03_probabilistic.py AAPL
python examples/04_dynamic_probabilistic.py AAPL
python examples/05_transfer_learning.py AAPL MSFT
python examples/06_rag_and_ai.py AAPL          # RAG + Claude insights
```

## CLI

```bash
lstm-forecast forecast AAPL --features --insights --plot --allow-synthetic
lstm-forecast serve --port 8000
```

## API & dashboard

```bash
uvicorn lstm_forecast.api.main:app --port 8000   # docs at /docs
streamlit run dashboard/app.py
# or both:
docker compose up --build
```

Forecasts can run synchronously (`POST /forecast`) or as a background job
(`POST /jobs/forecast` → `job_id`, then poll `GET /jobs/{id}`); fitted models are cached
in-process so repeated identical requests skip retraining. The dashboard adds an AI provider
selector and a calibration (reliability) plot of empirical vs nominal interval coverage.

## Enable Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Then `generate_insights`, the `/chat` endpoint, and `suggest_tuning` use Claude
(`claude-opus-4-8`, adaptive thinking). Without the key they return deterministic fallbacks.
