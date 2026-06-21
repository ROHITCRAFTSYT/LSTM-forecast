# Architecture

```
data ingest → feature engineering → reversible transform pipeline
     → [RAG analog retrieval] ⟶ feature fusion ⟶ PyTorch LSTM(+attention, quantile/conformal head)
     → backtest / conformal intervals → benchmark vs baselines
     → Claude: insights · chat (RAG over docs+results) · tuning suggestions
Surfaces:  Python API   ·   FastAPI service   ·   Streamlit dashboard
```

The **core library** (`src/lstm_forecast`) is consumed by the API and dashboard. No business
logic lives in the serving layers.

## Modules

| Package | Responsibility |
| --- | --- |
| `data` | loaders (yfinance/CSV/synthetic) + causal finance features |
| `transforms` | reversible, leakage-safe `SeriesTransform` ops + `Transformer`/`Reverter` |
| `models` | windowing, LSTM(+attention), point/quantile heads, `Trainer` |
| `forecasting` | `Forecaster` (public API), baselines, conformal, backtest |
| `rag` | window embedder, vector `AnalogStore`, `AnalogRetriever` |
| `ai` | Claude client, insights, structured tuner, doc index, chat assistant |
| `evaluation` | point / interval / probabilistic metrics |
| `api` | FastAPI service (schemas, service layer, routers) |

## Key design decisions

### Delta-mode targets
The model predicts the **change from the last observed value** rather than the absolute
level. On near-random-walk financial series this anchors the forecast to the naive baseline
(predict-zero-delta) and lets the model learn only the deviations — without it, the LSTM
regresses toward the training mean and underperforms naive badly. Toggle via
`ModelSpec.target_mode = "level" | "delta"`.

### Leakage-safe transforms with positional inversion
Every transform takes integer positions `t` alongside values, so trend/seasonal components
are evaluated correctly at **future** positions. Transforms are fit on the training split
only; the `Forecaster` refits on the full series for the production forecast and inverts
predictions at their true future positions.

### Two-model evaluation flow
`fit_predict` trains (1) a *test-evaluation* model (transformer fit on train only) used for
the honest baseline benchmark and conformal calibration, then (2) a *production* model
(transformer refit on all data) for the future forecast. This keeps the benchmark unbiased.

### Conformal intervals
Static intervals use the finite-sample-corrected quantile of absolute test residuals.
Dynamic intervals come from a rolling-origin backtest: the residual matrix
`(n_windows, horizon)` gives a per-step radius, so widths grow with the horizon.

### RAG analog conditioning
Windows are z-normalised so Euclidean nearest-neighbours are shape-matches. For each
timestep, the retriever summarises *what followed* the k most similar historical windows
into two extra channels (expected move + analog dispersion), concatenated to the model
input. Self-matches are dropped to avoid trivial leakage.

### Provider-agnostic AI integration
A single client (`ai/client.py`) wraps a pluggable `LLMProvider` (`ai/providers.py`):
Anthropic (default), OpenAI, Google Gemini, Ollama (local), or any OpenAI-compatible
endpoint. Providers implement only `complete`/`stream` over a normalised message format;
structured output (`parse`) is implemented once in the client via JSON-schema prompting, and
the chat assistant uses a portable retrieve-then-answer pattern — so every feature behaves
identically across providers. Selection is via `LSTM_FORECAST_AI__PROVIDER`. Every caller
checks `client.available` and falls back to deterministic behaviour when no provider is
configured, so the system is never blocked on the LLM.

### Model persistence
`Forecaster.save()` serialises the fitted ensemble weights, the fitted transformer, the
spec, and the calibration residuals; `Forecaster.load()` + `forecast_future()` reproduce
forecasts with no retraining — used to avoid retraining on every API request.
