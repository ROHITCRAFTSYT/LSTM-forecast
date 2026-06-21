# lstm-forecast — Deep Explainer

A complete, ground-up explanation of **what this project is, how every part works, and why
it was built the way it was**. This is the long-form companion to the
[README](README.md) and [architecture doc](docs/architecture.md).

> ⚠️ **Not financial advice.** This is a research/engineering framework. Forecasts are
> uncertain and markets are not guaranteed to be predictable.

---

## Table of contents

1. [What it is, in one paragraph](#1-what-it-is-in-one-paragraph)
2. [The problem & the philosophy](#2-the-problem--the-philosophy)
3. [End-to-end data flow](#3-end-to-end-data-flow)
4. [Data loading & feature engineering](#4-data-loading--feature-engineering)
5. [Reversible, leakage-safe transforms](#5-reversible-leakage-safe-transforms)
6. [The model: windowing, LSTM, attention, heads](#6-the-model-windowing-lstm-attention-heads)
7. [Delta-mode targets — the single most important design choice](#7-delta-mode-targets--the-single-most-important-design-choice)
8. [Training: the Trainer](#8-training-the-trainer)
9. [The Forecaster: the two-model evaluation flow](#9-the-forecaster-the-two-model-evaluation-flow)
10. [Uncertainty: conformal & dynamic intervals](#10-uncertainty-conformal--dynamic-intervals)
11. [Honest benchmarking & the Diebold–Mariano test](#11-honest-benchmarking--the-dieboldmariano-test)
12. [Retrieval-augmented forecasting (RAG)](#12-retrieval-augmented-forecasting-rag)
13. [Ensembling](#13-ensembling)
14. [Cross-validated tuning](#14-cross-validated-tuning)
15. [The provider-agnostic AI layer](#15-the-provider-agnostic-ai-layer)
16. [Transfer learning](#16-transfer-learning)
17. [Model persistence](#17-model-persistence)
18. [Serving: REST API, jobs, model cache](#18-serving-rest-api-jobs-model-cache)
19. [The dashboard](#19-the-dashboard)
20. [Configuration](#20-configuration)
21. [Testing, typing, CI](#21-testing-typing-ci)
22. [Design decisions & trade-offs](#22-design-decisions--trade-offs)
23. [Limitations & honesty](#23-limitations--honesty)
24. [How to extend it](#24-how-to-extend-it)

---

## 1. What it is, in one paragraph

`lstm-forecast` is a Python system for **forecasting financial time series** with a custom
**PyTorch LSTM** (with attention and probabilistic output heads). Around that core it adds
the things that make a forecaster trustworthy and shippable: a leakage-safe reversible
transform pipeline, **conformal prediction intervals** with coverage guarantees, **honest
benchmarking** against classical baselines with a statistical significance test,
**retrieval-augmented forecasting** (a non-parametric memory of recurring patterns),
**ensembling** and **cross-validated tuning**, and a **provider-agnostic LLM layer** for
natural-language insights, a chat assistant, and tuning suggestions. It's exposed three
ways — a pip-installable **library**, a **FastAPI** service, and a **Streamlit** dashboard —
all Dockerized, tested (74 offline tests), type-checked, and CI'd on Python 3.10–3.12.

---

## 2. The problem & the philosophy

The task is **multi-step time-series forecasting**: given a univariate (optionally
multivariate) history, predict the next *H* values. The project is opinionated about three
things most tutorials skip:

- **Uncertainty is the product, not an afterthought.** A point forecast you can't put error
  bars on is close to useless for decisions. Every forecast ships with an interval that has
  a real, checkable coverage property.
- **"Better" must be measured.** It's easy to claim a neural net beats a baseline. We
  actually score the model against naive/drift/seasonal-naive/ARIMA/ETS on held-out data and
  run a statistical test for significance. On near-random-walk data, a *tie* with naive is
  the honest, expected outcome — and the framework says so rather than hiding it.
- **No leakage, ever.** Anything that touches the future during training (fitting a scaler
  on the whole series, peeking at test data) silently inflates results. The transform and
  evaluation design is built specifically to prevent that.

**Layering principle:** the **core library** (`src/lstm_forecast/{data,transforms,models,
forecasting,rag,ai,evaluation}`) contains all logic and never imports FastAPI or Streamlit.
The **serving layers** (`api/`, `dashboard/`) only *consume* the core. This keeps the engine
testable and reusable.

---

## 3. End-to-end data flow

```
load_prices ──► add_finance_features ──► Transformer.fit(train) ──► transform
                                                  │
                       AnalogRetriever.feature_channels (optional RAG)
                                                  ▼
                          make_windows(lags, horizon)  ──►  (X, Y)
                                                  ▼
                LSTMForecaster (LSTM + attention) × ensemble  ──► raw Δ predictions
                                                  ▼
              + anchor (last value)  ──►  Transformer.inverse_transform(future positions)
                                                  ▼
        conformal / dynamic intervals     baselines + Diebold–Mariano     LLM insight
                                                  ▼
                        ForecastResult  ──►  library return / API JSON / dashboard
```

The key idea threaded through everything: **integer positions travel with the values** so
trend/seasonal components can be evaluated at *future* positions, which is what lets a
forecast made in transformed space be inverted back to real prices.

---

## 4. Data loading & feature engineering

**Loaders** (`data/loaders.py`):
- `load_prices(ticker, allow_synthetic_fallback=...)` — pulls daily OHLCV from Yahoo Finance
  (`yfinance`), caches to a local parquet, and **falls back to a deterministic synthetic
  generator** if the provider/network/extra is unavailable. This is why the whole system —
  including CI and the smoke run — works fully offline.
- `load_csv(path)` — your own data.
- `load_synthetic_prices(...)` — a geometric-random-walk + seasonality generator used for
  tests, demos, and the fallback. All loaders return a tidy OHLCV `DataFrame` with a
  `DatetimeIndex` and a guaranteed `close` column.

**Features** (`data/features.py`) — all **causal** (value at time *t* uses only data up to
*t*):
- `log_returns`, `rolling_volatility` (realised vol),
- `rsi` (Wilder's RSI), `macd` (line/signal/histogram), `bollinger` (band width + %B),
- `calendar_features` (cyclically-encoded day-of-week / month),
- `fourier_terms(period, n_harmonics)` for smooth seasonality (e.g. weekly ≈ 5 trading days).

`add_finance_features(df, ...)` bundles these and drops the warmup rows that contain NaNs.
The `close` column is preserved as the forecasting **target**; the rest become exogenous
inputs in multivariate mode.

---

## 5. Reversible, leakage-safe transforms

Financial series are non-stationary (trends, seasonality, changing scale). Neural nets learn
far better on stationary, scaled signals — but then you must **invert** the model's output
back to real prices, *including at future positions the transform was never fit on*. That's
the hard part this module solves.

**The interface** (`transforms/ops.py`): every transform implements

```python
fit(y, t)            # estimate params from TRAINING values y at integer positions t
transform(y, t)      # forward
inverse_transform(y, t)   # invert — valid at ARBITRARY positions t (incl. the future)
```

Passing `t` (integer positions relative to the series start) is the whole trick. For a
detrend, the trend is a polynomial of `t`; to invert a forecast at positions
`n … n+H-1` you simply evaluate that polynomial there and add it back.

**The transforms:**
- `DetrendTransform(poly_order)` — fits `np.polyfit(t, y)`, subtracts/adds `polyval`.
- `DeseasonTransform(period)` — estimates a centered seasonal profile indexed by `t % period`.
- `RobustScaleTransform` — centers by median, scales by IQR (robust to outliers — preferred
  for finance). `StandardScaleTransform` is the mean/std alternative.
- `LogTransform` — for strictly-positive series.
- `DifferenceTransform` — anchored first differences (`inverse = anchor + cumsum`).

**Composition** (`transforms/pipeline.py`): `Transformer([...])` fits each op on the
progressively-transformed training data and applies them in order; `inverse_transform`
applies the inverses in **reverse** order. `Reverter` is a thin handle bound to the same
fitted state. `default_finance_transformer()` returns the recommended stack:
**Detrend → Deseason → RobustScale**.

**Leakage safety:** transforms are fit on the *training slice only*. The `Forecaster`
enforces this (see §9). Every transform has a round-trip test and a future-position
inversion test.

---

## 6. The model: windowing, LSTM, attention, heads

**Windowing** (`models/dataset.py`): `make_windows(features, lags, horizon, target_idx=0)`
slides a window over the `(T, F)` feature matrix (column 0 = target) to produce
`X: (n_samples, lags, F)` and `Y: (n_samples, horizon)`. The model predicts the **whole
horizon at once** (direct multi-step) — this avoids the error accumulation of recursive
one-step-ahead prediction.

**The network** (`models/lstm.py`): `LSTMForecaster` is a stacked `nn.LSTM` followed by
either:
- **additive (Bahdanau) attention** that pools across all time steps with learned weights
  (so the model can emphasise the informative lags), or
- the last hidden state (attention is toggleable for ablation),

then an output **head** (`models/heads.py`):
- `PointHead` → one value per horizon step, or
- `QuantileHead` → several quantiles per step, trained with the **pinball (quantile) loss**.

Output shape is always `(batch, horizon, n_outputs)` where `n_outputs` is 1 (point) or
`len(quantiles)`.

---

## 7. Delta-mode targets — the single most important design choice

A naive LSTM trained to predict **absolute price levels** on a trending/random-walk series
tends to **regress toward the training mean**: it predicts ~67 when the last price was ~78,
and gets crushed by the naive "tomorrow = today" baseline (we observed RMSE 9 vs 2.6 in
early testing).

The fix (`ModelSpec.target_mode="delta"`, the default): the model predicts the **change from
the last observed value of the input window**, not the absolute level. At inference we add
that last value back. Consequences:
- **Naive becomes "predict zero delta"** — the model only has to learn *deviations* from the
  random walk, which is exactly the learnable signal.
- The forecast is **anchored** near the last price, so it's instantly competitive with naive
  and can pull ahead when real structure exists.

This is implemented in `Forecaster._train_models` (subtract the per-window anchor from `Y`)
and `_predict_horizon` (add the anchor back before inverse-transforming). `target_mode="level"`
remains available for series where it makes sense.

---

## 8. Training: the Trainer

`models/trainer.py` (`Trainer` + `TrainerConfig`):
- **Time-aware validation split** — the *latest* `val_fraction` of windows is the validation
  set (never shuffled across the split), so validation always measures forecasting the future.
- **Loss** — MSE for point heads, pinball for quantile heads.
- **Early stopping** — restores the best-val weights; gradient clipping for stability; Adam.
- **Inference** — `predict_point` (median quantile for quantile heads), and
  `predict_mc_dropout` for Monte-Carlo-dropout epistemic samples.

Reproducibility is handled by global seeding (`utils.set_seed`) and device selection
(`utils.resolve_device`: cuda → mps → cpu).

---

## 9. The Forecaster: the two-model evaluation flow

`forecasting/forecaster.py` is the headline API. `fit_predict()` does something subtle and
important — it trains **two** models to keep the benchmark unbiased:

**Stage 1 — test evaluation (unbiased benchmark + calibration).**
- Split off the last `test_length` points as a held-out test set.
- Fit the transformer on the **training portion only** (no leakage).
- Train a model with `horizon = test_length`, forecast the test window, invert to real scale.
- Score the model **and** all baselines on that test set (§11).
- The test residuals (`actual − pred`) double as the **conformal calibration set** (§10).

**Stage 2 — production forecast.**
- Refit the transformer on **all** data, train a fresh model with `horizon = future_dates`,
  and forecast the actual future.

**Stage 3 — intervals.**
- Static conformal intervals from the Stage-1 residuals, or dynamic intervals from a
  backtest (§10).

Everything is returned in a `ForecastResult` (point, lower, upper, future dates, test
arrays, `metrics`, `interval` coverage, `significance`, optional `backtest_result`). It has
`metrics_frame()` (RMSE-sorted model-vs-baselines table) and `to_dict()` (JSON for the API).

---

## 10. Uncertainty: conformal & dynamic intervals

**Why conformal?** Split-conformal prediction gives **finite-sample, distribution-free
marginal coverage**: a 90% interval contains the truth ~90% of the time *regardless* of the
model's internal assumptions, as long as the calibration data is exchangeable with the
future.

**Static intervals** (`forecasting/conformal.py`): the radius is the finite-sample-corrected
`(1−α)(1 + 1/n)` empirical quantile of the absolute calibration residuals. Every horizon
step gets the same radius — a constant-width band.

**Dynamic intervals** (`forecasting/backtest.py`): real forecast error grows with the
horizon, so a flat band is wrong far out. A **rolling-origin backtest** refits the model at
many earlier cutoffs and records the error at each horizon step, producing a residual matrix
of shape `(n_windows, horizon)`. The radius at step *k* is the quantile of
`|residuals[:, k]|`, so the band **widens with the horizon**. (Backtesting refits the model
many times, so it's the slow path — `run_backtest=True`.)

Interval quality is reported via empirical **coverage vs nominal** and **mean width**, and
visualised in the dashboard's calibration plot (`evaluation.calibration_curve`), which sweeps
several nominal levels and plots empirical-vs-nominal coverage against the ideal diagonal.

---

## 11. Honest benchmarking & the Diebold–Mariano test

Every benchmarked run scores the model against five baselines (`forecasting/baselines.py`):
**naive** (last value), **drift** (linear), **seasonal-naive**, **ARIMA**, **ETS**
(the statistical ones fall back to naive if `statsmodels` fails to converge). Metrics
(`evaluation/metrics.py`): RMSE, MAE, MAPE, sMAPE, R², MASE, plus interval coverage/width and
pinball loss.

But a lower RMSE could be luck. So we run the **Diebold–Mariano test**
(`evaluation/significance.py`) comparing the model's errors against naive's:
- loss differential `d = |e_model|² − |e_naive|²`,
- Newey-West-style variance over the horizon lags,
- the **Harvey-Leybourne-Newbold** small-sample correction (test windows are short),
- a two-sided p-value from the t-distribution.

The result (`ForecastResult.significance["vs_naive"]`) says **who won and whether it's
statistically significant**. This is deliberately uncheatable: on random-walk data it
honestly reports `winner=naive`.

---

## 12. Retrieval-augmented forecasting (RAG)

This is the most novel piece. The intuition: *"the recent price shape looks like situations
from the past — what tended to happen next?"*

**Embedding** (`rag/embedder.py`): each historical window of length `window_len` is
**z-normalised** (subtract mean, divide by std). Euclidean distance on z-normalised windows
is monotonic in *shape* dissimilarity, so nearest neighbours are "the same pattern at a
different level/scale." Each window stores its **successor value** (what came next).

**Index** (`rag/store.py`): `AnalogStore` uses **FAISS** (`IndexFlatL2`) when installed, and a
**NumPy brute-force** k-NN fallback otherwise — so RAG works with zero extra dependencies.

**Conditioning** (`rag/retriever.py`): `AnalogRetriever.feature_channels(series)` produces,
for every timestep, two extra channels:
- **channel 0** = mean of the *k* analogs' next values − the window's last value (an expected
  move), and
- **channel 1** = std of those next values (analog disagreement / regime uncertainty).

These are concatenated to the model's input features, giving it a **non-parametric memory of
recurring regimes** alongside the parametric LSTM. Self-matches (distance ≈ 0) are dropped to
avoid trivially leaking the actual next value. The index is built from the **transformed
training series only** (leakage-safe).

---

## 13. Ensembling

`ModelSpec.ensemble = N` trains *N* models with different seeds (`seed + member`) on the same
data and **averages their raw predictions** (before the delta-anchor and inverse-transform).
This reduces variance from random initialisation and produces more stable, better-calibrated
point forecasts. Backtesting forces `ensemble = 1` to keep refit cost bounded.

---

## 14. Cross-validated tuning

Hyperparameter search done right: **the LLM proposes, the data decides.**

- `ai.suggest_tuning(y)` asks the LLM for a small candidate grid (or returns a sensible
  default grid offline) as a validated `TuningSuggestion`.
- `forecasting.tuning.specs_from_suggestion()` turns that into concrete `ModelSpec`s.
- `Forecaster.tune(specs)` runs **walk-forward cross-validation** (`walk_forward_cv`): each
  candidate is evaluated on several expanding-window folds (epochs capped for affordability),
  and the lowest mean-CV-RMSE candidate is adopted as `self.spec`.

So the LLM only narrows the search space; cross-validation makes the final call. This is why
the model's own test metric is *always* computed (even with `benchmark=False`) — CV depends
on it.

---

## 15. The provider-agnostic AI layer

The AI layer (`ai/`) is **optional and pluggable**. A single chokepoint, `AIClient`
(`ai/client.py`), delegates to an `LLMProvider` (`ai/providers.py`):

| Provider | Backend |
| --- | --- |
| `anthropic` (default) | Claude via the Anthropic SDK (adaptive thinking + effort) |
| `openai` | OpenAI Chat Completions |
| `openai_compatible` | any OpenAI-style endpoint (OpenRouter, Together, Groq, vLLM) via `base_url` |
| `ollama` | local models, keyless, OpenAI-compatible at `localhost:11434/v1` |
| `google` | Gemini via `google-generativeai` |

Providers implement only `complete` / `stream` over a normalised message format. Two things
are made **uniform across providers** in the client itself:
- **Structured output** (`parse`) — instead of provider-specific structured APIs, the client
  prompts for JSON matching a Pydantic schema, extracts it (tolerating ``` fences/prose), and
  validates — retrying once with the error fed back. Works identically everywhere.
- **The chat assistant** (`ai/assistant.py`) uses a portable **retrieve-then-answer** pattern
  (TF-IDF doc retrieval via `ai/doc_index.py` + the current run's metrics/forecast as
  context), so it needs no provider-specific tool-calling.

**Three features** sit on top: `generate_insights` (NL forecast narrative, streaming),
`suggest_tuning` (structured grid), and `ChatAssistant` (grounded Q&A). **All degrade
gracefully**: with no key (or no SDK) `AIClient.available` is `False` and each feature returns
a deterministic template/fallback. Selection is pure config — no code change. Keys are
auto-detected from `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` or the explicit
`LSTM_FORECAST_AI__API_KEY`; Ollama needs none.

---

## 16. Transfer learning

`Forecaster.transfer_predict(transfer_from=other)` applies a model trained on one series to
**new data with no retraining** — two scenarios: fresh data from the same series, or a
different-but-similar series. It reuses the source's fitted models, transformer, and
calibration residuals (for the interval radius). Useful when you have a short series and a
well-fit model on a longer, related one.

---

## 17. Model persistence

Training every time is wasteful. `Forecaster.save(path)` serialises (via `torch.save`):
the ensemble's weights + constructor args, the fitted transformer, the spec, the calibration
residuals, and the series/exog history. `Forecaster.load(path)` reconstructs everything and
`forecast_future()` produces a forecast with **no retraining** (intervals reuse the stored
residuals). A round-trip test asserts the reloaded model reproduces the original forecast.

---

## 18. Serving: REST API, jobs, model cache

**FastAPI** (`api/`) with a clean service layer (`api/service.py`) — routes contain no
business logic:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness + AI provider + device |
| `POST /forecast` | forecast + conformal intervals + baseline metrics + significance |
| `POST /backtest` | forecast + dynamic (backtested) intervals |
| `POST /insights` | run a forecast, return the NL narrative |
| `POST /chat` | RAG chat grounded in docs + an optional run |
| `POST /transfer` | train on a source series, forecast a target |
| `POST /jobs/forecast` → `GET /jobs/{id}` | async training as a background job |

**Async jobs** (`api/jobs.py`): an in-process `JobManager` (a `ThreadPoolExecutor` + a
lock-guarded dict) so long trainings don't block the request — submit returns a `job_id`,
poll for the result. **Model cache** (`service.run_forecast_cached`): fitted `Forecaster`s
are cached by a hash of the training-relevant request fields, so repeated identical requests
skip retraining and go straight to `forecast_future()`. Both are **in-process / single-worker
by design** — documented as such; a production deployment would back them with Redis +
Celery/RQ and persist models to disk.

---

## 19. The dashboard

`dashboard/app.py` (Streamlit) is a thin UI over the library: pick a ticker and parameters,
run a forecast, and see the interval plot (Plotly), the **benchmark table** (model vs
baselines), the **interval coverage** metric, a **calibration/reliability plot**, an AI
**provider selector** (switch Claude/OpenAI/Gemini/Ollama live), the **AI insight**, and a
**chat** panel grounded in the run. No business logic lives here.

---

## 20. Configuration

`config.py` (pydantic-settings) — everything is environment-driven (prefix
`LSTM_FORECAST_`, nested with `__`) and **everything is optional**. Key knobs: the AI
`provider`/`api_key`/`model`/`base_url`/`effort`, the torch `device`, the global `seed`, the
cache directory, and API host/port/CORS. Defaults make the system run with zero config (and
no AI key). See [`.env.example`](.env.example).

---

## 21. Testing, typing, CI

- **74 tests** (`tests/`) run **fully offline**: synthetic data, the NumPy RAG fallback, and
  **mocked LLM calls** (no key, no network). Coverage includes transform round-trips, an
  overfit-tiny-batch sanity check, conformal coverage, backtest shapes, RAG channel shapes &
  causality, metric correctness, the DM test, tuning/CV, save/load round-trip, and the API
  (via `TestClient`).
- **Types:** `mypy src` is clean (the project is fully type-hinted). A real CI catch: numpy's
  stricter type stubs on the 3.10 runner flagged a shape-typed array assignment — fixed by
  broadening an annotation.
- **Lint/format:** `ruff` (lint + import order + format).
- **CI** (`.github/workflows/ci.yml`): lint + type + test on Python **3.10 / 3.11 / 3.12**,
  then builds the Docker images. A separate workflow deploys the docs site.

---

## 22. Design decisions & trade-offs

- **Direct multi-horizon vs recursive** → direct (predict all *H* at once) avoids compounding
  one-step errors, at the cost of a fixed horizon per trained model (hence the two-model flow).
- **Delta targets vs level targets** → delta by default; without it the model loses to naive
  (§7).
- **Conformal vs parametric intervals** → conformal for distribution-free coverage guarantees,
  with a backtested dynamic variant for horizon-aware widths.
- **Provider-agnostic AI via prompted JSON** → uniform structured output across providers at
  the cost of not using each provider's native structured-output API (a deliberate
  portability trade).
- **In-process jobs/cache** → simplest correct thing; explicitly not a distributed queue.
- **Synthetic fallback everywhere** → the whole system is runnable and testable offline with
  no keys; demos that show the model "winning" use a *structured* synthetic series and say so.

---

## 23. Limitations & honesty

- **Markets are close to a random walk.** On raw daily prices, naive is near-optimal; a tie
  is a good result. The framework measures this per-series rather than overclaiming.
- **Conformal coverage is marginal** and assumes exchangeability — it degrades under regime
  shifts.
- **RAG analogs are shape-based** and can retrieve spurious matches in low-signal regimes.
- **The LLM layer is explanatory, not authoritative** — it can produce plausible-but-wrong
  narratives; it only ever sees a numeric summary + retrieved docs.
- **Not financial advice**, and no live trading/execution. See [`docs/model_card.md`](docs/model_card.md).

---

## 24. How to extend it

- **New transform** → subclass `SeriesTransform` (implement `fit/transform/inverse_transform`),
  add a round-trip + future-position test.
- **New baseline** → implement the `BaseForecaster` protocol, add it to `baseline_registry`.
- **New metric** → add to `evaluation/metrics.py` with a correctness test.
- **New LLM provider** → add an `LLMProvider` in `ai/providers.py` and wire it into
  `build_provider`.
- **New model architecture** → it only needs to consume `(X, Y)` windows and return
  `(batch, horizon, n_outputs)`; swap it in `Forecaster._train_models`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup and the quality gate
(`ruff` + `mypy` + `pytest`).
